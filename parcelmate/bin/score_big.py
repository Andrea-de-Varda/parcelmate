"""Scoring for trees whose connectivity is tiled float16 (LOG.md Iteration 25).

Same rows as `score.py` (tree, variant, metric, fit, eval, value), computed so that each
175 GB half is read as few times as possible:

  within domain, per domain     one pass over half A for the block means of every partition
                                (real, pnull, rand), one pass over half B for every
                                prediction and the ceiling (r2 and r), one more pass over
                                half A for the in-sample fidelities.
  across domains, per fit domain the block means from the within pass are reused; one pass
                                over each evaluation domain's half B scores every partition
                                and the ceiling (r).

Reliability (ARI between partitions), the restart-split ceiling and the triviality metrics
need no matrix at all; hubness uses the row sums the half stores. Not computed here:
`reliability_within_coassoc` (its N x N counts are 174 GB at 295k units) and the `null`
tree (the pipeline scored on shifted data; its connectivity is purged after the null
parcellation, and the tree was kept for the record only). Every absent input is collected
and the run refuses to write a partial table, as `score.py` does.
"""

import csv
import os

import numpy as np

from parcelmate.bigconn import TiledMatrix, is_tiled
from parcelmate.bin.score import conn_path, parc_path, random_partition, wants_across
from parcelmate.constants import HALF_NAMES, OUTPUT_DIR
from parcelmate.metrics import (
    hard_labels, reliability, reliability_ceiling, stream_agreements, stream_block_means,
    triviality,
)
from parcelmate.util import derive_seed, load_h5_data, read_attrs, stderr, variants_tag


def tree_is_tiled(cfg):
    """Whether the config's real tree holds tiled halves (decided from the first file found)."""
    root = cfg.get('output_dir', OUTPUT_DIR)
    for domain in cfg.get('connectivity', {}).get('domains', []):
        p = conn_path(root, domain, HALF_NAMES[0])
        if os.path.exists(p):
            return is_tiled(p)
    return False


def load_parc(path):
    d = load_h5_data(path, verbose=False)
    return d


def score_config_big(cfg, out=None, cross_domain=True, allow_partial=False, verbose=True,
                     variants=None):
    root = cfg.get('output_dir', OUTPUT_DIR)
    null_root = root.rstrip('/') + '_null'
    all_variants = sorted((cfg.get('parcellation_variants') or {'default': {}}).keys())
    if variants:
        unknown = [v for v in variants if v not in all_variants]
        assert not unknown, 'unknown variant(s) %s; config has %s' % (unknown, all_variants)
        variants = sorted(variants)
    else:
        variants = all_variants
    score_cfg = cfg.get('score') or {}
    domains = list(score_cfg.get('domains') or cfg.get('connectivity', {}).get('domains') or [])
    seed = cfg.get('seed', 0) or 0
    assert domains, 'No domains in the config; nothing to score'
    pairs = None
    if score_cfg.get('across_pairs') is not None:
        pairs = {(str(a), str(b)) for a, b in score_cfg['across_pairs']}
    stderr('Scoring (tiled) %d variant(s) over %d domain(s)\n' % (len(variants), len(domains)))

    rows, missing = [], []

    def row(tree, variant, metric, fit, eval_, value):
        rows.append(dict(tree=tree, variant=variant, metric=metric, fit=fit, eval=eval_, value=float(value)))

    # Partitions per (domain, half): real, and the references. Loaded once.
    parcs = {}
    for domain in domains:
        for variant in variants:
            for key in HALF_NAMES:
                fr = parc_path(root, variant, domain, key)
                fn = parc_path(null_root, variant, domain, key)
                if not os.path.exists(fr):
                    missing.append('real/%s/%s: parcellated %s' % (variant, domain, key))
                    continue
                if not os.path.exists(fn):
                    missing.append('pnull/%s/%s: null-tree parcellation of %s' % (variant, domain, key))
                    continue
                d = load_parc(fr)
                dn = load_parc(fn)
                k = int(read_attrs(fr).get('n_networks', d['parcellation'].shape[1]))
                n_units = d['parcellation'].shape[0]
                parcs[(domain, variant, key)] = dict(
                    real=d, null=dn['parcellation'], k=k, n_units=n_units,
                    rand_a=random_partition(n_units, k, derive_seed(seed, 'rand_partition', variant, domain, 'a')),
                    rand_b=random_partition(n_units, k, derive_seed(seed, 'rand_partition', variant, domain, 'b')))

    block_means = {}   # (domain, variant) -> {name: (M, labels)} on real half A
    for domain in domains:
        pa, pb = conn_path(root, domain, HALF_NAMES[0]), conn_path(root, domain, HALF_NAMES[1])
        if not (os.path.exists(pa) and os.path.exists(pb)):
            missing.append('real/%s: tiled split-half connectivity' % domain)
            continue
        Ra, Rb = TiledMatrix(pa), TiledMatrix(pb)
        # 1. block means on half A, every partition of every variant, one pass
        parts = {}
        for variant in variants:
            a = parcs.get((domain, variant, HALF_NAMES[0]))
            if a is None:
                continue
            parts[(variant, 'real')] = a['real']['parcellation']
            parts[(variant, 'pnull')] = a['null']
            parts[(variant, 'rand')] = a['rand_a']
        if not parts:
            continue
        if verbose:
            stderr('  %-12s block means on half A (%d partitions)\n' % (domain, len(parts)))
        means = stream_block_means(Ra, parts)
        block_means[domain] = means
        # 2. held-out on half B, plus the ceiling, one pass
        if verbose:
            stderr('  %-12s held-out fidelity on half B\n' % domain)
        held = stream_agreements(Rb, means, ceiling=Ra)
        for metric, measure in (('fidelity_within', 'r2'), ('fidelity_within_r', 'r')):
            row('real', '(ceiling)', metric, domain, domain, held['(ceiling)'][measure])
            for key, v in held.items():
                if key == '(ceiling)':
                    continue
                variant, tree = key
                row(tree, variant, metric, domain, domain, v[measure])
        # 3. in-sample on half A, one pass
        if verbose:
            stderr('  %-12s in-sample fidelity on half A\n' % domain)
        ins = stream_agreements(Ra, means)
        for (variant, tree), v in ins.items():
            row(tree, variant, 'fidelity_within_insample', domain, domain, v['r2'])
        # 4. matrix-free metrics
        for variant in variants:
            a, b = parcs.get((domain, variant, HALF_NAMES[0])), parcs.get((domain, variant, HALF_NAMES[1]))
            if a is None or b is None:
                continue
            Pa, Pb = a['real']['parcellation'], b['real']['parcellation']
            row('real', variant, 'reliability_within', domain, domain, reliability(Pa, Pb))
            row('pnull', variant, 'reliability_within', domain, domain,
                (reliability(a['null'], Pb) + reliability(Pa, b['null'])) / 2.0)
            row('rand', variant, 'reliability_within', domain, domain,
                (reliability(a['rand_a'], Pb) + reliability(Pa, b['rand_b'])) / 2.0)
            da = a['real']
            if 'parcellation_split1' in da and 'parcellation_split2' in da:
                row('real', variant, 'reliability_ceiling', domain, domain,
                    reliability_ceiling(da['parcellation_split1'], da['parcellation_split2']))
            else:
                missing.append('real/%s/%s: restart-split consensuses' % (variant, domain))
            for tree, P in (('real', Pa), ('pnull', a['null']), ('rand', a['rand_a'])):
                for name, value in triviality(P, Ra.coordinates, strength=Ra.strength).items():
                    row(tree, variant, 'triviality_%s' % name, domain, domain, value)
            if verbose:
                stderr('  %-12s %-14s done\n' % (domain, variant))
        Ra.close()
        Rb.close()

    if cross_domain:
        for fit_domain in domains:
            if fit_domain not in block_means or not wants_across(pairs, fit_domain):
                continue
            pa = conn_path(root, fit_domain, HALF_NAMES[0])
            Ra = TiledMatrix(pa)
            for eval_domain in domains:
                if eval_domain == fit_domain or not wants_across(pairs, fit_domain, eval_domain):
                    continue
                pb = conn_path(root, eval_domain, HALF_NAMES[1])
                if not os.path.exists(pb):
                    continue
                Rb = TiledMatrix(pb)
                if verbose:
                    stderr('  across %s -> %s\n' % (fit_domain, eval_domain))
                res = stream_agreements(Rb, block_means[fit_domain], ceiling=Ra)
                row('real', '(ceiling)', 'fidelity_across_halves', fit_domain, eval_domain, res['(ceiling)']['r'])
                for key, v in res.items():
                    if key == '(ceiling)':
                        continue
                    variant, tree = key
                    row(tree, variant, 'fidelity_across_halves', fit_domain, eval_domain, v['r'])
                for variant in variants:
                    a = parcs.get((fit_domain, variant, HALF_NAMES[0]))
                    b = parcs.get((eval_domain, variant, HALF_NAMES[1]))
                    if a is None or b is None:
                        continue
                    Pa, Pb = a['real']['parcellation'], b['real']['parcellation']
                    row('real', variant, 'reliability_across_halves', fit_domain, eval_domain, reliability(Pa, Pb))
                    row('pnull', variant, 'reliability_across_halves', fit_domain, eval_domain,
                        (reliability(a['null'], Pb) + reliability(Pa, b['null'])) / 2.0)
                    row('rand', variant, 'reliability_across_halves', fit_domain, eval_domain,
                        (reliability(a['rand_a'], Pb) + reliability(Pa, b['rand_b'])) / 2.0)
                Rb.close()
            Ra.close()

    if missing:
        stderr('\n%d missing input(s):\n' % len(missing))
        for m in missing[:40]:
            stderr('  - %s\n' % m)
        if not allow_partial:
            raise SystemExit('\nRefusing to write partial results.')
    default_name = 'scores.csv' if variants == all_variants else 'scores_%s.csv' % variants_tag(variants)
    out_path = out or os.path.join(root, 'metrics', default_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['tree', 'variant', 'metric', 'fit', 'eval', 'value'])
        w.writeheader()
        w.writerows(rows)
    stderr('\nWrote %d rows to %s\n' % (len(rows), out_path))
    return rows, out_path
