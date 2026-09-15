"""Score parcellation variants for reliability and fidelity against three references.

    python -m parcelmate.bin.score configs/ladder.yml

Reads the real and null trees produced by the connectivity + split_halves + parcellation
steps and writes a tidy CSV plus a printed summary. Every measurement is written out
individually -- one row per tree, variant, metric and domain pair, unnormalized -- so that
plotting and any derived quantity happen downstream rather than being baked in here.

Four trees appear in the output:

  real   the pipeline on real data.
  null   the pipeline on circularly shifted data, scored on shifted data. The original
         design. Kept for the record, but not directly comparable to `real`: R^2 is a ratio
         and the two sit on different denominators, which is how a structureless matrix
         once out-scored real data (LOG.md Iteration 9).
  pnull  the null PARTITION evaluated on REAL data. For fidelity the partition is fit on
         shifted half A, its block means are estimated on REAL half A, and it predicts REAL
         half B -- same target, same denominator, so real - pnull is the credit the
         clustering earns beyond what a partition carrying only per-unit properties
         (autocorrelation, hubness) earns. For reliability, which is symmetric in the two
         halves, it is the mean of ARI(P_null_A, P_real_B) and ARI(P_real_A, P_null_B):
         both are "one real parcellation against one null parcellation" and averaging them
         removes an arbitrary choice of which half carries the shift. This is the primary
         comparison (LOG.md Iteration 12).
  rand   a seeded random partition of the same k, evaluated on real data: what "any 50
         blocks" gets, so that pnull's contribution above it is attributable to per-unit
         structure rather than to block-model capacity. A collapsed pnull partition (few
         blocks filled) has less capacity than a full one, and `triviality_n_effective_
         networks` under the pnull tree says how much.

Both real metrics get a ceiling, partialling out different nuisances. The fidelity ceiling
predicts held-out connectivity from the fitting matrix directly, removing the model
limitation to expose sampling noise between halves. The reliability ceiling compares two
consensuses built from disjoint halves of the same restarts on the same data, removing the
data difference to expose algorithmic instability.

Within-domain uses the two split halves. Across-domain is reported twice: `*_across` fits on
one domain's sample-average and evaluates on another's, and `*_across_halves` fits on half A
of one domain and evaluates on half B of another. The second is the data-matched one -- same
token budget and same disjoint-set structure as the within-domain metrics -- and is the one
the partition-null references are computed for.

Which matrix an arm sees (|r| or |z|) is read from the parcellation file's own provenance,
never from a config, so the scorer cannot disagree with the parcellation about the input.
"""

import argparse
import csv
import os

import numpy as np

from parcelmate.cfg import get_cfg
from parcelmate.constants import (
    CONNECTIVITY_NAME, EXTENSION, HALF_NAMES, OUTPUT_DIR, PARCELLATION_NAME,
)
from parcelmate.metrics import (
    coassociation_correlation, coassociation_counts, coassociation_reliability,
    domain_average, fidelity, fidelity_ceiling, fidelity_insample, map_reliability,
    reliability, reliability_ceiling, summarize, triviality,
)
from parcelmate.util import (
    connectivity_matrix, derive_seed, h5_keys, load_h5_array, load_h5_data, read_attrs,
    stderr, variants_tag,
)

# Within-domain, both halves share a scale, so variance explained is meaningful and is the
# stricter measure. Across domains they do not -- whitespace sits at mean |r| 0.37 against
# wikitext at 0.047 -- so an R^2 there scores the scale mismatch rather than whether the
# structure transfers. Correlation is reported for both, so the across/within ratio (the
# continuous domain-generality measure) compares like with like.
WITHIN_MEASURES = (('fidelity_within', 'r2'), ('fidelity_within_r', 'r'))


def conn_path(root, domain, key):
    return os.path.join(root, CONNECTIVITY_NAME,
                        '%s_%s_%s%s' % (CONNECTIVITY_NAME, domain, key, EXTENSION))


def parc_path(root, variant, domain, key):
    return os.path.join(root, variant, PARCELLATION_NAME,
                        '%s_%s_%s%s' % (PARCELLATION_NAME, domain, key, EXTENSION))


def load_connectivity(path, normalize=None):
    """The same matrix the parcellation saw: the identical call `run_parcellation` makes."""
    return connectivity_matrix(load_h5_data(path, verbose=False), normalize)


def load_noise_scale(path):
    """Per-unit mean null standard deviation, or None where no surrogates were run.

    Feeds the `ami_noise_scale` diagnostic (metrics.triviality): a |z| arm could score well
    by sorting units on how measurable they are, and this is how that would be seen.
    """
    # Key check then a single-dataset read: `load_h5_data` would pull the ~400 MB
    # connectivity matrix into memory too, and this is called once per arm per domain.
    if 'surrogate_var' not in h5_keys(path):
        return None
    v = np.nan_to_num(np.asarray(load_h5_array(path, 'surrogate_var'), dtype=np.float64))
    return np.sqrt(np.maximum(v, 0.0)).mean(axis=1)


def variant_normalize(root, variant, domain, key):
    """What the parcellation file says it was fit on. The single source of truth."""
    n = read_attrs(parc_path(root, variant, domain, key)).get('normalize', '')
    return None if n in ('', 'None', None) else str(n)


def ceiling_label(normalize):
    return '(ceiling)' if normalize is None else '(ceiling:%s)' % normalize


class MatrixCache:
    """Load each (path, normalize) once; `clear()` between domains keeps memory flat."""

    def __init__(self):
        self._d = {}

    def get(self, path, normalize):
        k = (path, normalize)
        if k not in self._d:
            self._d[k] = load_connectivity(path, normalize)
        return self._d[k]

    def clear(self):
        self._d.clear()


def random_partition(n_units, n_networks, seed):
    rng = np.random.RandomState(seed % (2 ** 32))
    return np.eye(int(n_networks))[rng.randint(0, int(n_networks), size=n_units)]


def wants_across(pairs, fit_domain, eval_domain=None):
    """Whether an across-domain comparison is requested. `pairs` None means every ordered pair."""
    if pairs is None:
        return True
    if eval_domain is None:
        return any(f == fit_domain for f, _ in pairs)
    return (fit_domain, eval_domain) in pairs


def score_tree(root, tree, variants, domains, rows, missing, cross_domain=True, verbose=True,
               pairs=None):
    """Score one tree on its own data. Appends result rows and any absent inputs.

    Nothing is skipped silently: every input that should exist and does not is recorded, so
    `score_config` can refuse to write a results file that only looks complete. A
    parcellation job that times out part way is the realistic failure here, and it would
    otherwise produce a scores.csv indistinguishable from a finished one.
    """
    cache = MatrixCache()
    for domain in domains:
        pa, pb = conn_path(root, domain, HALF_NAMES[0]), conn_path(root, domain, HALF_NAMES[1])
        if not (os.path.exists(pa) and os.path.exists(pb)):
            missing.append('%s/%s: split-half connectivity' % (tree, domain))
            continue
        noise_scale = load_noise_scale(pa)
        modes_done = set()

        for variant in variants:
            fa, fb = parc_path(root, variant, domain, HALF_NAMES[0]), \
                parc_path(root, variant, domain, HALF_NAMES[1])
            if not (os.path.exists(fa) and os.path.exists(fb)):
                missing.append('%s/%s/%s: parcellated halves' % (tree, variant, domain))
                continue
            normalize = variant_normalize(root, variant, domain, HALF_NAMES[0])
            R_a, R_b = cache.get(pa, normalize), cache.get(pb, normalize)

            # A property of the data, not of any variant: how much of half B is predictable
            # from half A with no compression at all. One per input mode in use.
            if normalize not in modes_done:
                modes_done.add(normalize)
                for metric_name, measure in WITHIN_MEASURES:
                    rows.append(dict(tree=tree, variant=ceiling_label(normalize),
                                     metric=metric_name, fit=domain, eval=domain,
                                     value=fidelity_ceiling(R_a, R_b, measure=measure)))

            da, db = load_h5_data(fa, verbose=False), load_h5_data(fb, verbose=False)
            P_a, P_b = da['parcellation'], db['parcellation']

            rows.append(dict(tree=tree, variant=variant, metric='reliability_within',
                             fit=domain, eval=domain, value=reliability(P_a, P_b)))
            if 'ica_maps' in da and 'ica_maps' in db:
                # The soft object an ICA arm produces, judged on its own terms.
                rows.append(dict(tree=tree, variant=variant, metric='reliability_within_maps',
                                 fit=domain, eval=domain,
                                 value=map_reliability(da['ica_maps'], db['ica_maps'])))
            if 'samples' in da and 'samples' in db:
                # The restart ensembles judged without label matching (Iteration 19, T3).
                rows.append(dict(tree=tree, variant=variant,
                                 metric='reliability_within_coassoc', fit=domain, eval=domain,
                                 value=coassociation_reliability(da['samples'], db['samples'])))
            for metric_name, measure in WITHIN_MEASURES:
                rows.append(dict(tree=tree, variant=variant, metric=metric_name,
                                 fit=domain, eval=domain,
                                 value=fidelity(R_a, R_b, P_a, measure=measure)))
            # The tighter reference: what this partition achieves on the half it was fit
            # to. Held-out over in-sample separates a partition that overfits its half from
            # a model class that simply does not describe this connectome.
            rows.append(dict(tree=tree, variant=variant, metric='fidelity_within_insample',
                             fit=domain, eval=domain, value=fidelity_insample(R_a, P_a)))

            # Reliability ceiling: two consensuses from disjoint halves of the SAME
            # restarts on the SAME data, so the residual is algorithmic instability alone.
            if 'parcellation_split1' in da and 'parcellation_split2' in da:
                rows.append(dict(
                    tree=tree, variant=variant, metric='reliability_ceiling',
                    fit=domain, eval=domain,
                    value=reliability_ceiling(da['parcellation_split1'],
                                              da['parcellation_split2'])))
            else:
                missing.append('%s/%s/%s: restart-split consensuses (re-run parcellation)'
                               % (tree, variant, domain))

            for name, value in triviality(P_a, da['coordinates'], connectivity=R_a,
                                          noise_scale=noise_scale).items():
                rows.append(dict(tree=tree, variant=variant, metric='triviality_%s' % name,
                                 fit=domain, eval=domain, value=float(value)))
            if verbose:
                stderr('  %-5s %-12s %-14s done\n' % (tree, domain, variant))
        cache.clear()

    if not cross_domain:
        return

    # Across-domain, twice, with different amounts of data on each side.
    #
    #   'avg'    fit on one domain's 4-sample average, evaluate on another's (~394k tokens
    #            per side). More tokens per side, so less estimation noise.
    #   'halves' fit on half A of one domain, evaluate on half B of another (~197k tokens
    #            per side, disjoint token sets) -- EXACTLY the data budget and the
    #            fit/evaluate structure of the within-domain metrics.
    #
    # Only the halves version licenses the comparison "within X vs across Y".
    for fit_key, eval_key, suffix in (('avg', 'avg', ''),
                                      (HALF_NAMES[0], HALF_NAMES[1], '_halves')):
        score_across(root, tree, variants, domains, rows, missing,
                     fit_key, eval_key, suffix, verbose=verbose, pairs=pairs)


def score_across(root, tree, variants, domains, rows, missing,
                 fit_key, eval_key, suffix, verbose=True, pairs=None):
    """Fit a parcellation on one domain and evaluate it on another, within one tree.

    `pairs`, a set of (fit, eval) domain names, restricts the comparisons; None scores
    every ordered pair (LOG.md Iteration 19: pooled pseudo-domains make most pairs
    meaningless, since a pool shares data with most other domains).
    """
    cache = MatrixCache()
    for fit_domain in domains:
        if not wants_across(pairs, fit_domain):
            continue
        p_fit = conn_path(root, fit_domain, fit_key)
        if not os.path.exists(p_fit):
            missing.append('%s/%s: %s connectivity' % (tree, fit_domain, fit_key))
            continue
        parcs = {}
        for variant in variants:
            f = parc_path(root, variant, fit_domain, fit_key)
            if os.path.exists(f):
                parcs[variant] = (load_h5_data(f, verbose=False)['parcellation'],
                                  variant_normalize(root, variant, fit_domain, fit_key))
            else:
                missing.append('%s/%s/%s: %s parcellation'
                               % (tree, variant, fit_domain, fit_key))
        modes = sorted({m for _, m in parcs.values()}, key=str)
        for eval_domain in domains:
            if eval_domain == fit_domain or not wants_across(pairs, fit_domain, eval_domain):
                continue
            p_eval = conn_path(root, eval_domain, eval_key)
            if not os.path.exists(p_eval):
                continue
            for normalize in modes:
                # Correlation, not R^2: see WITHIN_MEASURES above.
                rows.append(dict(tree=tree, variant=ceiling_label(normalize),
                                 metric='fidelity_across' + suffix,
                                 fit=fit_domain, eval=eval_domain,
                                 value=fidelity_ceiling(cache.get(p_fit, normalize),
                                                        cache.get(p_eval, normalize),
                                                        measure='r')))
            for variant, (P, normalize) in parcs.items():
                rows.append(dict(tree=tree, variant=variant,
                                 metric='fidelity_across' + suffix,
                                 fit=fit_domain, eval=eval_domain,
                                 value=fidelity(cache.get(p_fit, normalize),
                                                cache.get(p_eval, normalize), P,
                                                measure='r')))
                f_eval = parc_path(root, variant, eval_domain, eval_key)
                if os.path.exists(f_eval):
                    rows.append(dict(
                        tree=tree, variant=variant,
                        metric='reliability_across' + suffix,
                        fit=fit_domain, eval=eval_domain,
                        value=reliability(
                            P, load_h5_data(f_eval, verbose=False)['parcellation'])))
            # Keep only the fit-side matrices between eval domains.
            for normalize in modes:
                cache._d.pop((p_eval, normalize), None)
        if verbose:
            stderr('  %-5s %-12s across(%s->%s) done\n'
                   % (tree, fit_domain, fit_key, eval_key))
        cache.clear()


def score_partition_nulls(root, null_root, variants, domains, rows, missing, seed,
                          cross_domain=True, verbose=True, pairs=None):
    """Trees `pnull` and `rand`: reference partitions evaluated on the REAL data.

    The real metric fits P on half A and predicts half B. Here the partition comes from
    somewhere that knows nothing about the real cross-unit structure -- the null tree's
    parcellation of shifted half A, or a seeded random draw -- while the block means are
    still estimated on REAL half A and the target is still REAL half B. Everything except
    the partition is identical to the real row, so the difference is the partition.

    The matrix each reference sees is whatever the corresponding real arm saw (|r| or
    |z|), read from the real arm's provenance, so an arm and its references are always on
    the same input.
    """
    cache = MatrixCache()
    for domain in domains:
        pa, pb = conn_path(root, domain, HALF_NAMES[0]), conn_path(root, domain, HALF_NAMES[1])
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue   # already recorded as missing by score_tree('real')
        for variant in variants:
            fa, fb = (parc_path(root, variant, domain, HALF_NAMES[0]),
                      parc_path(root, variant, domain, HALF_NAMES[1]))
            fn_a = parc_path(null_root, variant, domain, HALF_NAMES[0])
            fn_b = parc_path(null_root, variant, domain, HALF_NAMES[1])
            if not (os.path.exists(fa) and os.path.exists(fb)):
                continue   # recorded by score_tree('real')
            if not (os.path.exists(fn_a) and os.path.exists(fn_b)):
                missing.append('pnull/%s/%s: null-tree parcellation of both halves'
                               % (variant, domain))
                continue
            normalize = variant_normalize(root, variant, domain, HALF_NAMES[0])
            R_a, R_b = cache.get(pa, normalize), cache.get(pb, normalize)
            da, db = load_h5_data(fa, verbose=False), load_h5_data(fb, verbose=False)
            dna, dnb = load_h5_data(fn_a, verbose=False), load_h5_data(fn_b, verbose=False)
            P_a, P_b = da['parcellation'], db['parcellation']
            P_null_a, P_null_b = dna['parcellation'], dnb['parcellation']
            if all('samples' in d for d in (da, db, dna, dnb)):
                # Co-association reliability against the null ensembles, both orientations
                # averaged as for the ARI reference below.
                C_a, C_b = coassociation_counts(da['samples']), coassociation_counts(db['samples'])
                rows.append(dict(
                    tree='pnull', variant=variant, metric='reliability_within_coassoc',
                    fit=domain, eval=domain,
                    value=(coassociation_correlation(coassociation_counts(dna['samples']), C_b)
                           + coassociation_correlation(C_a, coassociation_counts(dnb['samples'])))
                    / 2.0))
                del C_a, C_b
            k = int(read_attrs(fb).get('n_networks', P_b.shape[1]))
            n_units = P_b.shape[0]

            def rand_p(tag):
                return random_partition(n_units, k,
                                        derive_seed(seed, 'rand_partition', variant, domain, tag))

            # FIDELITY is directional -- the real row fits on half A and predicts half B --
            # so the reference must keep that exact direction and differ only in the
            # partition. Only the A-side reference is admissible here.
            refs = {'pnull': P_null_a, 'rand': rand_p('a')}
            # RELIABILITY is symmetric in the two halves: the real row is ARI(P_A, P_B), so
            # shifting A and shifting B are two equally valid versions of "one real
            # parcellation against one null parcellation" and the difference between them
            # is sampling noise. Averaging both halves the variance for free and removes an
            # arbitrary choice of which half carries the shift.
            rel_refs = {
                'pnull': (reliability(P_null_a, P_b) + reliability(P_a, P_null_b)) / 2.0,
                'rand': (reliability(rand_p('a'), P_b) + reliability(P_a, rand_p('b'))) / 2.0,
            }
            for tree, P_ref in refs.items():
                rows.append(dict(tree=tree, variant=variant, metric='reliability_within',
                                 fit=domain, eval=domain, value=rel_refs[tree]))
                for metric_name, measure in WITHIN_MEASURES:
                    rows.append(dict(tree=tree, variant=variant, metric=metric_name,
                                     fit=domain, eval=domain,
                                     value=fidelity(R_a, R_b, P_ref, measure=measure)))
                rows.append(dict(tree=tree, variant=variant,
                                 metric='fidelity_within_insample',
                                 fit=domain, eval=domain,
                                 value=fidelity_insample(R_a, P_ref)))
                # How degenerate is the reference partition? A null partition that fills
                # 17 of 50 blocks has less capacity than the real one, and this is where
                # that shows.
                for name, value in triviality(P_ref, db['coordinates'],
                                              connectivity=R_a).items():
                    rows.append(dict(tree=tree, variant=variant,
                                     metric='triviality_%s' % name,
                                     fit=domain, eval=domain, value=float(value)))
            if verbose:
                stderr('  %-5s %-12s %-14s done\n' % ('pnull', domain, variant))
        cache.clear()

    if not cross_domain:
        return

    # Across domains, data-matched halves only: the null partition of half A of one domain,
    # block means on real half A of that domain, predicting real half B of another; and
    # its agreement with the real parcellation of that other domain's half B.
    fit_key, eval_key = HALF_NAMES
    for fit_domain in domains:
        if not wants_across(pairs, fit_domain):
            continue
        p_fit = conn_path(root, fit_domain, fit_key)
        if not os.path.exists(p_fit):
            continue
        refs = {}
        for variant in variants:
            fn = parc_path(null_root, variant, fit_domain, fit_key)
            fr = parc_path(root, variant, fit_domain, fit_key)
            if not (os.path.exists(fn) and os.path.exists(fr)):
                continue   # recorded above / by score_tree
            normalize = variant_normalize(root, variant, fit_domain, fit_key)
            P_real = load_h5_data(fr, verbose=False)['parcellation']
            k = int(read_attrs(fr).get('n_networks', P_real.shape[1]))
            refs[variant] = (normalize, P_real, {
                'pnull': load_h5_data(fn, verbose=False)['parcellation'],
                'rand': random_partition(P_real.shape[0], k,
                                         derive_seed(seed, 'rand_partition', variant,
                                                     fit_domain, 'a')),
            }, k)
        for eval_domain in domains:
            if eval_domain == fit_domain or not wants_across(pairs, fit_domain, eval_domain):
                continue
            p_eval = conn_path(root, eval_domain, eval_key)
            if not os.path.exists(p_eval):
                continue
            for variant, (normalize, P_fit_real, by_tree, k) in refs.items():
                R_fit, R_eval = cache.get(p_fit, normalize), cache.get(p_eval, normalize)
                f_eval = parc_path(root, variant, eval_domain, eval_key)
                fn_eval = parc_path(null_root, variant, eval_domain, eval_key)
                P_eval = load_h5_data(f_eval, verbose=False)['parcellation'] \
                    if os.path.exists(f_eval) else None
                # The eval-side references, for the symmetric reliability comparison.
                eval_refs = {}
                if P_eval is not None:
                    eval_refs['rand'] = random_partition(
                        P_eval.shape[0], k,
                        derive_seed(seed, 'rand_partition', variant, eval_domain, 'b'))
                    if os.path.exists(fn_eval):
                        eval_refs['pnull'] = load_h5_data(
                            fn_eval, verbose=False)['parcellation']
                for tree, P_ref in by_tree.items():
                    # Fidelity keeps the real row's direction: fit side null, eval side real.
                    rows.append(dict(tree=tree, variant=variant,
                                     metric='fidelity_across_halves',
                                     fit=fit_domain, eval=eval_domain,
                                     value=fidelity(R_fit, R_eval, P_ref, measure='r')))
                    # Reliability is symmetric, so average shifting the fit side with
                    # shifting the eval side. See the within-domain block.
                    if P_eval is not None and tree in eval_refs:
                        rows.append(dict(tree=tree, variant=variant,
                                         metric='reliability_across_halves',
                                         fit=fit_domain, eval=eval_domain,
                                         value=(reliability(P_ref, P_eval)
                                                + reliability(P_fit_real,
                                                              eval_refs[tree])) / 2.0))
            for key in [k for k in cache._d if k[0] == p_eval]:
                cache._d.pop(key, None)
        if verbose:
            stderr('  %-5s %-12s across(halves) done\n' % ('pnull', fit_domain))
        cache.clear()


def score_config(cfg, out=None, cross_domain=True, allow_partial=False, verbose=True,
                 variants=None):
    """Score every variant in `cfg` across all trees. Returns (rows, out_path).

    Split out from `main` so the pipeline driver can call it as a step.
    """
    root = cfg.get('output_dir', OUTPUT_DIR)
    null_root = root.rstrip('/') + '_null'
    all_variants = sorted((cfg.get('parcellation_variants') or {'default': {}}).keys())
    if variants:
        unknown = [v for v in variants if v not in all_variants]
        assert not unknown, 'unknown variant(s) %s; config has %s' % (unknown, all_variants)
        variants = sorted(variants)
    else:
        variants = all_variants
    # An optional `score` section overrides the domains (so pooled pseudo-domains can be
    # scored, LOG.md Iteration 19) and restricts the across-domain pairs.
    score_cfg = cfg.get('score') or {}
    domains = list(score_cfg.get('domains') or cfg.get('connectivity', {}).get('domains') or [])
    seed = cfg.get('seed', 0) or 0
    assert domains, 'No domains in the config; nothing to score'
    pairs = None
    if score_cfg.get('across_pairs') is not None:
        pairs = set()
        for p in score_cfg['across_pairs']:
            assert len(p) == 2 and p[0] != p[1] and p[0] in domains and p[1] in domains, (
                'across pair %r must name two different domains from %s' % (p, domains))
            pairs.add((str(p[0]), str(p[1])))

    stderr('Scoring %d variant(s) over %d domain(s), %s across-domain pairs\n' % (
        len(variants), len(domains), 'all' if pairs is None else len(pairs)))
    rows, missing = [], []
    for tree, tree_root in (('real', root), ('null', null_root)):
        if not os.path.isdir(tree_root):
            missing.append('%s tree absent at %s' % (tree, tree_root))
            continue
        score_tree(tree_root, tree, variants, domains, rows, missing,
                   cross_domain=cross_domain, verbose=verbose, pairs=pairs)
    if os.path.isdir(root) and os.path.isdir(null_root):
        score_partition_nulls(root, null_root, variants, domains, rows, missing, seed,
                              cross_domain=cross_domain, verbose=verbose, pairs=pairs)

    # Refuse to publish a results file that merely looks complete. The realistic failure is
    # a parcellation job hitting its wall clock part way through; without this the scoring
    # would exit 0 over whatever happened to be on disk.
    if missing:
        stderr('\n%d missing input(s):\n' % len(missing))
        for m in missing[:40]:
            stderr('  - %s\n' % m)
        if len(missing) > 40:
            stderr('  ... and %d more\n' % (len(missing) - 40))
        if not allow_partial:
            raise SystemExit(
                '\nRefusing to write partial results. Re-run the missing steps, or pass '
                '--allow-partial if an incomplete table is genuinely what you want.')
        stderr('\n--allow-partial given; writing an incomplete table.\n')

    # A restricted variant set writes to its own file, so a partial table never overwrites
    # the full one; the full score of the same tree can still be run later.
    default_name = 'scores.csv' if variants == all_variants \
        else 'scores_%s.csv' % variants_tag(variants)
    out_path = out or os.path.join(root, 'metrics', default_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['tree', 'variant', 'metric', 'fit', 'eval', 'value'])
        w.writeheader()
        w.writerows(rows)
    stderr('\nWrote %d rows to %s\n' % (len(rows), out_path))

    # The CSV holds every individual measurement. The summary below is a reading aid;
    # anything plotted later should come from the CSV, not from here.
    summary = summarize(rows, verbose=False)
    print('\n%-14s %-26s %7s %7s %7s %7s %10s %10s' % (
        'variant', 'metric', 'real', 'null', 'pnull', 'rand', 'real-pnull', 'real-rand'))
    print('-' * 96)
    for metric in ('reliability_within', 'reliability_within_maps', 'reliability_within_coassoc',
                   'reliability_ceiling',
                   'reliability_across', 'reliability_across_halves',
                   'fidelity_within', 'fidelity_within_insample', 'fidelity_within_r',
                   'fidelity_across', 'fidelity_across_halves',
                   'triviality_ami_layer', 'triviality_ami_dimension', 'triviality_ami_hubness',
                   'triviality_ami_noise_scale',
                   'triviality_median_max_membership', 'triviality_n_effective_networks'):
        labels = variants + sorted({r['variant'] for r in summary
                                    if r['variant'].startswith('(ceiling')})
        for variant in labels:
            vals = [r for r in summary if r['metric'] == metric and r['variant'] == variant]
            if not vals:
                continue

            def avg(field):
                xs = [v[field] for v in vals if v[field] is not None and np.isfinite(v[field])]
                return float(np.mean(xs)) if xs else float('nan')

            is_ceiling = variant.startswith('(ceiling')
            print('%-14s %-26s %7.3f %7.3f %7.3f %7.3f %10s %10s' % (
                variant, metric, avg('real'), avg('null'), avg('pnull'), avg('rand'),
                '-' if is_ceiling else '%.3f' % domain_average(summary, metric, variant,
                                                               'delta_pnull'),
                '-' if is_ceiling else '%.3f' % domain_average(summary, metric, variant,
                                                               'delta_rand')))

    print('\nread real-pnull: the credit the clustering earns beyond a partition that knows')
    print('only per-unit properties, on the SAME real data. real-rand: beyond any partition')
    print('of the same k. `null` is the pipeline scored on shifted data, kept for the record;')
    print('it is not on the same denominator as `real` and should not be subtracted from it.')
    print('fidelity_within is R^2; *_r and *_across* are Pearson r. Domains weighted equally;')
    print('whitespace and codeparrot are degenerate and should be reported separately.')

    return rows, out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('config_path', nargs='?', default=None)
    ap.add_argument('-o', '--out', default=None,
                    help='CSV path (default <output_dir>/metrics/scores.csv)')
    ap.add_argument('--no-cross-domain', action='store_true',
                    help='Skip the across-domain comparisons, which dominate the runtime.')
    ap.add_argument('--allow-partial', action='store_true',
                    help='Write results even though inputs are missing. Off by default: a '
                         'partial scores.csv is indistinguishable from a complete one.')
    args = ap.parse_args()
    cfg = get_cfg(args.config_path) if args.config_path else {}
    score_config(cfg, out=args.out, cross_domain=not args.no_cross_domain,
                 allow_partial=args.allow_partial)


if __name__ == '__main__':
    main()
