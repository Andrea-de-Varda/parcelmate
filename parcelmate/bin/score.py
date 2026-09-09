"""Score parcellation variants for reliability and fidelity against a circular-shift null.

    python -m parcelmate.bin.score configs/reliability.yml

Reads the real and null trees produced by the connectivity + split_halves + parcellation
steps, and writes a tidy CSV plus a printed summary. Every number is reported for both
trees so the null-calibrated difference is available; fidelity additionally gets an
uncompressed ceiling (predicting held-out connectivity from the fitting matrix directly).

Within-domain uses the two split halves. Across-domain fits on one domain's sample-average
and evaluates on another's, which is the continuous measure of domain-generality -- how
much a parcellation transfers -- as opposed to the binary survive-or-die clique count.
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
    domain_average, fidelity, fidelity_ceiling, reliability, summarize, triviality,
)
from parcelmate.util import load_h5_data, stderr


def conn_path(root, domain, key):
    return os.path.join(root, CONNECTIVITY_NAME,
                        '%s_%s_%s%s' % (CONNECTIVITY_NAME, domain, key, EXTENSION))


def parc_path(root, variant, domain, key):
    return os.path.join(root, variant, PARCELLATION_NAME,
                        '%s_%s_%s%s' % (PARCELLATION_NAME, domain, key, EXTENSION))


def load_connectivity(path):
    """Absolute-valued, NaN-free connectivity, matching what the parcellation saw."""
    return np.abs(np.nan_to_num(load_h5_data(path, verbose=False)['connectivity']))


def score_tree(root, tree, variants, domains, rows, cross_domain=True, verbose=True):
    coordinates = None
    for domain in domains:
        pa, pb = conn_path(root, domain, HALF_NAMES[0]), conn_path(root, domain, HALF_NAMES[1])
        if not (os.path.exists(pa) and os.path.exists(pb)):
            stderr('  %s/%s: split halves missing, skipping\n' % (tree, domain))
            continue
        R_a, R_b = load_connectivity(pa), load_connectivity(pb)

        # Ceiling is a property of the data, not of any variant: how much of half B is
        # predictable from half A at all, with no compression.
        ceiling = fidelity_ceiling(R_a, R_b)
        rows.append(dict(tree=tree, variant='(ceiling)', metric='fidelity_within',
                         fit=domain, eval=domain, value=ceiling))
        if verbose:
            stderr('  %-5s %-12s ceiling R2 = %6.3f\n' % (tree, domain, ceiling))

        for variant in variants:
            fa, fb = parc_path(root, variant, domain, HALF_NAMES[0]), \
                parc_path(root, variant, domain, HALF_NAMES[1])
            if not (os.path.exists(fa) and os.path.exists(fb)):
                stderr('  %s/%s/%s: parcellated halves missing, skipping\n'
                       % (tree, variant, domain))
                continue
            da, db = load_h5_data(fa, verbose=False), load_h5_data(fb, verbose=False)
            P_a, P_b = da['parcellation'], db['parcellation']
            if coordinates is None:
                coordinates = da['coordinates']

            ari = reliability(P_a, P_b)
            r2_hard = fidelity(R_a, R_b, P_a, soft=False)
            r2_soft = fidelity(R_a, R_b, P_a, soft=True)
            rows.append(dict(tree=tree, variant=variant, metric='reliability_within',
                             fit=domain, eval=domain, value=ari))
            rows.append(dict(tree=tree, variant=variant, metric='fidelity_within',
                             fit=domain, eval=domain, value=r2_hard))
            rows.append(dict(tree=tree, variant=variant, metric='fidelity_within_soft',
                             fit=domain, eval=domain, value=r2_soft))
            for name, value in triviality(P_a, da['coordinates'], connectivity=R_a).items():
                rows.append(dict(tree=tree, variant=variant, metric='triviality_%s' % name,
                                 fit=domain, eval=domain, value=float(value)))
            if verbose:
                stderr('  %-5s %-12s %-14s ARI = %6.3f   R2 = %6.3f\n'
                       % (tree, domain, variant, ari, r2_hard))
        del R_a, R_b

    if not cross_domain:
        return

    # Across-domain: fit on one domain's average, evaluate on another's. Averages are
    # loaded one pair at a time -- holding seven 400 MB matrices at once is unnecessary.
    for fit_domain in domains:
        p_fit = conn_path(root, fit_domain, 'avg')
        if not os.path.exists(p_fit):
            continue
        R_fit = load_connectivity(p_fit)
        parcs = {}
        for variant in variants:
            f = parc_path(root, variant, fit_domain, 'avg')
            if os.path.exists(f):
                parcs[variant] = load_h5_data(f, verbose=False)['parcellation']
        for eval_domain in domains:
            if eval_domain == fit_domain:
                continue
            p_eval = conn_path(root, eval_domain, 'avg')
            if not os.path.exists(p_eval):
                continue
            R_eval = load_connectivity(p_eval)
            rows.append(dict(tree=tree, variant='(ceiling)', metric='fidelity_across',
                             fit=fit_domain, eval=eval_domain,
                             value=fidelity_ceiling(R_fit, R_eval)))
            for variant, P in parcs.items():
                rows.append(dict(tree=tree, variant=variant, metric='fidelity_across',
                                 fit=fit_domain, eval=eval_domain,
                                 value=fidelity(R_fit, R_eval, P, soft=False)))
            del R_eval
        del R_fit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('config_path', nargs='?', default=None)
    ap.add_argument('-o', '--out', default=None, help='CSV path (default <output_dir>/metrics/scores.csv)')
    ap.add_argument('--no-cross-domain', action='store_true',
                    help='Skip the across-domain comparisons, which dominate the runtime.')
    args = ap.parse_args()

    cfg = get_cfg(args.config_path) if args.config_path else {}
    root = cfg.get('output_dir', OUTPUT_DIR)
    null_root = root.rstrip('/') + '_null'
    variants = sorted((cfg.get('parcellation_variants') or {'default': {}}).keys())
    domains = list(cfg.get('connectivity', {}).get('domains') or [])
    assert domains, 'No domains in the config; nothing to score'

    stderr('Scoring %d variant(s) over %d domain(s)\n' % (len(variants), len(domains)))
    rows = []
    for tree, tree_root in (('real', root), ('null', null_root)):
        if not os.path.isdir(tree_root):
            stderr('  %s tree missing at %s, skipping\n' % (tree, tree_root))
            continue
        score_tree(tree_root, tree, variants, domains, rows,
                   cross_domain=not args.no_cross_domain)

    out_path = args.out or os.path.join(root, 'metrics', 'scores.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['tree', 'variant', 'metric', 'fit', 'eval', 'value'])
        w.writeheader()
        w.writerows(rows)
    stderr('\nWrote %d rows to %s\n' % (len(rows), out_path))

    summary = summarize(rows, verbose=False)
    print('\n%-14s %-22s %8s %8s %8s' % ('variant', 'metric', 'real', 'null', 'real-null'))
    print('-' * 66)
    for metric in ('reliability_within', 'fidelity_within', 'fidelity_within_soft',
                   'fidelity_across', 'triviality_ami_layer', 'triviality_ami_hubness'):
        for variant in variants + ['(ceiling)']:
            vals = [r for r in summary if r['metric'] == metric and r['variant'] == variant]
            if not vals:
                continue
            def avg(field):
                xs = [v[field] for v in vals if v[field] is not None and np.isfinite(v[field])]
                return float(np.mean(xs)) if xs else float('nan')
            print('%-14s %-22s %8.3f %8.3f %8.3f' % (
                variant, metric, avg('real'), avg('null'),
                domain_average(summary, metric, variant)))
    print('\nDomains weighted equally. Fidelity: block means estimated on the fitting half,')
    print('applied to the held-out half. Ceiling predicts held-out connectivity directly,')
    print('with no parcellation, so it bounds what any compression could achieve.')


if __name__ == '__main__':
    main()
