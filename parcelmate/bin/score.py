"""Score parcellation variants for reliability and fidelity against a circular-shift null.

    python -m parcelmate.bin.score configs/reliability.yml

Reads the real and null trees produced by the connectivity + split_halves + parcellation
steps, and writes a tidy CSV plus a printed summary. Every measurement is written out
individually -- one row per tree, variant, metric and domain pair, unnormalized -- so that
plotting and any derived quantity happen downstream rather than being baked in here.

Both metrics get a ceiling, and they partial out different nuisances. The fidelity ceiling
predicts held-out connectivity from the fitting matrix directly, removing the model
limitation to expose sampling noise between halves. The reliability ceiling compares two
consensuses built from disjoint halves of the same restarts on the same data, removing the
data difference to expose algorithmic instability.

Within-domain uses the two split halves. Across-domain is reported twice: `*_across` fits on
one domain's sample-average and evaluates on another's, and `*_across_halves` fits on half A
of one domain and evaluates on half B of another. The second is the data-matched one -- same
token budget and same disjoint-set structure as the within-domain metrics -- and is what
licenses comparing a within-domain number to an across-domain one directly.
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
    domain_average, fidelity, fidelity_ceiling, fidelity_insample, reliability,
    reliability_ceiling, summarize, triviality,
)

# Within-domain, both halves share a scale, so variance explained is meaningful and is the
# stricter measure. Across domains they do not -- whitespace sits at mean |r| 0.37 against
# wikitext at 0.047 -- so an R^2 there scores the scale mismatch rather than whether the
# structure transfers. Correlation is reported for both, so the across/within ratio (the
# continuous domain-generality measure) compares like with like.
WITHIN_MEASURES = (('fidelity_within', 'r2'), ('fidelity_within_r', 'r'))
from parcelmate.util import load_h5_data, stderr, surrogate_normalized


def conn_path(root, domain, key):
    return os.path.join(root, CONNECTIVITY_NAME,
                        '%s_%s_%s%s' % (CONNECTIVITY_NAME, domain, key, EXTENSION))


def parc_path(root, variant, domain, key):
    return os.path.join(root, variant, PARCELLATION_NAME,
                        '%s_%s_%s%s' % (PARCELLATION_NAME, domain, key, EXTENSION))


def load_connectivity(path):
    """The same matrix the parcellation saw -- |r|, or |z| where null variances exist.

    Deliberately the identical call `run_parcellation` makes, so "what the parcellation
    clustered" and "what the metric scores" cannot drift apart.
    """
    return surrogate_normalized(load_h5_data(path, verbose=False))


def score_tree(root, tree, variants, domains, rows, missing, cross_domain=True, verbose=True):
    """Score one tree. Appends result rows to `rows` and any absent inputs to `missing`.

    Nothing is skipped silently: every input that should exist and does not is recorded, so
    `main` can refuse to write a results file that only looks complete. A parcellation job
    that times out part way is the realistic failure here, and it would otherwise produce a
    scores.csv indistinguishable from a finished one.
    """
    for domain in domains:
        pa, pb = conn_path(root, domain, HALF_NAMES[0]), conn_path(root, domain, HALF_NAMES[1])
        if not (os.path.exists(pa) and os.path.exists(pb)):
            missing.append('%s/%s: split-half connectivity' % (tree, domain))
            continue
        R_a, R_b = load_connectivity(pa), load_connectivity(pb)

        # A property of the data, not of any variant: how much of half B is predictable
        # from half A with no compression at all.
        for metric_name, measure in WITHIN_MEASURES:
            rows.append(dict(tree=tree, variant='(ceiling)', metric=metric_name,
                             fit=domain, eval=domain,
                             value=fidelity_ceiling(R_a, R_b, measure=measure)))

        for variant in variants:
            fa, fb = parc_path(root, variant, domain, HALF_NAMES[0]), \
                parc_path(root, variant, domain, HALF_NAMES[1])
            if not (os.path.exists(fa) and os.path.exists(fb)):
                missing.append('%s/%s/%s: parcellated halves' % (tree, variant, domain))
                continue
            da, db = load_h5_data(fa, verbose=False), load_h5_data(fb, verbose=False)
            P_a, P_b = da['parcellation'], db['parcellation']

            rows.append(dict(tree=tree, variant=variant, metric='reliability_within',
                             fit=domain, eval=domain, value=reliability(P_a, P_b)))
            for metric_name, measure in WITHIN_MEASURES:
                rows.append(dict(tree=tree, variant=variant, metric=metric_name,
                                 fit=domain, eval=domain,
                                 value=fidelity(R_a, R_b, P_a, measure=measure)))
            # The tighter reference: what this partition achieves on the half it was fit
            # to. The uncompressed ceiling says what 50M free parameters can do; this says
            # what ~1,275 can. Held-out over in-sample separates a partition that overfits
            # its half from a model class that simply does not describe this connectome.
            rows.append(dict(tree=tree, variant=variant, metric='fidelity_within_insample',
                             fit=domain, eval=domain,
                             value=fidelity_insample(R_a, P_a)))

            # Reliability ceiling: two consensuses from disjoint halves of the SAME
            # restarts on the SAME data, so the residual is algorithmic instability alone.
            # Arms differ by nearly an order of magnitude here, so cross-half ARI is not
            # comparable between them without it.
            if 'parcellation_split1' in da and 'parcellation_split2' in da:
                rows.append(dict(
                    tree=tree, variant=variant, metric='reliability_ceiling',
                    fit=domain, eval=domain,
                    value=reliability_ceiling(da['parcellation_split1'],
                                              da['parcellation_split2'])))
            else:
                missing.append('%s/%s/%s: restart-split consensuses (re-run parcellation)'
                               % (tree, variant, domain))

            for name, value in triviality(P_a, da['coordinates'], connectivity=R_a).items():
                rows.append(dict(tree=tree, variant=variant, metric='triviality_%s' % name,
                                 fit=domain, eval=domain, value=float(value)))
            if verbose:
                stderr('  %-5s %-12s %-14s done\n' % (tree, domain, variant))
        del R_a, R_b

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
    # Both are kept because they answer slightly different questions, but only the halves
    # version licenses the comparison "within 0.51 vs across 0.08". With the avg version the
    # across row has twice the tokens, and although that mismatch runs in the across row's
    # favour -- less noise, not more -- reporting a within/across ratio across a 2x data
    # difference is not something to do in a paper.
    for fit_key, eval_key, suffix in (('avg', 'avg', ''),
                                      (HALF_NAMES[0], HALF_NAMES[1], '_halves')):
        score_across(root, tree, variants, domains, rows, missing,
                     fit_key, eval_key, suffix, verbose=verbose)


def score_across(root, tree, variants, domains, rows, missing,
                 fit_key, eval_key, suffix, verbose=True):
    """Fit a parcellation on one domain and evaluate it on another.

    `fit_key` and `eval_key` select which connectivity file stands for each side, so the
    same code produces both the sample-average comparison and the data-matched half-vs-half
    comparison. `suffix` is appended to the metric names.
    """
    # Matrices are loaded a pair at a time; holding seven 400 MB matrices at once is
    # unnecessary.
    for fit_domain in domains:
        p_fit = conn_path(root, fit_domain, fit_key)
        if not os.path.exists(p_fit):
            missing.append('%s/%s: %s connectivity' % (tree, fit_domain, fit_key))
            continue
        R_fit = load_connectivity(p_fit)
        parcs = {}
        for variant in variants:
            f = parc_path(root, variant, fit_domain, fit_key)
            if os.path.exists(f):
                parcs[variant] = load_h5_data(f, verbose=False)['parcellation']
            else:
                missing.append('%s/%s/%s: %s parcellation'
                               % (tree, variant, fit_domain, fit_key))
        for eval_domain in domains:
            if eval_domain == fit_domain:
                continue
            p_eval = conn_path(root, eval_domain, eval_key)
            if not os.path.exists(p_eval):
                continue
            R_eval = load_connectivity(p_eval)
            # Correlation, not R^2: see WITHIN_MEASURES above.
            rows.append(dict(tree=tree, variant='(ceiling)',
                             metric='fidelity_across' + suffix,
                             fit=fit_domain, eval=eval_domain,
                             value=fidelity_ceiling(R_fit, R_eval, measure='r')))
            for variant, P in parcs.items():
                rows.append(dict(tree=tree, variant=variant,
                                 metric='fidelity_across' + suffix,
                                 fit=fit_domain, eval=eval_domain,
                                 value=fidelity(R_fit, R_eval, P, measure='r')))
                # Reliability across domains: does the same partition reappear when the
                # model reads different text? The continuous counterpart of the
                # reciprocal-best-match clique, which only ever answered yes or no.
                f_eval = parc_path(root, variant, eval_domain, eval_key)
                if os.path.exists(f_eval):
                    rows.append(dict(
                        tree=tree, variant=variant,
                        metric='reliability_across' + suffix,
                        fit=fit_domain, eval=eval_domain,
                        value=reliability(
                            P, load_h5_data(f_eval, verbose=False)['parcellation'])))
            del R_eval
        if verbose:
            stderr('  %-5s %-12s across(%s->%s) done\n'
                   % (tree, fit_domain, fit_key, eval_key))
        del R_fit


def score_config(cfg, out=None, cross_domain=True, allow_partial=False, verbose=True):
    """Score every variant in `cfg` across both trees. Returns (rows, out_path).

    Split out from `main` so the pipeline driver can call it as a step, which keeps job
    generation uniform: `make_jobs ... -s score` produces a script like any other stage
    rather than needing a hand-written sbatch for this one module.
    """
    root = cfg.get('output_dir', OUTPUT_DIR)
    null_root = root.rstrip('/') + '_null'
    variants = sorted((cfg.get('parcellation_variants') or {'default': {}}).keys())
    domains = list(cfg.get('connectivity', {}).get('domains') or [])
    assert domains, 'No domains in the config; nothing to score'

    stderr('Scoring %d variant(s) over %d domain(s)\n' % (len(variants), len(domains)))
    rows, missing = [], []
    for tree, tree_root in (('real', root), ('null', null_root)):
        if not os.path.isdir(tree_root):
            missing.append('%s tree absent at %s' % (tree, tree_root))
            continue
        score_tree(tree_root, tree, variants, domains, rows, missing,
                   cross_domain=cross_domain, verbose=verbose)

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

    out_path = out or os.path.join(root, 'metrics', 'scores.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['tree', 'variant', 'metric', 'fit', 'eval', 'value'])
        w.writeheader()
        w.writerows(rows)
    stderr('\nWrote %d rows to %s\n' % (len(rows), out_path))

    # The CSV holds every individual measurement -- one row per tree, variant, metric and
    # domain pair, unnormalized. Deltas and ratios below are a reading aid; anything
    # plotted later should come from the CSV, not from these summaries.
    summary = summarize(rows, verbose=False)
    print('\n%-14s %-24s %8s %8s %10s' % ('variant', 'metric', 'real', 'null', 'real-null'))
    print('-' * 68)
    for metric in ('reliability_within', 'reliability_ceiling', 'reliability_across',
                   'fidelity_within', 'fidelity_within_insample', 'fidelity_within_r',
                   'fidelity_across', 'reliability_across_halves',
                   'fidelity_across_halves',
                   'triviality_ami_layer', 'triviality_ami_hubness',
                   'triviality_median_max_membership', 'triviality_n_effective_networks'):
        for variant in variants + ['(ceiling)']:
            vals = [r for r in summary if r['metric'] == metric and r['variant'] == variant]
            if not vals:
                continue

            def avg(field):
                xs = [v[field] for v in vals if v[field] is not None and np.isfinite(v[field])]
                return float(np.mean(xs)) if xs else float('nan')

            # No delta for the uncompressed reference. Subtracting its null from its real
            # value is not a quantity: the null reference is negative, because predicting
            # one noise matrix from another is worse than predicting the mean, so the
            # difference reads as a large positive number that means nothing.
            delta = ('%10s' % '-' if variant == '(ceiling)'
                     else '%10.3f' % domain_average(summary, metric, variant))
            print('%-14s %-24s %8.3f %8.3f %s' % (
                variant, metric, avg('real'), avg('null'), delta))

    print('\nfidelity_within is R^2; fidelity_within_r and fidelity_across are Pearson r,')
    print('which is scale-invariant and therefore the only valid across-domain measure.')
    print('Domains weighted equally here; per-domain values are in the CSV and should be')
    print('reported separately, since whitespace and codeparrot are degenerate cases that')
    print('a block model fits far too easily.')
    print('Fidelity: block means estimated on the fitting half,')
    print('applied to the held-out half, hard argmax labels. Two ceilings, partialling out')
    print('different nuisances: the fidelity ceiling removes the model limitation to expose')
    print('data noise; the reliability ceiling removes the data difference to expose')
    print('algorithmic instability. Compare each arm against its OWN reliability ceiling --')
    print('the arms differ by nearly an order of magnitude in that floor.')

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
