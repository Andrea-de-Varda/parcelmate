"""Cross-fit yardstick for hard-label reliability (LOG.md Iteration 19, T3).

    PYTHONPATH=. python analysis/yardstick.py --root results/final_mlp \
        [--variants vmf_lloyd100 ...] [--domains ...] [--out ...]

For each domain and each listed arm: the arm's consensus partition of half A gives centroids
on half A's standardized Fisher profiles, and Lloyd k-means runs on half B's profiles from
those centroids (and the same from B to A). The ARI between the starting partition and where
Lloyd lands is how close a converged k-means partition of one half can stay to the other's
when steered as close to it as the data allow: an upper reference for split-half reliability
at this k, `metrics.crossfit_yardstick`. The ordinary split-half ARI and the inertia ratio of
the steered partition against the half's own consensus are written alongside.

`pnull` rows repeat the procedure with the null tree's partitions (centroids still on the REAL
profiles, as for every partition-null reference): how much a partition that knows only
per-unit properties persists across halves under the same steering.

The profiles are rebuilt exactly as `sample_parcellations` builds them for a standardized
Fisher arm (|r|, Fisher transform, zero diagonal, row z-score), in float32, without any PCA
the arm may have applied: the yardstick is a property of the data at this k, and PCA-100
Lloyd matched full-profile Lloyd on every metric (Iteration 18). Arms whose provenance says
anything else are refused. One domain's two profile matrices are in memory at a time
(~0.8 GB at 9,984 units).
"""

import argparse
import csv
import os

import numpy as np

from parcelmate.constants import CONNECTIVITY_NAME, EXTENSION, HALF_NAMES, PARCELLATION_NAME
from parcelmate.data import fisher
from parcelmate.metrics import crossfit_yardstick, hard_labels
from parcelmate.util import connectivity_matrix, load_h5_data, read_attrs, stderr

PROSE = ('wikitext', 'bookcorpus', 'agnews', 'tldr17')


def conn_file(root, domain, key):
    return os.path.join(root, CONNECTIVITY_NAME, '%s_%s_%s%s' % (CONNECTIVITY_NAME, domain, key, EXTENSION))


def parc_file(root, variant, domain, key):
    return os.path.join(root, variant, PARCELLATION_NAME,
                        '%s_%s_%s%s' % (PARCELLATION_NAME, domain, key, EXTENSION))


def standardized_profiles(path):
    X = np.clip(connectivity_matrix(load_h5_data(path, verbose=False)).astype(np.float32), -1.0, 1.0)
    X = fisher(X)
    np.fill_diagonal(X, 0.0)
    X -= X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    X /= np.where(sd > 0, sd, 1.0)
    return X


def check_arm(path):
    """Refuse an arm whose clustering did not see standardized Fisher |r| profiles."""
    a = read_attrs(path)

    def on(key):
        return str(a.get(key)) in ('True', '1')

    ok = (on('fisher_transform_applied') and on('standardize_profiles')
          and not on('binarize_connectivity') and not on('sparsify_fisher')
          and not on('sparsify_profiles') and str(a.get('normalize')) in ('None', ''))
    assert ok, ('%s was not clustered on standardized Fisher |r| profiles; the yardstick '
                'would compare it with the wrong feature space' % path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True, help='experiment tree, e.g. results/final_mlp')
    ap.add_argument('--variants', nargs='+', default=['vmf_lloyd100'])
    ap.add_argument('--domains', nargs='+', default=list(PROSE))
    ap.add_argument('--max-iter', type=int, default=300)
    ap.add_argument('--out', default=None, help='CSV (default <root>/metrics/yardstick.csv)')
    args = ap.parse_args()
    root = args.root.rstrip('/')
    null_root = root + '_null'
    out = args.out or os.path.join(root, 'metrics', 'yardstick.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # Every input first, so a missing file fails before an hour of Lloyd rather than after.
    for domain in args.domains:
        for key in HALF_NAMES:
            assert os.path.exists(conn_file(root, domain, key)), conn_file(root, domain, key)
            for variant in args.variants:
                for tree_root in (root, null_root):
                    f = parc_file(tree_root, variant, domain, key)
                    assert os.path.exists(f), 'missing %s' % f
                    check_arm(f)

    rows = []
    for domain in args.domains:
        X = {key: standardized_profiles(conn_file(root, domain, key)) for key in HALF_NAMES}
        for variant in args.variants:
            real = {key: hard_labels(load_h5_data(parc_file(root, variant, domain, key),
                                                  verbose=False)['parcellation']) for key in HALF_NAMES}
            null = {key: hard_labels(load_h5_data(parc_file(null_root, variant, domain, key),
                                                  verbose=False)['parcellation']) for key in HALF_NAMES}
            for tree, labels in (('real', real), ('pnull', null)):
                for fit_key, eval_key in (HALF_NAMES, HALF_NAMES[::-1]):
                    res = crossfit_yardstick(X[fit_key], X[eval_key], labels[fit_key],
                                             labels_eval=real[eval_key], max_iter=args.max_iter)
                    direction = '%s->%s' % (fit_key, eval_key)
                    for metric, value in (('yardstick_ari', res['ari']),
                                          ('reliability_ari', res['reliability']),
                                          ('inertia_ratio', res['inertia_ratio']),
                                          ('n_iter', res['n_iter']),
                                          ('n_clusters', res['n_clusters'])):
                        rows.append(dict(tree=tree, variant=variant, domain=domain,
                                         direction=direction, metric=metric, value=float(value)))
                    stderr('  %-10s %-22s %-5s %-12s yardstick %.3f  reliability %.3f  inertia ratio %.3f  (%d it)\n'
                           % (domain, variant, tree, direction, res['ari'], res['reliability'],
                              res['inertia_ratio'], res['n_iter']))
        del X

    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['tree', 'variant', 'domain', 'direction', 'metric', 'value'])
        w.writeheader()
        w.writerows(rows)
    stderr('\nWrote %d rows to %s\n' % (len(rows), out))
    for variant in args.variants:
        for tree in ('real', 'pnull'):
            def m(metric):
                return np.mean([r['value'] for r in rows if r['variant'] == variant
                                and r['tree'] == tree and r['metric'] == metric])
            print('%-22s %-5s yardstick %.3f  reliability %.3f  inertia ratio %.3f'
                  % (variant, tree, m('yardstick_ari'), m('reliability_ari'), m('inertia_ratio')))


if __name__ == '__main__':
    main()
