"""How much of the residual-stream connectome is the residual stream (LOG.md Iteration 14, YOLO 3).

    PYTHONPATH=. python analysis/residual_chain.py [--root results/ladder] [--out ...]

Dimension d at layer l and dimension d at layer l+1 differ by one block's update, so a
residual-stream connectome contains 768 "chains" of 13 units by construction. This script
measures (a) how strong those chain pairs are relative to everything else, per domain, and
(b) how much each ladder arm's parcellation follows them: AMI with the dimension index, and
the fraction of chain-adjacent pairs that land in the same cluster against the fraction
expected from the cluster sizes alone. Reads the real half-A matrices and parcellations of
the existing ladder tree; writes one CSV. CPU, one matrix in memory at a time (~400 MB).
"""

import argparse
import csv
import os

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

from parcelmate.constants import CONNECTIVITY_NAME, HALF_NAMES, PARCELLATION_NAME
from parcelmate.metrics import hard_labels
from parcelmate.util import connectivity_matrix, load_h5_array, load_h5_data, stderr

PROSE = ('wikitext', 'bookcorpus', 'agnews', 'tldr17')
ARMS = ('legacy', 'current', 'fisher_pca', 'nopca_fisher', 'vmf_profile', 'vmf_z')


def chain_pairs(coordinates):
    """Index pairs (i, j) with the same dimension at layers l and l+1."""
    layer, dim = coordinates[:, 0], coordinates[:, 1]
    lookup = {(int(l), int(d)): i for i, (l, d) in enumerate(coordinates)}
    i_ix, j_ix = [], []
    for i, (l, d) in enumerate(coordinates):
        j = lookup.get((int(l) + 1, int(d)))
        if j is not None:
            i_ix.append(i)
            j_ix.append(j)
    return np.asarray(i_ix), np.asarray(j_ix)


def same_dim_nonadjacent_pairs(coordinates, rng, n_max=200000):
    layer, dim = coordinates[:, 0], coordinates[:, 1]
    by_dim = {}
    for i, d in enumerate(dim):
        by_dim.setdefault(int(d), []).append(i)
    i_ix, j_ix = [], []
    for d, units in by_dim.items():
        units = np.asarray(units)
        for a in range(len(units)):
            for b in range(a + 2, len(units)):  # skip adjacent (a, a+1)
                i_ix.append(units[a])
                j_ix.append(units[b])
    i_ix, j_ix = np.asarray(i_ix), np.asarray(j_ix)
    if len(i_ix) > n_max:
        sel = rng.choice(len(i_ix), n_max, replace=False)
        i_ix, j_ix = i_ix[sel], j_ix[sel]
    return i_ix, j_ix


def co_cluster_rate(labels, i_ix, j_ix):
    """Fraction of the given pairs in the same cluster, and the size-only expectation."""
    same = float((labels[i_ix] == labels[j_ix]).mean())
    p = np.bincount(labels) / float(len(labels))
    return same, float((p ** 2).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='results/ladder')
    ap.add_argument('--out', default=None)
    ap.add_argument('--domains', nargs='+', default=list(PROSE))
    ap.add_argument('--arms', nargs='+', default=list(ARMS))
    args = ap.parse_args()
    out = args.out or os.path.join(args.root, 'metrics', 'residual_chain.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rng = np.random.RandomState(0)
    rows = []

    def add(domain, arm, metric, value):
        rows.append(dict(domain=domain, arm=arm, metric=metric, value=float(value)))
        stderr('  %-10s %-12s %-36s %.4f\n' % (domain, arm, metric, value))

    for domain in args.domains:
        path = os.path.join(args.root, CONNECTIVITY_NAME,
                            '%s_%s_%s.h5' % (CONNECTIVITY_NAME, domain, HALF_NAMES[0]))
        if not os.path.exists(path):
            stderr('missing %s\n' % path)
            continue
        stderr('%s\n' % domain)
        coords = load_h5_array(path, 'coordinates')
        R = connectivity_matrix({'connectivity': load_h5_array(path, 'connectivity')})
        np.fill_diagonal(R, np.nan)
        n = R.shape[0]
        ci, cj = chain_pairs(coords)
        si, sj = same_dim_nonadjacent_pairs(coords, rng)
        chain = R[ci, cj]
        same_dim = R[si, sj]
        rand_i, rand_j = rng.randint(0, n, 200000), rng.randint(0, n, 200000)
        keep = rand_i != rand_j
        other = R[rand_i[keep], rand_j[keep]]
        add(domain, '-', 'n_chain_pairs', len(chain))
        add(domain, '-', 'mean_abs_r_chain_adjacent', np.nanmean(chain))
        add(domain, '-', 'median_abs_r_chain_adjacent', np.nanmedian(chain))
        add(domain, '-', 'mean_abs_r_same_dim_nonadjacent', np.nanmean(same_dim))
        add(domain, '-', 'mean_abs_r_random_pairs', np.nanmean(other))
        add(domain, '-', 'p90_abs_r_random_pairs', np.nanpercentile(other, 90))
        add(domain, '-', 'p99_abs_r_random_pairs', np.nanpercentile(other, 99))
        # Where does the adjacent same-dimension unit rank among each unit's partners?
        # (Rows of |r| sorted descending; rank 0 = strongest partner.)
        ranks = np.empty(len(ci))
        for m, (i, j) in enumerate(zip(ci, cj)):
            row = R[i]
            ranks[m] = np.sum(row > row[j])  # NaN diagonal compares False
        add(domain, '-', 'frac_chain_partner_is_top1', np.mean(ranks == 0))
        add(domain, '-', 'frac_chain_partner_in_top10', np.mean(ranks < 10))
        add(domain, '-', 'median_rank_of_chain_partner', np.median(ranks))
        # Share of each unit's total strength carried by its (up to 2) chain neighbours.
        strength = np.nansum(R, axis=1)
        chain_strength = np.zeros(n)
        np.add.at(chain_strength, ci, chain)
        np.add.at(chain_strength, cj, chain)
        add(domain, '-', 'median_share_of_strength_in_chain', np.median(chain_strength / strength))
        del R

        for arm in args.arms:
            ppath = os.path.join(args.root, arm, PARCELLATION_NAME,
                                 '%s_%s_%s.h5' % (PARCELLATION_NAME, domain, HALF_NAMES[0]))
            if not os.path.exists(ppath):
                stderr('  missing %s\n' % ppath)
                continue
            labels = hard_labels(load_h5_data(ppath, verbose=False)['parcellation'])
            add(domain, arm, 'ami_dimension', adjusted_mutual_info_score(coords[:, 1], labels))
            add(domain, arm, 'ami_layer', adjusted_mutual_info_score(coords[:, 0], labels))
            same, expect = co_cluster_rate(labels, ci, cj)
            add(domain, arm, 'chain_cocluster_rate', same)
            add(domain, arm, 'chain_cocluster_expected', expect)
            add(domain, arm, 'chain_cocluster_lift', same / expect if expect > 0 else float('nan'))
            # Whole chains: fraction of dimensions whose 13 units all share one label.
            by_dim = {}
            for i, d in enumerate(coords[:, 1]):
                by_dim.setdefault(int(d), []).append(labels[i])
            add(domain, arm, 'frac_dimensions_single_cluster',
                np.mean([len(set(v)) == 1 for v in by_dim.values()]))
            # Null-tree parcellation of the same arm, for reference.
            npath = ppath.replace(args.root.rstrip('/'), args.root.rstrip('/') + '_null', 1)
            if os.path.exists(npath):
                nlabels = hard_labels(load_h5_data(npath, verbose=False)['parcellation'])
                add(domain, arm + '(null)', 'ami_dimension',
                    adjusted_mutual_info_score(coords[:, 1], nlabels))
                nsame, nexp = co_cluster_rate(nlabels, ci, cj)
                add(domain, arm + '(null)', 'chain_cocluster_lift', nsame / nexp if nexp > 0 else float('nan'))

    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['domain', 'arm', 'metric', 'value'])
        w.writeheader()
        w.writerows(rows)
    stderr('\nWrote %d rows to %s\n' % (len(rows), out))


if __name__ == '__main__':
    main()
