"""Cross-dataset consensus from the stored restarts, and whether it reproduces (LOG.md
Iteration 38).

    python -m parcelmate.bin.restart_consensus --trees results/qwen35/qwen3.5-2b results/qwen35/qwen3.5-2b-rest \
        --datasets wikitext bookcorpus agnews tldr17 --out results/qwen35/consensus_2b_prose

Every dataset's partition files keep their k-means restarts (`samples`, n_restarts x units),
each fitted within its own dataset (no connectome is pooled). `stability.restart_consensus`
builds one partition from the restarts of all the datasets. Two reproducibility checks:

  restarts   (Andrea, 2026-10-07) consensus from restarts 1..R/2 of every dataset against
             consensus from restarts R/2+1..R, same data (half A): does the consensus depend
             on which restarts it was built from?
  data       consensus from all restarts of the half-A partitions against the half-B ones:
             independent tokens, the replication that matters for the claims

and as references: single-dataset partitions between halves, and the same consensus built
on the null partitions. Writes summary.csv, units.npz (labels, confidence) into --out.
"""

import argparse
import csv
import os

import h5py
import numpy as np

from parcelmate.bin.stability import write
from parcelmate.metrics import hard_labels
from parcelmate.stability import adjusted_rand, restart_consensus
from parcelmate.util import stderr

os.umask(0o002)


def find(trees, tree, d, h, variant):
    for root in trees:
        r = root if tree == 'real' else root.rstrip('/') + '_null'
        p = os.path.join(r, variant, 'parcellation', 'parcellation_%s_%s.h5' % (d, h))
        if os.path.exists(p):
            return p
    raise FileNotFoundError('%s %s %s' % (tree, d, h))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trees', nargs='+', required=True)
    ap.add_argument('--datasets', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--variant', default='final')
    ap.add_argument('--n-init', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows, store = [], {}
    for tree in ('real', 'null'):
        S, P, coords, k = {}, {}, None, None
        for d in args.datasets:
            for h in ('halfA', 'halfB'):
                with h5py.File(find(args.trees, tree, d, h, args.variant), 'r') as f:
                    S[(d, h)] = f['samples'][()].astype(np.int64)
                    P[(d, h)] = hard_labels(f['parcellation'][()])
                    c = f['coordinates'][()]
                    k = f['parcellation'].shape[1]
                assert coords is None or np.array_equal(c, coords)
                coords = c
        R = min(len(v) for v in S.values())
        half = R // 2
        stderr('%s: %d datasets, %d restarts each, k = %d, %d units\n' % (tree, len(args.datasets), R, k, len(coords)))
        res = {}
        for name, h, sl in (('A_first', 'halfA', slice(0, half)), ('A_second', 'halfA', slice(half, R)),
                            ('A_all', 'halfA', slice(0, R)), ('B_all', 'halfB', slice(0, R))):
            labs = np.concatenate([S[(d, h)][sl] for d in args.datasets])
            res[name] = restart_consensus(labs, k, n_init=args.n_init, seed=args.seed)
            store['%s_%s_labels' % (tree, name)] = res[name][0].astype(np.int16)
            store['%s_%s_confidence' % (tree, name)] = res[name][1].astype(np.float32)
            stderr('  %s: consensus over %d labelings, mean confidence %.3f\n' % (name, len(labs), res[name][1].mean()))
        single_halves = [adjusted_rand(P[(d, 'halfA')], P[(d, 'halfB')]) for d in args.datasets]
        single_restarts = []
        for d in args.datasets:
            a = restart_consensus(S[(d, 'halfA')][:half], k, n_init=max(2, args.n_init // 2), seed=args.seed)[0]
            b = restart_consensus(S[(d, 'halfA')][half:], k, n_init=max(2, args.n_init // 2), seed=args.seed)[0]
            single_restarts.append(adjusted_rand(a, b))
        cons_vs_own = [adjusted_rand(res['A_all'][0], P[(d, 'halfA')]) for d in args.datasets]
        row = dict(tree=tree, n_datasets=len(args.datasets), n_restarts=R,
                   restarts_ari=adjusted_rand(res['A_first'][0], res['A_second'][0]),
                   restarts_confidence_corr=float(np.corrcoef(res['A_first'][1], res['A_second'][1])[0, 1]),
                   data_ari=adjusted_rand(res['A_all'][0], res['B_all'][0]),
                   data_confidence_corr=float(np.corrcoef(res['A_all'][1], res['B_all'][1])[0, 1]),
                   single_dataset_halves_ari_mean=float(np.mean(single_halves)),
                   single_dataset_restarts_ari_mean=float(np.mean(single_restarts)),
                   consensus_vs_each_dataset_ari_mean=float(np.mean(cons_vs_own)),
                   confidence_mean=float(res['A_all'][1].mean()),
                   confidence_q10=float(np.quantile(res['A_all'][1], 0.1)),
                   confidence_q90=float(np.quantile(res['A_all'][1], 0.9)))
        for d, a, b, c in zip(args.datasets, single_halves, single_restarts, cons_vs_own):
            row['single_halves_ari_%s' % d] = a
            row['single_restarts_ari_%s' % d] = b
            row['consensus_vs_%s_ari' % d] = c
        rows.append(row)
        stderr('  %s: restarts ARI %.3f (single dataset %.3f) | data ARI %.3f (single dataset %.3f) | consensus vs each dataset %.3f\n' % (
            tree, row['restarts_ari'], row['single_dataset_restarts_ari_mean'], row['data_ari'],
            row['single_dataset_halves_ari_mean'], row['consensus_vs_each_dataset_ari_mean']))
    store['coordinates'] = coords
    write(os.path.join(args.out, 'summary.csv'), rows)
    np.savez_compressed(os.path.join(args.out, 'units.npz'), **store)


if __name__ == '__main__':
    main()
