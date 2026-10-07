"""Stability of networks across datasets, from stored partitions (LOG.md Iteration 36).

    python -m parcelmate.bin.stability --trees results/qwen35/qwen3.5-2b results/qwen35/qwen3.5-2b-rest \
        --out results/qwen35/stability_2b

Reads `<tree>/final/parcellation/parcellation_<dataset>_<half>.h5` for every dataset found
in the given trees and their `_null` twins (hard labels only; all partitions must share one
unit order). Writes into --out, group-writable:

  networks.csv        A: one row per tree, dataset, half and network (`network_stability`)
  meta_networks.csv   B: one row per tree, half set and meta-network: span, networks, core
                      size, and the Dice of its core with the best core of the other half set
  consensus.csv       C: per tree and half set, the consensus ARI between half sets against
                      the single-dataset split-half ARIs, and the distribution of unit scores
                      (also for layer-shuffled partitions, the chance reference)
  stable_sets.npz     what later analyses use: per tree and half set, the consensus labels
                      and unit scores, and per meta-network its core units and member networks
  summary.csv         the headline numbers
"""

import argparse
import csv
import glob
import os

import h5py
import numpy as np

from parcelmate.metrics import hard_labels
from parcelmate.stability import (
    adjusted_rand, consensus, layer_shuffle, match_meta, meta_networks, network_stability,
)
from parcelmate.util import stderr

os.umask(0o002)


def write(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for kk in r:
            if kk not in keys:
                keys.append(kk)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    stderr('wrote %s (%d rows)\n' % (path, len(rows)))


def load(trees, variant='final'):
    parts, coords, k = {}, None, None
    for root in trees:
        for tree, r in (('real', root), ('null', root.rstrip('/') + '_null')):
            for p in sorted(glob.glob(os.path.join(r, variant, 'parcellation', 'parcellation_*_half*.h5'))):
                name = os.path.basename(p)[len('parcellation_'):-len('.h5')]
                d, h = name.rsplit('_', 1)
                with h5py.File(p, 'r') as f:
                    P = f['parcellation'][()]
                    c = f['coordinates'][()]
                if coords is None:
                    coords, k = c, P.shape[1]
                assert np.array_equal(c, coords), 'unit order differs in %s' % p
                parts[(tree, d, h)] = hard_labels(P).astype(np.int64)
                stderr('  %s %s %s\n' % (tree, d, h))
    return parts, np.asarray(coords)[:, 0].astype(np.int64), k


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trees', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--variant', default='final')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-init', type=int, default=10)
    ap.add_argument('--datasets', nargs='+', default=None, help='restrict to these datasets')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    parts, layer, k = load(args.trees, args.variant)
    if args.datasets:
        parts = {key: v for key, v in parts.items() if key[1] in args.datasets}
    datasets = sorted({d for (_, d, _) in parts})
    stderr('%d partitions, datasets %s, k = %d, %d units\n' % (len(parts), ', '.join(datasets), k, len(layer)))

    # A
    rows_a = network_stability(parts, layer, k, rng=rng)
    write(os.path.join(args.out, 'networks.csv'), rows_a)

    # B and C, per tree and per half set (all half-A partitions, all half-B partitions)
    rows_b, rows_c, summ = [], [], []
    store = {}
    for tree in ('real', 'null'):
        metas = {}
        cons = {}
        for h in ('halfA', 'halfB'):
            labs = {d: parts[(tree, d, h)] for d in datasets}
            metas[h], _ = meta_networks(labs, layer, k, rng=rng)
            c, score = consensus(labs, k, n_init=args.n_init, seed=args.seed)
            shuffled = {d: layer_shuffle(v, layer, rng) for d, v in labs.items()}
            _, score_chance = consensus(shuffled, k, n_init=max(2, args.n_init // 5), seed=args.seed)
            cons[h] = (c, score, score_chance)
            store['%s_%s_consensus' % (tree, h)] = c.astype(np.int16)
            store['%s_%s_score' % (tree, h)] = score.astype(np.float32)
            for m_i, m in enumerate(metas[h]):
                store['%s_%s_meta%d_core' % (tree, h, m_i)] = m['core'].astype(np.int32)
                store['%s_%s_meta%d_nodes' % (tree, h, m_i)] = np.array(['%s:%d' % x for x in m['nodes']])
        for h, oh in (('halfA', 'halfB'), ('halfB', 'halfA')):
            rel = match_meta(metas[h], metas[oh])
            for m_i, (m, r) in enumerate(zip(metas[h], rel)):
                rows_b.append(dict(tree=tree, half=h, meta=m_i, span=m['span'], n_networks=m['n_networks'],
                                   pure=m['pure'], core_size=len(m['core']), core_dice_other_half=float(r),
                                   nodes=' '.join('%s:%d' % x for x in m['nodes'])))
        ari_cons = adjusted_rand(cons['halfA'][0], cons['halfB'][0])
        ari_single = [adjusted_rand(parts[(tree, d, 'halfA')], parts[(tree, d, 'halfB')]) for d in datasets]
        score_r = float(np.corrcoef(cons['halfA'][1], cons['halfB'][1])[0, 1])
        for h in ('halfA', 'halfB'):
            c, score, chance = cons[h]
            rows_c.append(dict(tree=tree, half=h, consensus_ari_between_halves=ari_cons,
                               single_dataset_ari_mean=float(np.mean(ari_single)),
                               score_mean=float(score.mean()), score_chance_mean=float(chance.mean()),
                               frac_score_ge_0p8=float((score >= 0.8).mean()),
                               frac_score_ge_0p8_chance=float((chance >= 0.8).mean()),
                               frac_score_eq_1=float((score == 1).mean()),
                               unit_score_corr_between_halves=score_r))
        for h in ('halfA', 'halfB'):
            ms = metas[h]
            spans = np.array([m['span'] for m in ms])
            rel = match_meta(ms, metas['halfB' if h == 'halfA' else 'halfA'])
            ra = [r for r in rows_a if r['tree'] == tree and r['half'] == h]
            summ.append(dict(
                tree=tree, half=h,
                A_generality_median=float(np.nanmedian([r['generality'] for r in ra])),
                A_within_median=float(np.median([r['within'] for r in ra])),
                A_across_median=float(np.median([r['across_mean'] for r in ra])),
                A_chance_median=float(np.median([r['chance'] for r in ra])),
                A_frac_present_all=float(np.mean([r['n_datasets_present'] == r['n_other_datasets'] for r in ra])),
                B_n_meta_span5=int((spans == len(datasets)).sum()),
                B_n_meta_span_ge4=int((spans >= len(datasets) - 1).sum()),
                B_core_units_span_ge4=int(sum(len(m['core']) for m in ms if m['span'] >= len(datasets) - 1)),
                B_core_dice_other_half_span_ge4=float(np.mean(rel[spans >= len(datasets) - 1])) if (spans >= len(datasets) - 1).any() else float('nan'),
                C_consensus_ari=ari_cons, C_single_ari_mean=float(np.mean(ari_single)),
                C_frac_core=float((cons[h][1] >= 0.8).mean()),
                C_frac_core_chance=float((cons[h][2] >= 0.8).mean())))
            stderr('  %s %s: A generality %.2f (within %.2f across %.2f chance %.2f) | B %d meta span 5, %d span >= 4 | C ARI %.2f vs %.2f, core %.0f%% (chance %.0f%%)\n' % (
                tree, h, summ[-1]['A_generality_median'], summ[-1]['A_within_median'], summ[-1]['A_across_median'],
                summ[-1]['A_chance_median'], summ[-1]['B_n_meta_span5'], summ[-1]['B_n_meta_span_ge4'],
                ari_cons, np.mean(ari_single), 100 * summ[-1]['C_frac_core'], 100 * summ[-1]['C_frac_core_chance']))
    write(os.path.join(args.out, 'meta_networks.csv'), rows_b)
    write(os.path.join(args.out, 'consensus.csv'), rows_c)
    write(os.path.join(args.out, 'summary.csv'), summ)
    store['layer'] = layer.astype(np.int16)
    store['datasets'] = np.array(datasets)
    np.savez_compressed(os.path.join(args.out, 'stable_sets.npz'), **store)
    stderr('wrote %s\n' % os.path.join(args.out, 'stable_sets.npz'))


if __name__ == '__main__':
    main()
