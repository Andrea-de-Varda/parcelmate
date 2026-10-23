"""Fixed-k participation of every unit, real and circular-shift trees (LOG.md Iteration 45,
revised 2026-10-22; plots/rich_club/README.md section 4.4).

    python -m parcelmate.bin.richclub_topk configs/richclub/<model>-resid-pool4-tiles.yml [--block 512]

Needs the stored tiles and the outputs of `parcelmate.bin.richclub` (units_<tree>.npz, for the
live units and the strength ranking). For each tree and direction (networks from the selection
half's k = 100 partition, partners from the evaluation half), every live unit's top-k partners
for k = 1, 2, 10% of the live units (`richclub.topk_pass`). Each unit's effective number of
networks reached, 1 / (1 - P), is set against random spread over all live units and against
random spread within the layers its partners come from. Writes, in
<output_dir>/metrics/richclub/:

  topk_units_<tree>.npz      per unit: P, E_random, E_layer, own_network, own_layer
  topk_participation.csv     medians and means by strength-rank bin, for the club of 1% and
                             for all live units; both trees
"""

import argparse
import csv
import os

import numpy as np
import yaml

from parcelmate.bigconn import TiledMatrix
from parcelmate.bin.score import conn_path, parc_path
from parcelmate.metrics import hard_labels
from parcelmate.richclub import TOPK_FRACS, topk_pass
from parcelmate.util import load_h5_data, stderr

os.umask(0o002)

DIRECTIONS = (('halfA', 'halfB'), ('halfB', 'halfA'))
N_BINS = 16
CLUB_FRAC = 0.01


def summarise(rows, tree, d, frac, o, rank, L, edges):
    P, Er, El = o['P'], o['E_random'], o['E_layer']
    ok = np.isfinite(P) & np.isfinite(Er) & np.isfinite(El) & (P < 1) & (rank >= 1)

    def add(mask, **kw):
        m = mask & ok
        rows.append(dict(tree=tree, direction=d, k_frac=frac, n_units=int(m.sum()),
                         neff_median=float(np.median(1 / (1 - P[m]))),
                         neff_random_median=float(np.median(1 / (1 - Er[m]))),
                         neff_layer_median=float(np.median(1 / (1 - El[m]))),
                         ratio_random_median=float(np.median((1 - Er[m]) / (1 - P[m]))),
                         ratio_layer_median=float(np.median((1 - El[m]) / (1 - P[m]))),
                         own_network_mean=float(np.mean(o['own_network'][m])),
                         own_layer_mean=float(np.mean(o['own_layer'][m])), **kw))
    for lo, hi in zip(edges[:-1], edges[1:]):
        add((rank >= lo) & (rank < hi), kind='rank_bin', rank_lo=int(lo), rank_hi=int(hi - 1))
    n_club = int(round(CLUB_FRAC * L))
    add(rank <= n_club, kind='club_1pct', rank_lo=1, rank_hi=n_club)
    add(rank >= 1, kind='all_live', rank_lo=1, rank_hi=L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config')
    ap.add_argument('--block', type=int, default=1024)
    ap.add_argument('--trees', nargs='+', default=['real', 'null'])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root = cfg['output_dir'].rstrip('/')
    pool = cfg['connectivity']['pool_as']
    out = os.path.join(root, 'metrics', 'richclub')
    rows = []
    for tree in args.trees:
        troot = root if tree == 'real' else root + '_null'
        u = np.load(os.path.join(out, 'units_%s.npz' % tree))
        live = u['live']
        layers = u['coordinates'][:, 0].astype(np.int64)
        L = int(live.sum())
        ks = {frac: max(2, int(round(frac * L))) for frac in TOPK_FRACS}
        edges = np.unique(np.round(np.geomspace(1, L + 1, N_BINS + 1)).astype(np.int64))
        saved = {}
        for sel, ev in DIRECTIONS:
            d = '%s->%s' % (sel[-1], ev[-1])
            labels = hard_labels(load_h5_data(parc_path(troot, 'final', pool, sel), verbose=False)['parcellation'])
            M = TiledMatrix(conn_path(troot, pool, ev))
            stderr('tree %s, %s: k = %s\n' % (tree, d, ', '.join(str(k) for k in ks.values())))
            res = topk_pass(M, ks.values(), labels, layers, live, block=args.block, label='top-k %s %s' % (tree, d))
            M.close()
            rank = np.zeros(len(live), dtype=np.int64)
            order = u['order_strength_%s' % sel]
            rank[order] = np.arange(1, len(order) + 1)
            for frac, k in ks.items():
                summarise(rows, tree, d, frac, res[k], rank, L, edges)
                for key, v in res[k].items():
                    saved['%s_%s_k%g' % (key, d.replace('->', ''), frac)] = v
        np.savez_compressed(os.path.join(out, 'topk_units_%s.npz' % tree), **saved)
    path = os.path.join(out, 'topk_participation.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    stderr('wrote %s\n' % path)


if __name__ == '__main__':
    main()
