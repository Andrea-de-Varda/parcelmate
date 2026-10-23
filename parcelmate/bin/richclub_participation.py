"""Participation against the random-spread reference (LOG.md Iteration 45, revised 2026-10-22).

    python -m parcelmate.bin.richclub_participation results/richclub/<model>-resid-pool4-tiles

Reads <tree>/metrics/richclub/units_real.npz (per-unit participation and degree from
`parcelmate.bin.richclub`) and the k = 100 partitions in <tree>/final/parcellation. For each
direction (networks from the selection half, links from the evaluation half) and each
thresholded graph (top 1, 2, 10% of links), every live unit's observed participation P is
set against E[P], its expectation if its links went to random partners regardless of network
(`richclub.expected_participation`). Units are grouped by their strength rank on the selection
half (log-spaced rank bins), so hubs are read against the trend over all strengths rather than
against a cut-off. Also reported on the scale of the effective number of networks a unit's
links reach, 1 / (1 - P), against 1 / (1 - E[P]). Writes <tree>/metrics/richclub/participation_random.csv.
"""

import argparse
import csv
import os

import numpy as np

from parcelmate.bin.score import parc_path
from parcelmate.metrics import hard_labels
from parcelmate.richclub import PART_DENSITIES, expected_participation
from parcelmate.util import load_h5_data

os.umask(0o002)

DIRECTIONS = (('halfA', 'halfB'), ('halfB', 'halfA'))
N_BINS = 16
CLUB_FRAC = 0.01


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('tree')
    args = ap.parse_args()
    out = os.path.join(args.tree, 'metrics', 'richclub')
    u = np.load(os.path.join(out, 'units_real.npz'))
    live = u['live']
    L = int(live.sum())
    edges = np.unique(np.round(np.geomspace(1, L + 1, N_BINS + 1)).astype(np.int64))
    rows = []
    for sel, ev in DIRECTIONS:
        lab = hard_labels(load_h5_data(parc_path(args.tree, 'final', 'pooled', sel), verbose=False)['parcellation'])
        order = u['order_strength_%s' % sel]          # live units only, strongest first
        rank = np.full(len(live), -1, dtype=np.int64)
        rank[order] = np.arange(1, len(order) + 1)
        d = '%s->%s' % (sel[-1], ev[-1])
        for dens in PART_DENSITIES:
            g = 'top%g%%' % (100 * dens)
            P = u['participation_%s%s_%s' % (sel[-1], ev[-1], g.replace('%', 'pct'))]
            k = u['degree%g_%s' % (dens, ev)]
            E = expected_participation(k, lab, live)
            ok = live & np.isfinite(P) & np.isfinite(E) & (E > 0)   # one link: P = E = 0, no ratio

            def summary(mask, **kw):
                m = mask & ok
                return dict(direction=d, graph=g, n_units=int(m.sum()),
                            P_mean=float(P[m].mean()), E_mean=float(E[m].mean()),
                            ratio_mean=float((P[m] / E[m]).mean()),
                            # effective number of networks reached, 1 / (1 - P): readable where P is
                            # compressed near its maximum (100 networks: random spread gives P = 0.99)
                            neff_median=float(np.median(1 / (1 - P[m]))),
                            neff_E_median=float(np.median(1 / (1 - E[m]))),
                            neff_ratio_median=float(np.median((1 - E[m]) / (1 - P[m]))),
                            share_below_half=float((P[m] < 0.5 * E[m]).mean()),
                            k_median=float(np.median(k[m])), **kw)
            for lo, hi in zip(edges[:-1], edges[1:]):
                rows.append(summary((rank >= lo) & (rank < hi), kind='rank_bin', rank_lo=int(lo), rank_hi=int(hi - 1)))
            n_club = int(round(CLUB_FRAC * L))
            rows.append(summary((rank >= 1) & (rank <= n_club), kind='club_1pct', rank_lo=1, rank_hi=n_club))
            rows.append(summary(rank >= 1, kind='all_live', rank_lo=1, rank_hi=L))
    path = os.path.join(out, 'participation_random.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('wrote', path)


if __name__ == '__main__':
    main()
