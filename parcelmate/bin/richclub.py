"""Rich club of the pooled residual connectome, real and circular-shift trees
(LOG.md Iteration 45, info/RICH_CLUB_PLAN.md).

    python -m parcelmate.bin.richclub configs/richclub/qwen3.5-2b-resid-pool4-tiles.yml

Reads the tiled halves <output_dir>/connectivity/connectivity_<pool>_half{A,B}.h5 and the
null tree's (<output_dir>_null), and the k = 100 partitions of each half from
<parc_root>/final/parcellation (default parc_root: the config's `richclub.parc_root`, else
output_dir). Per tree, three passes over each half (strengths with network strengths and
the exact |r| histogram; binary degree at fixed densities; the hub couplings of the other
half's rankings), then:

  curves.csv         rho(n), the hub statistics and, on real rows, the one-sided
                     Mann-Whitney p of the real hubs' log c_i against the null tree's hubs
  layers.csv         layer composition of the club against all live units
  participation.csv  participation of club members (selection-half partition, evaluation-
                     half weights) against the units ranked just below, networks spanned
  units_<tree>.npz   per-unit strengths, degrees, participation, rankings (for later stages)
  hubs_<tree>.npz    every hub's c_i(n)
  verdict.json       the pre-stated criterion (info/RICH_CLUB_PLAN.md section 5)

All written to <output_dir>/metrics/richclub/.
"""

import argparse
import csv
import json
import os

import numpy as np
import yaml
from scipy.stats import mannwhitneyu

from parcelmate.bigconn import TiledMatrix
from parcelmate.bin.score import conn_path, parc_path
from parcelmate.metrics import hard_labels
from parcelmate.richclub import (
    DENSITIES, DEFAULT_BLOCK, club_stats, degree_pass, density_thresholds, effective_number,
    eval_pass, longest_run, n_grid, participation, ranking, strength_pass,
)
from parcelmate.util import load_h5_data, stderr

os.umask(0o002)

HALVES = ('halfA', 'halfB')
DIRECTIONS = (('halfA', 'halfB'), ('halfB', 'halfA'))   # (selection, evaluation)
MIN_RUN = 3        # grid points of n over which the criterion must hold contiguously
ALPHA = 0.05


def direction_name(sel, ev):
    return '%s->%s' % (sel[-1], ev[-1])


def load_labels(parc_root, pool, key):
    path = parc_path(parc_root, 'final', pool, key)
    if not os.path.exists(path):
        stderr('  no partition at %s: participation skipped\n' % path)
        return None
    return hard_labels(load_h5_data(path, verbose=False)['parcellation'])


def analyse_tree(conn_root, parc_root, pool, block, in_sample=False, grid=None):
    """Every measure of one tree (real or null). Returns a dict of arrays."""
    M = {h: TiledMatrix(conn_path(conn_root, pool, h)) for h in HALVES}
    labels = {h: load_labels(parc_root, pool, h) for h in HALVES}
    parts = {h: lab for h, lab in labels.items() if lab is not None}
    first = {h: strength_pass(M[h], parts, block=block, label='strength %s' % h) for h in HALVES}
    strength = {h: first[h]['strength'] for h in HALVES}
    live = (strength['halfA'] > 0) & (strength['halfB'] > 0)
    L = int(live.sum())
    n_pairs = L * (L - 1)
    thr = {h: density_thresholds(first[h]['hist'], n_pairs) for h in HALVES}
    deg = {h: degree_pass(M[h], {d: thr[h][d][0] for d in DENSITIES}, block=block, label='degree %s' % h)
           for h in HALVES}
    orders = {}
    for h in HALVES:
        orders[h] = {'strength': ranking(strength[h], live)}
        for d in DENSITIES:
            orders[h]['degree%g' % d] = ranking(deg[h][d].astype(np.float64), live, tiebreak=strength[h])
    if grid is None:
        grid = n_grid(L)
    pairs = list(DIRECTIONS) + ([(h, h) for h in HALVES] if in_sample else [])
    stats = {}
    for sel, ev in pairs:
        numer = eval_pass(M[ev], orders[sel], grid, block=block, label='eval %s' % direction_name(sel, ev))
        for name, order in orders[sel].items():
            stats[(sel, ev, name)] = club_stats(numer[name], order, strength[ev], grid)
    for m in M.values():
        m.close()
    coords = TiledMatrix(conn_path(conn_root, pool, 'halfA')).coordinates
    return dict(strength=strength, live=live, L=L, thr=thr, deg=deg, orders=orders, grid=grid,
                stats=stats, labels=labels, kappa={h: first[h]['kappa'] for h in HALVES},
                coordinates=coords)


def jaccard(a, b):
    a, b = set(a.tolist()), set(b.tolist())
    return len(a & b) / float(len(a | b))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config')
    ap.add_argument('--parc-root', default=None, help='tree holding final/parcellation (default: config)')
    ap.add_argument('--block', type=int, default=DEFAULT_BLOCK)
    ap.add_argument('--in-sample', action='store_true',
                    help='also evaluate each half on itself (biased; diagnostic only)')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root = cfg['output_dir'].rstrip('/')
    pool = cfg['connectivity']['pool_as']
    parc_root = (args.parc_root or (cfg.get('richclub') or {}).get('parc_root') or root).rstrip('/')
    trees = {'real': (root, parc_root), 'null': (root + '_null', parc_root + '_null')}
    res = {}
    grid = None
    for tree, (conn_root, proot) in trees.items():
        stderr('tree %s: connectivity %s, partitions %s\n' % (tree, conn_root, proot))
        res[tree] = analyse_tree(conn_root, proot, pool, args.block, in_sample=args.in_sample, grid=grid)
        grid = res[tree]['grid']       # the null tree uses the real tree's grid
    out = os.path.join(root, 'metrics', 'richclub')
    os.makedirs(out, exist_ok=True)

    # curves.csv
    rows = []
    for tree, r in res.items():
        for (sel, ev, name), st in sorted(r['stats'].items()):
            for g, n in enumerate(r['grid']):
                c = st['c'][g]
                c = c[np.isfinite(c) & (c > 0)]
                row = dict(tree=tree, direction=direction_name(sel, ev), ranking=name, n=int(n),
                           n_live=r['L'], rho=st['rho'][g], W=st['W'][g], EW=st['EW'][g],
                           mean_log_c=float(np.mean(np.log(c))) if len(c) else np.nan,
                           median_c=float(np.median(c)) if len(c) else np.nan,
                           share_c_gt1=float(np.mean(c > 1)) if len(c) else np.nan,
                           jaccard_AB=np.nan, jaccard_random=np.nan, p_mw=np.nan, rho_null=np.nan)
                if name == 'strength':
                    row['jaccard_AB'] = jaccard(r['orders']['halfA']['strength'][:n], r['orders']['halfB']['strength'][:n])
                    inter = n * n / float(r['L'])
                    row['jaccard_random'] = inter / (2 * n - inter)
                if tree == 'real' and (sel, ev, name) in res['null']['stats']:
                    stn = res['null']['stats'][(sel, ev, name)]
                    cn = stn['c'][g]
                    cn = cn[np.isfinite(cn) & (cn > 0)]
                    row['rho_null'] = stn['rho'][g]
                    if len(c) and len(cn):
                        row['p_mw'] = mannwhitneyu(np.log(c), np.log(cn), alternative='greater').pvalue
                rows.append(row)
    write_csv(os.path.join(out, 'curves.csv'), rows)

    # verdict.json: the pre-stated criterion, per ranking (strength is primary)
    verdict = dict(alpha=ALPHA, min_run=MIN_RUN, grid=[int(n) for n in grid], rankings={})
    real = {(row['direction'], row['ranking'], row['n']): row for row in rows if row['tree'] == 'real'}
    for name in res['real']['orders']['halfA']:
        flags, detail = [], []
        for n in grid:
            a, b = real.get(('A->B', name, int(n))), real.get(('B->A', name, int(n)))
            ok = bool(a and b and a['rho'] > 1 and b['rho'] > 1 and a['p_mw'] < ALPHA and b['p_mw'] < ALPHA
                      and a['rho'] > a['rho_null'] and b['rho'] > b['rho_null'])
            flags.append(ok)
            detail.append(dict(n=int(n), holds=ok))
        start, length = longest_run(flags)
        verdict['rankings'][name] = dict(
            H1=length >= MIN_RUN, longest_run=length,
            n_range=[int(grid[start]), int(grid[start + length - 1])] if length else None, by_n=detail)
    verdict['n_live'] = res['real']['L']
    verdict['thresholds'] = {tree: {h: {str(d): r['thr'][h][d] for d in DENSITIES} for h in HALVES}
                             for tree, r in res.items()}
    with open(os.path.join(out, 'verdict.json'), 'w') as f:
        json.dump(verdict, f, indent=1)

    # layers.csv and participation.csv (strength ranking, both directions, both trees)
    lrows, prows = [], []
    for tree, r in res.items():
        layer = r['coordinates'][:, 0]
        n_layers = int(layer.max()) + 1
        all_share = np.bincount(layer[r['live']], minlength=n_layers) / float(r['L'])
        for sel, ev in DIRECTIONS:
            order = r['orders'][sel]['strength']
            lab = r['labels'][sel]
            P = None
            if lab is not None:
                k = int(lab.max()) + 1
                P = participation(r['kappa'][ev][sel], r['strength'][ev])
            for n in r['grid']:
                n = int(n)
                club = order[:n]
                share = np.bincount(layer[club], minlength=n_layers) / float(n)
                for l in range(n_layers):
                    lrows.append(dict(tree=tree, direction=direction_name(sel, ev), n=n, layer=l,
                                      share_club=share[l], share_all=all_share[l]))
                if P is not None:
                    below = order[n:2 * n]
                    prows.append(dict(
                        tree=tree, direction=direction_name(sel, ev), n=n,
                        P_club_mean=float(np.nanmean(P[club])), P_club_median=float(np.nanmedian(P[club])),
                        P_below_mean=float(np.nanmean(P[below])) if len(below) else np.nan,
                        P_below_median=float(np.nanmedian(P[below])) if len(below) else np.nan,
                        P_live_median=float(np.nanmedian(P[r['live']])),
                        networks_spanned=int(len(np.unique(lab[club]))),
                        networks_effective=effective_number(lab[club], k)))
        np.savez_compressed(
            os.path.join(out, 'units_%s.npz' % tree), coordinates=r['coordinates'], live=r['live'],
            **{'strength_%s' % h: r['strength'][h] for h in HALVES},
            **{'degree%g_%s' % (d, h): r['deg'][h][d] for h in HALVES for d in DENSITIES},
            **{'order_%s_%s' % (name, h): o for h in HALVES for name, o in r['orders'][h].items()},
            **{'participation_%s' % direction_name(sel, ev).replace('->', ''):
               participation(r['kappa'][ev][sel], r['strength'][ev])
               for sel, ev in DIRECTIONS if r['labels'][sel] is not None})
        hubs = {}
        for (sel, ev, name), st in r['stats'].items():
            C = np.full((int(r['grid'][-1]), len(r['grid'])), np.nan)
            for g, n in enumerate(r['grid']):
                C[:int(n), g] = st['c'][g]
            hubs['c_%s_%s' % (direction_name(sel, ev).replace('->', ''), name)] = C
        np.savez_compressed(os.path.join(out, 'hubs_%s.npz' % tree), grid=r['grid'], **hubs)
    write_csv(os.path.join(out, 'layers.csv'), lrows)
    write_csv(os.path.join(out, 'participation.csv'), prows)
    stderr('H1 (strength ranking): %s\n' % verdict['rankings']['strength']['H1'])
    stderr('wrote %s\n' % out)


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
