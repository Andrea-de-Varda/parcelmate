"""Verification for the rich-club analysis (LOG.md Iteration 45, info/RICH_CLUB_PLAN.md).

    PYTHONPATH=. python tests/verify_iter45_richclub.py

Covers: the streamed strengths, histogram thresholds, degrees, hub couplings, rho(n) and
c_i(n) equal a dense brute-force computation (and do not depend on the block size); a
planted rich club gives rho > 1 and c_i > 1; a rank-1 matrix (one shared factor) and an iid
random weighted matrix give rho close to 1; with variance shared by a subset of units in one
half only, selecting and evaluating on that half gives rho > 1 (the selection bias the
cross-half design removes) while evaluating on the other half gives rho close to 1; participation on a known case; the same-layer / other-layer split against brute force; a club made only of
within-layer coupling gives rho > 1 but other-layer rho close to 1; the CLI end to end on tiled
float16 halves of a planted club (H1) and of its circular-shift null tree.
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import yaml

from parcelmate.richclub import (
    DenseRows, club_stats, degree_pass, density_thresholds, eval_pass, longest_run, n_grid,
    participation, ranking, strength_pass,
)

failures = []
n_checks = [0]
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
ENV = dict(os.environ, PYTHONPATH='.')


def check(name, cond):
    n_checks[0] += 1
    print('%s %s' % ('OK' if cond else 'FAIL', name))
    if not cond:
        failures.append(name)


def corr(X):
    Z = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    return Z @ Z.T / X.shape[1]


def run(R_sel, R_ev, grid, block=64):
    """rho and c on R_ev for the strength ranking of R_sel (all units live)."""
    s_sel = strength_pass(DenseRows(R_sel), block=block, verbose=False)['strength']
    s_ev = strength_pass(DenseRows(R_ev), block=block, verbose=False)['strength']
    order = ranking(s_sel, np.ones(len(s_sel), bool))
    numer = eval_pass(DenseRows(R_ev), {'s': order}, grid, block=block, verbose=False)['s']
    return club_stats(numer, order, s_ev, grid)


os.makedirs(os.path.join(HERE, '.tmp'), exist_ok=True)
tmp = tempfile.mkdtemp(dir=os.path.join(HERE, '.tmp'))
rng = np.random.RandomState(0)

# ---------------------------------------------------------------- 1. brute force
N, T = 300, 400
F = rng.randn(5, T)
Xa = rng.randn(N, 5) @ F * 0.5 + rng.randn(N, T)
Xb = rng.randn(N, 5) @ F * 0.5 + rng.randn(N, T)
Ra = corr(Xa).astype(np.float16).astype(np.float32)   # the stored float16 values
Rb = corr(Xb).astype(np.float16).astype(np.float32)
Wa, Wb = np.abs(Ra), np.abs(Rb)
np.fill_diagonal(Wa, 0)
np.fill_diagonal(Wb, 0)
labels = rng.randint(0, 7, N)
pa = strength_pass(DenseRows(Ra), {'p': labels}, block=37, verbose=False)
pa2 = strength_pass(DenseRows(Ra), {'p': labels}, block=300, verbose=False)
check('strength = dense row sums without the diagonal, block size irrelevant',
      np.allclose(pa['strength'], Wa.sum(1, dtype=np.float64), rtol=1e-6)
      and np.allclose(pa['strength'], pa2['strength'], rtol=1e-9))
H = np.zeros((N, 7))
H[np.arange(N), labels] = 1
check('network strengths = dense W @ onehot', np.allclose(pa['kappa']['p'], Wa @ H, rtol=1e-5))
offdiag = Wa[~np.eye(N, dtype=bool)]
check('histogram counts every off-diagonal entry once', pa['hist'].sum() == N * (N - 1))
thr = density_thresholds(pa['hist'], N * (N - 1), densities=(0.01, 0.05))
ok = True
for d, (t, achieved) in thr.items():
    cnt = (offdiag >= t).sum()
    smaller = offdiag[offdiag < t]
    nxt = smaller.max() if len(smaller) else None
    ok &= cnt <= d * N * (N - 1) and np.isclose(achieved, cnt / float(N * (N - 1)))
    ok &= nxt is None or (offdiag >= nxt).sum() > d * N * (N - 1)   # t is the smallest admissible
check('density thresholds exact (largest count <= target, next value lower exceeds it)', ok)
deg, kb = degree_pass(DenseRows(Ra), {d: t for d, (t, _) in thr.items()}, partitions={'p': labels},
                      part_keys=(0.05,), block=50, verbose=False)
check('degree = dense count of |r| >= threshold',
      all(np.array_equal(deg[d], (Wa >= t).sum(1)) for d, (t, _) in thr.items()))
check('per-network edge counts on the thresholded graph = dense (W >= t) @ onehot, only for part_keys',
      np.allclose(kb[(0.05, 'p')], (Wa >= thr[0.05][0]).astype(float) @ H) and set(kb) == {(0.05, 'p')})
grid = n_grid(N, n_min=10, frac=0.5, points=8)
st = run(Ra, Rb, grid, block=41)
st2 = run(Ra, Rb, grid, block=300)
order = np.argsort(-Wa.sum(1), kind='stable')
sb = Wb.sum(1)
two_w = sb.sum()
ok_rho, ok_c = True, True
for g, n in enumerate(grid):
    R = order[:n]
    sub = Wb[np.ix_(R, R)]
    E = np.outer(sb[R], sb[R]) / two_w
    np.fill_diagonal(E, 0)
    ok_rho &= np.isclose(st['rho'][g], sub.sum() / E.sum(), rtol=1e-6)
    ok_c &= np.allclose(st['c'][g], sub.sum(1) / E.sum(1), rtol=1e-6)
check('rho(n) = brute force on the dense matrices (selection A, evaluation B)', ok_rho)
check('c_i(n) = brute force for every hub at every n', ok_c)
lay3 = rng.randint(0, 3, N)
s_a = strength_pass(DenseRows(Ra), block=64, verbose=False)['strength']
o_a = ranking(s_a, np.ones(N, bool))
nm = eval_pass(DenseRows(Rb), {'s': o_a}, grid, layers=lay3, block=53, verbose=False)['s']
sb64 = Wb.astype(np.float64).sum(1)                # float64 strengths, as strength_pass gives
tw64 = sb64.sum()
sp = club_stats(nm[0], o_a, sb64, grid, numer_same=nm[1], layers=lay3)
ok = np.allclose(sp['rho'], st['rho'], rtol=1e-6)
for g, n in enumerate(grid):
    R = o_a[:n]
    same = lay3[R][:, None] == lay3[R][None, :]
    sub = Wb[np.ix_(R, R)].astype(np.float64)      # float64: the streamed sums are float64
    E = np.outer(sb64[R], sb64[R]) / tw64
    np.fill_diagonal(E, 0)
    ok &= np.isclose(sp['rho_same'][g], (sub * same).sum() / (E * same).sum(), rtol=1e-6)
    ok &= np.isclose(sp['rho_other'][g], (sub * ~same).sum() / (E * ~same).sum(), rtol=1e-6)
    ok &= np.allclose(sp['c_other'][g], (sub * ~same).sum(1) / (E * ~same).sum(1), rtol=1e-6)
    ok &= np.allclose(sp['c_same'][g], (sub * same).sum(1) / (E * same).sum(1), rtol=1e-6, equal_nan=True)
check('same-layer / other-layer split of rho and c_i = brute force', ok)
check('rho and c independent of the block size',
      np.allclose(st['rho'], st2['rho'], rtol=1e-9) and all(np.allclose(a, b) for a, b in zip(st['c'], st2['c'])))
r_tie = ranking(np.array([1., 3., 3., 2.]), np.array([True, True, True, False]), tiebreak=np.array([0., 1., 5., 9.]))
check('ranking: dead units out, ties broken by the tiebreak then by index', list(r_tie) == [2, 1, 0])
check('longest_run', longest_run([0, 1, 1, 0, 1, 1, 1, 0]) == (4, 3) and longest_run([0, 0]) == (-1, 0))

# ---------------------------------------------------------------- 2. synthetic graphs
N = 1000
grid = n_grid(N, n_min=20, frac=0.2, points=6)
# Planted club: 40 units with large weights among themselves, on top of iid weights.
W = rng.uniform(0, 0.1, (N, N))
W = np.triu(W, 1)
W = W + W.T
club = rng.choice(N, 40, replace=False)
W2 = W.copy()
W2[np.ix_(club, club)] += 0.5
np.fill_diagonal(W2, 0)
Wn = W.copy() + rng.uniform(0, 0.02, (N, N))
Wn = (Wn + Wn.T) / 2
Wn[np.ix_(club, club)] += 0.5
np.fill_diagonal(Wn, 0)
st = run(W2, Wn, grid)
g40 = int(np.searchsorted(grid, 40))
check('planted club: rho > 1.5 at n = %d and c_i > 1 for every hub' % grid[g40 - 1],
      st['rho'][g40 - 1] > 1.5 and np.all(st['c'][g40 - 1] > 1))
# Rank 1: w_ij = a_i a_j (one shared factor): N1 predicts it exactly up to the diagonal.
a = rng.uniform(0.1, 1, N)
R1 = np.outer(a, a)
np.fill_diagonal(R1, 0)
st = run(R1, R1, grid)
check('rank-1 matrix: rho within 3%% of 1 at every n (%.3f-%.3f)' % (st['rho'].min(), st['rho'].max()),
      np.all(np.abs(st['rho'] - 1) < 0.03))
# iid random weights, independent halves. The grid starts at n = 100 (4,950 pairs): at n = 20
# (190 pairs) the sampling sd of rho is about 4% for these weights, too noisy to test at 2%.
grid = n_grid(N, n_min=100, frac=0.2, points=4)
Wr1 = rng.uniform(0, 1, (N, N)); Wr1 = (Wr1 + Wr1.T) / 2; np.fill_diagonal(Wr1, 0)
Wr2 = rng.uniform(0, 1, (N, N)); Wr2 = (Wr2 + Wr2.T) / 2; np.fill_diagonal(Wr2, 0)
st = run(Wr1, Wr2, grid)
check('iid random weights: rho within 2%% of 1 (%.3f-%.3f)' % (st['rho'].min(), st['rho'].max()),
      np.all(np.abs(st['rho'] - 1) < 0.02))
# Selection bias. On iid noise it is negligible: one edge is 1/N of a unit's strength, so
# selecting strong units barely inflates their mutual edges. What the cross-half design
# protects against is variance shared WITHIN ONE HALF only (e.g. a document or burst that
# drives a subset of units in one sample): those units are strong in that half because they
# co-vary there. Planted: 100 of 2,000 units share a factor in half A only.
N, T = 2000, 300
grid = n_grid(N, n_min=100, frac=0.1, points=3)
burst = rng.choice(N, 100, replace=False)
Xa = rng.randn(N, T)
Xa[burst] += 0.6 * rng.randn(1, T)
Rn_a, Rn_b = corr(Xa), corr(rng.randn(N, T))
st_in = run(Rn_a, Rn_a, grid)
st_x = run(Rn_a, Rn_b, grid)
check('half-specific shared variance, selected and evaluated on the same half: rho > 1.5 at n = 100 (%.3f)'
      % st_in['rho'][0], st_in['rho'][0] > 1.5)
check('the same hubs evaluated on the other half: rho within 3.5%% of 1 (%.3f-%.3f; sampling sd about 1%%)'
      % (st_x['rho'].min(), st_x['rho'].max()), np.all(np.abs(st_x['rho'] - 1) < 0.035))

# A LAYER club: units of one layer couple more among themselves, nothing else. The hubs are
# that layer's units, so rho > 1. Their coupling to hubs of other layers is BELOW what N1
# predicts (N1 spreads their inflated strength over every partner, but it stays in the
# layer), so rho_other <= 1 wherever other-layer hubs exist (none while n <= 300).
N, T = 1200, 2000
lay = np.arange(N) // 300
X = rng.randn(N, T)
X[lay == 2] += 0.7 * rng.randn(1, T)
Rl_a = corr(X)
X = rng.randn(N, T)
X[lay == 2] += 0.7 * rng.randn(1, T)
Rl_b = corr(X)
X = rng.randn(N, T)
grid = n_grid(N, n_min=100, frac=0.4, points=4)
s_a = strength_pass(DenseRows(Rl_a), verbose=False)['strength']
s_b = strength_pass(DenseRows(Rl_b), verbose=False)['strength']
o_a = ranking(s_a, np.ones(N, bool))
nm = eval_pass(DenseRows(Rl_b), {'s': o_a}, grid, layers=lay, verbose=False)['s']
sp = club_stats(nm[0], o_a, s_b, grid, numer_same=nm[1], layers=lay)
defined = grid > 300
check('layer club: hubs all in the planted layer, rho > 1.2 (%.2f), rho_other undefined to n = 300 and <= 1 after (%s)' % (
      sp['rho'][0], ', '.join('%.3f' % x for x in sp['rho_other'])),
      np.all(lay[o_a[:300]] == 2) and sp['rho'][0] > 1.2 and np.all(np.isnan(sp['rho_other'][~defined]))
      and defined.any() and np.all(sp['rho_other'][defined] <= 1.0))

# ---------------------------------------------------------------- 3. participation
kap = np.array([[2., 0., 0.], [1., 1., 0.], [1., 1., 1.], [0., 0., 0.]])
P = participation(kap, kap.sum(1))
check('participation: 0 inside one network, 1/2 over two, 2/3 over three, NaN without links',
      np.allclose(P[:3], [0, 0.5, 2 / 3]) and np.isnan(P[3]))

from parcelmate.richclub import expected_participation
lab = np.repeat(np.arange(5), [10, 20, 30, 40, 100])
live_ = np.ones(len(lab), bool)
live_[-3:] = False
kk = np.array([1, 5, 40] + [10] * (len(lab) - 3))
E = expected_participation(kk, lab, live_)
rs = np.random.RandomState(1)
ok = True
for i in (0, 1, 2, 50, 150):
    others = np.array([j for j in np.nonzero(live_)[0] if j != i])
    sims = []
    for _ in range(4000):
        pick = rs.choice(others, kk[i], replace=False)
        f = np.bincount(lab[pick], minlength=5) / float(kk[i])
        sims.append(1 - (f ** 2).sum())
    ok &= abs(np.mean(sims) - E[i]) < 0.01
check('expected participation under random spread = Monte Carlo (hypergeometric), k = 1 gives 0',
      ok and abs(E[0]) < 1e-12 and np.all(np.isnan(E[-3:])))

# ---------------------------------------------------------------- 4. CLI end to end
L_layers, per = 4, 300
N, T, k = L_layers * per, 3000, 12
mod = np.repeat(np.arange(k), N // k)                    # 12 networks of 100 units
hub = np.zeros(N, bool)
for m in range(k):
    hub[np.nonzero(mod == m)[0][:5]] = True            # 5 hubs per network, 60 in all
coords = np.stack([np.arange(N) // per, np.arange(N) % per], 1).astype(np.int32)


def timecourses(seed):
    r = np.random.RandomState(seed)
    X = 0.5 * r.randn(k, T)[mod] + r.randn(N, T)
    X[hub] += 0.9 * r.randn(1, T)                       # the club factor
    X[-7:] = 0.0                                         # dead units (constant)
    return X


def corr_dead(X):
    sd = X.std(1, keepdims=True)
    Z = np.where(sd > 0, (X - X.mean(1, keepdims=True)) / np.where(sd > 0, sd, 1), 0)
    return Z @ Z.T / X.shape[1]


def write_tiled(path, R):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, 'w') as f:
        f.create_dataset('connectivity', data=R.astype(np.float16))
        f.create_dataset('unit_strength', data=np.abs(R).sum(1))
        f.create_dataset('coordinates', data=coords)
        f.attrs['storage'] = 'tiled_fp16'


def write_parc(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    P = np.zeros((N, k), np.float32)
    P[np.arange(N), labels] = 1
    with h5py.File(path, 'w') as f:
        f.create_dataset('parcellation', data=P)


root = os.path.join(tmp, 'tree')
for t, (seed_a, seed_b) in (('', (1, 2)), ('_null', (3, 4))):
    for h, seed in (('halfA', seed_a), ('halfB', seed_b)):
        X = timecourses(seed)
        if t == '_null':
            rs = np.random.RandomState(seed + 100)
            X = np.stack([np.roll(x, rs.randint(T)) for x in X])
        write_tiled(os.path.join(root + t, 'connectivity', 'connectivity_pooled_%s.h5' % h), corr_dead(X))
        write_parc(os.path.join(root + t, 'final', 'parcellation', 'parcellation_pooled_%s.h5' % h),
                   mod if t == '' else rng.randint(0, k, N))
cfg_path = os.path.join(tmp, 'cfg.yml')
with open(cfg_path, 'w') as f:
    yaml.safe_dump(dict(output_dir=root, connectivity=dict(pool_as='pooled')), f)
r = subprocess.run([sys.executable, '-m', 'parcelmate.bin.richclub', cfg_path, '--block', '128'],
                   cwd=ROOT, env=ENV, capture_output=True, text=True)
out = os.path.join(root, 'metrics', 'richclub')
check('CLI runs and writes its outputs', r.returncode == 0 and all(os.path.exists(os.path.join(out, f)) for f in (
    'curves.csv', 'layers.csv', 'participation.csv', 'verdict.json', 'units_real.npz', 'units_null.npz',
    'hubs_real.npz', 'hubs_null.npz')))
if r.returncode != 0:
    print(r.stderr[-3000:])
else:
    v = json.load(open(os.path.join(out, 'verdict.json')))
    rows = list(csv.DictReader(open(os.path.join(out, 'curves.csv'))))
    u = np.load(os.path.join(out, 'units_real.npz'))
    check('dead units excluded: %d live of %d' % (v['n_live'], N), v['n_live'] == N - 7 and u['live'][-7:].sum() == 0)
    check('CLI: H1 on the planted club (strength ranking), over %d grid points' % v['rankings']['strength']['longest_run'],
          v['rankings']['strength']['H1'])
    r60 = {(x['tree'], x['direction']): x for x in rows if x['ranking'] == 'strength'
           and int(x['n']) == min(int(y['n']) for y in rows if int(y['n']) >= 60)}
    check('CLI: null tree rho close to 1 (%s), real rho > 1.5 (%s)' % (
        ', '.join('%.3f' % float(r60[('null', d)]['rho']) for d in ('A->B', 'B->A')),
        ', '.join('%.3f' % float(r60[('real', d)]['rho']) for d in ('A->B', 'B->A'))),
        all(abs(float(r60[('null', d)]['rho']) - 1) < 0.05 and float(r60[('real', d)]['rho']) > 1.5
            for d in ('A->B', 'B->A')))
    top = set(u['order_strength_halfA'][:60].tolist())
    check('CLI: the selected top 60 are the planted hubs (%d/60)' % len(top & set(np.nonzero(hub)[0].tolist())),
          len(top & set(np.nonzero(hub)[0].tolist())) >= 57)
    prow = list(csv.DictReader(open(os.path.join(out, 'participation.csv'))))
    n60 = int(r60[('real', 'A->B')]['n'])
    part = {(x['tree'], x['direction'], x['graph']): x for x in prow if int(x['n']) == n60}
    check('CLI: participation on the weighted graph and the 1, 2, 10% graphs',
          {x['graph'] for x in prow} == {'weighted', 'top1%', 'top2%', 'top10%'})
    check('CLI: the planted club spans all 12 networks and its members participate more than the units below (all graphs)',
          int(part[('real', 'A->B', 'weighted')]['networks_spanned']) == 12
          and all(float(part[('real', 'A->B', g)]['P_club_mean']) > float(part[('real', 'A->B', g)]['P_below_mean'])
                  for g in ('weighted', 'top1%', 'top2%', 'top10%')))
    check('CLI: the planted club holds on other-layer coupling (its hubs span all layers)',
          v['rankings_other_layers']['strength']['H1'] and float(r60[('real', 'A->B')]['rho_other']) > 1.5)
    check('CLI outputs group-writable', oct(os.stat(os.path.join(out, 'curves.csv')).st_mode & 0o777) in ('0o664', '0o666'))

# ---------------------------------------------------------------- 5. fixed-k participation
from parcelmate.richclub import layer_matched_expectation, layer_network_table, topk_pass
rs = np.random.RandomState(2)
N4, T4 = 400, 600
lay4 = np.arange(N4) // 100
net4 = rs.randint(0, 8, N4)
live4 = np.ones(N4, bool)
live4[-5:] = False
X = rs.randn(N4, T4)
X[-5:] = 0
W4 = np.abs(corr_dead(X))
res = topk_pass(DenseRows(W4), [5, 20], net4, lay4, live4, block=37, verbose=False)
res2 = topk_pass(DenseRows(W4), [5, 20], net4, lay4, live4, block=400, verbose=False)
ok = True
for k in (5, 20):
    for i in np.nonzero(live4)[0]:
        w = W4[i].copy()
        w[i] = -1
        w[~live4] = -1
        t = np.argsort(-w, kind='stable')[:k]
        f = np.bincount(net4[t], minlength=8) / k
        ok &= np.isclose(res[k]['P'][i], 1 - (f ** 2).sum())
        ok &= np.isclose(res[k]['own_network'][i], (net4[t] == net4[i]).mean())
        ok &= np.isclose(res[k]['own_layer'][i], (lay4[t] == lay4[i]).mean())
    ok &= all(np.allclose(res[k][key], res2[k][key], equal_nan=True) for key in res[k])
ok &= all(np.all(np.isnan(res[k]['P'][~live4])) for k in (5, 20))
check('top-k participation and own-network / own-layer shares = brute force, any block size, dead units NaN', ok)

counts, sizes = layer_network_table(net4, lay4, live4)
C = np.array([[3., 10., 0., 7.]])
E = layer_matched_expectation(C, counts, sizes)[0]
sims = []
for _ in range(6000):
    labs = np.concatenate([rs.choice(net4[(lay4 == l) & live4], int(c), replace=False) for l, c in enumerate(C[0]) if c])
    f = np.bincount(labs, minlength=8) / C.sum()
    sims.append(1 - (f ** 2).sum())
check('layer-matched expectation = Monte Carlo (%.4f vs %.4f)' % (E, np.mean(sims)), abs(E - np.mean(sims)) < 0.005)

# Planted: units couple only within their layer; networks are random within layers. Then the
# partners are concentrated relative to random spread over all units (they all share one layer,
# and networks are uneven across layers here), but not relative to the layer-matched reference.
netL = np.where(lay4 < 2, rs.randint(0, 4, N4), rs.randint(4, 8, N4))   # networks 0-3 in layers 0-1
X = rs.randn(N4, T4)
for l in range(4):
    X[lay4 == l] += 0.8 * rs.randn(1, T4)
Wl = np.abs(corr_dead(X))
allv = np.ones(N4, bool)
rl = topk_pass(DenseRows(Wl), [20], netL, lay4, allv, verbose=False)[20]
ratio_rand = np.median((1 - rl['E_random']) / (1 - rl['P']))
ratio_layer = np.median((1 - rl['E_layer']) / (1 - rl['P']))
check('layer-only coupling: below random spread (%.2f) but at the layer-matched reference (%.2f)' % (ratio_rand, ratio_layer),
      ratio_rand < 0.75 and abs(ratio_layer - 1) < 0.1)
# Planted: within each layer, units also couple within their network: below the layer reference too.
X = rs.randn(N4, T4)
for l in range(4):
    X[lay4 == l] += 0.5 * rs.randn(1, T4)
for g in range(8):
    X[netL == g] += 0.8 * rs.randn(1, T4)
Wn = np.abs(corr_dead(X))
rn = topk_pass(DenseRows(Wn), [20], netL, lay4, allv, verbose=False)[20]
check('network coupling beyond layer: below the layer-matched reference (%.2f)' % np.median((1 - rn['E_layer']) / (1 - rn['P'])),
      np.median((1 - rn['E_layer']) / (1 - rn['P'])) < 0.6)

r = subprocess.run([sys.executable, '-m', 'parcelmate.bin.richclub_topk', cfg_path, '--block', '100'],
                   cwd=ROOT, env=ENV, capture_output=True, text=True)
tp = os.path.join(root, 'metrics', 'richclub', 'topk_participation.csv')
ok = r.returncode == 0 and os.path.exists(tp)
if ok:
    rows_t = list(csv.DictReader(open(tp)))
    club = [x for x in rows_t if x['kind'] == 'club_1pct' and x['tree'] == 'real' and x['k_frac'] == '0.01']
    ok = len({x['tree'] for x in rows_t}) == 2 and len(club) == 2 and all(np.isfinite(float(x['ratio_layer_median'])) for x in club)
else:
    print(r.stderr[-2000:])
check('top-k CLI end to end: both trees, both directions, finite ratios', ok)

shutil.rmtree(tmp)
print('\n%d checks, %d failure(s)' % (n_checks[0], len(failures)))
for f in failures:
    print('  FAIL', f)
sys.exit(1 if failures else 0)
