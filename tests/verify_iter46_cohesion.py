"""Verification for Iteration 46 (cohesion of the networks a circuit spans).

    PYTHONPATH=. python tests/verify_iter46_cohesion.py

CPU only. Covers: normalised coupling; cohesion and specificity equal brute-force loops;
within-bin permutations; a planted super-network is detected against layer-matched networks
while random circuits give calibrated p; a set of hub networks reads high cohesion but
specificity near 1, unlike a planted super-network; and `network_coupling` end to end on
tiled float16 halves equals dense block means of |r|.
"""

import os
import shutil
import sys
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import h5py
import numpy as np

from parcelmate.cohesion import (cohesion_stats, cohesion_test, layer_bins, network_layers,
                                 normalized_coupling, permute_within_bins)

failures = []
n_checks = [0]
HERE = os.path.dirname(os.path.abspath(__file__))


def check(name, cond):
    n_checks[0] += 1
    print('%s %s' % ('OK' if cond else 'FAIL', name))
    if not cond:
        failures.append(name)


rs = np.random.RandomState(0)

# ---------------------------------------------------------------- normalised coupling
B = rs.uniform(0.05, 0.2, (6, 6)); B = (B + B.T) / 2; np.fill_diagonal(B, rs.uniform(0.3, 0.5, 6))
Bn = normalized_coupling(B)
check('normalised coupling: diagonal 1, symmetric, B_ij / sqrt(B_ii B_jj)',
      np.allclose(np.diag(Bn), 1) and np.allclose(Bn, Bn.T)
      and np.isclose(Bn[1, 3], B[1, 3] / np.sqrt(B[1, 1] * B[3, 3])))

# ---------------------------------------------------------------- brute force
k = 12
Bn = normalized_coupling(np.abs(rs.normal(0.1, 0.05, (k, k))) + np.diag(rs.uniform(0.4, 0.6, k)))
Bn = (Bn + Bn.T) / 2; np.fill_diagonal(Bn, 1)
ok = True
for _ in range(20):
    p = rs.dirichlet(np.ones(k) * 0.5) * (rs.rand(k) > 0.4)
    p = p / p.sum()
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            if i != j:
                num += p[i] * p[j] * Bn[i, j]; den += p[i] * p[j]
    E = np.array([np.mean([Bn[i, j] for j in range(k) if j != i]) for i in range(k)])
    C, S = cohesion_stats(p, Bn)
    ok &= np.isclose(C, num / den) and np.isclose(S, (num / den) / (p @ E))
check('cohesion and specificity equal brute-force double loops', ok)
C1, S1 = cohesion_stats(np.eye(k)[3], Bn)
check('a single-network circuit has no pairs: cohesion NaN', np.isnan(C1) and np.isnan(S1))
P = rs.dirichlet(np.ones(k), size=5)
Cv, Sv = cohesion_stats(P, Bn)
check('vectorised over rows equals one row at a time',
      np.allclose(Cv, [cohesion_stats(r, Bn)[0] for r in P]) and np.allclose(Sv, [cohesion_stats(r, Bn)[1] for r in P]))

# ---------------------------------------------------------------- bins and permutations
cent = rs.uniform(0, 24, 40)
bins = layer_bins(cent, 4)
perms = permute_within_bins(bins, 200, rs)
check('layer bins: equal sizes, ordered by layer position',
      np.bincount(bins).tolist() == [10] * 4 and all(cent[bins == b].max() <= cent[bins == b + 1].min() for b in range(3)))
check('within-bin permutations are permutations that keep every network in its bin',
      all(sorted(r) == list(range(40)) for r in perms) and np.all(bins[perms] == bins[None, :]))
labels = np.repeat(np.arange(4), 3)
coords = np.c_[np.array([0, 0, 2, 4, 4, 4, 1, 1, 1, 7, 9, 8]), np.arange(12)]
check('network layer centre = mean layer of its units',
      np.allclose(network_layers(labels, coords, 4), [2 / 3, 4.0, 1.0, 8.0]))

# ---------------------------------------------------------------- planted structure
k, nb = 40, 4
bins = np.repeat(np.arange(nb), k // nb)     # networks 0-9 bin 0, 10-19 bin 1, ...
base = rs.uniform(0.15, 0.25, (k, k)); base = (base + base.T) / 2
club = [0, 1, 2, 3, 4, 5]                     # a super-network inside bin 0 (layer-matched swaps leave it)
Bc = base.copy()
for i in club:
    for j in club:
        Bc[i, j] = 0.7
np.fill_diagonal(Bc, 1.0)
p = np.zeros(k); p[club] = [0.3, 0.2, 0.2, 0.1, 0.1, 0.1]
r = cohesion_test(p, Bc, bins, 1000, np.random.RandomState(1))
check('planted super-network: cohesion far above layer-matched networks (p %.3f), specificity > 1.5 (%.2f)'
      % (r['p_cohesion'], r['specificity']), r['p_cohesion'] < 0.01 and r['specificity'] > 1.5)
ps = []
for s in range(300):
    q = np.zeros(k); idx = rs.choice(k, 6, replace=False); q[idx] = rs.dirichlet(np.ones(6))
    ps.append(cohesion_test(q, normalized_coupling(base + np.eye(k) * 0.8), bins, 200, np.random.RandomState(s))['p_cohesion'])
frac = np.mean(np.asarray(ps) < 0.05)
check('no structure: p < 0.05 in about 5%% of random circuits (%.3f)' % frac, 0.01 <= frac <= 0.10)
hubs = [0, 1, 10, 20, 21, 30]     # spread over the layer bins, so matched swaps reach non-hubs
Bh = base.copy()
for i in hubs:
    Bh[i, :] = Bh[:, i] = 0.6
np.fill_diagonal(Bh, 1.0)
q = np.zeros(k); q[hubs] = 1 / 6
rh = cohesion_test(q, Bh, bins, 1000, np.random.RandomState(2))
check('hub networks (coupled to everyone): high cohesion (p %.3f) but specificity near 1 (%.2f), unlike a club'
      % (rh['p_cohesion'], rh['specificity']), rh['p_cohesion'] < 0.01 and 0.85 < rh['specificity'] < 1.3)

# ---------------------------------------------------------------- network_coupling end to end
from parcelmate.bigconn import write_tiled_half
from parcelmate.bin.network_coupling import run_tree
os.makedirs(os.path.join(HERE, '.tmp'), exist_ok=True)
tmp = tempfile.mkdtemp(dir=os.path.join(HERE, '.tmp'))
n, T, kk = 60, 400, 5
coords = np.c_[np.repeat(np.arange(3), 20), np.tile(np.arange(20), 3)].astype(np.int32)
ok = True
for tree in ('t', 't_null'):
    root = os.path.join(tmp, tree)
    os.makedirs(os.path.join(root, 'final', 'parcellation'))
    dense = {}
    for h in ('halfA', 'halfB'):
        z = rs.normal(size=(n, T)); z[:20] += rs.normal(size=T); z = (z - z.mean(1, keepdims=True)) / z.std(1, keepdims=True)
        stats = dict(unit_means=np.zeros(n), unit_stds=np.ones(n), n_obs=np.asarray(T), coordinates=coords)
        write_tiled_half([z.astype(np.float16)], None, os.path.join(root, 'connectivity', 'connectivity_pooled_%s.h5' % h),
                         None, stats, {}, device='cpu', verbose=False)
        with h5py.File(os.path.join(root, 'connectivity', 'connectivity_pooled_%s.h5' % h), 'r') as f:
            dense[h] = np.abs(np.nan_to_num(f['connectivity'][()].astype(np.float64)))
        P = np.eye(kk)[rs.randint(0, kk, n)]
        with h5py.File(os.path.join(root, 'final', 'parcellation', 'parcellation_pooled_%s.h5' % h), 'w') as f:
            f['parcellation'] = P; f['coordinates'] = coords
    run_tree(root, 'final', 'pooled')
    zf = np.load(os.path.join(root, 'metrics', 'coupling', 'coupling.npz'))
    for fit in ('halfA', 'halfB'):
        lab = zf['labels_%s' % fit]
        for ev in ('halfA', 'halfB'):
            R = dense[ev].copy(); np.fill_diagonal(R, 0)
            M = np.zeros((kk, kk))
            for a in range(kk):
                for b in range(kk):
                    ia, ib = np.flatnonzero(lab == a), np.flatnonzero(lab == b)
                    cnt = len(ia) * len(ib) - (len(ia) if a == b else 0)
                    M[a, b] = R[np.ix_(ia, ib)].sum() / cnt if cnt else 0
            ok &= np.allclose(zf['M_%s_on_%s' % (fit, ev)], M, atol=1e-6)
check('network_coupling on tiled float16 halves equals dense block means of |r| (both trees, all fit x eval pairs)', ok)
shutil.rmtree(tmp)

print('\n%d checks, %d failure(s)' % (n_checks[0], len(failures)))
for f_ in failures:
    print('  FAIL', f_)
sys.exit(1 if failures else 0)
