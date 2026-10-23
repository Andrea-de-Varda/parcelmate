"""Verification for Iteration 46 (do the networks a circuit spans prefer each other?).

    PYTHONPATH=. python tests/verify_iter46_cohesion.py

CPU only. Covers: normalised coupling; cohesion and the preference ratio equal brute-force
loops; network depth; on synthetic coupling that depends only on depth distance, layer-
concentrated circuits score a preference of about 1 while the unmatched in/out ratio is well
above 1 (why depth is matched); a planted group of networks scores above 1, an avoiding group
below 1, and hub networks coupled to everyone about 1 (sociability cancels); and
`network_coupling` end to end on tiled float16 halves equals dense block means of |r|.
"""

import os
import shutil
import sys
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import h5py
import numpy as np

from parcelmate.cohesion import cohesion, network_layers, normalized_coupling, preference, unmatched_preference

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
depth = rs.uniform(0, 24, k)
okC = okR = True
for _ in range(20):
    p = rs.dirichlet(np.ones(k) * 0.5) * (rs.rand(k) > 0.4)
    p = p / p.sum()
    inside = [i for i in range(k) if p[i] > 0]
    outside = [i for i in range(k) if p[i] == 0]
    num = den = denO = 0.0
    for i in inside:
        for j in inside:
            if i == j:
                continue
            near = sorted(outside, key=lambda m: (abs(depth[m] - depth[j]), m))[:3]
            num += p[i] * p[j] * Bn[i, j]; den += p[i] * p[j]
            denO += p[i] * p[j] * np.mean([Bn[i, m] for m in near])
    okC &= np.isclose(cohesion(p, Bn), num / den)
    okR &= np.isclose(preference(p, Bn, depth, 3), num / denO)
check('cohesion equals a brute-force double loop', okC)
check('preference ratio equals a brute-force loop (3 outside networks nearest in depth to each partner)', okR)
check('a single-network circuit has no pairs: cohesion and preference NaN',
      np.isnan(cohesion(np.eye(k)[3], Bn)) and np.isnan(preference(np.eye(k)[3], Bn, depth)))
check('a circuit covering every network has no outsiders: preference NaN',
      np.isnan(preference(np.ones(k) / k, Bn, depth)))
labels = np.repeat(np.arange(4), 3)
coords = np.c_[np.array([0, 0, 2, 4, 4, 4, 1, 1, 1, 7, 9, 8]), np.arange(12)]
check('network depth = mean layer of its units',
      np.allclose(network_layers(labels, coords, 4), [2 / 3, 4.0, 1.0, 8.0]))

# ---------------------------------------------------------------- planted structure
# 60 networks spread over 24 layers; coupling falls with depth distance, as in the model.
k = 60
depth = np.sort(rs.uniform(0, 24, k))
dist = np.abs(depth[:, None] - depth[None, :])
base = 0.65 - 0.02 * dist + rs.normal(0, 0.01, (k, k)); base = (base + base.T) / 2
np.fill_diagonal(base, 1.0)


pref, unm = [], []
for s in range(200):
    start = rs.randint(0, k - 12)
    idx = start + rs.choice(12, 5, replace=False)        # five networks within a narrow depth range
    q = np.zeros(k); q[idx] = rs.dirichlet(np.ones(5))
    pref.append(preference(q, base, depth)); unm.append(unmatched_preference(q, base))
check('depth only: layer-concentrated circuits score preference about 1 (median %.3f) while the unmatched ratio is inflated (median %.3f)'
      % (np.median(pref), np.median(unm)), abs(np.median(pref) - 1) < 0.03 and np.median(unm) > 1.15)
group = [10, 14, 20, 25, 33]
Bg = base.copy()
for i in group:
    for j in group:
        if i != j:
            Bg[i, j] += 0.15
q = np.zeros(k); q[group] = [0.3, 0.2, 0.2, 0.15, 0.15]
rg = preference(q, Bg, depth)
check('planted group (+0.15 among five networks): preference > 1.2 (%.2f)' % rg, rg > 1.2)
Ba = base.copy()
for i in group:
    for j in group:
        if i != j:
            Ba[i, j] -= 0.15
ra = preference(q, Ba, depth)
check('planted avoidance (-0.15): preference < 0.85 (%.2f)' % ra, ra < 0.85)
Bh = base.copy()
for i in group:
    Bh[i, :] = Bh[:, i] = 0.8          # hubs: coupled at the same high level to every network, each other included
np.fill_diagonal(Bh, 1.0)
rh = preference(q, Bh, depth)
check('hub networks (0.8 to every network): preference 1 (%.3f), cohesion raised (%.2f vs %.2f)'
      % (rh, cohesion(q, Bh), cohesion(q, base)), abs(rh - 1) < 0.01 and cohesion(q, Bh) > cohesion(q, base) + 0.1)
Bs = base.copy()
for i in group:
    Bs[i, :] += 0.2; Bs[:, i] += 0.2
np.fill_diagonal(Bs, 1.0)
rs_ = preference(q, Bs, depth)
check('sociable networks (+0.2 to everyone, so +0.4 to each other): preference > 1 (%.2f); they are coupled to each other more than to outsiders'
      % rs_, rs_ > 1.1)

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
