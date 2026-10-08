"""Verification for the Iteration 36 additions (network stability across datasets).

    PYTHONPATH=. python tests/verify_iter18_stability.py

A planted case: 6 layers x 500 units, k = 30, five datasets with two halves each. Units
0-1,199 form 12 STABLE networks (the same in every dataset and half, 5% of units relabelled
per partition), units 1,200-2,399 form 12 DATASET-SPECIFIC networks (reliable within a
dataset, both halves equal up to 5% noise, but drawn afresh per dataset), and units
2,400-2,999 are 6 LAYER-LOCAL networks (a fixed fifth of one layer each), identical
everywhere. Stable and specific networks are spread over all layers.

Covers: best-match Dice on a known case; layer shuffling keeps per-layer composition; A
separates stable (generality near 1, present in all other datasets) from dataset-specific
networks (near 0) while both are reliable within a dataset; a layer-local network that is
the same subset of its layer everywhere counts as stable (layer shuffling removes only what
the per-layer composition predicts: chance about 0.2 for it, not 1), while a partition
that is nothing but layers is at chance; B finds the 12 stable and 6 layer-local networks
as span-5 meta-networks, reproduced across half sets, and none made of dataset-specific
networks; C gives stable units high consensus
scores and dataset-specific units low ones, above layer-shuffled chance; the CLI end to end.
"""

import csv
import os
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy as np

from parcelmate.stability import (
    best_dice, consensus, layer_shuffle, meta_networks, network_stability,
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


os.makedirs(os.path.join(HERE, '.tmp'), exist_ok=True)
tmp = tempfile.mkdtemp(dir=os.path.join(HERE, '.tmp'))
rng = np.random.RandomState(0)

a = np.array([0, 0, 1, 1, 2, 2])
b = np.array([1, 1, 0, 0, 0, 0])
d, j, cont, _ = best_dice(a, b, 3, 2)
check('best_dice: exact match 1, a network inside a merged one has Dice 2/3 and containment 1',
      np.isclose(d[0], 1) and j[0] == 1 and np.isclose(d[1], 2 / 3) and np.isclose(cont[1], 1))

L, W, k = 6, 500, 30
N = L * W
layer = np.arange(N) // W
# Units in layer order; networks span two adjacent layers except the bands.
stable_units = np.concatenate([np.arange(l * W, l * W + 200) for l in range(L)])      # 1,200
specific_units = np.concatenate([np.arange(l * W + 200, l * W + 400) for l in range(L)])  # 1,200
band_units = np.concatenate([np.arange(l * W + 400, l * W + 500) for l in range(L)])     # 600
stable_lab = np.repeat(np.arange(12), 100)                         # 12 networks over the stable units
order = rng.permutation(len(stable_units))
base = np.full(N, -1, dtype=np.int64)   # -1: dataset-specific units, set per dataset
base[stable_units[order]] = stable_lab
base[band_units] = 24 + layer[band_units]                          # networks 24-29: one per layer
datasets = ['d1', 'd2', 'd3', 'd4', 'd5']
spec = {d: rng.permutation(np.repeat(np.arange(12, 24), 100)) for d in datasets}


def noisy(lab, rs, p=0.05):
    out = lab.copy()
    m = rs.rand(N) < p
    out[m] = rs.randint(0, k, m.sum())
    return out


parts = {}
for tree in ('real',):
    for d in datasets:
        lab = base.copy()
        lab[specific_units] = spec[d]
        for h in ('halfA', 'halfB'):
            parts[(tree, d, h)] = noisy(lab, rng)

sh = layer_shuffle(parts[('real', 'd1', 'halfA')], layer, rng)
check('layer_shuffle keeps every layer\'s label composition',
      all(np.array_equal(np.sort(sh[layer == l]), np.sort(parts[('real', 'd1', 'halfA')][layer == l])) for l in range(L)))

rows = network_stability(parts, layer, k, n_shuffle=5, rng=np.random.RandomState(1))
by = {}
for r in rows:
    if r['dataset'] == 'd1' and r['half'] == 'halfA':
        by[r['network']] = r
st = [by[i] for i in range(12)]
sp = [by[i] for i in range(12, 24)]
bd = [by[i] for i in range(24, 30)]
check('A: stable networks generalise (generality > 0.8, present in all 4 other datasets)',
      all(r['generality'] > 0.8 and r['n_datasets_present'] == 4 for r in st))
check('A: dataset-specific networks are reliable within a dataset but do not generalise (generality < 0.2)',
      all(r['within'] > 0.85 and r['generality'] < 0.2 for r in sp))
check('A: layer-local networks fixed within their layer are stable (chance well below the match)',
      all(r['across_mean'] > 0.85 and r['chance'] < 0.4 and r['n_datasets_present'] == 4 for r in bd))
whole = {('real', dd, hh): layer.copy() for dd in datasets for hh in ('halfA', 'halfB')}
rw = network_stability(whole, layer, L, n_shuffle=3, rng=np.random.RandomState(4))
check('A: a partition that is exactly the layers is at chance (match = layer-shuffled chance, nothing present)',
      all(abs(r['across_mean'] - r['chance']) < 1e-9 and r['n_datasets_present'] == 0 for r in rw))

labs = {d: parts[('real', d, 'halfA')] for d in datasets}
labsB = {d: parts[('real', d, 'halfB')] for d in datasets}
ma, _ = meta_networks(labs, layer, k, rng=np.random.RandomState(2))
mb, _ = meta_networks(labsB, layer, k, rng=np.random.RandomState(3))
span5 = [m for m in ma if m['span'] == 5]
stable_sets = [set(np.flatnonzero(base == i)) for i in range(12)]
stable_sets += [set(np.flatnonzero(base == i)) for i in range(24, 30)]
check('B: the 12 stable and 6 layer-local networks are span-5 meta-networks whose cores recover them (Dice > 0.85)',
      sum(1 for s in stable_sets if any(2 * len(s & set(m['core'])) / (len(s) + len(m['core'])) > 0.85 for m in span5)) == 18)
check('B: no span-5 meta-network is built from dataset-specific units',
      all(np.mean(np.isin(m['core'], specific_units)) < 0.1 for m in span5) and len(span5) == 18)
from parcelmate.stability import match_meta
check('B: span-5 meta-networks reproduce across half sets (core Dice > 0.85)',
      np.all(match_meta(span5, [m for m in mb if m['span'] == 5]) > 0.85))

c, score = consensus(labs, k, n_init=5, seed=0)
_, chance = consensus({d: layer_shuffle(v, layer, rng) for d, v in labs.items()}, k, n_init=2, seed=0)
check('C: stable units score high, dataset-specific units low, layer-shuffled partitions low',
      np.median(score[stable_units]) >= 0.8 and np.median(score[specific_units]) <= 0.6
      and np.mean(chance >= 0.8) < np.mean(score >= 0.8))

# D on dense matrices: a network planted in dataset X only is cohesive in X, not in Y
from parcelmate.stability import cohesion_across, network_cohesion
n2, k2 = 400, 4
lab2 = np.repeat(np.arange(k2), n2 // k2)
def mat(blocks, rs):
    M = 0.05 + 0.02 * rs.rand(n2, n2)
    for b in blocks:
        idx = np.flatnonzero(lab2 == b)
        M[np.ix_(idx, idx)] += 0.3
    M = (M + M.T) / 2
    np.fill_diagonal(M, 1.0)
    return M
rs7 = np.random.RandomState(7)
co = cohesion_across({'X': mat([0, 1], rs7), 'Y': mat([0], rs7)}, {('X', 'real'): lab2})
cx = {(r['eval'], r['network']): r['cohesion'] for r in co}
check('D: network_cohesion = within minus mean between; a network cohesive in X and not in Y is caught',
      cx[('X', 0)] > 0.25 and cx[('Y', 0)] > 0.25 and cx[('X', 1)] > 0.25 and abs(cx[('Y', 1)]) < 0.05
      and np.allclose(network_cohesion(np.array([[3., 1.], [1., 2.]]))[0], [2., 1.]))

# stable sets vs circuits
from parcelmate.stability import set_enrichment, set_graded
S = np.zeros(N, dtype=bool); S[stable_units] = True
circ_in = rng.choice(stable_units, 60, replace=False)
circ_rand = rng.choice(N, 60, replace=False)
e_in = set_enrichment(circ_in, S, W, L, 500, np.random.RandomState(5))
e_rand = set_enrichment(circ_rand, S, W, L, 500, np.random.RandomState(6))
att = rng.randn(N); att[stable_units] += 1.0
gi = set_graded(att, S, W, L, 100, np.random.RandomState(7))
g0 = set_graded(rng.randn(N), S, W, L, 100, np.random.RandomState(8))
check('stable-set enrichment: exact layer-matched expectation, a circuit inside the set is enriched, a random one is not',
      abs(e_in['expected'] - 0.4) < 1e-9 and e_in['ratio'] > 2 and e_in['p'] < 0.01 and e_rand['p'] > 0.01
      and gi['p'] < 0.01 and gi['rank_diff'] > 0 and g0['p'] > 0.01)

# restart consensus (Iteration 38): planted shared structure across 3 'datasets' plus noise
from parcelmate.stability import restart_consensus, adjusted_rand as _ari
rs8 = np.random.RandomState(21)
truth = np.repeat(np.arange(10), 60)
def noisy(lab, p):
    out = lab.copy(); m = rs8.rand(len(lab)) < p
    out[m] = rs8.randint(0, 10, m.sum()); return rs8.permutation(10)[out]
labs = np.stack([noisy(truth, 0.45) for _ in range(3 * 20)])
c1, f1 = restart_consensus(labs[::2], 10, n_components=20, n_init=5, seed=0)
c2, f2 = restart_consensus(labs[1::2], 10, n_components=20, n_init=5, seed=1)
check('restart consensus recovers a structure shared by noisy labelings (ARI with truth > 0.95) and reproduces across restart sets',
      _ari(c1, truth) > 0.95 and _ari(c1, c2) > 0.95 and np.median([_ari(l, truth) for l in labs]) < 0.4
      and f1.min() > 0 and f1.max() <= 1)

# CLI
for d in datasets:
    for tree, sfx in (('real', ''), ('null', '_null')):
        pdir = os.path.join(tmp, 'tree' + sfx, 'final', 'parcellation')
        os.makedirs(pdir, exist_ok=True)
        for h in ('halfA', 'halfB'):
            lab = parts[('real', d, h)] if tree == 'real' else rng.permutation(parts[('real', d, h)])
            with h5py.File(os.path.join(pdir, 'parcellation_%s_%s.h5' % (d, h)), 'w') as f:
                f.create_dataset('parcellation', data=np.eye(k, dtype=np.float32)[lab])
                f.create_dataset('coordinates', data=np.stack([layer, np.arange(N) % W], 1))
out = os.path.join(tmp, 'stab')
r = subprocess.run([sys.executable, '-m', 'parcelmate.bin.stability', '--trees', os.path.join(tmp, 'tree'),
                    '--out', out, '--n-init', '3'], cwd=ROOT, env=ENV, capture_output=True, text=True)
summ = {(x['tree'], x['half']): x for x in csv.DictReader(open(os.path.join(out, 'summary.csv')))} if r.returncode == 0 else {}
check('CLI: all tables and the stable sets written, group-writable',
      r.returncode == 0 and all(os.path.exists(os.path.join(out, f)) for f in
                                ('networks.csv', 'meta_networks.csv', 'consensus.csv', 'summary.csv', 'stable_sets.npz'))
      and oct(os.stat(os.path.join(out, 'summary.csv')).st_mode & 0o777) in ('0o664', '0o666'))
check('CLI: 18 span-5 meta-networks on the real tree, none on the shuffled null tree',
      bool(summ) and int(summ[('real', 'halfA')]['B_n_meta_span5']) == 18 and int(summ[('null', 'halfA')]['B_n_meta_span5']) == 0)
if r.returncode != 0:
    print(r.stderr[-2000:])

shutil.rmtree(tmp)
print('\n%d checks, %d failure(s)' % (n_checks[0], len(failures)))
for f in failures:
    print('  FAIL', f)
sys.exit(1 if failures else 0)
