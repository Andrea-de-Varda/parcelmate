"""Verification for the Iteration 30 additions (circuits against networks).

    PYTHONPATH=. python tests/verify_iter16_circuits.py [path/to/LLM_Modularity]

Covers: the circuit selection and the overlap ratio against the original repository's
`get_sorted_indices` / `get_top_neurons` / `compute_overlap_matrix` (skipped without a
clone); the flat-index mapping under shuffled coordinates; the layer-matched null keeping
each layer's count; test 1 on planted cases with LAYER-LOCAL networks, the hard case: a
circuit planted in two networks is strongly concentrated, a circuit random within one layer
is not, although a null ignoring layers would call it concentrated; test 2 finding exactly
the planted networks after FDR; test 3 finding shared structure beyond layers when tasks
share networks within one layer, and nothing when they do not; the CLI end to end.
"""

import csv
import os
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy as np

from parcelmate.circuits import (
    benjamini_hochberg, circuit_indices, concentration, concentration_test, enrichment,
    flat_labels, layer_matched_draws, overlap_matrix, structure_test,
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

# ---------------------------------------------------------------- against the original
orig = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, '..', 'LLM_Modularity')
if os.path.isdir(os.path.join(orig, 'scripts')):
    import torch
    sys.path.insert(0, os.path.join(orig, 'scripts'))
    sys.path.insert(0, orig)
    import run_overlap as ro
    ok = True
    sets = []
    for t in range(3):
        A = rng.randn(6, 400) * (rng.rand(6, 400) > 0.3)
        d = os.path.join(tmp, 'orig%d' % t)
        os.makedirs(d)
        path = os.path.join(d, 'neuron_attribution.pt')
        torch.save(torch.tensor(A), path)
        sidx, total = ro.get_sorted_indices(path, 'positive', 'neurons')
        for pct in (0.1, 1, 5):
            theirs, k = ro.get_top_neurons(sidx, pct, total)
            mine = circuit_indices(A, pct)
            ok &= set((int(i) // 400, int(i) % 400) for i in mine) == set(theirs) and len(mine) == k
        theirs5, k5 = ro.get_top_neurons(sidx, 5, total)
        sets.append(('t%d' % t, theirs5, k5))
    check('circuit selection equals the original (top pct%% of all units by positive score)', ok)
    Mo, _ = ro.compute_overlap_matrix(sets)
    mine_sets = [np.array([l * 400 + u for l, u in s[1]]) for s in sets]
    check('overlap ratio equals the original compute_overlap_matrix', np.allclose(Mo, overlap_matrix(mine_sets)))
else:
    print('SKIP original-code comparison (no clone at %s)' % orig)

# ---------------------------------------------------------------- mapping and null
L, W = 4, 500
k = 40
# Layer-local networks: network c lives in layer c // 10, 50 units each.
lab_flat = np.repeat(np.arange(k), W * L // k)
coords = np.stack([np.arange(L * W) // W, np.arange(L * W) % W], 1)
perm = rng.permutation(L * W)
P = np.eye(k)[lab_flat[perm]]
check('flat_labels maps shuffled (layer, neuron) rows back to their units',
      np.array_equal(flat_labels(P, coords[perm], L, W), lab_flat))
circ = np.concatenate([rng.choice(50, 20, replace=False), 50 + rng.choice(50, 20, replace=False),
                       W + rng.choice(W, 10, replace=False)])
draws = layer_matched_draws(circ, W, L, 50, np.random.RandomState(1))
check('layer-matched draws keep the size and every layer\'s count, without repeats',
      draws.shape == (50, 50) and all(np.array_equal(np.bincount(d // W, minlength=L), [40, 10, 0, 0])
                                      and len(set(d)) == 50 for d in draws))

# ---------------------------------------------------------------- test 1 and 2 on planted cases
A_circ = np.concatenate([rng.choice(50, 20, replace=False), 50 + rng.choice(50, 20, replace=False)])  # networks 0, 1
B_circ = rng.choice(W, 40, replace=False)                                                           # random in layer 0
resA, nullA = concentration_test(A_circ, lab_flat, k, W, L, 1000, np.random.RandomState(2))
resB, nullB = concentration_test(B_circ, lab_flat, k, W, L, 1000, np.random.RandomState(3))
check('test 1: a circuit planted in two networks is concentrated (2 effective networks, z << 0, p at its floor)',
      abs(resA['effective_networks'] - 2) < 1e-9 and resA['entropy_z'] < -5 and resA['p_concentrated'] < 0.002)
check('test 1: a circuit random within one layer is not concentrated under the layer-matched null',
      resB['p_concentrated'] > 0.05 and abs(resB['entropy_z']) < 2.5)
naive = np.array([concentration(lab_flat[rng.choice(L * W, 40, replace=False)], k)['entropy'] for _ in range(1000)])
check('the layer match matters: a null ignoring layers would call the within-layer circuit concentrated',
      (1 + (naive <= resB['entropy']).sum()) / 1001.0 < 0.01)
o, e, ratio, p = enrichment(A_circ, lab_flat, k, nullA)
q = benjamini_hochberg(p)
check('test 2: exactly the two planted networks are enriched after FDR',
      set(np.flatnonzero(q < 0.05)) == {0, 1} and ratio[0] > 3)
_, _, _, pB = enrichment(B_circ, lab_flat, k, nullB)
check('test 2: no network is enriched for the within-layer random circuit', (benjamini_hochberg(pB) < 0.05).sum() == 0)
from scipy.stats import false_discovery_control
pp = np.random.RandomState(9).rand(50) ** 3
check('benjamini_hochberg equals scipy.stats.false_discovery_control (BH)',
      np.allclose(benjamini_hochberg(pp), false_discovery_control(pp, method='bh'))
      and np.allclose(benjamini_hochberg([0.01, 0.04, 0.03, 0.5]), [0.04, 0.16 / 3, 0.16 / 3, 0.5]))

# ---------------------------------------------------------------- test 3
def planted_task(nets, rs):
    return np.concatenate([n * 50 + rs.choice(50, 20, replace=False) for n in nets])
rs = np.random.RandomState(4)
circuits = [planted_task((0, 1), rs) for _ in range(4)] + [planted_task((2, 3), rs) for _ in range(4)]
doms = ['X'] * 4 + ['Y'] * 4
st = structure_test(circuits, lab_flat, k, W, L, doms, n_perm=500, rng=np.random.RandomState(5))
check('test 3: tasks sharing networks within ONE layer show structure beyond layers and a domain contrast',
      st['partial_overlap_network_given_layer'] > 0.3 and st['p_partial'] < 0.05
      and st['domain_contrast_network'] > 0.5 and st['p_domain_contrast'] < 0.05
      and abs(st['domain_contrast_layer']) < 1e-9)
rand_c = [rs.choice(W, 40, replace=False) for _ in range(8)]
st0 = structure_test(rand_c, lab_flat, k, W, L, doms, n_perm=500, rng=np.random.RandomState(6))
check('test 3: tasks random within one layer show no domain contrast', st0['p_domain_contrast'] > 0.05)
# (cross-domain pairs sit BELOW chance here, since both domains use layer 0, so only the
# contrast is expected to be positive, not the mean over all pairs)
check('test 3 (non-shared units): planted shared networks give a same-domain excess contrast',
      st['domain_contrast_excess'] > 0.3 and st['p_domain_contrast_excess'] < 0.05)
# The circularity: circuits that share units, on a partition that knows nothing about them.
# Pairs overlap by varying amounts; the raw measure then tracks overlap on ANY partition, the
# non-shared excess measure does not.
rs2 = np.random.RandomState(7)
core = rs2.choice(L * W, 40, replace=False)
shared_c = [np.concatenate([core[:rs2.randint(0, 40)], rs2.choice(L * W, 40, replace=False)])[:40] for _ in range(10)]
shared_c = [np.unique(c) for c in shared_c]
rand_part = rs2.randint(0, k, L * W)
st_c = structure_test(shared_c, rand_part, k, W, L, ['X'] * 5 + ['Y'] * 5, n_perm=300, rng=np.random.RandomState(8))
check('test 3 circularity: on a random partition the raw measure tracks overlap, the non-shared excess does not',
      st_c['spearman_overlap_network'] > 0.5 and abs(st_c['mean_excess_similarity']) < 0.05
      and st_c['p_overlap_excess'] > 0.05)

# ---------------------------------------------------------------- CLI end to end
patching = os.path.join(tmp, 'patching')
nets = os.path.join(tmp, 'nets')
for dom, net_sets in (('Lan', [(0, 1), (0, 1)]), ('MD', [(12, 13), (12, 13)])):
    for i, ns in enumerate(net_sets):
        a = np.zeros((L, W))
        for u in planted_task(ns, rs):
            a[u // W, u % W] = 1.0 + rs.rand()
        a += 0.01 * rs.randn(L, W)
        d = os.path.join(patching, dom, '%s_task%d' % (dom, i))
        os.makedirs(d)
        np.save(os.path.join(d, 'neuron_attribution.npy'), a)
for tree_suffix, labels in (('', lab_flat), ('_null', rs.permutation(lab_flat))):
    d = os.path.join(nets + tree_suffix, 'final', 'parcellation')
    os.makedirs(d)
    for dom in ('wikitext', 'bookcorpus'):
        for h in ('halfA', 'halfB'):
            with h5py.File(os.path.join(d, 'parcellation_%s_%s.h5' % (dom, h)), 'w') as f:
                f.create_dataset('parcellation', data=np.eye(k)[labels])
                f.create_dataset('coordinates', data=coords)
r = subprocess.run([sys.executable, '-m', 'parcelmate.bin.compare_circuits', '--patching', patching,
                    '--networks', nets, '--pct', '2', '--draws', '200', '--perm', '100'],
                   cwd=ROOT, env=ENV, capture_output=True, text=True)
out = os.path.join(nets, 'circuits')
summ = list(csv.DictReader(open(os.path.join(out, 'summary.csv')))) if r.returncode == 0 else []
check('CLI: a summary row per text domain x half x tree, and all four tables written',
      r.returncode == 0 and len(summ) == 2 * 2 * 2
      and all(os.path.exists(os.path.join(out, f)) for f in ('concentration.csv', 'enrichment.csv', 'structure.csv')))
real = [s for s in summ if s['tree'] == 'real']
null = [s for s in summ if s['tree'] == 'null']
check('CLI: planted circuits are concentrated on the real partition, not on the shuffled null partition',
      all(float(s['frac_tasks_p05']) == 1.0 for s in real) and all(float(s['frac_tasks_p05']) <= 0.5 for s in null))
check('CLI: outputs are group writable', oct(os.stat(os.path.join(out, 'summary.csv')).st_mode & 0o777) in ('0o664', '0o666'))

shutil.rmtree(tmp)
print('\n%d checks, %d failure(s)' % (n_checks[0], len(failures)))
for f in failures:
    print('  FAIL', f)
sys.exit(1 if failures else 0)
