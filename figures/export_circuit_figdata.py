"""Per-task and per-pair data for the explanatory circuit figures (LOG.md Iteration 36).

    PYTHONPATH=. python figures/export_circuit_figdata.py

Worked example: Qwen3.5-2B, networks fitted on wikitext, half A, real and null partition,
0.1% circuits. Needs results/patching/Qwen_Qwen3-5-2B/*/*/neuron_attribution.npy and
results/qwen35/qwen3.5-2b{,_null}/final/parcellation/parcellation_wikitext_halfA.h5.

Writes figures/circuits_example_2b_wikitext.npz: task names and domains; per tree the
circuit's count per network, the layer-matched expectation, the effective number of
networks of the circuit and of each of the 1,000 random sets (same seeds as
compare_circuits, so these are the production draws); the task x task circuit overlap and
the non-shared-unit excess network similarity (200 draws per pair, own seed).
"""

import os

import h5py
import numpy as np

from parcelmate.bin.compare_circuits import load_tasks
from parcelmate.circuits import (
    circuit_indices, concentration, excess_network_similarity, flat_labels, layer_matched_draws,
    overlap_matrix,
)
from parcelmate.util import derive_seed

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DOMAIN, HALF, PCT, SEED, DRAWS = 'wikitext', 'halfA', 0.1, 42, 1000
ORDER = ['Lan', 'MD', 'ToM', 'phys']

tasks = load_tasks(os.path.join(ROOT, 'results', 'patching', 'Qwen_Qwen3-5-2B'))
tasks = sorted(tasks, key=lambda t: (ORDER.index(t['domain']), t['task']))
L, W = tasks[0]['attribution'].shape
circuits = [circuit_indices(t['attribution'], PCT) for t in tasks]
out = dict(task=np.array([t['task'] for t in tasks]), domain=np.array([t['domain'] for t in tasks]),
           overlap=(lambda O: (O + O.T) / 2)(overlap_matrix(circuits)))
for tree, root in (('real', 'qwen3.5-2b'), ('null', 'qwen3.5-2b_null')):
    path = os.path.join(ROOT, 'results', 'qwen35', root, 'final', 'parcellation', 'parcellation_%s_%s.h5' % (DOMAIN, HALF))
    with h5py.File(path, 'r') as f:
        lab = flat_labels(np.asarray(f['parcellation']), np.asarray(f['coordinates']), L, W)
    k = int(lab.max()) + 1
    obs, exp_, eff, eff_draws = [], [], [], []
    for t, c in zip(tasks, circuits):
        rng = np.random.RandomState(derive_seed(SEED, 'circuit_null', DOMAIN, HALF, tree, PCT, t['task']) % (2 ** 32))
        draws = lab[layer_matched_draws(c, W, L, DRAWS, rng)]
        counts = np.stack([np.bincount(r, minlength=k) for r in draws]).astype(float)
        p = counts / counts.sum(1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            H = -np.where(p > 0, p * np.log(p), 0.0).sum(1)
        obs.append(np.bincount(lab[c], minlength=k))
        exp_.append(counts.mean(0))
        eff.append(concentration(lab[c], k)['effective_networks'])
        eff_draws.append(np.exp(H))
    out['%s_observed' % tree] = np.array(obs)
    out['%s_expected' % tree] = np.array(exp_)
    out['%s_eff' % tree] = np.array(eff)
    out['%s_eff_draws' % tree] = np.array(eff_draws, dtype=np.float32)
    out['%s_similarity' % tree] = excess_network_similarity(circuits, lab, k, W, L, 200,
                                                            np.random.RandomState(derive_seed(SEED, 'fig_similarity', tree) % (2 ** 32)))
    print(tree, 'done: median effective networks %.1f, random %.1f' % (np.median(out['%s_eff' % tree]),
                                                                       np.median(out['%s_eff_draws' % tree].mean(1))))
np.savez_compressed(os.path.join(ROOT, 'figures', 'circuits_example_2b_wikitext.npz'), **out)
print('wrote figures/circuits_example_2b_wikitext.npz')
