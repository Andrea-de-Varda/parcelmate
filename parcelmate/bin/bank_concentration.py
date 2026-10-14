"""Test 1 of the circuit comparison (concentration) for every task of several attribution sets.

    python -m parcelmate.bin.bank_concentration \
        --patching results/patching_bank/Qwen_Qwen3-5-2B results/patching/Qwen_Qwen3-5-2B \
        --networks results/qwen35/qwen3.5-2b-resid-pool4 --out figures/bank_concentration_2b.csv

Only the concentration test of `compare_circuits` (LOG.md Iteration 30), for every task found
as `<root>/<group>/<task>/neuron_attribution.npy` under each --patching root (the task bank:
group = source; LLM_Modularity: group = task domain), on every `final` partition of
--networks and its `_null` tree. Same circuit definition (top --pct% of units by positive
attribution), same layer-matched null and the same seed keys as `compare_circuits`, so a task
scored by both gets identical numbers. One row per task x partition: the circuit's effective
number of networks and the mean over the random sets, with z and p.
"""

import argparse
import csv
import glob
import os

import numpy as np

from parcelmate.bin.compare_circuits import load_partitions
from parcelmate.circuits import circuit_indices, concentration_test, flat_labels
from parcelmate.util import derive_seed, stderr

os.umask(0o002)


def load_tasks(roots):
    tasks = []
    for root in roots:
        for p in sorted(glob.glob(os.path.join(root, '*', '*', 'neuron_attribution.npy'))):
            task_dir = os.path.dirname(p)
            tasks.append(dict(root=os.path.basename(os.path.dirname(root.rstrip('/'))),
                              group=os.path.basename(os.path.dirname(task_dir)),
                              task=os.path.basename(task_dir), attribution=np.load(p)))
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--patching', nargs='+', required=True)
    ap.add_argument('--networks', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pct', type=float, default=0.1)
    ap.add_argument('--draws', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--variant', default='final')
    args = ap.parse_args()

    tasks = load_tasks(args.patching)
    assert tasks, 'no neuron_attribution.npy under %s' % args.patching
    n_layers, width = tasks[0]['attribution'].shape
    assert all(t['attribution'].shape == (n_layers, width) for t in tasks), 'attribution shapes differ'
    parts = load_partitions(args.networks, args.variant)
    assert parts, 'no partitions under %s' % args.networks
    stderr('%d tasks, %d partitions, %d layers x %d neurons\n' % (len(tasks), len(parts), n_layers, width))

    circuits = [circuit_indices(t['attribution'], args.pct) for t in tasks]
    rows = []
    for (domain, half, tree), (P, coords) in sorted(parts.items()):
        k = P.shape[1]
        labels = flat_labels(P, coords, n_layers, width)
        for t, c in zip(tasks, circuits):
            rng = np.random.RandomState(derive_seed(args.seed, 'circuit_null', domain, half, tree, args.pct,
                                                    t['task']) % (2 ** 32))
            res, _ = concentration_test(c, labels, k, width, n_layers, args.draws, rng)
            rows.append(dict(text_domain=domain, half=half, tree=tree, pct=args.pct, root=t['root'],
                             group=t['group'], task=t['task'], **res))
        stderr('  %s %s %s: median effective networks %.1f, random %.1f\n' % (
            domain, half, tree, np.median([r['effective_networks'] for r in rows[-len(tasks):]]),
            np.median([r['null_effective_networks_mean'] for r in rows[-len(tasks):]])))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    stderr('wrote %s (%d rows)\n' % (args.out, len(rows)))


if __name__ == '__main__':
    main()
