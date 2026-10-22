"""Cohesion of the networks each circuit spans, against layer-matched networks (LOG.md Iteration 46).

    python -m parcelmate.bin.circuit_cohesion \
        --patching results/patching_bank/Qwen_Qwen3-5-2B results/patching/Qwen_Qwen3-5-2B \
        --coupling results/richclub/qwen3.5-2b-resid-pool4-tiles/metrics/coupling/coupling.npz \
        --out figures/circuit_cohesion_2b.csv

--coupling is the real tree's file from `network_coupling`; the null tree's file is read from
the matching `_null` path. For each tree and each direction (partition fitted on one half,
coupling measured on the other half's connectome), every task's circuit (top --pct% of units by
positive attribution, as in the concentration test) gives shares over the networks; its
cohesion and specificity (`parcelmate.cohesion`) are tested against --draws permutations of
network identities within --bins bins of layer position. One row per task x tree x direction.
"""

import argparse
import csv
import os

import numpy as np

from parcelmate.bin.bank_concentration import load_tasks
from parcelmate.circuits import circuit_indices
from parcelmate.cohesion import cohesion_test, layer_bins, network_layers, normalized_coupling
from parcelmate.constants import HALF_NAMES
from parcelmate.util import derive_seed, stderr


def null_path(path):
    tree = path.split(os.sep + 'metrics' + os.sep)[0]
    return path.replace(tree, tree.rstrip('/') + '_null', 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--patching', nargs='+', required=True)
    ap.add_argument('--coupling', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pct', type=float, default=0.1)
    ap.add_argument('--bins', type=int, default=10)
    ap.add_argument('--draws', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    tasks = load_tasks(args.patching)
    n_layers, width = tasks[0]['attribution'].shape
    circuits = [circuit_indices(t['attribution'], args.pct) for t in tasks]
    rows = []
    for tree, path in (('real', args.coupling), ('null', null_path(args.coupling))):
        z = np.load(path)
        coords = z['coordinates'].astype(np.int64)
        flat = coords[:, 0] * width + coords[:, 1]
        for fit in HALF_NAMES:
            ev = [h for h in HALF_NAMES if h != fit][0]
            labels = z['labels_%s' % fit]
            M = z['M_%s_on_%s' % (fit, ev)]
            k = M.shape[0]
            unit_net = np.empty(n_layers * width, dtype=np.int64)
            unit_net[flat] = labels
            Bn = normalized_coupling(M)
            bins = layer_bins(network_layers(labels, coords, k), args.bins)
            off = Bn[~np.eye(k, dtype=bool)]
            stderr('%s fit %s eval %s: between-network coupling median %.3f (normalised)\n' % (
                tree, fit, ev, np.median(off)))
            for t, c in zip(tasks, circuits):
                shares = np.bincount(unit_net[c], minlength=k) / float(len(c))
                rng = np.random.RandomState(derive_seed(args.seed, 'cohesion', tree, fit, args.pct, t['task']) % (2 ** 32))
                res = cohesion_test(shares, Bn, bins, args.draws, rng)
                rows.append(dict(tree=tree, fit=fit, eval=ev, pct=args.pct, root=t['root'], group=t['group'],
                                 task=t['task'], median_between_coupling=float(np.median(off)), **res))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    stderr('wrote %s (%d rows)\n' % (args.out, len(rows)))


if __name__ == '__main__':
    main()
