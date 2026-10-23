"""Do the networks a circuit spans prefer each other? (LOG.md Iteration 46.)

    python -m parcelmate.bin.circuit_cohesion \
        --patching results/patching_bank/Qwen_Qwen3-5-2B results/patching/Qwen_Qwen3-5-2B \
        --coupling results/richclub/qwen3.5-2b-resid-pool4-tiles/metrics/coupling/coupling.npz \
        --out figures/circuit_cohesion_2b.csv

--coupling is the real tree's file from `network_coupling`; the null tree's file is read from
the matching `_null` path. For each tree and each direction (partition fitted on one half,
coupling measured on the other half's connectome), every task's circuit (top --pct% of units
by positive attribution, as in the concentration test) gives shares over the networks, and
`parcelmate.cohesion` gives its cohesion (descriptive) and its preference ratio (coupling among
its networks relative to depth-matched outside networks). One row per task x tree x direction.
"""

import argparse
import csv
import os

import numpy as np

from parcelmate.bin.bank_concentration import load_tasks
from parcelmate.circuits import circuit_indices
from parcelmate.cohesion import (cohesion, depth_spread, network_layers, normalized_coupling, preference,
                                 unmatched_preference)
from parcelmate.constants import HALF_NAMES
from parcelmate.util import stderr


def null_path(path):
    tree = path.split(os.sep + 'metrics' + os.sep)[0]
    return path.replace(tree, tree.rstrip('/') + '_null', 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--patching', nargs='+', required=True)
    ap.add_argument('--coupling', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pct', type=float, default=0.1)
    ap.add_argument('--n-near', type=int, default=3, help='outside networks per circuit network, nearest in depth')
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
            Bn = normalized_coupling(z['M_%s_on_%s' % (fit, ev)])
            k = Bn.shape[0]
            unit_net = np.empty(n_layers * width, dtype=np.int64)
            unit_net[flat] = labels
            depth = network_layers(labels, coords, k)
            between = float(np.median(Bn[~np.eye(k, dtype=bool)]))
            stderr('%s fit %s eval %s: between-network coupling median %.3f\n' % (tree, fit, ev, between))
            for t, c in zip(tasks, circuits):
                p = np.bincount(unit_net[c], minlength=k) / float(len(c))
                nz = p[p > 0]
                rows.append(dict(tree=tree, fit=fit, eval=ev, pct=args.pct, n_near=args.n_near, root=t['root'],
                                 group=t['group'], task=t['task'], cohesion=cohesion(p, Bn),
                                 preference=preference(p, Bn, depth, args.n_near),
                                 preference_unmatched=unmatched_preference(p, Bn), depth_spread=depth_spread(p, depth),
                                 n_networks_spanned=int((p > 0).sum()),
                                 effective_networks=float(np.exp(-np.sum(nz * np.log(nz)))),
                                 median_between_coupling=between))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    stderr('wrote %s (%d rows)\n' % (args.out, len(rows)))


if __name__ == '__main__':
    main()
