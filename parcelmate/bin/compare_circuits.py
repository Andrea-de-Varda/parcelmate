"""Compare attribution-patching circuits with connectivity networks (LOG.md Iteration 30).

    python -m parcelmate.bin.compare_circuits \
        --patching results/patching/Qwen_Qwen3-5-2B --networks results/qwen35/qwen3.5-2b

Reads every `<task domain>/<task>/neuron_attribution.npy` under --patching (only tasks that
passed the inclusion rule have one) and every `parcellation_<domain>_<half>.h5` of the
`final` variant under --networks and its `_null` tree. For each text domain, half, tree and
circuit size (--pct) it runs the three tests of `parcelmate.circuits` and writes, into
--out (default <networks>/circuits):

  concentration.csv  one row per task: observed entropy, effective networks, max and top-3
                     shares, the layer-matched null's mean and sd, z, and p (test 1)
  enrichment.csv     one row per task x network: observed, expected, ratio, p, BH q (test 2)
  structure.csv      one row per measure: overlap against network similarity, controlling
                     for layer similarity, and the same-domain contrast (test 3)
  summary.csv        per text domain, half, tree and pct: median z, fraction of tasks with
                     p < 0.05, median effective networks observed and under the null

Files are group readable and writable.
"""

import argparse
import csv
import glob
import os

import h5py
import numpy as np

from parcelmate.circuits import (
    benjamini_hochberg, circuit_indices, concentration_test, enrichment, flat_labels,
    structure_test,
)
from parcelmate.util import derive_seed, stderr

os.umask(0o002)


def write(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    stderr('wrote %s (%d rows)\n' % (path, len(rows)))


def load_tasks(patching):
    tasks = []
    for p in sorted(glob.glob(os.path.join(patching, '*', '*', 'neuron_attribution.npy'))):
        task_dir = os.path.dirname(p)
        tasks.append(dict(domain=os.path.basename(os.path.dirname(task_dir)),
                          task=os.path.basename(task_dir), attribution=np.load(p)))
    return tasks


def load_partitions(networks, variant='final'):
    out = {}
    for tree, root in (('real', networks), ('null', networks.rstrip('/') + '_null')):
        for p in sorted(glob.glob(os.path.join(root, variant, 'parcellation', 'parcellation_*_half*.h5'))):
            name = os.path.basename(p)[len('parcellation_'):-len('.h5')]
            domain, half = name.rsplit('_', 1)
            with h5py.File(p, 'r') as f:
                out[(domain, half, tree)] = (np.asarray(f['parcellation']), np.asarray(f['coordinates']))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--patching', required=True)
    ap.add_argument('--networks', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--domains', nargs='+', default=None, help='text domains (default: every one found)')
    ap.add_argument('--pct', nargs='+', type=float, default=[0.1, 1.0])
    ap.add_argument('--draws', type=int, default=1000)
    ap.add_argument('--perm', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--variant', default='final')
    args = ap.parse_args()
    out = args.out or os.path.join(args.networks, 'circuits')
    os.makedirs(out, exist_ok=True)

    tasks = load_tasks(args.patching)
    assert tasks, 'no neuron_attribution.npy under %s' % args.patching
    n_layers, width = tasks[0]['attribution'].shape
    assert all(t['attribution'].shape == (n_layers, width) for t in tasks), 'attribution shapes differ'
    parts = load_partitions(args.networks, args.variant)
    assert parts, 'no partitions under %s' % args.networks
    stderr('%d tasks (%s), %d partitions, %d layers x %d neurons\n' % (
        len(tasks), ', '.join(sorted({t['domain'] for t in tasks})), len(parts), n_layers, width))

    conc, enr, struct, summ = [], [], [], []
    for (domain, half, tree), (P, coords) in sorted(parts.items()):
        if args.domains and domain not in args.domains:
            continue
        k = P.shape[1]
        labels = flat_labels(P, coords, n_layers, width)
        for pct in args.pct:
            circuits = [circuit_indices(t['attribution'], pct) for t in tasks]
            zs, ps, eff, effn = [], [], [], []
            for t, c in zip(tasks, circuits):
                rng = np.random.RandomState(derive_seed(args.seed, 'circuit_null', domain, half, tree, pct, t['task']) % (2 ** 32))
                res, null_counts = concentration_test(c, labels, k, width, n_layers, args.draws, rng)
                conc.append(dict(text_domain=domain, half=half, tree=tree, pct=pct,
                                 task_domain=t['domain'], task=t['task'], **res))
                zs.append(res['entropy_z']); ps.append(res['p_concentrated'])
                eff.append(res['effective_networks']); effn.append(res['null_effective_networks_mean'])
                o, e, ratio, p = enrichment(c, labels, k, null_counts)
                for net in range(k):
                    enr.append(dict(text_domain=domain, half=half, tree=tree, pct=pct,
                                    task_domain=t['domain'], task=t['task'], network=net,
                                    observed=o[net], expected=e[net], ratio=ratio[net], p=p[net]))
            # BH within this partition and circuit size, over all task x network pairs
            block = [r for r in enr if (r['text_domain'], r['half'], r['tree'], r['pct']) == (domain, half, tree, pct)]
            for r, q in zip(block, benjamini_hochberg([r['p'] for r in block])):
                r['q'] = q
            rng = np.random.RandomState(derive_seed(args.seed, 'circuit_structure', domain, half, tree, pct) % (2 ** 32))
            for name, v in structure_test(circuits, labels, k, width, n_layers, [t['domain'] for t in tasks],
                                          args.perm, rng).items():
                struct.append(dict(text_domain=domain, half=half, tree=tree, pct=pct, measure=name, value=v))
            summ.append(dict(text_domain=domain, half=half, tree=tree, pct=pct, n_tasks=len(tasks),
                             median_entropy_z=float(np.median(zs)),
                             frac_tasks_p05=float(np.mean(np.asarray(ps) < 0.05)),
                             median_effective_networks=float(np.median(eff)),
                             median_null_effective_networks=float(np.median(effn)),
                             n_enriched_q05=int(sum(r['q'] < 0.05 for r in block))))
            stderr('  %-10s %-5s %-4s pct %-4g median z %6.2f, %3.0f%% of tasks p<0.05, eff. networks %.1f vs %.1f\n' % (
                domain, half, tree, pct, summ[-1]['median_entropy_z'], 100 * summ[-1]['frac_tasks_p05'],
                summ[-1]['median_effective_networks'], summ[-1]['median_null_effective_networks']))
    write(os.path.join(out, 'concentration.csv'), conc)
    write(os.path.join(out, 'enrichment.csv'), enr)
    write(os.path.join(out, 'structure.csv'), struct)
    write(os.path.join(out, 'summary.csv'), summ)


if __name__ == '__main__':
    main()
