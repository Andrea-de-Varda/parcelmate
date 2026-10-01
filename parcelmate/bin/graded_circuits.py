"""Graded comparison of attribution maps with networks, on all units (LOG.md Iteration 35).

    python -m parcelmate.bin.graded_circuits --patching results/patching/Qwen_Qwen3-5-2B \
        --networks results/qwen35/qwen3.5-2b [--variant final] [--out DIR]

For every partition (`<variant>/parcellation/parcellation_<domain>_<half>.h5`, real tree and
`_null` tree) and every task, plus each task domain's averaged map (each task rescaled to
mean |a| = 1, as in step B): `graded_network_variance`. Writes into --out (default
<networks>/circuits for the final variant, <networks>/circuits_<variant> otherwise):

  graded_tasks.csv    one row per text domain, half, tree, map (task or task domain)
  graded_summary.csv  per text domain, half and task group: median excess on the real and the
                      null partition, share of tasks p < 0.05 on each, and the paired
                      real - null excess (one-sided Wilcoxon, real > null)
"""

import argparse
import collections
import csv
import os

import numpy as np
from scipy import stats

from parcelmate.bin.compare_circuits import load_partitions, load_tasks, write
from parcelmate.circuits import domain_attribution, flat_labels, graded_network_variance
from parcelmate.util import derive_seed, stderr

os.umask(0o002)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--patching', required=True)
    ap.add_argument('--networks', required=True)
    ap.add_argument('--variant', default='final')
    ap.add_argument('--out', default=None)
    ap.add_argument('--perm', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    out = args.out or os.path.join(args.networks, 'circuits' if args.variant == 'final' else 'circuits_' + args.variant)
    os.makedirs(out, exist_ok=True)
    tasks = load_tasks(args.patching)
    n_layers, width = tasks[0]['attribution'].shape
    names = [t['task'] for t in tasks]
    doms = [t['domain'] for t in tasks]
    dmaps = domain_attribution([t['attribution'] for t in tasks], doms, rescale=True)
    A = np.stack([t['attribution'].ravel() for t in tasks] + [dmaps[d].ravel() for d in sorted(dmaps)])
    kinds = ['task'] * len(tasks) + ['domain'] * len(dmaps)
    labels_map = names + sorted(dmaps)
    domains_map = doms + sorted(dmaps)
    parts = load_partitions(args.networks, args.variant)
    assert parts, 'no partitions for variant %s under %s' % (args.variant, args.networks)
    rows = []
    for (domain, half, tree), (P, coords) in sorted(parts.items()):
        k = P.shape[1]
        lab = flat_labels(P, coords, n_layers, width)
        rng = np.random.RandomState(derive_seed(args.seed, 'graded', domain, half, tree, args.variant) % (2 ** 32))
        res = graded_network_variance(A, lab, k, width, n_layers, args.perm, rng)
        for kind, name, td, r in zip(kinds, labels_map, domains_map, res):
            rows.append(dict(text_domain=domain, half=half, tree=tree, variant=args.variant, k=k, kind=kind,
                             task_domain=td, map=name, **r))
        tr = [r for r, kd in zip(res, kinds) if kd == 'task']
        stderr('  %-10s %-5s %-4s k %3d: median excess %.4f (eta2 %.4f vs %.4f), %3.0f%% of tasks p<0.05\n' % (
            domain, half, tree, k, np.median([r['excess'] for r in tr]), np.median([r['eta2_network'] for r in tr]),
            np.median([r['null_mean'] for r in tr]), 100 * np.mean([r['p'] < 0.05 for r in tr])))
    write(os.path.join(out, 'graded_tasks.csv'), rows)
    by = {(r['text_domain'], r['half'], r['tree'], r['map']): r for r in rows if r['kind'] == 'task'}
    groups = collections.defaultdict(list)
    for (td, half, tree, m), r in by.items():
        if tree != 'real' or (td, half, 'null', m) not in by:
            continue
        pair = (r, by[(td, half, 'null', m)])
        groups[(td, half, 'all')].append(pair)
        groups[(td, half, r['task_domain'])].append(pair)
    summ = []
    for (td, half, g), pairs in sorted(groups.items()):
        er = np.array([a['excess'] for a, b in pairs])
        en = np.array([b['excess'] for a, b in pairs])
        d = er - en
        summ.append(dict(text_domain=td, half=half, variant=args.variant, task_group=g, n=len(pairs),
                         median_excess_real=float(np.median(er)), median_excess_null=float(np.median(en)),
                         frac_p05_real=float(np.mean([a['p'] < 0.05 for a, b in pairs])),
                         frac_p05_null=float(np.mean([b['p'] < 0.05 for a, b in pairs])),
                         mean_diff=float(d.mean()), frac_real_gt_null=float((d > 0).mean()),
                         p_wilcoxon=float(stats.wilcoxon(d, alternative='greater').pvalue) if np.any(d != 0) else 1.0))
    write(os.path.join(out, 'graded_summary.csv'), summ)


if __name__ == '__main__':
    main()
