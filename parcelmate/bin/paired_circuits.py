"""Real against null partition, paired per task (LOG.md Iteration 33).

    python -m parcelmate.bin.paired_circuits results/qwen35/qwen3.5-4b-pool5 [more trees...]

Reads `<tree>/circuits/concentration.csv` and `domain_concentration.csv` (written by
`compare_circuits`) and writes, into `<tree>/circuits/`:

  paired_tasks.csv    one row per text domain, half, pct and task: the entropy deficit
                      (observed minus layer-matched mean, nats) on the real and the null
                      partition, their difference, and the same for z
  paired_summary.csv  one row per text domain, half, pct and task group (all tasks, then
                      each task domain): `paired_real_null` on the deficits
  paired_domains.csv  step B circuits (rescaled averaging): deficits on real and null and
                      their difference, per text domain, half, pct and task domain
"""

import argparse
import collections
import csv
import os

import numpy as np

from parcelmate.circuits import paired_real_null
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


def deficit(r):
    return float(r['entropy']) - float(r['null_entropy_mean'])


def pair(rows, keys):
    by = {}
    for r in rows:
        by[tuple(r[k] for k in keys) + (r['tree'],)] = r
    out = []
    for k, real in sorted(by.items()):
        if k[-1] != 'real':
            continue
        null = by.get(k[:-1] + ('null',))
        if null is None:
            continue
        out.append(dict(zip(keys, k[:-1]), deficit_real=deficit(real), deficit_null=deficit(null),
                        deficit_diff=deficit(real) - deficit(null),
                        z_real=float(real['entropy_z']), z_null=float(null['entropy_z']),
                        p_real=float(real['p_concentrated']), p_null=float(null['p_concentrated'])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('trees', nargs='+')
    ap.add_argument('--perm', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--sub', default='circuits', help='results subdirectory (circuits_k10 for a coarse variant)')
    args = ap.parse_args()
    for tree in args.trees:
        d = os.path.join(tree, args.sub)
        conc = list(csv.DictReader(open(os.path.join(d, 'concentration.csv'))))
        tasks = pair(conc, ('text_domain', 'half', 'pct', 'task_domain', 'task'))
        write(os.path.join(d, 'paired_tasks.csv'), tasks)
        groups = collections.defaultdict(list)
        for t in tasks:
            groups[(t['text_domain'], t['half'], t['pct'], 'all')].append(t)
            groups[(t['text_domain'], t['half'], t['pct'], t['task_domain'])].append(t)
        summ = []
        for (td, half, pct, grp), ts in sorted(groups.items()):
            rng = np.random.RandomState(derive_seed(args.seed, 'paired', td, half, pct, grp) % (2 ** 32))
            res = paired_real_null([t['deficit_real'] for t in ts], [t['deficit_null'] for t in ts], args.perm, rng)
            summ.append(dict(text_domain=td, half=half, pct=pct, task_group=grp, **res))
            if grp == 'all':
                stderr('  %-12s %-10s %-5s pct %-4s mean diff %+.3f, %3.0f%% of tasks real < null, p %.4f\n' % (
                    os.path.basename(tree.rstrip('/')), td, half, pct, res['mean_diff'],
                    100 * res['frac_real_more_concentrated'], res['p_wilcoxon']))
        write(os.path.join(d, 'paired_summary.csv'), summ)
        dom = [r for r in csv.DictReader(open(os.path.join(d, 'domain_concentration.csv'))) if r['averaging'] == 'rescaled']
        write(os.path.join(d, 'paired_domains.csv'), pair(dom, ('text_domain', 'half', 'pct', 'task_domain')))


if __name__ == '__main__':
    main()
