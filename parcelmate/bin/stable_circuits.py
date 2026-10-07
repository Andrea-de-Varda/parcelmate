"""Do circuits favour networks that are stable across datasets? (LOG.md Iteration 36.)

    python -m parcelmate.bin.stable_circuits --patching results/patching/Qwen_Qwen3-5-2B \
        --stability results/qwen35/stability_2b --circuits results/qwen35/qwen3.5-2b results/qwen35/qwen3.5-2b-rest

The hypothesis (Andrea, 2026-10-06), tested rather than assumed. (i) Unit level: for each
task, the share of its circuit (top 1% and 0.1%) inside the stable sets of
`bin/stability.py` (B: cores of meta-networks reaching at least 4 of the 5 datasets; C:
units whose consensus score is at least 0.8), against layer-matched random sets, and the
graded version on all units (attribution rank inside minus outside the set, membership
permuted within layers); the same sets from the null partitions as the neuron-property
reference. (ii) Network level: in every dataset's partition, Spearman between a network's
cross-dataset generality (A) and its circuit enrichment (observed / expected circuit units
summed over tasks, from `enrichment.csv`), partial on log size and effective layers.
Writes units.csv, units_summary.csv, networks.csv into --stability/circuits/.
"""

import argparse
import collections
import csv
import os

import numpy as np
from scipy import stats

from parcelmate.bin.compare_circuits import load_tasks
from parcelmate.bin.stability import write
from parcelmate.circuits import circuit_indices, partial_spearman
from parcelmate.stability import set_enrichment, set_graded
from parcelmate.util import derive_seed, stderr

os.umask(0o002)


def stable_sets(npz, n_datasets):
    """{(tree, half, kind): boolean mask} from stable_sets.npz."""
    z = np.load(npz, allow_pickle=False)
    n = len(z['layer'])
    out = {}
    for tree in ('real', 'null'):
        for h in ('halfA', 'halfB'):
            m = np.zeros(n, dtype=bool)
            i = 0
            while '%s_%s_meta%d_core' % (tree, h, i) in z:
                nodes = z['%s_%s_meta%d_nodes' % (tree, h, i)]
                if len({str(x).split(':')[0] for x in nodes}) >= n_datasets - 1:
                    m[z['%s_%s_meta%d_core' % (tree, h, i)]] = True
                i += 1
            out[(tree, h, 'B_meta_cores')] = m
            out[(tree, h, 'C_consensus_core')] = z['%s_%s_score' % (tree, h)] >= 0.8
    return out, z


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--patching', required=True)
    ap.add_argument('--stability', required=True)
    ap.add_argument('--circuits', nargs='+', required=True, help='trees whose circuits/enrichment.csv to read')
    ap.add_argument('--pct', nargs='+', type=float, default=[1.0, 0.1])
    ap.add_argument('--draws', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    out = os.path.join(args.stability, 'circuits')
    os.makedirs(out, exist_ok=True)
    tasks = load_tasks(args.patching)
    n_layers, width = tasks[0]['attribution'].shape
    datasets = [str(x) for x in np.load(os.path.join(args.stability, 'stable_sets.npz'))['datasets']]
    sets, _ = stable_sets(os.path.join(args.stability, 'stable_sets.npz'), len(datasets))

    # (i) unit level
    rows = []
    for (tree, h, kind), mask in sorted(sets.items()):
        stderr('  %s %s %s: %d units (%.1f%%)\n' % (tree, h, kind, mask.sum(), 100 * mask.mean()))
        for t in tasks:
            rng = np.random.RandomState(derive_seed(args.seed, 'stable', tree, h, kind, t['task']) % (2 ** 32))
            g = set_graded(t['attribution'], mask, width, n_layers, 200, rng)
            for pct in args.pct:
                e = set_enrichment(circuit_indices(t['attribution'], pct), mask, width, n_layers, args.draws, rng)
                rows.append(dict(tree=tree, half=h, set=kind, set_units=int(mask.sum()), pct=pct,
                                 task_domain=t['domain'], task=t['task'], **e, rank_diff=g['rank_diff'], p_graded=g['p']))
    write(os.path.join(out, 'units.csv'), rows)
    summ = []
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r['tree'], r['set'], r['pct'], 'all')].append(r)
        groups[(r['tree'], r['set'], r['pct'], r['task_domain'])].append(r)
    for (tree, kind, pct, g), rr in sorted(groups.items()):
        per = collections.defaultdict(list)
        for r in rr:
            per[r['task']].append(r)
        lr = np.array([np.mean([np.log(max(x['ratio'], 1e-6)) for x in v]) for v in per.values()])
        rd = np.array([np.mean([x['rank_diff'] for x in v]) for v in per.values()])
        summ.append(dict(tree=tree, set=kind, pct=pct, task_group=g, n_tasks=len(lr),
                         median_ratio=float(np.exp(np.median(lr))),
                         frac_tasks_ratio_gt1=float((lr > 0).mean()),
                         p_wilcoxon_ratio=float(stats.wilcoxon(lr, alternative='greater').pvalue) if np.any(lr != 0) else 1.0,
                         frac_tasks_p05=float(np.mean([np.mean([x['p'] for x in v]) < 0.05 for v in per.values()])),
                         mean_rank_diff=float(rd.mean()),
                         p_wilcoxon_rank=float(stats.wilcoxon(rd, alternative='greater').pvalue) if np.any(rd != 0) else 1.0))
        if g == 'all':
            s = summ[-1]
            stderr('  %-4s %-16s pct %-4g circuit share in set %.2fx chance (%.0f%% of tasks > 1, p %.2g); rank diff %+.4f (p %.2g)\n' % (
                tree, kind, pct, s['median_ratio'], 100 * s['frac_tasks_ratio_gt1'], s['p_wilcoxon_ratio'],
                s['mean_rank_diff'], s['p_wilcoxon_rank']))
    write(os.path.join(out, 'units_summary.csv'), summ)

    # (ii) network level
    nets = collections.defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(args.stability, 'networks.csv'))):
        nets[(r['tree'], r['dataset'], r['half'])][int(r['network'])] = r
    rows2 = []
    for root in args.circuits:
        p = os.path.join(root, 'circuits', 'enrichment.csv')
        if not os.path.exists(p):
            continue
        acc = collections.defaultdict(lambda: np.zeros((2, 1000)))
        for r in csv.DictReader(open(p)):
            key = (r['tree'], r['text_domain'], r['half'], r['pct'])
            acc[key][0, int(r['network'])] += float(r['observed'])
            acc[key][1, int(r['network'])] += float(r['expected'])
        for (tree, d, h, pct), A in sorted(acc.items()):
            info = nets.get((tree, d, h))
            if not info:
                continue
            ids = sorted(info)
            gen = np.array([float(info[i]['generality']) for i in ids])
            enr = np.log((A[0, ids] + 0.5) / (A[1, ids] + 0.5))
            size = np.log([float(info[i]['size']) for i in ids])
            lay = np.array([float(info[i]['effective_layers']) for i in ids])
            ok = np.isfinite(gen)
            rho = stats.spearmanr(gen[ok], enr[ok])
            # partial on size and layer span: residualize ranks on both
            from scipy.stats import rankdata
            Z = np.column_stack([np.ones(ok.sum()), rankdata(size[ok]), rankdata(lay[ok])])
            res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
            pr = float(np.corrcoef(res(rankdata(gen[ok])), res(rankdata(enr[ok])))[0, 1])
            rows2.append(dict(tree=tree, dataset=d, half=h, pct=pct, n_networks=int(ok.sum()),
                              spearman_generality_enrichment=float(rho[0]), p=float(rho[1]),
                              partial_on_size_layers=pr))
            stderr('  network level %-4s %-10s %s pct %-4s rho %+.2f (p %.3f), partial %+.2f\n' % (
                tree, d, h, pct, rho[0], rho[1], pr))
    write(os.path.join(out, 'networks.csv'), rows2)


if __name__ == '__main__':
    main()
