"""Qwen3.5-2B: patching circuits against ordinary and residual-connectome networks
(LOG.md Iteration 40).

    python figures/make_circuits_residual_figure.py

For each of the four prose datasets, the ordinary networks (results/qwen35/qwen3.5-2b for
wikitext and bookcorpus, qwen3.5-2b-rest for agnews and tldr17) against the residual ones
(results/qwen35/qwen3.5-2b-resid: token type and position removed before correlating), each
with its null partition. Four measures, 0.1% circuits, means over the two halves:
  A  tasks whose circuit is more concentrated on the real networks than on the null partition
  B  held-out enrichment: a left-out task's share in its domain's top-5 networks, / chance
  C  tasks whose attribution the networks explain beyond layer (graded test, all neurons)
  D  shared structure: circuit overlap against network similarity of the non-shared units
Writes plots/circuits_vs_networks/residual_vs_ordinary.{svg,png} and
figures/circuits_residual_vs_ordinary.csv.
"""

import collections
import csv
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, ROOT)
from parcelmate.circuits import heldout_enrichment

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'DejaVu Sans'
RES = os.path.join(ROOT, 'results', 'qwen35')
OUT = os.path.join(ROOT, 'plots', 'circuits_vs_networks')
DATASETS = [('wikitext', 'qwen3.5-2b'), ('bookcorpus', 'qwen3.5-2b'), ('agnews', 'qwen3.5-2b-rest'), ('tldr17', 'qwen3.5-2b-rest')]
LABEL = {'wikitext': 'wiki', 'bookcorpus': 'book', 'agnews': 'news', 'tldr17': 'tldr'}
ORD, RESID, NULL = '#9a9a9a', '#2c5f8a', '#d9d9d9'


def rd(p):
    return list(csv.DictReader(open(p)))


def measures(tree, td):
    d = os.path.join(RES, tree, 'circuits')
    pt = [r for r in rd(os.path.join(d, 'paired_tasks.csv')) if r['text_domain'] == td and r['pct'] == '0.1']
    per = collections.defaultdict(list)
    for r in pt:
        per[r['task']].append(float(r['deficit_diff']))
    paired = float(np.mean([np.mean(v) < 0 for v in per.values()]))
    rows = [r for r in rd(os.path.join(d, 'enrichment.csv')) if r['text_domain'] == td and r['pct'] == '0.1']
    he = {}
    for tr in ('real', 'null'):
        vals = []
        for h in ('halfA', 'halfB'):
            rr = [r for r in rows if r['tree'] == tr and r['half'] == h]
            tasks = sorted({r['task'] for r in rr})
            k = 1 + max(int(r['network']) for r in rr)
            O, E = np.zeros((len(tasks), k)), np.zeros((len(tasks), k))
            dom = {}
            for r in rr:
                i = tasks.index(r['task'])
                O[i, int(r['network'])] = float(r['observed'])
                E[i, int(r['network'])] = float(r['expected'])
                dom[r['task']] = r['task_domain']
            vals += [x['ratio'] for x in heldout_enrichment(O, E, [dom[t] for t in tasks], 5)]
        he[tr] = float(np.nanmean(vals))
    g = [r for r in rd(os.path.join(d, 'graded_summary.csv')) if r['text_domain'] == td and r['task_group'] == 'all']
    st = collections.defaultdict(list)
    for r in rd(os.path.join(d, 'structure.csv')):
        if r['text_domain'] == td and r['pct'] == '0.1' and r['measure'] == 'spearman_overlap_excess':
            st[r['tree']].append(float(r['value']))
    return dict(paired=paired, held_real=he['real'], held_null=he['null'],
                graded_real=float(np.mean([float(x['frac_p05_real']) for x in g])),
                graded_null=float(np.mean([float(x['frac_p05_null']) for x in g])),
                struct_real=float(np.mean(st['real'])), struct_null=float(np.mean(st['null'])))


def main():
    M = {(td, kind): measures(tree if kind == 'ordinary' else 'qwen3.5-2b-resid', td)
         for td, tree in DATASETS for kind in ('ordinary', 'residual')}
    fig, axes = plt.subplots(1, 4, figsize=(15 * .85, 3.4 * .85), dpi=300)
    plt.subplots_adjust(wspace=0.38)
    panels = [('A', 'Tasks more concentrated\non real than null', 'paired', None, 'share of tasks'),
              ('B', 'Domains reuse networks\n(held-out enrichment)', 'held_real', 'held_null', '× chance'),
              ('C', 'All neurons, no cut\n(graded test)', 'graded_real', 'graded_null', 'share of tasks significant'),
              ('D', 'Shared circuits →\nshared networks', 'struct_real', 'struct_null', 'Spearman ρ')]
    x = np.arange(len(DATASETS))
    for ax, (letter, title, key, nkey, ylab) in zip(axes, panels):
        for j, (kind, col, dx) in enumerate((('ordinary', ORD, -0.17), ('residual', RESID, 0.17))):
            vals = [M[(td, kind)][key] for td, _ in DATASETS]
            ax.bar(x + dx, vals, width=0.32, color=col, edgecolor='black', lw=0.6, zorder=3)
            if nkey:
                nv = [M[(td, kind)][nkey] for td, _ in DATASETS]
                ax.scatter(x + dx, nv, marker='_', s=140, color='black', lw=1.6, zorder=4)
        if key == 'paired':
            ax.axhline(0.5, color='#333333', lw=0.9, ls='--', zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[td] for td, _ in DATASETS], fontsize=8)
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.set_title(title, fontsize=9, weight='bold')
        ax.text(-0.2, 1.08, letter, transform=ax.transAxes, fontsize=13, weight='bold', va='bottom')
        ax.spines[['top', 'right']].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_linewidth(1.5)
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax.tick_params(labelsize=8)
    axes[-1].legend(handles=[mpl.patches.Patch(fc=ORD, ec='black', label='ordinary networks'),
                             mpl.patches.Patch(fc=RESID, ec='black', label='residual networks'),
                             Line2D([], [], marker='_', ls='', ms=12, mew=1.6, color='black', label='their null partition'),
                             Line2D([], [], ls='--', color='#333333', label='chance (A)')],
                    frameon=False, fontsize=7.5, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    os.makedirs(OUT, exist_ok=True)
    for ext in ('svg', 'png'):
        fig.savefig(os.path.join(OUT, 'residual_vs_ordinary.%s' % ext), format=ext, dpi=300, bbox_inches='tight')
    with open(os.path.join(HERE, 'circuits_residual_vs_ordinary.csv'), 'w', newline='') as f:
        keys = ['dataset', 'networks'] + list(next(iter(M.values())).keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for (td, kind), v in M.items():
            w.writerow(dict(dataset=td, networks=kind, **v))
    for (td, kind), v in sorted(M.items()):
        print('%-10s %-9s ' % (td, kind) + ' '.join('%s %.2f' % (k, val) for k, val in v.items()))


if __name__ == '__main__':
    main()
