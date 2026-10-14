"""Per-task concentration of attribution-patching circuits in the connectivity networks, every
task of the task bank and LLM_Modularity (LOG.md Iteration 44).

    python figures/make_bank_concentration_figure.py \
        --scores figures/bank_concentration_2b_resid_pool4.csv --model 2B --stem bank_concentration_2b

One row per task, grouped by domain (bracket and name on the left), the task name with the
data source in gray below it. Two panels share the rows: the real networks and the null
partition (the same pipeline on circularly shifted timecourses). In each, the circuit's
effective number of networks (filled dot) is paired with the mean over 1,000 random unit
sets of the same size and layer profile (open dot). Values are the means over the two
split-half partitions. A task is drawn in its domain colour when its circuit is more
concentrated than the random sets (p < 0.05) on both halves, in gray otherwise. The rows
are split over pages at domain boundaries; each page is saved as SVG (editable text) and PNG.
"""

import argparse
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'DejaVu Sans'

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Language red, Formal blue, Physics orange, Social green as in the LLM_Modularity figures
# (MODULARITY_DOMAIN_COLORS); the other domains take neighbouring or distinct hues.
COLORS = {
    'Syntax': '#C0392B', 'Morphology & lexical semantics': '#D9776D',
    'Arithmetic': '#2471A3', 'Logic, code & multiple choice': '#5DADE2',
    'Factual knowledge': '#B7950B', 'Entity tracking & binding': '#148F77',
    'In-context string tasks': '#7D3C98', 'Physical reasoning': '#E67E22', 'Social reasoning': '#27AE60',
}
GRAY_FILL, GRAY_EDGE, CITE_GRAY = '#cccccc', '#999999', '#8c8c8c'
PAGES = [['Syntax'], ['Morphology & lexical semantics', 'Arithmetic', 'Logic, code & multiple choice'],
         ['Factual knowledge'],
         ['Entity tracking & binding', 'In-context string tasks', 'Physical reasoning', 'Social reasoning']]
ROW = 0.175          # inches per task row
BRACKET_X = 0.2      # domain bracket position in the label column (0-1), just left of the longest name
GAP = 0.9            # rows of space between domains


def summarise(scores, labels):
    # `null` is a tree name, not a missing value
    s = pd.read_csv(scores, keep_default_na=False, na_values=['', 'nan', 'NaN'])
    s = s[s.pct == 0.1]
    g = s.groupby(['tree', 'group', 'task'])
    out = g.agg(circuit=('effective_networks', 'mean'), random=('null_effective_networks_mean', 'mean'),
                p_max=('p_concentrated', 'max'), n_halves=('half', 'nunique')).reset_index()
    assert (out.n_halves == 2).all(), 'every task should have both halves'
    out['sig'] = out.p_max < 0.05
    lab = pd.read_csv(labels)
    out = out.merge(lab, on=['group', 'task'], how='left', validate='many_to_one')
    assert out.name.notna().all(), 'tasks missing from %s: %s' % (labels, out[out.name.isna()].task.tolist())
    return out


def layout(lab, domains):
    """y position of every task row, top to bottom, with a gap between domains."""
    rows, spans, y = [], [], 0.0
    for d in domains:
        sub = lab[lab.domain == d].sort_values(['citation', 'name'])
        y0 = y
        for _, r in sub.iterrows():
            rows.append((r.group, r.task, y))
            y += 1
        spans.append((d, y0, y - 1))
        y += GAP
    return rows, spans, y - GAP


def page(data, lab, domains, model, xlim, path):
    rows, spans, height = layout(lab, domains)
    fig_h = height * ROW + 1.25
    fig = plt.figure(figsize=(7.2, fig_h), dpi=300)
    top, bottom = 1 - 0.38 / fig_h, 0.72 / fig_h
    label_w = 0.37
    axes = [fig.add_axes([label_w + 0.005 + i * 0.31, bottom, 0.27, top - bottom]) for i in range(2)]
    lab_ax = fig.add_axes([0.0, bottom, label_w, top - bottom])
    ylim = (height - 1 + 0.6, -0.6)
    lab_ax.set_xlim(0, 1)
    lab_ax.set_ylim(*ylim)
    lab_ax.axis('off')

    pos = {(g, t): y for g, t, y in rows}
    for (gr, task), y in pos.items():
        r = lab[(lab.group == gr) & (lab.task == task)].iloc[0]
        lab_ax.text(0.985, y - 0.13, r['name'], ha='right', va='center', fontsize=6.2)
        lab_ax.text(0.985, y + 0.33, r.citation, ha='right', va='center', fontsize=5.0, color=CITE_GRAY)
    for d, y0, y1 in spans:
        c = COLORS[d]
        x = BRACKET_X
        lab_ax.plot([x + 0.02, x, x, x + 0.02], [y0 - 0.35, y0 - 0.35, y1 + 0.35, y1 + 0.35],
                    color=c, lw=1.2, solid_capstyle='butt', clip_on=False)
        lab_ax.text(x - 0.03, (y0 + y1) / 2, d, rotation=90, ha='right', va='center', fontsize=7.5,
                    color=c, weight='bold')

    for ax, tree, title in ((axes[0], 'real', 'Real networks'),
                            (axes[1], 'null', 'Null partition (circular shift)')):
        sub = data[data.tree == tree]
        n_sig = n_all = 0
        for (gr, task), y in pos.items():
            r = sub[(sub.group == gr) & (sub.task == task)].iloc[0]
            c = COLORS[r.domain] if r.sig else GRAY_EDGE
            n_all += 1
            n_sig += int(r.sig)
            ax.hlines(y, min(r.circuit, r.random), max(r.circuit, r.random), color=c,
                      alpha=0.35 if r.sig else 0.5, lw=2.4, zorder=2)
            ax.scatter(r.random, y, s=16, facecolors='white', edgecolors=GRAY_EDGE, linewidths=0.9, zorder=3)
            ax.scatter(r.circuit, y, s=16, facecolors=c if r.sig else GRAY_FILL,
                       edgecolors='black' if r.sig else GRAY_EDGE, linewidths=0.5, zorder=4)
        ax.set_ylim(*ylim)
        ax.set_xlim(*xlim)
        ax.set_yticks([])
        ax.grid(axis='x', linestyle='--', alpha=0.5, zorder=1)
        ax.spines[['top', 'right']].set_visible(False)
        for s_ in ('left', 'bottom'):
            ax.spines[s_].set_linewidth(1.5)
        ax.tick_params(axis='x', labelsize=7.5, width=1.2, length=3)
        ax.set_xlabel('Effective number of networks', fontsize=8)
        ax.set_title('%s\n%d of %d circuits below random' % (title, n_sig, n_all), fontsize=8,
                     weight='bold', pad=4)

    handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor='#555555', markeredgecolor='black',
                      markersize=4.5, label='circuit (top 0.1% of units)'),
               Line2D([0], [0], marker='o', color='none', markerfacecolor='white', markeredgecolor=GRAY_EDGE,
                      markersize=4.5, label='random sets, same layers (mean)'),
               Line2D([0], [0], marker='o', color='none', markerfacecolor=GRAY_FILL, markeredgecolor=GRAY_EDGE,
                      markersize=4.5, label='not below random on both halves')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=6.8,
               bbox_to_anchor=(0.5 + label_w / 2 - 0.2, 0.0), handletextpad=0.3, columnspacing=1.2)
    fig.savefig(path + '.svg', format='svg', bbox_inches='tight')
    fig.savefig(path + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('wrote %s.{svg,png} (%d tasks)' % (os.path.relpath(path, ROOT), len(rows)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scores', required=True)
    ap.add_argument('--labels', default=os.path.join(HERE, 'bank_task_labels.csv'))
    ap.add_argument('--model', required=True)
    ap.add_argument('--stem', required=True)
    ap.add_argument('--outdir', default=os.path.join(ROOT, 'plots', 'circuits_vs_networks'))
    args = ap.parse_args()
    data = summarise(args.scores, args.labels)
    lab = data.drop_duplicates(['group', 'task'])
    vals = np.r_[data.circuit, data.random]
    xlim = (5 * np.floor(vals.min() / 5) - 2, 5 * np.ceil(vals.max() / 5) + 2)
    assert set(sum(PAGES, [])) == set(lab.domain), 'page domains do not cover the labels'
    for i, domains in enumerate(PAGES):
        page(data, lab, domains, args.model, xlim, os.path.join(args.outdir, '%s_p%d' % (args.stem, i + 1)))
    for tree in ('real', 'null'):
        d = data[data.tree == tree]
        print('%s: %d of %d circuits below random on both halves; median circuit %.1f, random %.1f' % (
            tree, d.sig.sum(), len(d), d.circuit.median(), d.random.median()))


if __name__ == '__main__':
    main()
