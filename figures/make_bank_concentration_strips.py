"""Per-task concentration of attribution-patching circuits in the connectivity networks, all
tasks in one figure: one target value and three null values per task (LOG.md Iteration 44).

    python figures/make_bank_concentration_strips.py \
        --scores figures/bank_concentration_2b_resid_pool4.csv --stem bank_concentration_2b_strips

Tasks run along x in four strips split at domain boundaries, grouped by domain (coloured
bracket and name above), each labelled with its name and, in gray, its data source (tilted).
Every task has four values of the effective number of networks it spans:

    target   the circuit (top 0.1% of units) on the real networks          filled circle, domain colour
    null 1   the circuit on the null partition (circular shift)            open circle, domain colour
    null 2   random unit sets, same size and layers, on the real networks  filled gray diamond
    null 3   the random sets on the null partition                         open gray diamond

Shape = unit set (circle circuit, diamond random), fill = networks (filled real, open null).
Values are means over the two split-half partitions; random sets are the mean over 1,000
draws. The target is drawn in its domain colour when it is below all three nulls and below
the random sets at p < 0.05 on both halves, in gray otherwise. Only tasks that passed the
inclusion rule have a circuit (both-correct accuracy >= 0.6 with >= 100 both-correct items),
and their attribution uses only aligned, both-correct items.
"""

import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import re
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.textpath import TextToPath

from make_bank_concentration_figure import COLORS, PAGES, summarise

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'DejaVu Sans'

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GRAY_FILL, GRAY_EDGE, RAND, CITE_GRAY = '#cccccc', '#999999', '#7a7a7a', '#8c8c8c'
GAP = 1.4            # columns of space between domains
ROT = 65             # label tilt (degrees)
NAME_PT, CITE_PT = 5.2, 4.5
# Abbreviations for this dense layout only (spelled out in the caption).
SHORT = [(r'^Subject-verb number', 'SV number'), (r'^Subject-verb agreement$', 'SV agreement'),
         (r'^Subject-verb', 'SV'), (r'^(German|French|Hebrew|Russian) subject-verb',
                                    lambda m: {'German': 'De', 'French': 'Fr', 'Hebrew': 'He', 'Russian': 'Ru'}[m.group(1)] + ' SV'),
         (r'object RC', 'obj. RC'), (r'subject RC', 'subj. RC'), (r'coordination', 'coord.'),
         (r'Nobel laureate', 'Nobel'), (r'agreement', 'agr.'), (r'Code output: ', 'Code: '),
         (r'Add/subtract', 'Add/sub'), (r'Multiply/divide', 'Mul/div'), (r' operands', ' op.'),
         (r'Located in \(admin. territory\)', 'Located in'), (r'Capitalized', 'Cap.'), (r'Lowercase', 'Lower.')]


def short_name(name):
    for pat, rep in SHORT:
        name = re.sub(pat, rep, name)
    return name


def short_cite(cite):
    # "Newman et al. (2021)" -> "Newman 2021"; "Feng & Steinhardt (2024)" -> "Feng 2024"
    m = re.match(r'(.+?)( et al\.| & [^(]+)? \((\d{4})\)', cite)
    return '%s %s' % (m.group(1), m.group(3))


_T2P = TextToPath()


def text_len_pt(s, size):
    w, _, _ = _T2P.get_text_width_height_descent(s, FontProperties(family='DejaVu Sans', size=size), ismath=False)
    return w
DX = 0.17            # horizontal offset: circuit values left of the column, random right


def columns(lab, domains):
    cols, spans, x = [], [], 0.0
    for d in domains:
        sub = lab[lab.domain == d].sort_values(['citation', 'name'])
        x0 = x
        for _, r in sub.iterrows():
            cols.append((r.group, r.task, x))
            x += 1
        spans.append((d, x0, x - 1))
        x += GAP
    return cols, spans, x - GAP


def strip(ax, data, lab, domains, ylim, xmax_all):
    cols, spans, width = columns(lab, domains)
    real = data[data.tree == 'real'].set_index(['group', 'task'])
    null = data[data.tree == 'null'].set_index(['group', 'task'])
    n_sig = 0
    for gr, task, x in cols:
        r, n = real.loc[(gr, task)], null.loc[(gr, task)]
        ok = bool(r.sig) and r.circuit < min(n.circuit, r.random, n.random)
        n_sig += ok
        c = COLORS[r.domain]
        tc = c if ok else GRAY_EDGE
        # circuit pair (left) and random pair (right), each joined by a thin line
        ax.plot([x - DX, x - DX], [r.circuit, n.circuit], color=tc, lw=1.0, alpha=0.5, zorder=2)
        ax.plot([x + DX, x + DX], [r.random, n.random], color=RAND, lw=1.0, alpha=0.35, zorder=2)
        ax.scatter(x + DX, n.random, s=11, marker='D', facecolors='white', edgecolors=RAND, linewidths=0.7, zorder=3)
        ax.scatter(x + DX, r.random, s=11, marker='D', facecolors=RAND, edgecolors=RAND, linewidths=0.5, zorder=3)
        ax.scatter(x - DX, n.circuit, s=14, marker='o', facecolors='white', edgecolors=tc, linewidths=0.9, zorder=4)
        ax.scatter(x - DX, r.circuit, s=16, marker='o', facecolors=c if ok else GRAY_FILL,
                   edgecolors='black' if ok else GRAY_EDGE, linewidths=0.5, zorder=5)
    # tilted labels, one line per task: the name, then the data source in gray, ending at the tick
    ax.set_xticks([x for _, _, x in cols])
    ax.set_xticklabels([])
    ax.tick_params(axis='x', length=2, width=0.8)
    th = np.deg2rad(ROT)
    base = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for gr, task, x in cols:
        r = lab[(lab.group == gr) & (lab.task == task)].iloc[0]
        cite = short_cite(r.citation)
        t_cite = mtransforms.offset_copy(base, fig=ax.figure, x=0, y=-3.5, units='points')
        ax.text(x, 0, cite, transform=t_cite, rotation=ROT, ha='right', va='center',
                rotation_mode='anchor', fontsize=CITE_PT, color=CITE_GRAY)
        L = text_len_pt(cite, CITE_PT) + 2.5   # name ends where the citation starts
        t_name = mtransforms.offset_copy(base, fig=ax.figure, x=-L * np.cos(th), y=-3.5 - L * np.sin(th), units='points')
        ax.text(x, 0, short_name(r['name']), transform=t_name, rotation=ROT, ha='right', va='center',
                rotation_mode='anchor', fontsize=NAME_PT)
    # domain brackets above the data
    for d, x0, x1 in spans:
        c = COLORS[d]
        y = 1.02
        tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.plot([x0 - 0.4, x0 - 0.4, x1 + 0.4, x1 + 0.4], [y - 0.03, y, y, y - 0.03], color=c, lw=1.1,
                transform=tr, clip_on=False, solid_capstyle='butt')
        title = '%s (%d)' % (d, int(x1 - x0 + 1))
        if (x1 - x0 + 1) < 10 or (len(title) > 22 and (x1 - x0 + 1) < 16):   # narrow span: two lines
            words = d.split(' ')
            k = (len(words) + 1) // 2
            title = '%s\n%s (%d)' % (' '.join(words[:k]), ' '.join(words[k:]), int(x1 - x0 + 1))
        ax.text((x0 + x1) / 2, y + 0.02, title, transform=tr, ha='center', va='bottom',
                fontsize=7, color=c, weight='bold', linespacing=1.0)
    ax.set_xlim(-0.8, xmax_all - 0.2)
    ax.set_ylim(*ylim)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
    ax.spines[['top', 'right']].set_visible(False)
    for s_ in ('left', 'bottom'):
        ax.spines[s_].set_linewidth(1.5)
    ax.tick_params(axis='y', labelsize=7, width=1.2, length=3)
    return n_sig, len(cols)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scores', required=True)
    ap.add_argument('--labels', default=os.path.join(HERE, 'bank_task_labels.csv'))
    ap.add_argument('--stem', required=True)
    ap.add_argument('--outdir', default=os.path.join(ROOT, 'plots', 'circuits_vs_networks'))
    args = ap.parse_args()
    data = summarise(args.scores, args.labels)
    lab = data.drop_duplicates(['group', 'task'])
    vals = np.r_[data.circuit, data.random]
    ylim = (max(0, 10 * np.floor(vals.min() / 10)), 10 * np.ceil(vals.max() / 10))
    widths = [columns(lab, d)[2] for d in PAGES]
    fig, axes = plt.subplots(len(PAGES), 1, figsize=(7.2, 12.0), dpi=300)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.955, bottom=0.1, hspace=1.45)
    tot = 0
    for ax, domains in zip(axes, PAGES):
        n_sig, n = strip(ax, data, lab, domains, ylim, max(widths) + 1)
        tot += n_sig
    fig.text(0.012, 0.5, 'Effective number of networks', rotation=90, ha='left', va='center', fontsize=9)
    handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='#555555', markeredgecolor='black', markersize=4.5,
               label='circuit, real networks (target)'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='white', markeredgecolor='#555555', markersize=4.5,
               label='circuit, null partition'),
        Line2D([0], [0], marker='D', color='none', markerfacecolor=RAND, markeredgecolor=RAND, markersize=3.6,
               label='random units, real networks'),
        Line2D([0], [0], marker='D', color='none', markerfacecolor='white', markeredgecolor=RAND, markersize=3.6,
               label='random units, null partition'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=GRAY_FILL, markeredgecolor=GRAY_EDGE, markersize=4.5,
               label='target not below all three'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=5, frameon=False, fontsize=6.3,
               bbox_to_anchor=(0.53, 1.0), handletextpad=0.25, columnspacing=1.0)
    path = os.path.join(args.outdir, args.stem)
    fig.savefig(path + '.svg', format='svg', bbox_inches='tight')
    fig.savefig(path + '.png', dpi=300, bbox_inches='tight')
    print('wrote %s.{svg,png}; target below all three nulls in %d of %d tasks' % (
        os.path.relpath(path, ROOT), tot, len(lab)))


if __name__ == '__main__':
    main()
