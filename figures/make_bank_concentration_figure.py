"""Per-task concentration of attribution-patching circuits in the pooled residual networks, every
task of the task bank and LLM_Modularity in one two-row figure (LOG.md Iterations 44 and 47).

    python figures/make_bank_concentration_figure.py --model 2B
    python figures/make_bank_concentration_figure.py --model 4B

reads figures/bank_concentration_<model>_resid_pool4.csv (written by
parcelmate.bin.bank_concentration) and figures/bank_task_labels.csv, and writes
plots/circuits_vs_networks/bank_concentration_<model>.{svg,png}.

Tasks run along x in two rows split at domain boundaries, grouped by domain (coloured bracket
and name above), each labelled with its name and, in gray, its data source (tilted). Every task
has four values of the effective number of networks it spans:

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
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.textpath import TextToPath

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
ROWS = [['Syntax', 'Morphology & lexical semantics', 'Arithmetic', 'Logic, code & multiple choice'],
        ['Factual knowledge', 'Entity tracking & binding', 'In-context string tasks', 'Physical reasoning',
         'Social reasoning']]
GRAY_FILL, GRAY_EDGE, RAND, CITE_GRAY = '#cccccc', '#999999', '#7a7a7a', '#8c8c8c'
GAP = 1.4            # columns of space between domains
ROT = 55             # label tilt (degrees)
NAME_PT, CITE_PT = 5.2, 4.5
YLIM = (0, 80)       # the same y range for 2B and 4B

# Shorter labels for this dense layout (spelled out in the caption): exact replacements for the
# longest names first, then patterns.
EXACT = {
    'Filler-gap, PP gap': 'Filler-gap (PP)', 'Filler-gap, subject gap': 'Filler-gap (subject)',
    'Garden path, NP/Z (transitive)': 'Garden path NP/Z', 'Wug test (novel plurals)': 'Wug test',
    'Hypernym (", X and other")': 'Hypernym (and other)', 'Hypernym ("is a type of")': 'Hypernym (type of)',
    'Addition/subtraction, digits': 'Add/sub', 'Word problems (+, -)': 'Word problems',
    'Docstring argument (Google style)': 'Docstring (Google)', 'Docstring argument (reST)': 'Docstring (reST)',
    'ARC-Challenge, answer position': 'ARC-Challenge', 'ARC-Easy, answer position': 'ARC-Easy',
    'Copy colors, answer position': 'Copy colors', 'Copy colors, position and color': 'Copy colors (+color)',
    'Boxes after one move': 'Boxes, 1 move', 'Boxes after one removal': 'Boxes, 1 removal',
    'Boxes, no operation': 'Boxes, no op.', 'Alphabetically first word': 'Alphabetic first',
    'Alphabetically last word': 'Alphabetic last', 'Next item in sequence': 'Next item',
    'Previous item in sequence': 'Previous item', 'Object affordances (PROST)': 'Affordances (PROST)',
    'Packing order (Newton)': 'Packing (Newton)', 'Emotion inference (few-shot)': 'Emotion (few-shot)',
    'Gendered pronoun (stereotype)': 'Gender stereotype', 'Superhero to alter ego': 'Superhero → alter ego',
    'City attributes (attribute CF)': 'City attr. (attr. CF)', 'City attributes (entity swap)': 'City attr. (entity)',
    'City attributes (template CF)': 'City attr. (template)', 'Headquarters location': 'HQ location',
    'Country of citizenship': 'Citizenship', 'Original language of film': 'Film language',
    'Twinned administrative body': 'Twin city', 'Location of formation': 'Place of formation',
    'Addition/subtraction, English words': 'Add/sub, English', 'Addition/subtraction, Spanish words': 'Add/sub, Spanish',
    'English reflexive, across object RC': 'En reflexive, obj. RC', 'English reflexive, simple': 'En reflexive, simple',
    'Propositional logic, symbolic': 'Prop. logic, symbolic',
}
SHORT = [(r' \(P\d+\)$', ''), (r'^NPI "(any|ever)"', r'NPI \1'), (r'^Subject-verb number', 'SV'),
         (r'^Reflexive number', 'Reflexive'), (r'^Subject-verb agreement$', 'SV agreement'),
         (r'^(German|French|Hebrew|Russian) subject-verb',
          lambda m: {'German': 'De', 'French': 'Fr', 'Hebrew': 'He', 'Russian': 'Ru'}[m.group(1)] + ' SV'),
         (r'^Subject-verb', 'SV'), (r'across ', ''), (r'within ', 'in '), (r'object RC', 'obj. RC'),
         (r'subject RC', 'subj. RC'), (r'long VP coordination', 'long VP coord.'), (r'VP coordination', 'VP coord.'),
         (r'Det-noun agreement', 'Det-noun'), (r'agreement', 'agr.'), (r', with adjective', ', with adj.'),
         (r'Nobel laureate', 'Nobel'), (r'Code output: ', 'Code: '), (r'Add/subtract', 'Add/sub'),
         (r'Multiply/divide', 'Mul/div'), (r', (\d) operands, digits', r', \1 op.'), (r', (\d) operands, words', r', \1 op., words'),
         (r'Addition, ', 'Add., '), (r'Subtraction, ', 'Sub., '), (r'one digit changed', '1 digit'),
         (r'random operands', 'random'), (r'Located in \(admin. territory\)', 'Located in'),
         (r'Capitalized', 'Cap.'), (r'Lowercase', 'Lower.'), (r'native language', 'native lang.'),
         (r' to ', ' → '), (r'Ordering ID: ', 'OI: '), (r', inanimate', ', inanim.')]

_T2P = TextToPath()


def summarise(scores, labels):
    """Means over the two halves per tree x task, with the both-halves significance flag and labels."""
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


def short_name(name):
    if name in EXACT:
        return EXACT[name]
    for pat, rep in SHORT:
        name = re.sub(pat, rep, name)
    return EXACT.get(name, name)


def short_cite(cite):
    # "Newman et al. (2021)" -> "Newman 2021"; "Feng & Steinhardt (2024)" -> "Feng 2024"
    m = re.match(r'(.+?)( et al\.| & [^(]+)? \((\d{4})\)', cite)
    return '%s %s' % (m.group(1), m.group(3))


def text_len_pt(s, size):
    w, _, _ = _T2P.get_text_width_height_descent(s, FontProperties(family='DejaVu Sans', size=size), ismath=False)
    return w


def columns(lab, domains):
    """x position of every task, left to right, with a gap between domains."""
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


def strip(ax, data, lab, domains, ylim, xmax):
    cols, spans, _ = columns(lab, domains)
    real = data[data.tree == 'real'].set_index(['group', 'task'])
    null = data[data.tree == 'null'].set_index(['group', 'task'])
    n_sig = 0
    for gr, task, x in cols:
        r, n = real.loc[(gr, task)], null.loc[(gr, task)]
        ok = bool(r.sig) and r.circuit < min(n.circuit, r.random, n.random)
        n_sig += ok
        c = COLORS[r.domain]
        tc = c if ok else GRAY_EDGE
        # circuit pair and random pair in one column, each joined by a thin line
        ax.plot([x, x], [r.circuit, n.circuit], color=tc, lw=1.0, alpha=0.5, zorder=2)
        ax.plot([x, x], [r.random, n.random], color=RAND, lw=1.0, alpha=0.35, zorder=2)
        ax.scatter(x, n.random, s=11, marker='D', facecolors='white', edgecolors=RAND, linewidths=0.7, zorder=3)
        ax.scatter(x, r.random, s=11, marker='D', facecolors=RAND, edgecolors=RAND, linewidths=0.5, zorder=3)
        ax.scatter(x, n.circuit, s=14, marker='o', facecolors='white', edgecolors=tc, linewidths=0.9, zorder=4)
        ax.scatter(x, r.circuit, s=16, marker='o', facecolors=c if ok else GRAY_FILL,
                   edgecolors='black' if ok else GRAY_EDGE, linewidths=0.5, zorder=5)
    ax.set_xlim(-0.8, xmax - 0.2)
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
    # domain brackets above the data; a title wider than its span is broken into lines that fit
    bb = ax.get_position()
    pt_per_col = bb.width * ax.figure.get_figwidth() * 72 / (ax.get_xlim()[1] - ax.get_xlim()[0])
    for d, x0, x1 in spans:
        c = COLORS[d]
        y = 1.02
        ax.plot([x0 - 0.4, x0 - 0.4, x1 + 0.4, x1 + 0.4], [y - 0.03, y, y, y - 0.03], color=c, lw=1.1,
                transform=base, clip_on=False, solid_capstyle='butt')
        title = '%s (%d)' % (d, int(x1 - x0 + 1))
        avail = (x1 - x0 + 1.6) * pt_per_col
        if text_len_pt(title, 7) > avail:
            lines, cur = [], ''
            for w in title.split(' '):
                trial = (cur + ' ' + w).strip()
                if cur and text_len_pt(trial, 7) > avail:
                    lines.append(cur)
                    cur = w
                else:
                    cur = trial
            title = '\n'.join(lines + [cur])
        ax.text((x0 + x1) / 2, y + 0.02, title, transform=base, ha='center', va='bottom',
                fontsize=7, color=c, weight='bold', linespacing=1.0)
    ax.set_ylim(*ylim)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
    ax.spines[['top', 'right']].set_visible(False)
    for s_ in ('left', 'bottom'):
        ax.spines[s_].set_linewidth(1.5)
    ax.tick_params(axis='y', labelsize=7, width=1.2, length=3)
    ax.set_ylabel('Effective number of networks', fontsize=8)
    return n_sig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, choices=('2B', '4B'))
    ap.add_argument('--scores', help='default figures/bank_concentration_<model>_resid_pool4.csv')
    ap.add_argument('--labels', default=os.path.join(HERE, 'bank_task_labels.csv'))
    ap.add_argument('--outdir', default=os.path.join(ROOT, 'plots', 'circuits_vs_networks'))
    args = ap.parse_args()
    m = args.model.lower()
    scores = args.scores or os.path.join(HERE, 'bank_concentration_%s_resid_pool4.csv' % m)
    data = summarise(scores, args.labels)
    lab = data.drop_duplicates(['group', 'task'])
    assert set(sum(ROWS, [])) == set(lab.domain), 'row domains do not cover the labels'
    vals = np.r_[data.circuit, data.random]
    assert vals.min() >= YLIM[0] and vals.max() <= YLIM[1], 'values outside the fixed y range'
    ylim = YLIM
    xmax = max(columns(lab, d)[2] for d in ROWS) + 1

    fig, axes = plt.subplots(len(ROWS), 1, figsize=(10.0, 6.0), dpi=300)
    fig.subplots_adjust(left=0.05, right=0.997, top=0.895, bottom=0.14, hspace=1.25)
    tot = sum(strip(ax, data, lab, domains, ylim, xmax) for ax, domains in zip(axes, ROWS))
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
               label='target not below random at p < .05 on both halves'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=5, frameon=False, fontsize=6.8,
               bbox_to_anchor=(0.52, 1.0), handletextpad=0.25, columnspacing=1.0)
    path = os.path.join(args.outdir, 'bank_concentration_%s' % m)
    fig.savefig(path + '.svg', format='svg', bbox_inches='tight')
    fig.savefig(path + '.png', dpi=300, bbox_inches='tight')
    print('wrote %s.{svg,png}; target below all three nulls in %d of %d tasks' % (
        os.path.relpath(path, ROOT), tot, len(lab)))
    for tree in ('real', 'null'):
        d = data[data.tree == tree]
        print('%s: %d of %d circuits below random on both halves; median circuit %.1f, random %.1f' % (
            tree, d.sig.sum(), len(d), d.circuit.median(), d.random.median()))


if __name__ == '__main__':
    main()
