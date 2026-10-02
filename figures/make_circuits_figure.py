"""Patching circuits against connectivity networks on Qwen3.5 (LOG.md Iterations 30-33).

    python figures/make_circuits_figure.py

Reads `results/qwen35/<tree>/circuits/` for the four network sets (2B single-dataset,
2B pooled, 4B wikitext, 4B pooled) and draws plots/circuits_vs_networks/overview.{svg,png} (and each panel on its own), six panels:

  A  step A: circuit overlap between tasks of the same task domain vs of different ones
  B  the headline paired test: per task, entropy deficit on the real minus the null partition
  C  the same for each task domain's averaged circuit (step B)
  D  test 3: Spearman(overlap, non-shared-unit network similarity), real vs null partition
  E  test 3: same-domain minus cross-domain network similarity, real vs null partition
  F  test 2: task x network pairs enriched at q < 0.05, real vs null partition
  (since Iteration 35: D = held-out enrichment per task domain; test 3 and test 2 move to
  E, F, G; the 2B agnews, tldr17 and codeparrot sets added)

Circuit size 0.1% of all MLP neurons except F (1%: at 0.1% too few units survive FDR).
Values averaged over the two halves; a per-half value is a small dot. The plotted values
are also written to figures/circuits_networks.csv.
"""

import collections
import csv
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

from parcelmate.circuits import heldout_enrichment

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'DejaVu Sans'

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
RES = os.path.join(ROOT, 'results', 'qwen35')

# Task domains: the house cognitive-domain palette and names.
DOMAIN_COLORS = {'Lan': '#C44E52', 'MD': '#4C72B0', 'ToM': '#55A868', 'phys': '#DD8452'}
DOMAIN_LABELS = {'Lan': 'Language', 'MD': 'Formal', 'ToM': 'Social', 'phys': 'Physical'}
DOMAIN_ORDER = ['Lan', 'MD', 'ToM', 'phys']
REAL, NULL = '#2c5f8a', '#9a9a9a'
NS_FILL, NS_EDGE = '#cccccc', '#999999'

# Network sets: (tree, text domain, label). "pooled" = five datasets.
SETS = [('qwen3.5-2b', 'bookcorpus', '2B\nbook'),
        ('qwen3.5-2b', 'wikitext', '2B\nwiki'),
        ('qwen3.5-2b-rest', 'agnews', '2B\nnews'),
        ('qwen3.5-2b-rest', 'tldr17', '2B\ntldr'),
        ('qwen3.5-2b-rest', 'codeparrot', '2B\ncode'),
        ('qwen3.5-2b-pool5', 'pooled', '2B\npool'),
        ('qwen3.5-4b', 'wikitext', '4B\nwiki'),
        ('qwen3.5-4b-pool5', 'pooled', '4B\npool')]
MODEL_TREE = {'2B': 'qwen3.5-2b', '4B': 'qwen3.5-4b'}


def read(tree, name):
    return list(csv.DictReader(open(os.path.join(RES, tree, 'circuits', name))))


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'


def style(ax, ygrid=True):
    ax.spines[['top', 'right']].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_linewidth(1.5)
    if ygrid:
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
    ax.tick_params(labelsize=8)


def title(ax, letter, text):
    ax.set_title(text, fontsize=9, weight='bold')
    ax.text(-0.18, 1.08, letter, transform=ax.transAxes, fontsize=12, weight='bold', va='bottom')


def set_xticks(ax):
    ax.set_xticks(range(len(SETS)))
    ax.set_xticklabels([s[2] for s in SETS], fontsize=6.8)
    ax.set_xlim(-0.6, len(SETS) - 0.4)


out_rows = []


def keep(panel, **kw):
    out_rows.append(dict(panel=panel, **kw))


fig, axes = plt.subplots(2, 4, figsize=(19 * .85, 7.4 * .85))
plt.subplots_adjust(wspace=0.5, hspace=0.72)

# ---------------------------------------------------------------- A: step A
ax = axes[0, 0]
y = 0
yt, yl = [], []
for d in DOMAIN_ORDER:
    for model in ('2B', '4B'):
        r = [x for x in read(MODEL_TREE[model], 'domain_overlap.csv') if x['pct'] == '0.1' and x['task_domain'] == d][0]
        w, a, p = float(r['mean_within']), float(r['mean_across']), float(r['p'])
        sig = p < 0.05
        c = DOMAIN_COLORS[d] if sig else NS_FILL
        ax.hlines(y, a, w, color=c, alpha=0.35 if sig else 0.8, lw=3, zorder=2)
        ax.scatter(a, y, s=34, facecolor='white', edgecolor=c if sig else NS_EDGE, lw=1.4, zorder=3)
        ax.scatter(w, y, s=34, facecolor=c, edgecolor='black' if sig else NS_EDGE, lw=0.6, zorder=4)
        ax.text(w + 0.006, y, stars(p), va='center', fontsize=7, color='0.25')
        yt.append(y)
        yl.append('%s  %s' % (DOMAIN_LABELS[d], model))
        keep('A', task_domain=d, model=model, within=w, across=a, p=p)
        y += 1
    y += 0.6
ax.set_yticks(yt)
ax.set_yticklabels(yl, fontsize=7.5)
ax.invert_yaxis()
ax.set_xlim(0, 0.215)
ax.set_xlabel('circuit overlap between tasks', fontsize=9)
style(ax, ygrid=False)
ax.grid(axis='x', linestyle='--', alpha=0.5, zorder=1)
title(ax, 'A', 'Patching: domains share circuits')
ax.legend(handles=[Line2D([], [], marker='o', ls='', mfc='0.3', mec='black', label='within domain'),
                   Line2D([], [], marker='o', ls='', mfc='white', mec='0.3', label='across domains')],
          frameon=False, fontsize=7, loc='upper left', bbox_to_anchor=(0.0, -0.17), ncol=2, handletextpad=0.2)

# ---------------------------------------------------------------- B: paired per task
ax = axes[0, 1]
rng = np.random.default_rng(0)
for i, (tree, td, label) in enumerate(SETS):
    rows = [r for r in read(tree, 'paired_tasks.csv') if r['pct'] == '0.1' and r['text_domain'] == td]
    per = collections.defaultdict(list)
    dom = {}
    for r in rows:
        per[r['task']].append(float(r['deficit_diff']))
        dom[r['task']] = r['task_domain']
    tasks = sorted(per)
    d = np.array([np.mean(per[t]) for t in tasks])
    p = stats.wilcoxon(d, alternative='less').pvalue
    x = i + rng.uniform(-0.2, 0.2, len(d))
    ax.scatter(x, d, s=11, c=[DOMAIN_COLORS[dom[t]] for t in tasks], alpha=0.75, edgecolors='none', zorder=3)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    ax.errorbar(i + 0.32, m, yerr=se, fmt='s', color='black', ms=4.5, capsize=2.5, lw=1.2, zorder=4)
    ax.text(i, 0.62, stars(p), ha='center', fontsize=8)
    keep('B', set=label.replace('\n', ' '), n_tasks=len(d), mean=m, se=se,
         frac_real_lt_null=float((d < 0).mean()), p_wilcoxon=p)
ax.axhline(0, color='#333333', lw=0.9, zorder=2)
ax.set_ylim(-1.15, 0.72)
set_xticks(ax)
ax.set_ylabel('real − null partition\n(entropy deficit, nats)', fontsize=8.5)
ax.text(0.02, 0.03, 'below 0: more concentrated\non the real networks', transform=ax.transAxes,
        fontsize=6.5, color='0.35')
style(ax)
title(ax, 'B', 'Per-task circuits: real − null')

# ---------------------------------------------------------------- C: domain circuits
ax = axes[0, 2]
offs = dict(zip(DOMAIN_ORDER, (-0.24, -0.08, 0.08, 0.24)))
for i, (tree, td, label) in enumerate(SETS):
    rows = [r for r in read(tree, 'paired_domains.csv') if r['pct'] == '0.1' and r['text_domain'] == td]
    for dname in DOMAIN_ORDER:
        v = [float(r['deficit_diff']) for r in rows if r['task_domain'] == dname]
        ax.scatter([i + offs[dname]] * len(v), v, s=8, color=DOMAIN_COLORS[dname], alpha=0.35, edgecolors='none', zorder=3)
        ax.scatter(i + offs[dname], np.mean(v), s=30, color=DOMAIN_COLORS[dname], edgecolors='black', lw=0.5, zorder=4)
        keep('C', set=label.replace('\n', ' '), task_domain=dname, mean=float(np.mean(v)))
ax.axhline(0, color='#333333', lw=0.9, zorder=2)
set_xticks(ax)
ax.set_ylabel('real − null partition\n(entropy deficit, nats)', fontsize=8.5)
style(ax)
title(ax, 'C', 'Domain circuits: real − null')


# ---------------------------------------------------------------- D, E, F: real vs null
def real_vs_null(ax, getter, ylabel, letter, text, pct='0.1', star_key=None):
    for i, (tree, td, label) in enumerate(SETS):
        vals, ps = {}, []
        for tr in ('null', 'real'):
            v, pv = getter(tree, td, tr, pct)
            vals[tr] = v
            if tr == 'real' and pv is not None:
                ps = pv
        ax.plot([i - 0.12, i + 0.12], [np.mean(vals['null']), np.mean(vals['real'])], color='0.6', lw=1.2, zorder=2)
        for tr, dx, col in (('null', -0.12, NULL), ('real', 0.12, REAL)):
            ax.scatter([i + dx] * len(vals[tr]), vals[tr], s=9, color=col, alpha=0.45, edgecolors='none', zorder=3)
            ax.scatter(i + dx, np.mean(vals[tr]), s=36, color=col, edgecolors='black', lw=0.6, zorder=4)
        if ps:
            ax.text(i + 0.12, max(vals['real']) * 1.0, '  ' + stars(max(ps)), fontsize=7, va='bottom', ha='center')
        keep(letter, set=label.replace('\n', ' '), real=float(np.mean(vals['real'])), null=float(np.mean(vals['null'])),
             p_real_max=max(ps) if ps else '')
    set_xticks(ax)
    ax.set_ylabel(ylabel, fontsize=8.5)
    style(ax)
    title(ax, letter, text)


def structure(measure, pmeasure):
    def get(tree, td, tr, pct):
        rows = read(tree, 'structure.csv')
        v = [float(r['value']) for r in rows if r['text_domain'] == td and r['tree'] == tr and r['pct'] == pct and r['measure'] == measure]
        p = [float(r['value']) for r in rows if r['text_domain'] == td and r['tree'] == tr and r['pct'] == pct and r['measure'] == pmeasure]
        return v, p
    return get


def enriched(tree, td, tr, pct):
    c = collections.Counter()
    for r in read(tree, 'enrichment.csv'):
        if r['text_domain'] == td and r['tree'] == tr and r['pct'] == pct:
            c[r['half']] += float(r['q']) < 0.05
    return [c.get('halfA', 0), c.get('halfB', 0)], None


real_vs_null(axes[1, 0], structure('spearman_overlap_excess', 'p_overlap_excess'),
             'Spearman ρ: overlap vs network\nsimilarity of non-shared units', 'E',
             'Shared circuits → shared networks')
axes[1, 0].set_ylim(-0.05, 0.85)
real_vs_null(axes[1, 1], structure('domain_contrast_excess', 'p_domain_contrast_excess'),
             'similarity excess,\nsame − cross domain', 'F',
             'Same-domain tasks closer')
axes[1, 1].set_ylim(0, 0.17)
real_vs_null(axes[1, 2], enriched, 'enriched task × network\npairs (q < 0.05)', 'G',
             'Enriched networks (1%)', pct='1.0')

# ---------------------------------------------------------------- D: held-out enrichment
ax = axes[0, 3]
ratios = {tr: collections.defaultdict(list) for tr in ('real', 'null')}   # domain -> one value per set x half
for tree, td, label in SETS:
    rows = [r for r in read(tree, 'enrichment.csv') if r['text_domain'] == td and r['pct'] == '0.1']
    for tr in ('real', 'null'):
        for half in ('halfA', 'halfB'):
            rr = [r for r in rows if r['tree'] == tr and r['half'] == half]
            tasks = sorted({r['task'] for r in rr})
            k = 1 + max(int(r['network']) for r in rr)
            O, E = np.zeros((len(tasks), k)), np.zeros((len(tasks), k))
            dom = {}
            for r in rr:
                i = tasks.index(r['task'])
                O[i, int(r['network'])] = float(r['observed'])
                E[i, int(r['network'])] = float(r['expected'])
                dom[r['task']] = r['task_domain']
            doms = [dom[t] for t in tasks]
            res = heldout_enrichment(O, E, doms, n_top=5)
            for dname in DOMAIN_ORDER:
                v = [x['ratio'] for x, dd in zip(res, doms) if dd == dname and np.isfinite(x['ratio'])]
                ratios[tr][dname].append(float(np.mean(v)))
                keep('D', set=label.replace('\n', ' '), half=half, tree=tr, task_domain=dname, ratio=float(np.mean(v)))
for j, dname in enumerate(DOMAIN_ORDER):
    rv, nv = np.array(ratios['real'][dname]), np.array(ratios['null'][dname])
    p = stats.wilcoxon(rv - nv, alternative='greater').pvalue
    xr = j + 0.14 + rng.uniform(-0.06, 0.06, len(rv))
    xn = j - 0.14 + rng.uniform(-0.06, 0.06, len(nv))
    ax.scatter(xn, nv, s=9, color=NULL, alpha=0.5, edgecolors='none', zorder=3)
    ax.scatter(xr, rv, s=9, color=DOMAIN_COLORS[dname], alpha=0.5, edgecolors='none', zorder=3)
    ax.plot([j - 0.14, j + 0.14], [nv.mean(), rv.mean()], color='0.6', lw=1.2, zorder=2)
    ax.scatter(j - 0.14, nv.mean(), s=40, color=NULL, edgecolors='black', lw=0.6, zorder=4)
    ax.scatter(j + 0.14, rv.mean(), s=40, color=DOMAIN_COLORS[dname], edgecolors='black', lw=0.6, zorder=4)
    ax.text(j, 4.35, stars(p), ha='center', fontsize=7.5)
    keep('D', task_domain=dname, mean_real=float(rv.mean()), mean_null=float(nv.mean()), p_wilcoxon_real_gt_null=p)
ax.axhline(1, color='#333333', lw=0.9, ls='--', zorder=2)
ax.set_ylim(0.6, 4.6)
ax.set_xticks(range(4))
ax.set_xticklabels([DOMAIN_LABELS[d] for d in DOMAIN_ORDER], fontsize=8)
ax.set_xlim(-0.6, 3.6)
ax.set_ylabel('held-out task\'s share in its domain\'s\ntop-5 networks ÷ chance', fontsize=8.5)
style(ax)
title(ax, 'D', 'Domains reuse networks')

# ---------------------------------------------------------------- legends
ax = axes[1, 3]
ax.axis('off')
ax.legend(handles=[Line2D([], [], marker='o', ls='', mfc=DOMAIN_COLORS[d], mec='black', mew=0.5, ms=7,
                          label=DOMAIN_LABELS[d]) for d in DOMAIN_ORDER]
          + [Line2D([], [], ls='', label='')]
          + [Line2D([], [], marker='o', ls='', mfc=REAL, mec='black', mew=0.6, ms=7, label='real networks'),
             Line2D([], [], marker='o', ls='', mfc=NULL, mec='black', mew=0.6, ms=7, label='null partition')],
          frameon=False, fontsize=8.5, loc='upper left', bbox_to_anchor=(0.0, 1.0), handletextpad=0.3)
ax.text(0.0, -0.05, 'book, wiki, news (agnews), tldr (tldr17),\ncode (codeparrot): networks fitted on one\ndataset; pool: on all five together.\nSmall dots: halves (B: tasks); large: means.',
        transform=ax.transAxes, fontsize=7.5, color='0.35', va='bottom')

fig.text(0.5, -0.02, 'Qwen3.5 attribution-patching circuits (top 0.1% of MLP neurons by attribution) against connectivity networks '
         '(k = 100). Every measure is relative to layer-matched random neurons. D: networks chosen on the other tasks of a domain, scored on the held-out task.',
         ha='center', fontsize=7, color='0.35', wrap=True)

OUT = os.path.join(ROOT, 'plots', 'circuits_vs_networks', 'supplementary')   # superseded by make_circuits_explained.py (Iteration 36)
os.makedirs(OUT, exist_ok=True)
for ext in ('svg', 'png'):
    fig.savefig(os.path.join(OUT, 'overview.%s' % ext), format=ext, dpi=300, bbox_inches='tight')
keys = []
for r in out_rows:
    for k in r:
        if k not in keys:
            keys.append(k)
with open(os.path.join(HERE, 'circuits_networks.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(out_rows)
print('wrote plots/circuits_vs_networks/supplementary/overview and figures/circuits_networks.csv')
