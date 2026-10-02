"""Supporting figures for plots/circuits_vs_networks/README.md (LOG.md Iterations 30-35).

    PYTHONPATH=. python figures/make_circuits_extra_figures.py

  schematic.{svg,png}    the two methods and how they are compared
  granularity.{svg,png}  2B single datasets at k = 10, 20, 100 (Iteration 35)
  graded.{svg,png}       the all-unit graded test (Iteration 35)

Reads results/qwen35/<tree>/circuits*/; the plotted values go to
figures/circuits_extra.csv.
"""

import collections
import csv
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats

from parcelmate.circuits import heldout_enrichment

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'DejaVu Sans'

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
RES = os.path.join(ROOT, 'results', 'qwen35')
OUT = os.path.join(ROOT, 'plots', 'circuits_vs_networks')
DOMAIN_COLORS = {'Lan': '#C44E52', 'MD': '#4C72B0', 'ToM': '#55A868', 'phys': '#DD8452'}
DOMAIN_LABELS = {'Lan': 'Language', 'MD': 'Formal', 'ToM': 'Social', 'phys': 'Physical'}
DOMAIN_ORDER = ['Lan', 'MD', 'ToM', 'phys']
REAL, NULL = '#2c5f8a', '#9a9a9a'
DATASET_COLORS = {'bookcorpus': '#8c6d31', 'wikitext': '#3182bd', 'agnews': '#31a354',
                  'tldr17': '#756bb1', 'codeparrot': '#636363'}
DATASET_LABELS = {'bookcorpus': 'bookcorpus', 'wikitext': 'wikitext', 'agnews': 'agnews',
                  'tldr17': 'tldr17', 'codeparrot': 'codeparrot (code)'}
rows_out = []


def rd(*parts):
    return list(csv.DictReader(open(os.path.join(RES, *parts))))


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_linewidth(1.5)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
    ax.tick_params(labelsize=8)


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ('svg', 'png'):
        fig.savefig(os.path.join(OUT, '%s.%s' % (name, ext)), format=ext, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------- schematic
def schematic():
    fig, ax = plt.subplots(figsize=(10.5 * .85, 4.2 * .85))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.2)
    ax.axis('off')

    def box(x, y, w, h, text, fc, ec='black', fs=7.5, weight='normal'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.12',
                                    fc=fc, ec=ec, lw=1.2, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, weight=weight, zorder=3)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=11,
                                     lw=1.3, color='0.25', zorder=1))

    pc, cc, mc = '#f6e3e3', '#e1ebf5', '#f2f2f2'
    ax.text(0.05, 3.95, 'Attribution patching  (task circuits)', fontsize=9.5, weight='bold', color='#8b2f33')
    box(0.05, 2.75, 2.0, 1.0, '46 minimal-pair tasks\n4 domains: Language,\nFormal, Social, Physical', pc)
    box(2.45, 2.75, 2.0, 1.0, 'attribution of every\nMLP neuron\n(activation × gradient)', pc)
    box(4.85, 2.75, 2.0, 1.0, 'circuit = top 0.1%\nof all neurons\n(147 at 2B, 294 at 4B)', pc)
    arrow(2.05, 3.25, 2.45, 3.25)
    arrow(4.45, 3.25, 4.85, 3.25)
    ax.text(0.05, 1.85, 'Connectivity  (networks)', fontsize=9.5, weight='bold', color='#1f4e79')
    box(0.05, 0.65, 2.0, 1.0, 'ordinary text, 5 datasets\n(4 prose + code),\nno tasks involved', cc)
    box(2.45, 0.65, 2.0, 1.0, 'activation timecourse\nof every MLP neuron\n→ |r| connectome', cc)
    box(4.85, 0.65, 2.0, 1.0, 'k = 100 networks\n(k-means, consensus)\nper dataset or pooled', cc)
    arrow(2.05, 1.15, 2.45, 1.15)
    arrow(4.45, 1.15, 4.85, 1.15)
    box(7.45, 1.2, 2.95, 2.0, 'Do circuits and networks\nfind the same neurons?\n\nconcentration · enrichment\nshared structure · graded',
        mc, fs=8.5)
    arrow(6.85, 3.25, 7.45, 2.6)
    arrow(6.85, 1.15, 7.45, 1.8)
    ax.text(7.45, 0.55, 'references:\n• random neurons with the circuit\'s per-layer counts\n• null partition: same pipeline on time-shifted\n   activations (keeps neuron properties, no co-activation)',
            fontsize=7, color='0.3', va='top')
    save(fig, 'schematic')


# ---------------------------------------------------------------- granularity
GRAN = [('bookcorpus', 'qwen3.5-2b', 'qwen3.5-2b-coarse'), ('wikitext', 'qwen3.5-2b', 'qwen3.5-2b-coarse'),
        ('agnews', 'qwen3.5-2b-rest', 'qwen3.5-2b-rest'), ('tldr17', 'qwen3.5-2b-rest', 'qwen3.5-2b-rest'),
        ('codeparrot', 'qwen3.5-2b-rest', 'qwen3.5-2b-rest')]
KS = [10, 20, 100]


def granularity_values(td, tree, sub, k):
    d = (tree, sub)
    pt = [r for r in rd(*d, 'paired_tasks.csv') if r['pct'] == '0.1' and r['text_domain'] == td]
    per = collections.defaultdict(list)
    for r in pt:
        per[r['task']].append(float(r['deficit_diff']))
    dd = np.array([np.mean(v) for v in per.values()])
    st = collections.defaultdict(dict)
    for r in rd(*d, 'structure.csv'):
        if r['pct'] == '0.1' and r['text_domain'] == td:
            st[(r['tree'], r['half'])][r['measure']] = float(r['value'])
    rho = {tr: np.mean([st[(tr, h)]['spearman_overlap_excess'] for h in ('halfA', 'halfB')]) for tr in ('real', 'null')}
    ntop = max(1, int(round(0.05 * k)))
    rows = [r for r in rd(*d, 'enrichment.csv') if r['pct'] == '0.1' and r['text_domain'] == td]
    he = {}
    for tr in ('real', 'null'):
        vals = []
        for h in ('halfA', 'halfB'):
            rr = [r for r in rows if r['tree'] == tr and r['half'] == h]
            tasks = sorted({r['task'] for r in rr})
            dom = {}
            O, E = np.zeros((len(tasks), k)), np.zeros((len(tasks), k))
            for r in rr:
                i = tasks.index(r['task'])
                O[i, int(r['network'])] = float(r['observed'])
                E[i, int(r['network'])] = float(r['expected'])
                dom[r['task']] = r['task_domain']
            vals.append(np.nanmean([x['ratio'] for x in heldout_enrichment(O, E, [dom[t] for t in tasks], ntop)]))
        he[tr] = float(np.mean(vals))
    return dict(frac_real_more_concentrated=float((dd < 0).mean()),
                p_paired=float(stats.wilcoxon(dd, alternative='less').pvalue),
                median_z_real=float(np.median([float(r['z_real']) for r in pt])),
                rho_real=float(rho['real']), rho_null=float(rho['null']),
                heldout_real=he['real'], heldout_null=he['null'], n_top=ntop)


def granularity():
    vals = {}
    for td, t100, tc in GRAN:
        for k in KS:
            tree, sub = (t100, 'circuits') if k == 100 else (tc, 'circuits_k%d' % k)
            if not os.path.exists(os.path.join(RES, tree, sub, 'paired_tasks.csv')):
                continue
            vals[(td, k)] = granularity_values(td, tree, sub, k)
            rows_out.append(dict(figure='granularity', dataset=td, k=k, **vals[(td, k)]))
    panels = [('median_z_real', 'median z of circuit entropy\n(real networks; < 0 = concentrated)', 'Concentration vs chance'),
              ('frac_real_more_concentrated', 'share of tasks more concentrated\non real than on null networks', 'Real vs null partition'),
              ('rho', 'Spearman ρ, real − null\n(overlap vs network similarity)', 'Shared circuits → shared networks'),
              ('heldout', 'held-out enrichment,\nreal ÷ null partition', 'Domains reuse networks')]
    fig, axes = plt.subplots(1, 4, figsize=(14 * .85, 3.3 * .85))
    plt.subplots_adjust(wspace=0.55)
    xs = np.arange(len(KS))
    for ax, (key, ylabel, ttl) in zip(axes, panels):
        for td, _, _ in GRAN:
            y = []
            for k in KS:
                v = vals.get((td, k))
                if v is None:
                    y.append(np.nan)
                elif key == 'rho':
                    y.append(v['rho_real'] - v['rho_null'])
                elif key == 'heldout':
                    y.append(v['heldout_real'] / v['heldout_null'])
                else:
                    y.append(v[key])
            ax.plot(xs, y, marker='o', ms=5, lw=1.6, color=DATASET_COLORS[td], label=DATASET_LABELS[td], zorder=3)
        ref = {'median_z_real': 0, 'frac_real_more_concentrated': 0.5, 'rho': 0, 'heldout': 1}[key]
        ax.axhline(ref, color='#333333', lw=0.9, ls='--', zorder=2)
        ax.set_xticks(xs)
        ax.set_xticklabels(['k = %d' % k for k in KS], fontsize=8)
        ax.set_xlim(-0.3, len(KS) - 0.7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(ttl, fontsize=9, weight='bold')
        style(ax)
    axes[-1].legend(frameon=False, fontsize=7.5, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    fig.text(0.5, -0.06, 'Qwen3.5-2B, networks fitted on one dataset at a time, 0.1% circuits, means of the two halves. '
             'Held-out enrichment uses the top 5% of networks (1 at k = 10 and 20, 5 at k = 100).',
             ha='center', fontsize=7, color='0.35')
    save(fig, 'granularity')


# ---------------------------------------------------------------- graded
SETS = [('qwen3.5-2b', 'bookcorpus', '2B\nbook'), ('qwen3.5-2b', 'wikitext', '2B\nwiki'),
        ('qwen3.5-2b-rest', 'agnews', '2B\nnews'), ('qwen3.5-2b-rest', 'tldr17', '2B\ntldr'),
        ('qwen3.5-2b-rest', 'codeparrot', '2B\ncode'), ('qwen3.5-2b-pool5', 'pooled', '2B\npool'),
        ('qwen3.5-4b', 'wikitext', '4B\nwiki'), ('qwen3.5-4b-pool5', 'pooled', '4B\npool')]


def graded():
    fig, axes = plt.subplots(1, 2, figsize=(10 * .85, 3.4 * .85), gridspec_kw=dict(width_ratios=[1.6, 1]))
    plt.subplots_adjust(wspace=0.35)
    ax = axes[0]
    for i, (tree, td, label) in enumerate(SETS):
        g = [r for r in rd(tree, 'circuits', 'graded_summary.csv') if r['task_group'] == 'all' and r['text_domain'] == td]
        fr = [float(r['frac_p05_real']) for r in g]
        fn = [float(r['frac_p05_null']) for r in g]
        p = max(float(r['p_wilcoxon']) for r in g)
        ax.plot([i - 0.12, i + 0.12], [np.mean(fn), np.mean(fr)], color='0.6', lw=1.2, zorder=2)
        ax.scatter([i - 0.12] * 2, fn, s=9, color=NULL, alpha=0.5, edgecolors='none', zorder=3)
        ax.scatter([i + 0.12] * 2, fr, s=9, color=REAL, alpha=0.5, edgecolors='none', zorder=3)
        ax.scatter(i - 0.12, np.mean(fn), s=36, color=NULL, edgecolors='black', lw=0.6, zorder=4)
        ax.scatter(i + 0.12, np.mean(fr), s=36, color=REAL, edgecolors='black', lw=0.6, zorder=4)
        ax.text(i, 0.66, stars(p), ha='center', fontsize=7.5)
        rows_out.append(dict(figure='graded', set=label.replace('\n', ' '), frac_p05_real=np.mean(fr),
                             frac_p05_null=np.mean(fn), p_paired_worst_half=p))
    ax.set_xticks(range(len(SETS)))
    ax.set_xticklabels([s[2] for s in SETS], fontsize=7)
    ax.set_ylim(0, 0.72)
    ax.set_ylabel('share of tasks whose attribution\nnetworks explain beyond layer (p < .05)', fontsize=8)
    ax.set_title('All neurons, no cut: per task', fontsize=9, weight='bold')
    style(ax)
    ax.legend(handles=[Line2D([], [], marker='o', ls='', mfc=REAL, mec='black', mew=0.6, label='real networks'),
                       Line2D([], [], marker='o', ls='', mfc=NULL, mec='black', mew=0.6, label='null partition')],
              frameon=False, fontsize=7, loc='upper left', bbox_to_anchor=(0.0, -0.2), ncol=2)
    ax = axes[1]
    rng = np.random.default_rng(0)
    for j, d in enumerate(DOMAIN_ORDER):
        v = {'real': [], 'null': []}
        for tree, td, label in SETS:
            for r in rd(tree, 'circuits', 'graded_tasks.csv'):
                if r['kind'] == 'domain' and r['map'] == d and r['text_domain'] == td:
                    v[r['tree']].append(1e4 * float(r['excess']))
        for tr, dx, col in (('null', -0.14, NULL), ('real', 0.14, DOMAIN_COLORS[d])):
            x = j + dx + rng.uniform(-0.05, 0.05, len(v[tr]))
            ax.scatter(x, v[tr], s=9, color=col, alpha=0.5, edgecolors='none', zorder=3)
            ax.scatter(j + dx, np.mean(v[tr]), s=40, color=col, edgecolors='black', lw=0.6, zorder=4)
        ax.plot([j - 0.14, j + 0.14], [np.mean(v['null']), np.mean(v['real'])], color='0.6', lw=1.2, zorder=2)
        rows_out.append(dict(figure='graded_domain', task_domain=d, excess_real_x1e4=np.mean(v['real']),
                             excess_null_x1e4=np.mean(v['null'])))
    ax.axhline(0, color='#333333', lw=0.9, zorder=2)
    ax.set_xticks(range(4))
    ax.set_xticklabels([DOMAIN_LABELS[d] for d in DOMAIN_ORDER], fontsize=8)
    ax.set_ylabel('variance explained by networks\nbeyond layer, excess (× 10⁻⁴)', fontsize=8)
    ax.set_title('Domain-averaged maps', fontsize=9, weight='bold')
    style(ax)
    save(fig, 'graded')


schematic()
granularity()
graded()
keys = []
for r in rows_out:
    for k in r:
        if k not in keys:
            keys.append(k)
with open(os.path.join(HERE, 'circuits_extra.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows_out)
print('wrote plots/circuits_vs_networks/{schematic,granularity,graded}.{svg,png} and figures/circuits_extra.csv')
