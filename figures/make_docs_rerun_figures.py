"""Shuffled-rows against document-coherent text (LOG.md Iteration 48): the reruns compared.

    python figures/make_docs_rerun_figures.py

Reads, for each pair of trees (old = shuffled rows, new = whole documents):
  results/richclub/<model>-resid-pool4-{tiles,docs}/metrics/richclub/curves.csv   (Pythia-70m, Qwen3.5-2B)
  figures/bank_concentration_2b_resid_pool4{,_docs}.csv and figures/bank_task_labels.csv
  results/{pythia_resid,pythia_resid_docs}/<model>/dynamics                    (Pythia-70m, 160m)
Writes plots/docs_rerun/*.{svg,png}:
  richclub     rich-club coefficient against club size (strength ranking, mean of the two
               directions); real (blue) and circular-shift null (gray), old dashed, new solid
  circuits     per domain, median of (effective networks of the circuit - of layer-matched
               random units), on the real networks (filled) and the null partition (open);
               old text left, new text right
  dynamics     seven training-dynamics measures, old (gray) against new (blue); dashed: null
               partition; one row per model
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import make_dynamics_compare as DC
from make_pythia_figures import log_x, xpos

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'DejaVu Sans'

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')
OUT = os.path.join(ROOT, 'plots', 'docs_rerun')

OLD, NEW = 'shuffled rows', 'whole documents'
REAL, NULL = '#2C5F8A', '#999999'
GRAY_OLD = '#A0A0A0'

# Same domain colours as the circuits-vs-networks figures (make_bank_concentration_figure.py).
COLORS = {
    'Syntax': '#C0392B', 'Morphology & lexical semantics': '#D9776D',
    'Arithmetic': '#2471A3', 'Logic, code & multiple choice': '#5DADE2',
    'Factual knowledge': '#B7950B', 'Entity tracking & binding': '#148F77',
    'In-context string tasks': '#7D3C98', 'Physical reasoning': '#E67E22', 'Social reasoning': '#27AE60',
}
SHORT = {'Syntax': 'Syntax', 'Morphology & lexical semantics': 'Morph. &\nlex. sem.', 'Arithmetic': 'Arithmetic',
         'Logic, code & multiple choice': 'Logic,\ncode, MC', 'Factual knowledge': 'Factual',
         'Entity tracking & binding': 'Entity\ntracking', 'In-context string tasks': 'In-context\nstrings',
         'Physical reasoning': 'Physical', 'Social reasoning': 'Social'}
ORDER = ['Syntax', 'Morphology & lexical semantics', 'Arithmetic', 'Logic, code & multiple choice',
         'Factual knowledge', 'Entity tracking & binding', 'In-context string tasks', 'Physical reasoning',
         'Social reasoning']


def chrome(ax, lw=1.5, grid='y'):
    ax.spines[['top', 'right']].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_linewidth(lw)
    if grid:
        ax.grid(axis=grid, linestyle='--', alpha=0.5, zorder=1)
    ax.set_axisbelow(True)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ('svg', 'png'):
        fig.savefig(os.path.join(OUT, '%s.%s' % (name, ext)), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('wrote plots/docs_rerun/%s.{svg,png}' % name)


# ---------------------------------------------------------------------------- rich club

def fig_richclub():
    models = (('pythia-70m', 'Pythia-70m'), ('qwen3.5-2b', 'Qwen3.5-2B'))
    fig, axes = plt.subplots(1, 2, figsize=(6.0 * .9, 2.5 * .9))
    for ax, (m, title) in zip(axes, models):
        for v, ls in (('tiles', '--'), ('docs', '-')):
            C = pd.read_csv(os.path.join(RES, 'richclub', '%s-resid-pool4-%s' % (m, v), 'metrics', 'richclub', 'curves.csv'))
            C = C[(C.ranking == 'strength') & (C.tree == 'real')].groupby('n')[['rho', 'rho_null']].mean()
            ax.plot(C.index, C.rho, color=REAL, lw=1.8, ls=ls, zorder=3)
            ax.plot(C.index, C.rho_null, color=NULL, lw=1.4, ls=ls, zorder=2)
        ax.axhline(1, color='black', lw=0.8, zorder=1)
        ax.set_xscale('log')
        ax.set_title(title, fontsize=9, weight='bold')
        ax.set_xlabel('Club size n', fontsize=10)
        ax.tick_params(labelsize=8)
        chrome(ax)
    axes[0].set_ylabel('Rich-club coefficient', fontsize=10)
    h = [Line2D([0], [0], color=REAL, lw=1.8, label='real'), Line2D([0], [0], color=NULL, lw=1.4, label='circular-shift null'),
         Line2D([0], [0], color='black', lw=1.2, ls='-', label=NEW), Line2D([0], [0], color='black', lw=1.2, ls='--', label=OLD)]
    axes[1].legend(handles=h, frameon=False, fontsize=7, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    save(fig, 'richclub')


# ---------------------------------------------------------------------------- circuits

def circuit_table(path):
    s = pd.read_csv(path, keep_default_na=False, na_values=['', 'nan', 'NaN'])  # `null` is a tree name
    s = s[s.pct == 0.1]
    g = s.groupby(['tree', 'group', 'task']).agg(circ=('effective_networks', 'mean'),
                                                 rand=('null_effective_networks_mean', 'mean')).reset_index()
    g['d'] = g.circ - g.rand
    lab = pd.read_csv(os.path.join(HERE, 'bank_task_labels.csv'))
    return g.merge(lab[['group', 'task', 'domain']], on=['group', 'task'], how='left', validate='many_to_one')


def fig_circuits():
    tabs = ((OLD, circuit_table(os.path.join(HERE, 'bank_concentration_2b_resid_pool4.csv'))),
            (NEW, circuit_table(os.path.join(HERE, 'bank_concentration_2b_resid_pool4_docs.csv'))))
    fig, axes = plt.subplots(1, 2, figsize=(10.0 * .9, 2.8 * .9), sharey=True)
    for ax, (name, T) in zip(axes, tabs):
        for x, dom in enumerate(ORDER):
            r = T[(T.domain == dom) & (T.tree == 'real')].d.median()
            n = T[(T.domain == dom) & (T.tree == 'null')].d.median()
            ax.plot([x, x], [r, n], color=COLORS[dom], lw=1.5, alpha=0.5, zorder=2)
            ax.scatter(x, r, s=34, color=COLORS[dom], edgecolors='black', lw=0.6, zorder=4)
            ax.scatter(x, n, s=34, facecolors='white', edgecolors=COLORS[dom], lw=1.4, zorder=3)
        ax.axhline(0, color='black', lw=1, zorder=1)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([SHORT[d].replace('\n', ' ') for d in ORDER], fontsize=6.5, rotation=45, ha='right', rotation_mode='anchor')
        for t, d in zip(ax.get_xticklabels(), ORDER):
            t.set_color(COLORS[d])
        ax.set_xlim(-0.6, len(ORDER) - 0.4)
        ax.set_title(name, fontsize=9, weight='bold')
        ax.tick_params(axis='y', labelsize=8)
        chrome(ax)
    axes[0].set_ylabel('Circuit − random', fontsize=10)
    h = [Line2D([0], [0], marker='o', color='none', markerfacecolor='#555555', markeredgecolor='black', ms=6, label='real networks'),
         Line2D([0], [0], marker='o', color='none', markerfacecolor='white', markeredgecolor='#555555', markeredgewidth=1.4, ms=6,
                label='null partition')]
    axes[1].legend(handles=h, frameon=False, fontsize=7.5, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    save(fig, 'circuits')


# ---------------------------------------------------------------------------- training dynamics

SHORT_TITLES = ['Dimensionality', 'Mean |r|', 'Top-1% concentration', 'Modularity (held out)',
                'Layers per network', 'Firing rate (median)', 'Token-class coherence']


def fig_dynamics():
    models = ('pythia-70m', 'pythia-160m')
    trees = ((OLD, 'pythia_resid', GRAY_OLD), (NEW, 'pythia_resid_docs', REAL))
    D = {(m, name): DC.load(os.path.join(RES, t, m, 'dynamics')) for m in models for name, t, _ in trees}
    n = len(DC.MEASURES)
    fig, axes = plt.subplots(len(models), n, figsize=(13.5 * .9, 4.6 * .9))
    for i, m in enumerate(models):
        for j, (title, get) in enumerate(DC.MEASURES):
            ax = axes[i, j]
            for name, _, col in trees:
                real, null = get(D[(m, name)])
                steps = sorted(real)
                ax.plot([xpos(s) for s in steps], [np.nanmean(real[s]) for s in steps], color=col, lw=1.6, marker='o',
                        ms=2.5, markeredgecolor='black', markeredgewidth=0.3, zorder=3)
                if null:
                    st = sorted(null)
                    ax.plot([xpos(s) for s in st], [np.nanmean(null[s]) for s in st], color=col, lw=1.0, ls='--', zorder=2)
            log_x(ax, steps)
            ax.tick_params(labelsize=6)
            chrome(ax)
            if i == 0:
                ax.set_title(SHORT_TITLES[j], fontsize=8, weight='bold')
            if j == 0:
                ax.set_ylabel(m.replace('pythia', 'Pythia'), fontsize=9)
            if i == len(models) - 1:
                ax.set_xlabel('Training step', fontsize=7.5)
    h = [Line2D([0], [0], color=GRAY_OLD, lw=1.6, label=OLD), Line2D([0], [0], color=REAL, lw=1.6, label=NEW),
         Line2D([0], [0], color='black', lw=1.0, ls='--', label='null partition')]
    fig.legend(handles=h, frameon=False, fontsize=8, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    save(fig, 'dynamics')


def main():
    fig_richclub()
    fig_circuits()
    fig_dynamics()


if __name__ == '__main__':
    main()
