"""The descriptive training-dynamics measures (LOG.md Iteration 28) against training step.

    python figures/make_dynamics_figures.py

Reads `results/pythia/pythia-<size>/dynamics/` for every size that has it and writes
`plots/pythia_descriptive.{svg,png}`: twelve panels in three rows, the same axes and
conventions as `plots/pythia_dynamics.svg` (log step with the initialisation at a labelled
pseudo-position; the real partition solid, the null partition dashed, a noise reference
dotted; per-domain values as faint dots behind each mean over domains and halves).

  connectome   dimensionality (participation ratio), mean |r|, concentration of each unit's
               connections on its top 1% of partners, inequality of unit strength (Gini)
  networks     modularity of the stored partition on the other half, effective number of
               layers per network, crispness, networks carried over to the next checkpoint
               (reference: two halves of the same checkpoint)
  activations  median firing rate, token-class coherence of the networks, language-modelling
  and model    loss on the same text, connectome similarity to the final checkpoint
               (reference: two halves of the same checkpoint)

CSV readers keep the tree label 'null' (pandas would read it as missing).
"""

import csv
import glob
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from make_pythia_figures import SIZE_COLORS, SIZE_MARKERS, SIZE_ORDER, log_x, style, xpos

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'DejaVu Sans'

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
RESULTS = os.path.join(ROOT, 'results', 'pythia')
PLOTS = os.path.join(ROOT, 'plots')


def read(path):
    if not os.path.exists(path):
        return []
    out = list(csv.DictReader(open(path)))
    for r in out:
        r['value'] = float(r['value'])
        r['step'] = int(r['step'])
    return out


def load(size):
    d = os.path.join(RESULTS, 'pythia-%s' % size, 'dynamics')
    if not os.path.isdir(d):
        return None
    conn = []
    for p in sorted(glob.glob(os.path.join(d, 'connectome_step*.csv'))):
        conn.extend(read(p))
    return dict(conn=conn, part=read(os.path.join(d, 'partitions.csv')),
                sim=read(os.path.join(d, 'similarity.csv')), sel=read(os.path.join(d, 'selectivity.csv')))


def by_domain(rows, **filt):
    """{step: array of per-domain means} over the rows matching `filt` (halves averaged)."""
    acc = {}
    for r in rows:
        if all((r.get(k) == v) if not callable(v) else v(r) for k, v in filt.items()):
            acc.setdefault(r['step'], {}).setdefault(r['domain'], []).append(r['value'])
    return {s: np.array([np.mean(v) for v in dom.values()]) for s, dom in acc.items()}


def draw(ax, data, color, marker, ls='-', dots=True, lw=1.6, alpha=1.0, zorder=3):
    if not data:
        return []
    steps = sorted(data)
    if dots:
        for s in steps:
            ax.scatter(np.full(len(data[s]), xpos(s)), data[s], s=6, color=color, alpha=0.25,
                       edgecolors='none', zorder=2)
    ax.plot([xpos(s) for s in steps], [np.nanmean(data[s]) for s in steps], color=color,
            marker=marker, markersize=4, lw=lw, ls=ls, alpha=alpha, markeredgecolor='black',
            markeredgewidth=0.4, zorder=zorder)
    return steps


def main():
    sizes = [s for s in SIZE_ORDER if load(s) is not None]
    assert sizes, 'no dynamics results under %s' % RESULTS
    fig, axes = plt.subplots(3, 4, figsize=(9.2 * .95, 6.6 * .95), dpi=300)
    axes = axes.ravel()
    titles = ['Dimensionality\n(participation ratio)', 'Mean |r|', 'Concentration on the\ntop 1% of partners',
              'Inequality of unit\nstrength (Gini)',
              'Modularity of the\nnetworks (held out)', 'Layers per network\n(effective number)',
              'Crispness (median\nmax membership)', 'Networks carried over\nto the next checkpoint',
              'Neuron firing rate\n(median)', 'Token-class coherence\nof the networks', 'Language-modelling\nloss',
              'Connectome similarity\nto the final checkpoint']
    ref = dict(ls='--', dots=False, lw=1.0, alpha=0.8, zorder=2)
    ceil = dict(ls=':', dots=False, lw=1.0, alpha=0.8, zorder=2)
    all_steps = set()
    for size in sizes:
        D = load(size)
        c, m = SIZE_COLORS.get(size, '0.3'), SIZE_MARKERS.get(size, 'o')
        conn, part, sim, sel = D['conn'], D['part'], D['sim'], D['sel']
        own = dict(partition='-')
        for ax, measure in zip(axes[:4], ('dim_participation_ratio', 'coupling_mean_absr',
                                          'coupling_concentration_top1pct', 'hub_strength_gini')):
            all_steps |= set(draw(ax, by_domain(conn, measure=measure, **own), c, m))
        held = lambda r: r['key'] != r['eval_key']
        draw(axes[4], by_domain(conn, measure='segregation_modularity', partition='real', held=held), c, m)
        draw(axes[4], by_domain(conn, measure='segregation_modularity', partition='null', held=held), c, None, **ref)
        halves = lambda r: r['key'] in ('halfA', 'halfB')
        for ax, measure in ((axes[5], 'depth_median_effective_layers'), (axes[6], 'crisp_median_max_membership')):
            draw(ax, by_domain(part, measure=measure, tree='real', h=halves), c, m)
            draw(ax, by_domain(part, measure=measure, tree='null', h=halves), c, None, **ref)
        draw(axes[7], by_domain(part, measure='trans_continued', tree='real', h=halves), c, m)
        draw(axes[7], by_domain(part, measure='halves_continued', tree='real'), c, None, **ceil)
        draw(axes[8], by_domain(conn, measure='act_firing_rate_median', **own), c, m)
        draw(axes[9], by_domain(sel, measure='selectivity_coherence_median', tree='real'), c, m)
        draw(axes[9], by_domain(sel, measure='selectivity_coherence_median', tree='null'), c, None, **ref)
        draw(axes[10], by_domain(conn, measure='lm_loss', **own), c, m)
        draw(axes[11], by_domain(sim, measure='similarity_to_final'), c, m)
        draw(axes[11], by_domain(sim, measure='similarity_between_halves'), c, None, **ceil)
    steps = sorted(all_steps) or [1, 143000]
    for i, (ax, title) in enumerate(zip(axes, titles)):
        style(ax)
        log_x(ax, steps + [143000])
        ax.set_title(title, fontsize=8, fontweight='bold')
        if i in (4, 6, 7, 9, 11):
            ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
        if i >= 8:
            ax.set_xlabel('Training step', fontsize=8.5)
    handles = [Line2D([0], [0], color=SIZE_COLORS.get(s, '0.3'), marker=SIZE_MARKERS.get(s, 'o'),
                      markersize=4, markeredgecolor='black', markeredgewidth=0.4, lw=1.6,
                      label='Pythia-%s' % s) for s in sizes]
    handles += [Line2D([0], [0], color='0.3', ls='-', lw=1.6, label='real partition / measure'),
                Line2D([0], [0], color='0.3', ls='--', lw=1.0, label='null partition'),
                Line2D([0], [0], color='0.3', ls=':', lw=1.0, label='two halves of one checkpoint')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.05),
               ncol=len(handles), frameon=False, fontsize=7.5)
    fig.tight_layout(w_pad=0.8, h_pad=1.0)
    os.makedirs(PLOTS, exist_ok=True)
    fig.savefig(os.path.join(PLOTS, 'pythia_descriptive.svg'), format='svg', bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS, 'pythia_descriptive.png'), dpi=300, bbox_inches='tight')
    print('wrote plots/pythia_descriptive.{svg,png} for sizes %s' % ', '.join(sizes))


if __name__ == '__main__':
    main()
