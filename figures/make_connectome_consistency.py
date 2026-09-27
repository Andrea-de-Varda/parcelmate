"""Connectome consistency over training, no parcellation (LOG.md Iteration 31, 2026-09-25).

    python figures/make_connectome_consistency.py

Reads the `(ceiling)` rows of `results/pythia/pythia-<size>/metrics/scores_all.csv`: the
Pearson r over all unit pairs (strict upper triangle) between two |r| connectomes of all MLP
neurons. Writes `plots/pythia_connectome_consistency.{svg,png}`, three panels against the
training step:

  A  split-half: half A vs half B of one domain (faint dots: the four domains)
  B  across datasets: half A of one domain vs half B of another (faint dots: 12 ordered pairs)
  C  across datasets per domain pair (Pythia-160m, both directions averaged), the two pairs
     that agree most at initialisation in colour
  D  word order: across-dataset r on word-shuffled text minus intact text (LOG.md Iteration
     32; faint dots: the 12 pairs)

Real solid, circular-shift null dashed; in A and B the word-shuffled text (Iteration 32,
`results/pythia_shuffled/`) dotted with open markers, where those tables exist. Chrome and x axis from make_pythia_figures.py.
"""

import collections
import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from make_pythia_figures import (
    PLOTS, RESULTS, ROOT, SIZE_COLORS, SIZE_MARKERS, log_x, style, xpos,
)

SIZES = ['70m', '160m']
LABEL = {'wikitext': 'Wiki', 'bookcorpus': 'Books', 'agnews': 'News', 'tldr17': 'Reddit'}
HIGHLIGHT = {('bookcorpus', 'wikitext'): '#e6550d', ('agnews', 'tldr17'): '#31a354'}


def load(size):
    out = collections.defaultdict(list)   # (tree, metric, step) -> [(fit, eval, value)]
    with open(os.path.join(RESULTS, 'pythia-%s' % size, 'metrics', 'scores_all.csv')) as f:
        for r in csv.DictReader(f):
            if r['variant'] == '(ceiling)' and r['metric'] in ('fidelity_within_r', 'fidelity_across_halves'):
                out[(r['tree'], r['metric'], int(r['step']))].append((r['fit'], r['eval'], float(r['value'])))
    return out


def load_shuffled(size):
    out = collections.defaultdict(list)   # (metric, step) -> [(fit, eval, value)]
    for p in glob.glob(os.path.join(ROOT, 'results', 'pythia_shuffled', 'pythia-%s' % size, 'step*',
                                    'metrics', 'connectome_consistency.csv')):
        for r in csv.DictReader(open(p)):
            out[(r['metric'], int(r['revision'].replace('step', '')))].append((r['fit'], r['eval'], float(r['value'])))
    return out


def panel(ax, data, metric, title, shuffled=None):
    for size in SIZES:
        d, c, m = data[size], SIZE_COLORS[size], SIZE_MARKERS[size]
        steps = sorted({k[2] for k in d if k[1] == metric})
        x = np.array([xpos(s) for s in steps])
        for s, xx in zip(steps, x):
            v = [t[2] for t in d[('real', metric, s)]]
            ax.scatter(np.full(len(v), xx), v, s=6, color=c, alpha=0.25, lw=0, zorder=2)
        real = [np.mean([t[2] for t in d[('real', metric, s)]]) for s in steps]
        null = [np.mean([t[2] for t in d[('null', metric, s)]]) for s in steps]
        ax.plot(x, real, color=c, marker=m, ms=4, mec='k', mew=0.5, lw=1.6, zorder=3)
        ax.plot(x, null, color=c, ls='--', lw=1.2, zorder=3)
        sh = (shuffled or {}).get(size)
        if sh:
            ss = sorted(k[1] for k in sh if k[0] == metric)
            ax.plot([xpos(t) for t in ss], [np.mean([v for *_, v in sh[(metric, t)]]) for t in ss], color=c,
                    ls=':', lw=1.4, marker=m, ms=4, mfc='white', mec=c, mew=0.9, zorder=4)
    log_x(ax, steps)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title, fontsize=9, fontweight='bold')


def main():
    data = {s: load(s) for s in SIZES}
    shuffled = {s: load_shuffled(s) for s in SIZES}
    has_shuf = any(shuffled.values())
    n = 4 if has_shuf else 3
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n * .9, 2.6 * .9))
    for ax in axes[1:3]:
        ax.sharey(axes[0])
    panel(axes[0], data, 'fidelity_within_r', 'Same dataset,\ntwo halves', shuffled)
    panel(axes[1], data, 'fidelity_across_halves', 'Across datasets', shuffled)
    if has_shuf:
        ax = axes[3]
        for size in SIZES:
            d, sh, c, m = data[size], shuffled[size], SIZE_COLORS[size], SIZE_MARKERS[size]
            steps = sorted(k[1] for k in sh if k[0] == 'fidelity_across_halves')
            mean = []
            for t in steps:
                o = {(f_, e): v for f_, e, v in d[('real', 'fidelity_across_halves', t)]}
                diff = [v - o[(f_, e)] for f_, e, v in sh[('fidelity_across_halves', t)]]
                ax.scatter(np.full(len(diff), xpos(t)), diff, s=6, color=c, alpha=0.25, lw=0, zorder=2)
                mean.append(np.mean(diff))
            ax.plot([xpos(t) for t in steps], mean, color=c, marker=m, ms=4, mec='k', mew=0.5, lw=1.6, zorder=3)
        ax.axhline(0, color='0.3', lw=0.9, zorder=1)
        log_x(ax, steps)
        ax.set_title('Effect of shuffling word\norder (across datasets)', fontsize=9, fontweight='bold')
        ax.set_ylabel('Δr (shuffled − intact)', fontsize=10)

    ax = axes[2]
    d = data['160m']
    steps = sorted({k[2] for k in d if k[1] == 'fidelity_across_halves'})
    pairs = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in steps:
        for f_, e, v in d[('real', 'fidelity_across_halves', s)]:
            pairs[tuple(sorted((f_, e)))][s].append(v)
    x = [xpos(s) for s in steps]
    for pair, by in sorted(pairs.items(), key=lambda kv: kv[0] in HIGHLIGHT):
        c = HIGHLIGHT.get(pair, '0.7')
        ax.plot(x, [np.mean(by[s]) for s in steps], color=c, lw=1.8 if pair in HIGHLIGHT else 1.0,
                zorder=3 if pair in HIGHLIGHT else 2)
    log_x(ax, steps)
    ax.set_title('Across datasets, per pair\n(Pythia-160m)', fontsize=9, fontweight='bold')
    for ax in axes[1:3]:
        ax.tick_params(labelleft=False)

    axes[0].set_ylabel('Connectome correlation (r)', fontsize=10)
    for ax in axes:
        style(ax)
        ax.set_xlabel('Training step', fontsize=10)

    handles = [Line2D([], [], color=SIZE_COLORS[s], marker=SIZE_MARKERS[s], mec='k', mew=0.5, lw=1.6,
                      label='Pythia-%s' % s) for s in SIZES]
    handles += [Line2D([], [], color='0.3', lw=1.6, label='intact text'),
                Line2D([], [], color='0.3', lw=1.2, ls='--', label='circular-shift null')]
    if has_shuf:
        handles += [Line2D([], [], color='0.3', lw=1.4, ls=':', marker='o', ms=4, mfc='white', label='words shuffled')]
    handles += [Line2D([], [], color=c, lw=1.8, label='%s–%s' % (LABEL[a], LABEL[b])) for (a, b), c in HIGHLIGHT.items()]
    handles += [Line2D([], [], color='0.7', lw=1.0, label='other pairs')]
    fig.legend(handles=handles, loc='lower center', ncol=8, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.1), handlelength=1.8, columnspacing=1.2)
    fig.tight_layout()
    for ext in ('svg', 'png'):
        fig.savefig(os.path.join(PLOTS, 'pythia_connectome_consistency.%s' % ext), format=ext,
                    dpi=300, bbox_inches='tight')
    print('wrote plots/pythia_connectome_consistency.{svg,png}')


if __name__ == '__main__':
    main()
