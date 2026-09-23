"""Training-dynamics figure for the Pythia runs (LOG.md Iterations 22 and 24).

    python figures/make_pythia_figures.py

Reads `results/pythia/pythia-<size>/metrics/scores_all.csv` and `checkpoints.csv` for every
size that has them (70m now, 160m when its chain completes) and writes
`plots/pythia_dynamics.{svg,png}`: four panels against the training step (log x; step 0
sits at a labelled pseudo-position left of step 1), one colour per model size, the per-domain
(or per-domain-pair) values as faint dots behind each mean. In every panel the real
partition is solid, the null partition dashed, and a ceiling dotted where it fits the axis:

  within-domain reliability (ARI), with the restart-split ceiling
  across-domain reliability (ARI)
  within-domain fidelity (r); the uncompressed ceiling, 0.99 throughout, is off the axis
  across-domain fidelity (r), with the uncompressed across-domain r

The cross-checkpoint agreement (checkpoints.csv) is no longer drawn (Andrea, 2026-09-23).

House style (scientific-figure-style): left and bottom spines only, 1.5 pt, dashed y-grid,
1/2/5 log ticks, editable-text SVG, frameless legend outside.
"""

import csv
import glob
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'DejaVu Sans'

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
RESULTS = os.path.join(ROOT, 'results', 'pythia')
PLOTS = os.path.join(ROOT, 'plots')
PROSE = ('wikitext', 'bookcorpus', 'agnews', 'tldr17')
VARIANT = 'final'

# One colour per model size, a blue ramp so that sizes read as an ordered family.
SIZE_COLORS = {'70m': '#6baed6', '160m': '#2171b5', '410m': '#08306b'}
SIZE_MARKERS = {'70m': 'o', '160m': 's', '410m': 'D'}
SIZE_ORDER = ['70m', '160m', '410m']
STEP0_X = 0.18  # pseudo-position of the initialisation on the log axis


def sci_ticks(val, _=None):
    exp = int(np.floor(np.log10(val)))
    base = val / (10 ** exp)
    if base == 1:
        return r'$10^{%d}$' % exp
    if base in (2, 5):
        return r'$%d\times10^{%d}$' % (int(base), exp)
    return ''


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_linewidth(1.5)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.tick_params(axis='both', labelsize=8)


def log_x(ax, steps):
    ax.set_xscale('log')
    # Six decades: decade ticks only (the 1/2/5 house locator gives up at this range),
    # labelled with the house sci-notation formatter; 2 and 5 stay as unlabelled minors.
    top = int(np.ceil(np.log10(max(steps))))
    ax.xaxis.set_major_locator(FixedLocator([10.0 ** k for k in range(0, top + 1)]))
    ax.xaxis.set_major_formatter(FuncFormatter(sci_ticks))
    ax.xaxis.set_minor_locator(FixedLocator([m * 10.0 ** k for k in range(0, top + 1) for m in (2, 5)]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(STEP0_X * 0.55, max(steps) * 1.8)
    # The initialisation is not on a log axis; mark it explicitly.
    ax.axvline(STEP0_X * 2.2, color='0.8', lw=0.8, ls=':', zorder=0)
    ax.annotate('init', (STEP0_X, 0), xycoords=('data', 'axes fraction'), xytext=(0, -14),
                textcoords='offset points', ha='center', fontsize=7, color='0.35')


def xpos(step):
    return STEP0_X if step == 0 else step


def load(size):
    root = os.path.join(RESULTS, 'pythia-%s' % size, 'metrics')
    if not os.path.exists(os.path.join(root, 'scores_all.csv')):
        return None
    scores = [r for r in csv.DictReader(open(os.path.join(root, 'scores_all.csv')))
              if r['fit'] in PROSE and r['eval'] in PROSE]
    for r in scores:
        r['value'] = float(r['value'])
        r['step'] = int(r['step'])
    ckpt = []
    if os.path.exists(os.path.join(root, 'checkpoints.csv')):
        ckpt = list(csv.DictReader(open(os.path.join(root, 'checkpoints.csv'))))
        for r in ckpt:
            r['value'] = float(r['value'])
            r['fit_step'], r['eval_step'] = int(r['fit_step']), int(r['eval_step'])
    return scores, ckpt


def series(scores, metric, tree='real', variant=VARIANT, delta=False):
    """{step: array of per-domain(-pair) values}, real or real minus pnull."""
    out = {}
    real = {(r['step'], r['fit'], r['eval']): r['value'] for r in scores
            if r['metric'] == metric and r['tree'] == tree and r['variant'] == variant}
    ref = {(r['step'], r['fit'], r['eval']): r['value'] for r in scores
           if r['metric'] == metric and r['tree'] == 'pnull' and r['variant'] == variant} if delta else {}
    for (step, f, e), v in real.items():
        if delta:
            if (step, f, e) not in ref:
                continue
            v = v - ref[(step, f, e)]
        out.setdefault(step, []).append(v)
    return {k: np.asarray(v) for k, v in out.items()}


def ckpt_series(ckpt, metric, tree):
    out = {}
    for r in ckpt:
        if r['metric'] == metric and r['tree'] == tree:
            out.setdefault(r['fit_step'], []).append(r['value'])
    return {k: np.asarray(v) for k, v in out.items()}


def draw(ax, data, color, marker, ls='-', dots=True, lw=1.6, alpha=1.0, zorder=3):
    steps = sorted(data)
    x = [xpos(s) for s in steps]
    y = [data[s].mean() for s in steps]
    if dots:
        for s in steps:
            ax.scatter(np.full(len(data[s]), xpos(s)), data[s], s=6, color=color, alpha=0.25,
                       edgecolors='none', zorder=2)
    ax.plot(x, y, color=color, marker=marker, markersize=4, lw=lw, ls=ls, alpha=alpha,
            markeredgecolor='black', markeredgewidth=0.4, zorder=zorder)
    return steps


def main():
    sizes = []
    for path in sorted(glob.glob(os.path.join(RESULTS, 'pythia-*'))):
        size = re.sub(r'.*pythia-', '', path)
        if load(size) is not None:
            sizes.append(size)
    sizes = [s for s in SIZE_ORDER if s in sizes] + [s for s in sizes if s not in SIZE_ORDER]
    assert sizes, 'no metrics under %s' % RESULTS

    # Every panel the same way (Andrea, 2026-09-23): the real partition solid, the null
    # partition (fit on circularly shifted data, evaluated on the real data) dashed, and a
    # ceiling dotted where it fits the axis. Earlier versions plotted fidelity as real minus
    # null and reliability raw, which hid that the fidelity null moves a lot over training.
    fig, axes = plt.subplots(2, 2, figsize=(5.4 * .95, 4.4 * .95), dpi=300)
    axes = axes.ravel()
    titles = ['Within-domain reliability (ARI)', 'Across-domain reliability (ARI)',
              'Within-domain fidelity (r)', 'Across-domain fidelity (r)']
    all_steps = set()
    for size in sizes:
        scores, _ = load(size)
        c, m = SIZE_COLORS.get(size, '0.3'), SIZE_MARKERS.get(size, 'o')
        ref = dict(ls='--', dots=False, lw=1.0, alpha=0.8, zorder=2)
        ceil = dict(ls=':', dots=False, lw=1.0, alpha=0.8, zorder=2)
        all_steps |= set(draw(axes[0], series(scores, 'reliability_within'), c, m))
        draw(axes[0], series(scores, 'reliability_within', tree='pnull'), c, None, **ref)
        draw(axes[0], series(scores, 'reliability_ceiling'), c, None, **ceil)
        draw(axes[1], series(scores, 'reliability_across_halves'), c, m)
        draw(axes[1], series(scores, 'reliability_across_halves', tree='pnull'), c, None, **ref)
        draw(axes[2], series(scores, 'fidelity_within_r'), c, m)
        draw(axes[2], series(scores, 'fidelity_within_r', tree='pnull'), c, None, **ref)
        draw(axes[3], series(scores, 'fidelity_across_halves'), c, m)
        draw(axes[3], series(scores, 'fidelity_across_halves', tree='pnull'), c, None, **ref)
        draw(axes[3], series(scores, 'fidelity_across_halves', variant='(ceiling)'), c, None, **ceil)
    steps = sorted(all_steps)
    for ax, title in zip(axes, titles):
        style(ax)
        log_x(ax, steps)
        ax.set_title(title, fontsize=8.5, fontweight='bold')
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    for ax in axes[2:]:
        ax.set_xlabel('Training step', fontsize=9)

    handles = [Line2D([0], [0], color=SIZE_COLORS.get(s, '0.3'), marker=SIZE_MARKERS.get(s, 'o'),
                      markersize=4, markeredgecolor='black', markeredgewidth=0.4, lw=1.6,
                      label='Pythia-%s' % s) for s in sizes]
    handles += [Line2D([0], [0], color='0.3', ls='-', lw=1.6, label='real partition'),
                Line2D([0], [0], color='0.3', ls='--', lw=1.0, label='null partition'),
                Line2D([0], [0], color='0.3', ls=':', lw=1.0, label='ceiling')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.09),
               ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(w_pad=1.0, h_pad=1.2)
    os.makedirs(PLOTS, exist_ok=True)
    fig.savefig(os.path.join(PLOTS, 'pythia_dynamics.svg'), format='svg', bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS, 'pythia_dynamics.png'), dpi=300, bbox_inches='tight')
    print('wrote plots/pythia_dynamics.{svg,png} for sizes %s' % ', '.join(sizes))


if __name__ == '__main__':
    main()
