"""Publication figures for the parcelmate reliability/fidelity ladder.

    python figures/make_figures.py      ->  plots/*.svg + *.png

Reads figures/scores.csv (copied from results/reliability/metrics/scores.csv on the
cluster, job 17366628). Every number plotted is a per-domain measurement averaged over the
four prose domains; the degenerate whitespace and codeparrot domains are excluded from the
averages and shown separately, because a block model fits them trivially well and they
dominate an equal-weighted mean.

Three figures:
  1. ladder_null_calibrated -- the main result: each arm's real vs null, the delta is the point
  2. ladder_mechanism       -- why: hubness contamination tracks the null's clusterability
  3. fidelity_vs_ceiling    -- held-out == in-sample, so the model class is the binding limit
"""

import csv
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.expanduser(
    '~/.claude/skills/scientific-figure-style/scripts'))
from figure_style import apply_style, save_fig, style_spines  # noqa: E402

apply_style()
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'DejaVu Sans'

HERE = os.path.dirname(os.path.abspath(__file__))
PROSE = ('wikitext', 'agnews', 'bookcorpus', 'tldr17')

# One colour per arm, held fixed across every figure. The ladder is a light-to-dark ramp so
# the progression reads off the page; `legacy` is gray because it is the inherited
# pre-bugfix baseline rather than a rung of the ladder.
ARM_COLORS = {
    'legacy':       '#999999',   # gray: the inherited pre-bugfix baseline, not a rung
    'current':      '#a6cee3',
    'fisher_pca':   '#4e9ac6',
    'nopca_fisher': '#1f6eb3',
    'vmf_profile':  '#08306b',
}
ARM_LABELS = {
    'legacy': 'legacy',
    'current': 'current',
    'fisher_pca': '+ magnitudes',
    'nopca_fisher': '+ no PCA',
    'vmf_profile': '+ standardised',
}
LADDER = ['legacy', 'current', 'fisher_pca', 'nopca_fisher', 'vmf_profile']
NS_FILL, NS_EDGE = '#cccccc', '#999999'


def load():
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, 'scores.csv')))]
    for r in rows:
        r['value'] = float(r['value'])
    return rows


ROWS = load()


def vals(metric, tree, variant, domains=PROSE):
    return [r['value'] for r in ROWS
            if r['metric'] == metric and r['tree'] == tree
            and r['variant'] == variant and r['fit'] in domains]


def mean(metric, tree, variant, domains=PROSE):
    v = vals(metric, tree, variant, domains)
    return float(np.mean(v)) if v else float('nan')


# ---------------------------------------------------------------------------------------
# Figure 1 -- the main result. Dumbbell: the delta between null and real IS the finding, and
# only one arm has a positive delta on both metrics.
# ---------------------------------------------------------------------------------------
def fig_ladder():
    fig, axes = plt.subplots(1, 2, figsize=(6.8 * 0.88, 2.9 * 0.88), dpi=300, sharey=True)
    panels = [('reliability_within', 'Reliability (ARI)', axes[0]),
              ('fidelity_within', 'Fidelity (R$^2$)', axes[1])]
    ypos = np.arange(len(LADDER))[::-1]

    for metric, xlabel, ax in panels:
        for i, arm in enumerate(LADDER[::-1]):
            y = ypos[i]
            real, null = mean(metric, 'real', arm), mean(metric, 'null', arm)
            beats = real > null
            # Gray when the arm fails to beat its own null -- the house convention for
            # "not significant". Colour is earned, not given.
            c = ARM_COLORS[arm] if beats else NS_EDGE
            ax.hlines(y, min(real, null), max(real, null),
                      color=c, alpha=0.35 if beats else 0.25, linewidth=3, zorder=2)
            # Faint per-domain real values, so the reader sees the spread behind the mean.
            for v in vals(metric, 'real', arm):
                ax.scatter(v, y, s=9, color=c, alpha=0.30, edgecolors='none', zorder=2.5)
            ax.scatter(null, y, s=42, facecolors='white', edgecolors=c,
                       linewidths=1.2, zorder=3)
            ax.scatter(real, y, s=42, color=c if beats else NS_FILL,
                       edgecolors='black' if beats else NS_EDGE,
                       linewidths=0.5, zorder=3.5)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.grid(axis='x', linestyle='--', alpha=0.5, zorder=1)
        style_spines(ax, drop_top_right=True)
        ax.tick_params(axis='both', which='major', labelsize=9.5)

    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([ARM_LABELS[a] for a in LADDER[::-1]], fontsize=9.5)
    axes[0].set_ylim(-0.7, len(LADDER) - 0.3)

    handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='white',
               markeredgecolor='black', markersize=6, label='null'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='black',
               markeredgecolor='black', markersize=6, label='real'),
        Line2D([0], [0], color=NS_EDGE, lw=3, alpha=0.35, label='fails to beat its null'),
    ]
    # Below the panels rather than inside: in the fidelity panel the nopca_fisher null
    # reaches 0.56, so any in-axes corner collides with the data.
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.10),
               ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout()
    save_fig(fig, 'ladder_null_calibrated')


# ---------------------------------------------------------------------------------------
# Figure 2 -- the mechanism. Hubness contamination and the null's block-fittability rise and
# fall together along the ladder, which is what identifies the confound.
# ---------------------------------------------------------------------------------------
def fig_mechanism():
    fig, axes = plt.subplots(1, 2, figsize=(6.4 * 0.88, 2.5 * 0.88), dpi=300)
    ladder = LADDER[1:]                      # the ladder proper; legacy is not a rung
    x = np.arange(len(ladder))

    specs = [
        (axes[0], 'triviality_ami_hubness', 'real',
         'AMI with hubness', 'Parcellation driven by\nconnection strength'),
        (axes[1], 'fidelity_within_insample', 'null',
         'Null fidelity (R$^2$)', 'Structure a block model\nfinds in pure noise'),
    ]
    for ax, metric, tree, ylabel, title in specs:
        ys = [mean(metric, tree, a) for a in ladder]
        ax.plot(x, ys, color='0.45', lw=1.2, zorder=2)
        for xi, a, y in zip(x, ladder, ys):
            for v in vals(metric, tree, a):
                ax.scatter(xi, v, s=10, color=ARM_COLORS[a], alpha=0.30,
                           edgecolors='none', zorder=2.5)
            ax.scatter(xi, y, s=55, color=ARM_COLORS[a], edgecolors='black',
                       linewidths=0.6, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([ARM_LABELS[a] for a in ladder], fontsize=8,
                           rotation=30, ha='right')
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=8.5, fontweight='bold')
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
        style_spines(ax, drop_top_right=True)
        ax.tick_params(axis='both', which='major', labelsize=9)

    fig.tight_layout()
    save_fig(fig, 'ladder_mechanism')


# ---------------------------------------------------------------------------------------
# Figure 3 -- held-out sits on top of in-sample, so the partitions do not overfit at all and
# the whole gap to the uncompressed reference is the block-model form.
# ---------------------------------------------------------------------------------------
def fig_ceiling():
    fig, ax = plt.subplots(figsize=(5.4 * 0.82, 2.9 * 0.82), dpi=300)
    ypos = np.arange(len(LADDER))[::-1]
    ceiling = mean('fidelity_within', 'real', '(ceiling)')

    # The two estimates coincide to two decimals -- which IS the finding -- so a dumbbell
    # degenerates to a single dot and reads as missing data. Offsetting them vertically
    # keeps both visible and makes the coincidence the thing you see.
    dy = 0.16
    for i, arm in enumerate(LADDER[::-1]):
        y, c = ypos[i], ARM_COLORS[arm]
        held, ins = mean('fidelity_within', 'real', arm), \
            mean('fidelity_within_insample', 'real', arm)
        ax.vlines(np.mean([held, ins]), y - dy, y + dy, color=c, alpha=0.45,
                  linewidth=1.2, zorder=2)
        ax.scatter(ins, y + dy, s=38, facecolors='white', edgecolors=c,
                   linewidths=1.2, zorder=3)
        ax.scatter(held, y - dy, s=38, color=c, edgecolors='black',
                   linewidths=0.5, zorder=3.5)

    ax.axvline(ceiling, color='crimson', linestyle='--', lw=1.4, alpha=0.9, zorder=2)
    ax.text(ceiling - 0.025, len(LADDER) - 0.55,
            'uncompressed\nreference %.2f' % ceiling, fontsize=8, color='crimson',
            ha='right', va='top')
    # The emptiness between the arms and the reference is the message: everything the
    # partitions achieve sits in the leftmost sixth of what is predictable.
    ax.annotate('', xy=(ceiling - 0.01, -0.45), xytext=(0.18, -0.45),
                arrowprops=dict(arrowstyle='<->', color='0.5', lw=1.0))
    ax.text((0.18 + ceiling) / 2, -0.30, 'unexplained by any 50-block partition',
            fontsize=7.5, color='0.35', ha='center', va='bottom')

    ax.set_yticks(ypos)
    ax.set_yticklabels([ARM_LABELS[a] for a in LADDER[::-1]], fontsize=9.5)
    ax.set_xlabel('Fidelity (R$^2$), prose domains', fontsize=11)
    ax.set_xlim(0, 1.04)
    ax.set_ylim(-0.8, len(LADDER) - 0.3)
    ax.grid(axis='x', linestyle='--', alpha=0.5, zorder=1)
    style_spines(ax, drop_top_right=True)
    ax.tick_params(axis='both', which='major', labelsize=9.5)

    handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='white',
               markeredgecolor='black', markersize=6, label='in-sample'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='black',
               markeredgecolor='black', markersize=6, label='held-out'),
    ]
    ax.legend(handles=handles, loc='upper right', frameon=False, fontsize=8.5,
              bbox_to_anchor=(1.0, 0.78))
    ax.text(0.245, 0.30, 'held-out = in-sample:\nno overfitting', transform=ax.transAxes,
            fontsize=8, ha='left',
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))
    fig.tight_layout()
    save_fig(fig, 'fidelity_vs_ceiling')


if __name__ == '__main__':
    os.makedirs('plots', exist_ok=True)
    fig_ladder()
    fig_mechanism()
    fig_ceiling()
    print('wrote plots/{ladder_null_calibrated,ladder_mechanism,fidelity_vs_ceiling}.{svg,png}')
