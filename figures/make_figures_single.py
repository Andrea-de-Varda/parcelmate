"""Single-run figures, superseded by make_figures.py (LOG.md Iteration 17).

Kept because it regenerates the Iteration 8/11/13 figures from one score file; the current
figures combine every run and live in make_figures.py.

Original docstring follows.

Publication figures for the parcelmate method ladder.

    python figures/make_figures.py                  -> plots/*.svg + *.png
    python figures/make_figures.py scores_norm.csv  -> the same figures for an older run

Reads figures/scores_ladder.csv by default (results/ladder/metrics/scores.csv on the
cluster, jobs 17379136 + 17379137, LOG.md Iteration 12). Every number plotted is a
per-domain measurement averaged over the four prose domains; whitespace, codeparrot and
random are excluded, because a block model fits their degenerate connectivity trivially
well and they dominate an equal-weighted mean.

THE REFERENCE IS `pnull`, NOT `null`. `null` is the whole pipeline run on circularly
shifted data and scored on shifted data: it sits on a different denominator from the real
measurement, which is how a structureless matrix once out-scored real data (Iteration 9).
`pnull` is the null PARTITION evaluated on the REAL data -- same target, same denominator --
so real - pnull is the credit the clustering earns beyond a partition carrying only
per-unit properties. `rand` is a random partition of the same k and comes out at 0.000
everywhere, so it is annotated rather than plotted.

Four figures:
  1. ladder_references  -- the main result: real vs pnull, within and across domains
  2. within_vs_across   -- the ordering flips; which arm actually generalizes
  3. reference_check    -- the reference itself: collapse makes it stronger, and how much
                           of each arm's score per-unit properties already recover
  4. fidelity_vs_ceiling -- held-out == in-sample, so the model class is the binding limit
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

# One colour per arm, fixed across every figure. The rungs are a light-to-dark ramp so the
# progression reads off the page; `legacy` is gray because it is the inherited pre-bugfix
# baseline rather than a rung.
ARM_COLORS = {
    'legacy':       '#999999',
    'current':      '#a6cee3',
    'fisher_pca':   '#6baed6',
    'nopca_fisher': '#3182bd',
    'vmf_profile':  '#08519c',
    'vmf_z':        '#08306b',
}
ARM_LABELS = {
    'legacy': 'legacy',
    'current': 'current',
    'fisher_pca': '+ magnitudes',
    'nopca_fisher': '+ no PCA',
    'vmf_profile': '+ standardised',
    'vmf_z': '+ effect size',
}
LADDER = ['legacy', 'current', 'fisher_pca', 'nopca_fisher', 'vmf_profile', 'vmf_z']
NS_FILL, NS_EDGE = '#cccccc', '#999999'

SCORES = sys.argv[1] if len(sys.argv) > 1 else 'scores_ladder.csv'
SUFFIX = '' if SCORES == 'scores_ladder.csv' else '_' + SCORES.split('.')[0]


def load():
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, SCORES)))]
    for r in rows:
        r['value'] = float(r['value'])
    return rows


ROWS = load()
ARMS = [a for a in LADDER if any(r['variant'] == a for r in ROWS)]
# Older runs predate the partition-level references and only carry the tree-level null.
REF = 'pnull' if any(r['tree'] == 'pnull' for r in ROWS) else 'null'
REF_LABEL = ('null partition, real data' if REF == 'pnull'
             else 'pipeline on shifted data')


def vals(metric, tree, variant, domains=PROSE):
    """Per-domain measurements. Both `fit` and `eval` are restricted: for the within-domain
    metrics the two are equal so the eval filter is a no-op, and for the across-domain ones
    it is what keeps the 12 ordered prose pairs."""
    return [r['value'] for r in ROWS
            if r['metric'] == metric and r['tree'] == tree and r['variant'] == variant
            and r['fit'] in domains and r['eval'] in domains]


def mean(metric, tree, variant, domains=PROSE):
    v = vals(metric, tree, variant, domains)
    return float(np.mean(v)) if v else float('nan')


def paired_deltas(metric, variant, tree=None, domains=PROSE):
    """Real minus reference, matched domain by domain (or pair by pair).

    The reference is computed on the same data as the real measurement, so the comparison
    is paired and the per-domain differences are the unit of evidence, not the difference
    of the means.
    """
    def keyed(t):
        return {(r['fit'], r['eval']): r['value'] for r in ROWS
                if r['metric'] == metric and r['tree'] == t and r['variant'] == variant
                and r['fit'] in domains and r['eval'] in domains}
    real, ref = keyed('real'), keyed(tree or REF)
    keys = sorted(set(real) & set(ref))
    return np.array([real[k] - ref[k] for k in keys])


def beats_ref(metric, variant, tree=None, domains=PROSE):
    """Colour is earned by a UNANIMOUS win: the arm must beat its reference in every domain.

    Comparing the two means is not enough -- with four prose domains no rank test reaches
    p < 0.05, so the defensible rule at this n is the sign test (4/4 is p = 0.0625
    one-sided, 12/12 is p = 0.00024), and it also stops a coin-flip advantage in the mean
    from being painted as a win.
    """
    d = paired_deltas(metric, variant, tree, domains)
    return bool(len(d)) and bool((d > 0).all())


def _dumbbell(ax, metric, arms, ypos):
    for i, arm in enumerate(arms[::-1]):
        y = ypos[i]
        real, ref = mean(metric, 'real', arm), mean(metric, REF, arm)
        beats = beats_ref(metric, arm)
        c = ARM_COLORS[arm] if beats else NS_EDGE
        ax.hlines(y, min(real, ref), max(real, ref),
                  color=c, alpha=0.35 if beats else 0.25, linewidth=3, zorder=2)
        for v in vals(metric, 'real', arm):
            ax.scatter(v, y, s=9, color=c, alpha=0.30, edgecolors='none', zorder=2.5)
        ax.scatter(ref, y, s=42, facecolors='white', edgecolors=c, linewidths=1.2, zorder=3)
        ax.scatter(real, y, s=42, color=c if beats else NS_FILL,
                   edgecolors='black' if beats else NS_EDGE, linewidths=0.5, zorder=3.5)


# ---------------------------------------------------------------------------------------
# Figure 1 -- the main result. Every arm clears the reference on both metrics; the gap IS
# the finding, and the rungs above `current` open it much wider than the rungs below.
# ---------------------------------------------------------------------------------------
def fig_references():
    fig, axes = plt.subplots(2, 2, figsize=(6.9 * 0.88, 5.4 * 0.88), dpi=300,
                             sharey=True, sharex='col')
    ypos = np.arange(len(ARMS))[::-1]
    panels = [
        (axes[0, 0], 'reliability_within', 'within domain\n(split halves)'),
        (axes[0, 1], 'fidelity_within_r', None),
        (axes[1, 0], 'reliability_across_halves', 'across domains\n(12 prose pairs)'),
        (axes[1, 1], 'fidelity_across_halves', None),
    ]
    for ax, metric, rowlabel in panels:
        if not any(r['metric'] == metric for r in ROWS):
            metric = metric.replace('_halves', '')       # pre-Iteration-11 scores
        _dumbbell(ax, metric, ARMS, ypos)
        n_win = sum(beats_ref(metric, a) for a in ARMS)
        # Top-right, not bottom-right: the bottom row of every panel is `legacy`, whose
        # dumbbell runs well to the right and collides with a bottom-anchored note.
        ax.text(0.98, 0.97, '%d/%d arms beat the reference' % (n_win, len(ARMS)),
                transform=ax.transAxes, fontsize=7.5, color='0.35', ha='right', va='top')
        ax.grid(axis='x', linestyle='--', alpha=0.5, zorder=1)
        style_spines(ax, drop_top_right=True)
        ax.tick_params(axis='both', which='major', labelsize=9.5)
        if rowlabel is not None:
            ax.set_ylabel(rowlabel, fontsize=9.5, fontweight='bold', labelpad=8)

    for ax, t in zip(axes[0], ('Reliability (ARI)', 'Fidelity (r)')):
        ax.set_title(t, fontsize=11)
    for ax in axes[1]:
        ax.set_xlim(left=0)
    for ax in axes[:, 0]:
        ax.set_yticks(ypos)
        ax.set_yticklabels([ARM_LABELS[a] for a in ARMS[::-1]], fontsize=9.5)
        ax.set_ylim(-0.7, len(ARMS) - 0.3)

    handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='white',
               markeredgecolor='black', markersize=6, label=REF_LABEL),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='black',
               markeredgecolor='black', markersize=6, label='real'),
        Line2D([0], [0], color=NS_EDGE, lw=3, alpha=0.35,
               label='does not beat it in every domain'),
    ]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.055),
               ncol=3, frameon=False, fontsize=8.5)
    if REF == 'pnull':
        fig.text(0.5, -0.085, 'a random partition of the same k scores 0.00 on every '
                 'metric and every arm', ha='center', fontsize=7.5, color='0.4')
    fig.tight_layout()
    save_fig(fig, 'ladder_references' + SUFFIX)


# ---------------------------------------------------------------------------------------
# Figure 2 -- the ordering is not the same question answered twice. Within domain the
# no-PCA arm wins; across domains it is the worst of the six and the standardised arm wins.
# ---------------------------------------------------------------------------------------
def fig_within_vs_across():
    if not any(r['metric'] == 'fidelity_across_halves' for r in ROWS):
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.6 * 0.88, 2.9 * 0.88), dpi=300)
    specs = [(axes[0], 'fidelity_within_r', 'fidelity_across_halves', 'Fidelity (r)'),
             (axes[1], 'reliability_within', 'reliability_across_halves',
              'Reliability (ARI)')]
    for ax, m_in, m_ac, title in specs:
        for arm in ARMS:
            x = mean(m_in, 'real', arm) - mean(m_in, REF, arm)
            y = mean(m_ac, 'real', arm) - mean(m_ac, REF, arm)
            ax.scatter(x, y, s=75, color=ARM_COLORS[arm], edgecolors='black',
                       linewidths=0.6, zorder=3)
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0], 0)
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color='0.6', lw=1.0, ls='--', zorder=1)
        ax.set_xlabel('within domain', fontsize=10)
        ax.set_ylabel('across domains', fontsize=10)
        ax.set_title(title + ', real $-$ reference', fontsize=8.5, fontweight='bold')
        # Everything sits below y = x: no arm transfers as well as it fits.
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        style_spines(ax, drop_top_right=True)
        ax.grid(linestyle='--', alpha=0.4, zorder=0)
        ax.tick_params(axis='both', which='major', labelsize=9)
    # A shared legend rather than per-point labels: with six points in a small panel the
    # labels land on top of neighbouring dots and read as if they belonged to them.
    fig.legend(handles=[Line2D([0], [0], marker='o', color='none',
                               markerfacecolor=ARM_COLORS[a], markeredgecolor='black',
                               markersize=6, label=ARM_LABELS[a]) for a in ARMS],
               loc='lower center', bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False,
               fontsize=8)
    fig.tight_layout()
    save_fig(fig, 'within_vs_across' + SUFFIX)


# ---------------------------------------------------------------------------------------
# Figure 3 -- two things about the reference itself. Left: a reference partition that
# collapses is a STRONGER reference, not a weaker one, so real - pnull is conservative
# exactly where the capacity worry said it would be inflated. Right: the pnull-to-rand gap
# is the share of each arm's performance that a partition knowing only per-unit properties
# already recovers.
# ---------------------------------------------------------------------------------------
def fig_reference_check():
    if REF != 'pnull' or not any(r['tree'] == 'rand' for r in ROWS):
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.8 * 0.88, 3.0 * 0.88), dpi=300)

    ax = axes[0]
    xs = [mean('triviality_n_effective_networks', 'pnull', a) for a in ARMS]
    ys = [mean('fidelity_within_r', 'pnull', a) for a in ARMS]
    for a, x, y in zip(ARMS, xs, ys):
        ax.scatter(x, y, s=75, color=ARM_COLORS[a], edgecolors='black', linewidths=0.6,
                   zorder=3)
    r = float(np.corrcoef(xs, ys)[0, 1])
    ax.set_xlabel('networks the reference partition fills', fontsize=10)
    ax.set_ylabel('reference fidelity (r)', fontsize=10)
    ax.set_title('A collapsed reference explains\nMORE, not less', fontsize=8.5,
                 fontweight='bold')
    ax.set_xlim(0, 55)
    ax.set_ylim(0, max(ys) * 1.18)        # headroom: the top dot was clipped by autoscale
    ax.text(0.04, 0.06, 'r = %+.2f' % r, transform=ax.transAxes, fontsize=8.5,
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

    ax = axes[1]
    ypos = np.arange(len(ARMS))[::-1]
    for i, arm in enumerate(ARMS[::-1]):
        y, c = ypos[i], ARM_COLORS[arm]
        real = mean('fidelity_within_r', 'real', arm)
        d_p = real - mean('fidelity_within_r', 'pnull', arm)
        d_r = real - mean('fidelity_within_r', 'rand', arm)
        ax.hlines(y, d_p, d_r, color=c, alpha=0.35, linewidth=3, zorder=2)
        ax.scatter(d_r, y, s=42, facecolors='white', edgecolors=c, linewidths=1.2, zorder=3)
        ax.scatter(d_p, y, s=42, color=c, edgecolors='black', linewidths=0.5, zorder=3.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels([ARM_LABELS[a] for a in ARMS[::-1]], fontsize=9)
    ax.set_xlabel('real $-$ reference, fidelity (r)', fontsize=10)
    ax.set_title('Gap = the share already recovered\nby per-unit properties', fontsize=8.5,
                 fontweight='bold')
    for ax, axis in zip(axes, ('y', 'x')):
        ax.grid(axis=axis, linestyle='--', alpha=0.5, zorder=1)
        style_spines(ax, drop_top_right=True)
        ax.tick_params(axis='both', which='major', labelsize=9)
    # Below the figure: six rows of dumbbells leave no in-axes corner that does not sit on
    # either the top or the bottom row.
    fig.legend(handles=[
        Line2D([0], [0], marker='o', color='none', markerfacecolor='black',
               markeredgecolor='black', markersize=6, label='vs null partition'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='white',
               markeredgecolor='black', markersize=6, label='vs random partition'),
    ], loc='lower center', bbox_to_anchor=(0.75, -0.09), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    save_fig(fig, 'reference_check' + SUFFIX)


# ---------------------------------------------------------------------------------------
# Figure 4 -- held-out sits on top of in-sample, so the partitions do not overfit at all
# and the whole gap to the uncompressed reference is the block-model form.
# ---------------------------------------------------------------------------------------
def fig_ceiling():
    fig, ax = plt.subplots(figsize=(5.6 * 0.82, 3.1 * 0.82), dpi=300)
    ypos = np.arange(len(ARMS))[::-1]
    ceiling = mean('fidelity_within', 'real', '(ceiling)')

    # The two estimates coincide to two decimals -- which IS the finding -- so a dumbbell
    # degenerates to a single dot and reads as missing data. Offsetting them vertically
    # keeps both visible and makes the coincidence the thing you see.
    dy = 0.16
    for i, arm in enumerate(ARMS[::-1]):
        y, c = ypos[i], ARM_COLORS[arm]
        held = mean('fidelity_within', 'real', arm)
        ins = mean('fidelity_within_insample', 'real', arm)
        ax.vlines(np.mean([held, ins]), y - dy, y + dy, color=c, alpha=0.45,
                  linewidth=1.2, zorder=2)
        ax.scatter(ins, y + dy, s=38, facecolors='white', edgecolors=c,
                   linewidths=1.2, zorder=3)
        ax.scatter(held, y - dy, s=38, color=c, edgecolors='black',
                   linewidths=0.5, zorder=3.5)

    ax.axvline(ceiling, color='crimson', linestyle='--', lw=1.4, alpha=0.9, zorder=2)
    ax.text(ceiling - 0.025, len(ARMS) - 0.55,
            'uncompressed\nreference %.2f' % ceiling, fontsize=8, color='crimson',
            ha='right', va='top')
    ax.annotate('', xy=(ceiling - 0.01, -0.5), xytext=(0.22, -0.5),
                arrowprops=dict(arrowstyle='<->', color='0.5', lw=1.0))
    ax.text((0.22 + ceiling) / 2, -0.35, 'unexplained by any 50-block partition',
            fontsize=7.5, color='0.35', ha='center', va='bottom')

    ax.set_yticks(ypos)
    ax.set_yticklabels([ARM_LABELS[a] for a in ARMS[::-1]], fontsize=9.5)
    ax.set_xlabel('Fidelity (R$^2$), prose domains', fontsize=11)
    ax.set_xlim(0, 1.04)
    ax.set_ylim(-0.85, len(ARMS) - 0.3)
    ax.grid(axis='x', linestyle='--', alpha=0.5, zorder=1)
    style_spines(ax, drop_top_right=True)
    ax.tick_params(axis='both', which='major', labelsize=9.5)
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='none', markerfacecolor='white',
               markeredgecolor='black', markersize=6, label='in-sample'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='black',
               markeredgecolor='black', markersize=6, label='held-out'),
    ], loc='upper right', frameon=False, fontsize=8.5, bbox_to_anchor=(1.0, 0.80))
    ax.text(0.26, 0.32, 'held-out = in-sample:\nno overfitting', transform=ax.transAxes,
            fontsize=8, ha='left',
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))
    fig.tight_layout()
    save_fig(fig, 'fidelity_vs_ceiling' + SUFFIX)


if __name__ == '__main__':
    os.makedirs('plots', exist_ok=True)
    fig_references()
    fig_within_vs_across()
    fig_reference_check()
    fig_ceiling()
    print('wrote plots/*%s.{svg,png} from figures/%s (reference: %s)'
          % (SUFFIX, SCORES, REF))
