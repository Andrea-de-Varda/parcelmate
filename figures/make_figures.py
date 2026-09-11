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


# Which experiment to plot. `scores.csv` is the unnormalized run (job 17366628, LOG.md
# Iteration 8); `scores_norm.csv` is the surrogate-normalized one (jobs 17368677/816/817,
# Iteration 11), where the null is an actual noise floor. Pass a filename to switch.
SCORES = sys.argv[1] if len(sys.argv) > 1 else 'scores_norm.csv'
SUFFIX = '' if SCORES == 'scores_norm.csv' else '_unnormalized'


def load():
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, SCORES)))]
    for r in rows:
        r['value'] = float(r['value'])
    return rows


ROWS = load()


def vals(metric, tree, variant, domains=PROSE):
    """Every per-domain measurement for one (metric, tree, arm).

    Both `fit` and `eval` are restricted to `domains`. For the within-domain metrics the two
    columns are equal so the eval filter is a no-op; for the across-domain metrics it is
    what keeps the 12 ordered prose pairs and drops any pair touching whitespace,
    codeparrot or random.
    """
    return [r['value'] for r in ROWS
            if r['metric'] == metric and r['tree'] == tree
            and r['variant'] == variant and r['fit'] in domains and r['eval'] in domains]


def mean(metric, tree, variant, domains=PROSE):
    v = vals(metric, tree, variant, domains)
    return float(np.mean(v)) if v else float('nan')


def paired_deltas(metric, variant, domains=PROSE):
    """Real minus null, matched domain by domain (or domain pair by domain pair).

    The null is computed on the same tokens as the real run, so the comparison is paired and
    the per-domain differences are the unit of evidence, not the difference of the means.
    """
    real = {(r['fit'], r['eval']): r['value'] for r in ROWS
            if r['metric'] == metric and r['tree'] == 'real' and r['variant'] == variant
            and r['fit'] in domains and r['eval'] in domains}
    null = {(r['fit'], r['eval']): r['value'] for r in ROWS
            if r['metric'] == metric and r['tree'] == 'null' and r['variant'] == variant
            and r['fit'] in domains and r['eval'] in domains}
    keys = sorted(set(real) & set(null))
    return np.array([real[k] - null[k] for k in keys])


def beats_null(metric, variant, domains=PROSE):
    """Colour is earned only by a UNANIMOUS win: the arm must beat its null in every domain.

    Comparing the two means is not enough. With four prose domains no rank test can reach
    p < 0.05, so the defensible rule available at this n is the sign test: 4/4 is p = 0.0625
    one-sided, 12/12 is p = 0.00024. It also happens to separate the data cleanly -- every
    real win here is unanimous, while `vmf_profile`'s across-domain fidelity advantage of
    +0.007 holds in only 5 of 12 pairs, which is a coin flip dressed as an effect and would
    otherwise have been coloured as a win.
    """
    d = paired_deltas(metric, variant, domains)
    return bool(len(d)) and bool((d > 0).all())


# ---------------------------------------------------------------------------------------
# Figure 1 -- the main result. Dumbbell: the delta between null and real IS the finding, and
# only one arm has a positive delta on both metrics.
# ---------------------------------------------------------------------------------------
def fig_ladder():
    """The main result, 2x2: {within, across} domain x {reliability, fidelity}.

    Fidelity is Pearson r in BOTH rows rather than R^2 in the top one, so the two rows are in
    the same units and the drop from within to across is readable directly. R^2 is the right
    summary within a domain (the two halves share a scale) but is not defined across domains,
    where the two connectomes differ in scale; figure 3 keeps R^2 for the ceiling comparison.

    Caveat carried in the row label: the two rows do not use the same amount of data. A
    within-domain row compares two half-domain connectomes (~197k tokens each); an
    across-domain row compares two sample-averaged connectomes (~394k tokens each). The
    mismatch favours the across row -- more tokens per side, less estimation noise -- so the
    collapse seen there cannot be blamed on noisier inputs.
    """
    fig, axes = plt.subplots(2, 2, figsize=(6.8 * 0.88, 5.1 * 0.88), dpi=300,
                             sharey=True, sharex='col')
    ypos = np.arange(len(LADDER))[::-1]

    panels = [
        (axes[0, 0], 'reliability_within', 'Reliability (ARI)',
         'within domain\n(split halves)'),
        (axes[0, 1], 'fidelity_within_r', 'Fidelity (r)', None),
        # The halves variant, not the `avg` one: it fits on half A of one domain and
        # evaluates on half B of another, the same token budget and the same disjoint-set
        # structure as the within-domain row above, so the two rows are comparable.
        (axes[1, 0], 'reliability_across_halves', 'Reliability (ARI)',
         'across domains\n(12 prose pairs)'),
        (axes[1, 1], 'fidelity_across_halves', 'Fidelity (r)', None),
    ]

    for ax, metric, coltitle, rowlabel in panels:
        if not any(r['metric'] == metric for r in ROWS):
            metric = metric.replace('_halves', '')   # pre-Iteration-11 scores.csv
        for i, arm in enumerate(LADDER[::-1]):
            y = ypos[i]
            real, null = mean(metric, 'real', arm), mean(metric, 'null', arm)
            beats = beats_null(metric, arm)
            # Gray when the arm fails to beat its own null in every domain -- the house
            # convention for "not significant". See beats_null: the mean difference alone
            # would paint a 5-of-12 coin flip as a win.
            c = ARM_COLORS[arm] if beats else NS_EDGE
            ax.hlines(y, min(real, null), max(real, null),
                      color=c, alpha=0.35 if beats else 0.25, linewidth=3, zorder=2)
            # Faint per-measurement real values (per domain above, per domain pair below),
            # so the reader sees the spread behind the mean.
            for v in vals(metric, 'real', arm):
                ax.scatter(v, y, s=9, color=c, alpha=0.30, edgecolors='none', zorder=2.5)
            ax.scatter(null, y, s=42, facecolors='white', edgecolors=c,
                       linewidths=1.2, zorder=3)
            ax.scatter(real, y, s=42, color=c if beats else NS_FILL,
                       edgecolors='black' if beats else NS_EDGE,
                       linewidths=0.5, zorder=3.5)
        # An all-grey panel is the actual finding in the bottom row, but an all-grey panel
        # also looks like a rendering failure. The count says which it is.
        n_win = sum(beats_null(metric, a) for a in LADDER)
        ax.text(0.97, 0.04, '%d/%d arms beat their null' % (n_win, len(LADDER)),
                transform=ax.transAxes, fontsize=7.5, color='0.35',
                ha='right', va='bottom')
        ax.grid(axis='x', linestyle='--', alpha=0.5, zorder=1)
        style_spines(ax, drop_top_right=True)
        ax.tick_params(axis='both', which='major', labelsize=9.5)
        if rowlabel is not None:
            ax.set_ylabel(rowlabel, fontsize=9.5, fontweight='bold', labelpad=8)

    # Column identity goes on the top row only; repeating it as an xlabel underneath was
    # pure duplication. Both columns start at zero so the bars are not visually inflated.
    for ax, t in zip(axes[0], ('Reliability (ARI)', 'Fidelity (r)')):
        ax.set_title(t, fontsize=11)
    for ax in axes[1]:
        ax.set_xlim(left=0)

    for ax in axes[:, 0]:
        ax.set_yticks(ypos)
        ax.set_yticklabels([ARM_LABELS[a] for a in LADDER[::-1]], fontsize=9.5)
        ax.set_ylim(-0.7, len(LADDER) - 0.3)

    handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='white',
               markeredgecolor='black', markersize=6, label='null (circular shift)'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='black',
               markeredgecolor='black', markersize=6, label='real'),
        Line2D([0], [0], color=NS_EDGE, lw=3, alpha=0.35, label='does not beat its null in every domain'),
    ]
    # Below the panels rather than inside: the nulls run high enough in both fidelity
    # panels that any in-axes corner collides with the data.
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.055),
               ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout()
    save_fig(fig, 'ladder_null_calibrated' + SUFFIX)


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
    save_fig(fig, 'ladder_mechanism' + SUFFIX)


# ---------------------------------------------------------------------------------------
# Figure 2b (normalized run) -- why two arms keep a high null reliability even though their
# null fidelity is zero. It is cluster collapse: reliability rewards a degenerate partition,
# which is precisely why it was never specified to be read alone.
# ---------------------------------------------------------------------------------------
def fig_degeneracy():
    fig, axes = plt.subplots(1, 2, figsize=(6.6 * 0.88, 2.7 * 0.88), dpi=300)

    ax = axes[0]
    pts = {a: (mean('triviality_n_effective_networks', 'null', a),
               mean('reliability_within', 'null', a)) for a in LADDER}
    for arm in LADDER:
        x, y = pts[arm]
        ax.scatter(x, y, s=70, color=ARM_COLORS[arm], edgecolors='black',
                   linewidths=0.6, zorder=3)
    # The three PCA arms land on the same point (50 networks filled, ARI 0.003), so one
    # label per arm produces three overlapping strings. Group coincident points and label
    # the group once.
    groups = []
    for arm in LADDER:
        x, y = pts[arm]
        for g in groups:
            gx, gy = pts[g[0]]
            if abs(x - gx) < 3 and abs(y - gy) < 0.05:
                g.append(arm)
                break
        else:
            groups.append([arm])
    for g in groups:
        x, y = pts[g[0]]
        label = '\n'.join(ARM_LABELS[a] for a in g)
        ax.annotate(label, (x, y), textcoords='offset points',
                    xytext=(0, -12 - 9 * (len(g) - 1) if y > 0.25 else 9),
                    fontsize=7.5, ha='center', va='top' if y > 0.25 else 'bottom',
                    color='0.25', linespacing=1.25)
    ax.set_xlabel('Networks actually filled, null', fontsize=10)
    ax.set_ylabel('Reliability (ARI), null', fontsize=10)
    ax.set_title('A degenerate partition\nreproduces itself', fontsize=8.5,
                 fontweight='bold')
    ax.set_xlim(0, 55)
    ax.set_ylim(-0.12, 0.68)

    # The companion panel: fidelity is not fooled by the same partitions.
    ax = axes[1]
    ypos = np.arange(len(LADDER))[::-1]
    for i, arm in enumerate(LADDER[::-1]):
        y, c = ypos[i], ARM_COLORS[arm]
        rel = mean('reliability_within', 'null', arm)
        fid = mean('fidelity_within', 'null', arm)
        ax.hlines(y, min(rel, fid), max(rel, fid), color=c, alpha=0.35, linewidth=3,
                  zorder=2)
        ax.scatter(rel, y, s=42, color=c, edgecolors='black', linewidths=0.5, zorder=3)
        ax.scatter(fid, y, s=42, facecolors='white', edgecolors=c, linewidths=1.2,
                   zorder=3.5)
    ax.axvline(0, color='0.5', lw=1.0, zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([ARM_LABELS[a] for a in LADDER[::-1]], fontsize=9)
    ax.set_xlabel('Null score', fontsize=10)
    ax.set_title('Fidelity is not fooled\nby the same partitions', fontsize=8.5,
                 fontweight='bold')
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='none', markerfacecolor='black',
               markeredgecolor='black', markersize=6, label='reliability'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='white',
               markeredgecolor='black', markersize=6, label='fidelity'),
    ], loc='center right', frameon=False, fontsize=8)

    for ax in axes:
        ax.grid(axis='y' if ax is axes[0] else 'x', linestyle='--', alpha=0.5, zorder=1)
        style_spines(ax, drop_top_right=True)
        ax.tick_params(axis='both', which='major', labelsize=9)
    fig.tight_layout()
    save_fig(fig, 'null_degeneracy')


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
    save_fig(fig, 'fidelity_vs_ceiling' + SUFFIX)


if __name__ == '__main__':
    os.makedirs('plots', exist_ok=True)
    fig_ladder()
    fig_ceiling()
    if SUFFIX:
        # The mechanism figure diagnoses the |r| artifact, so it belongs to the
        # unnormalized run. On the normalized one the null fidelity it plots is ~0 for
        # every arm and the panel says nothing.
        fig_mechanism()
    else:
        fig_degeneracy()
    print('wrote plots/*%s.{svg,png} from figures/%s' % (SUFFIX, SCORES))
