"""Publication figures across every arm of every run (LOG.md Iterations 13-18).

    python figures/make_figures.py      ->  plots/*.svg + *.png

Combines six score files into one comparison of 41 arms:

    scores_ladder.csv       the six-arm ladder, MiniBatch k-means, residual stream (Iter. 13)
    scores_yolo.csv         converged optimizer, Ward, block-model objective, k = 100 (Iter. 16)
    scores_yolo4.csv        spatial ICA and the sparsification rungs (Iter. 16)
    scores_yolo_mlp.csv     the same pipeline on MLP neurons instead of the residual stream
    scores_final_resid.csv  residual Ward at k = 150, 200 (final test T1, Iter. 18)
    scores_final_mlp.csv    final tests T1, T2, T5 on MLP neurons (Iter. 18)

Every number is a per-domain measurement averaged over the four prose domains; whitespace,
codeparrot and random are excluded because a block model fits their degenerate connectivity
trivially well and they would dominate an equal-weighted mean.

THE REFERENCE IS `pnull`: the null PARTITION (fit on circularly shifted data) evaluated on
the REAL data, so it sits on the same target and the same denominator as the real
measurement and `real - pnull` is the credit the clustering earns beyond a partition that
knows only per-unit properties. A random partition of the same k scores ~0 everywhere.

Arms are grouped by what the clustering actually sees, because that is what the results
turn on: binarized adjacency, Fisher magnitudes, standardized profiles, the block-model
objective, independent components, and the same standardized pipeline on a different unit.
Within a group the order is inherited pipeline -> converged optimizer -> k = 100.

Figures:
  1. arms_references   -- the main result: real vs the null partition, within and across
                          domains, all 24 arms on x with grouping brackets
  1b. arms_deltas      -- the same, as real minus null alone: easier to rank once figure 1
                          has shown where the null sits
  2. hubness_tradeoff  -- hubness against level (R^2), pattern (r) and transfer, with r
                          per unit type: on the residual stream hubness costs transfer,
                          on MLP neurons the block polish breaks that relation
  3. within_vs_across  -- like-for-like (r against r): no arm transfers as well as it fits
  4. maps_vs_labels    -- ICA judged on the object it produces, not only on its labels
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
NS_FILL, NS_EDGE = '#cccccc', '#999999'

SOURCES = {
    'scores_ladder.csv': '',        # residual stream, the Iteration 13 ladder
    'scores_yolo.csv': '',
    'scores_yolo4.csv': '',
    'scores_yolo_mlp.csv': 'mlp:',  # prefixed: same arm names, different units
    'scores_final_resid.csv': '',
    'scores_final_mlp.csv': 'mlp:',
}

# Groups, in plot order. One hue per group, light -> dark within it, so the family is
# readable from colour alone and the bracket names it. The order inside a group is the
# order the arms were built: inherited pipeline, converged optimizer, k = 100.
GROUPS = [
    ('Binarized', '#9ecae1', '#08306b', [
        ('legacy', 'legacy'),
        ('current', 'current'),
        ('binarize_row_lloyd', 'row thr.'),
        ('binarize_row_lloyd100', 'row thr. k100'),
        ('binarize_global_lloyd', 'global thr.'),
        ('binarize_global_lloyd100', 'global thr. k100'),
    ]),
    ('Fisher magnitudes', '#a1d99b', '#00441b', [
        ('fisher_pca', 'PCA'),
        ('nopca_fisher', 'no PCA'),
        ('fisher_pca_lloyd', 'PCA, Lloyd'),
        ('nopca_lloyd', 'no PCA, Lloyd'),
        ('sparse_fisher_lloyd', 'sparse'),
        ('sparse_fisher_lloyd100', 'sparse k100'),
    ]),
    ('Standardized', '#fdbe85', '#7f2704', [
        ('vmf_profile', 'MiniBatch'),
        ('vmf_z', 'effect size'),
        ('vmf_lloyd', 'Lloyd'),
        ('vmf_ward', 'Ward'),
        ('vmf_lloyd100', 'Lloyd k100'),
        ('vmf_ward100', 'Ward k100'),
        ('vmf_ward150', 'Ward k150'),
        ('vmf_ward200', 'Ward k200'),
    ]),
    ('Block-model', '#fc9272', '#67000d', [
        ('blockmodel', 'k50'),
        ('blockmodel100', 'k100'),
    ]),
    ('Spatial ICA', '#bcbddc', '#3f007d', [
        ('ica', 'k50'),
        ('ica100', 'k100'),
    ]),
    # MLP neurons (prefix `mlp:`), everything on standardized profiles.
    ('MLP standardized', '#c7c7c7', '#252525', [
        ('mlp:vmf_profile', 'MiniBatch'),
        ('mlp:vmf_lloyd', 'Lloyd'),
        ('mlp:vmf_ward', 'Ward'),
        ('mlp:vmf_lloyd100', 'Lloyd k100'),
        ('mlp:vmf_ward100', 'Ward k100'),
        ('mlp:vmf_ward150', 'Ward k150'),
        ('mlp:vmf_ward200', 'Ward k200'),
        ('mlp:vmf_wardlloyd100', 'Ward-init Lloyd k100'),
    ]),
    ('MLP denoised', '#a6dcd6', '#01665e', [
        ('mlp:vmf_pca20_ward100', 'PCA20 Ward'),
        ('mlp:vmf_pca20_lloyd100', 'PCA20 Lloyd'),
        ('mlp:vmf_pca100_ward100', 'PCA100 Ward'),
        ('mlp:vmf_pca100_lloyd100', 'PCA100 Lloyd'),
        ('mlp:vmf_sparse_ward100', 'sparse Ward'),
        ('mlp:vmf_sparse_lloyd100', 'sparse Lloyd'),
    ]),
    ('MLP block polish', '#f1b6da', '#8e0152', [
        ('mlp:vmf_ward100_bm', 'raw'),
        ('mlp:vmf_ward100_bm_double', 'double-centred'),
        ('mlp:vmf_ward100_bm_degree', 'degree-corrected'),
    ]),
]


def load():
    rows = []
    for fname, prefix in SOURCES.items():
        path = os.path.join(HERE, fname)
        if not os.path.exists(path):
            continue
        for r in csv.DictReader(open(path)):
            r['value'] = float(r['value'])
            r['variant'] = prefix + r['variant']
            rows.append(r)
    return rows


ROWS = load()
PRESENT = {r['variant'] for r in ROWS}
# Drop any arm a score file does not carry, so the script still runs on a partial set.
GROUPS = [(g, c0, c1, [(a, lab) for a, lab in arms if a in PRESENT])
          for g, c0, c1, arms in GROUPS]
GROUPS = [g for g in GROUPS if g[3]]
ARMS = [a for _, _, _, arms in GROUPS for a, _ in arms]


def _ramp(c0, c1, n):
    """n colours from c0 to c1. A single-arm group takes the dark end."""
    if n == 1:
        return [c1]
    a, b = np.array(mpl.colors.to_rgb(c0)), np.array(mpl.colors.to_rgb(c1))
    return [mpl.colors.to_hex(a + (b - a) * i / (n - 1)) for i in range(n)]


ARM_COLORS, ARM_LABELS, ARM_GROUP = {}, {}, {}
for gname, c0, c1, arms in GROUPS:
    for (arm, label), colour in zip(arms, _ramp(c0, c1, len(arms))):
        ARM_COLORS[arm], ARM_LABELS[arm], ARM_GROUP[arm] = colour, label, gname

def is_mlp(arm):
    return arm.startswith('mlp:')


# x positions, with a gap between groups and a wider one where the unit changes.
GAP, UNIT_GAP = 1.0, 2.0
XPOS, GROUP_SPAN = {}, []
UNIT_DIVIDER = None
_x = 0.0
for gi, (gname, _, _, arms) in enumerate(GROUPS):
    if gi and is_mlp(arms[0][0]) and not is_mlp(GROUPS[gi - 1][3][0][0]):
        _x += UNIT_GAP - GAP
        UNIT_DIVIDER = _x - UNIT_GAP / 2 - 0.5
    start = _x
    for arm, _ in arms:
        XPOS[arm] = _x
        _x += 1.0
    GROUP_SPAN.append((gname, start, _x - 1.0))
    _x += GAP
XMAX = _x - GAP
# The arm axis was 29 units wide at 10.7 in for 24 arms; keep that density as arms are added.
ARMS_FIG_W = 13.0 * 0.82 * (XMAX + 1.0) / 29.0


def vals(metric, tree, variant, domains=PROSE):
    """Per-domain measurements. `fit` and `eval` are both restricted: for the within-domain
    metrics they are equal so the eval filter is a no-op, and for the across-domain ones it
    is what keeps the 12 ordered prose pairs."""
    return [r['value'] for r in ROWS
            if r['metric'] == metric and r['tree'] == tree and r['variant'] == variant
            and r['fit'] in domains and r['eval'] in domains]


def mean(metric, tree, variant, domains=PROSE):
    v = vals(metric, tree, variant, domains)
    return float(np.mean(v)) if v else float('nan')


def paired_deltas(metric, variant, tree='pnull', domains=PROSE):
    """Real minus reference, matched domain by domain (or pair by pair).

    The reference is computed on the same data as the real measurement, so the comparison
    is paired and the per-domain differences are the unit of evidence, not the difference
    of the means.
    """
    def keyed(t):
        return {(r['fit'], r['eval']): r['value'] for r in ROWS
                if r['metric'] == metric and r['tree'] == t and r['variant'] == variant
                and r['fit'] in domains and r['eval'] in domains}
    real, ref = keyed('real'), keyed(tree)
    keys = sorted(set(real) & set(ref))
    return np.array([real[k] - ref[k] for k in keys])


def delta(metric, variant):
    d = paired_deltas(metric, variant)
    return float(np.mean(d)) if len(d) else float('nan')


def beats_ref(metric, variant):
    """Colour is earned by a UNANIMOUS win: the arm must beat its reference in every domain.

    Comparing the two means is not enough -- with four prose domains no rank test reaches
    p < 0.05, so the defensible rule at this n is the sign test (4/4 is p = 0.0625
    one-sided, 12/12 is p = 0.00024), and it also stops a coin-flip advantage in the mean
    from being painted as a win.
    """
    d = paired_deltas(metric, variant)
    return bool(len(d)) and bool((d > 0).all())


def brackets(ax, y, drop, label_gap, fontsize=8.0, lw=1.0, color='0.35'):
    """Grouping brackets below the plot, one per family, in axes coordinates.

    `y` is the bracket line, in axis-height units below the axes. It has to clear the
    rotated arm labels, whose length is set by the longest of them ("global thr. k100"),
    not by the axes; `label_gap` then separates each group name from its own bracket.
    """
    for gname, x0, x1 in GROUP_SPAN:
        ax.plot([x0 - 0.35, x1 + 0.35], [y, y], color=color, lw=lw,
                transform=ax.get_xaxis_transform(), clip_on=False)
        for x in (x0 - 0.35, x1 + 0.35):
            ax.plot([x, x], [y, y + drop], color=color, lw=lw,
                    transform=ax.get_xaxis_transform(), clip_on=False)
        # Every group label is tilted, not only the narrow ones: a mix of horizontal and
        # rotated labels reads as two kinds of thing rather than one row of group names.
        ax.text((x0 + x1) / 2, y - label_gap, gname, ha='right', va='center',
                rotation=30, rotation_mode='anchor',
                fontsize=fontsize, color='0.15', fontweight='bold',
                transform=ax.get_xaxis_transform(), clip_on=False)


def place_brackets(fig, axes_row):
    """Draw the group brackets below the rotated arm labels of each axis in `axes_row`.

    The depth of a rotated label in axes-fraction units depends on the longest label AND on
    the axes height, so a fixed offset collides with "global thr. k100" whenever the figure
    is resized. Measure the rendered labels instead. Call after `tight_layout`, which is
    what fixes the axes height.
    """
    fig.canvas.draw()
    for ax in axes_row:
        inv = ax.transAxes.inverted()
        depth = min(inv.transform((0, lbl.get_window_extent().y0))[1]
                    for lbl in ax.get_xticklabels())
        brackets(ax, y=depth - 0.05, drop=0.035, label_gap=0.09)


def arm_labels(axes_row):
    for ax in axes_row:
        ax.set_xticks([XPOS[a] for a in ARMS])
        ax.set_xticklabels([ARM_LABELS[a] for a in ARMS], rotation=90, fontsize=7.5)


def unit_divider(axes, top_row):
    """A dashed rule between residual-stream and MLP arms, named once in each top panel.

    The unit is the largest split in the comparison (residual-stream results carry the
    chain confound, LOG.md Iteration 18), so it gets a rule rather than one more bracket.
    Headroom is added so the two names never sit on a data point.
    """
    if UNIT_DIVIDER is None:
        return
    for ax in np.ravel(axes):
        ax.axvline(UNIT_DIVIDER, color='0.55', lw=1.0, ls=(0, (4, 3)), zorder=1.2)
    for ax in top_row:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.14 * (hi - lo))
        tr = ax.get_xaxis_transform()
        kw = dict(transform=tr, va='top', fontsize=8.5, color='0.3', fontstyle='italic')
        ax.text(UNIT_DIVIDER - 0.5, 0.985, 'residual stream', ha='right', **kw)
        ax.text(UNIT_DIVIDER + 0.5, 0.985, 'MLP neurons', ha='left', **kw)


def group_handles():
    return [Line2D([0], [0], marker='o', color='none', markerfacecolor=_ramp(c0, c1, 2)[1],
                   markeredgecolor='black', markersize=6, label=g)
            for g, c0, c1, _ in GROUPS]


def unit_handles():
    """Marker shape carries the unit in the scatter figures, colour the group."""
    return [Line2D([0], [0], marker=m, color='none', markerfacecolor='white',
                   markeredgecolor='black', markersize=6, label=lab)
            for m, lab in (('o', 'residual stream'), ('s', 'MLP neurons'))]


def best_per_unit(metric, value):
    """The arm with the largest real - null on `metric` among residual and among MLP arms.

    Ringed separately: residual-stream transfer is inflated by chain-following (Iteration
    18), so one overall winner would always be a residual arm and say nothing about MLP.
    """
    out = []
    for unit in (False, True):
        pool = [a for a in ARMS if is_mlp(a) == unit and np.isfinite(delta(metric, a))]
        if pool:
            best = max(pool, key=lambda a: delta(metric, a))
            out.append((best, value(best)))
    return out


def ring_best(ax, metric, value):
    for arm, y in best_per_unit(metric, value):
        ax.scatter(XPOS[arm], y, s=190, facecolors='none', edgecolors='0.25',
                   linewidths=1.1, zorder=4)
    ax.text(0.01, 0.80, 'ringed: best transfer\nwithin each unit type', transform=ax.transAxes,
            ha='left', va='top', fontsize=7.5, color='0.2')


PANELS = [
    ((0, 0), 'reliability_within', 'Reliability (ARI)', 'within domain\n(split halves)'),
    ((0, 1), 'fidelity_within_r', 'Fidelity (r)', None),
    ((1, 0), 'reliability_across_halves', 'Reliability (ARI)',
     'across domains\n(12 prose pairs)'),
    ((1, 1), 'fidelity_across_halves', 'Fidelity (r)', None),
]


# ---------------------------------------------------------------------------------------
# Figure 1 -- the main result. Arms on x (24 of them would make an unreadably tall figure
# on y), grouped by what the clustering sees. The gap between the open and filled dot is
# the finding: how much the partition earns beyond one that knows only per-unit properties.
# ---------------------------------------------------------------------------------------
def fig_references():
    fig, axes = plt.subplots(2, 2, figsize=(ARMS_FIG_W, 6.7 * 0.82), dpi=300, sharex=True)
    for (i, j), metric, title, rowlabel in PANELS:
        ax = axes[i, j]
        for arm in ARMS:
            x = XPOS[arm]
            real, ref = mean(metric, 'real', arm), mean(metric, 'pnull', arm)
            if not np.isfinite(real):
                continue
            beats = beats_ref(metric, arm)
            c = ARM_COLORS[arm] if beats else NS_EDGE
            ax.vlines(x, min(real, ref), max(real, ref), color=c,
                      alpha=0.45 if beats else 0.3, linewidth=3.2, zorder=2)
            # Faint per-domain values behind the mean, so the spread is visible.
            for v in vals(metric, 'real', arm):
                ax.scatter(x, v, s=7, color=c, alpha=0.28, edgecolors='none', zorder=2.5)
            ax.scatter(x, ref, s=34, facecolors='white', edgecolors=c, linewidths=1.1,
                       zorder=3)
            ax.scatter(x, real, s=42, color=c if beats else NS_FILL,
                       edgecolors='black' if beats else NS_EDGE, linewidths=0.5, zorder=3.5)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlim(-0.8, XMAX - 0.2)
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
        style_spines(ax, drop_top_right=True)
        ax.tick_params(axis='both', which='major', labelsize=9)
        if rowlabel is not None:
            ax.set_ylabel(rowlabel, fontsize=9.5, fontweight='bold', labelpad=8)

    unit_divider(axes, axes[0])
    # One callout, on the panel that carries the selection criterion: a ring on each unit
    # type's best arm plus a corner note, since a leader line would cross other arms.
    ring_best(axes[1, 1], 'fidelity_across_halves',
              lambda a: mean('fidelity_across_halves', 'real', a))

    arm_labels(axes[1])

    handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='white',
               markeredgecolor='black', markersize=6, label='null partition, real data'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='black',
               markeredgecolor='black', markersize=6, label='real'),
        Line2D([0], [0], color=NS_EDGE, lw=3, alpha=0.35,
               label='does not beat it in every domain'),
    ]
    # Below everything, including the bracket band, which hangs outside the axes; a
    # legend reserved inside the figure lands on top of it. bbox_inches='tight' at save
    # time keeps the negative offset in frame.
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.035),
               ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    place_brackets(fig, axes[1])
    save_fig(fig, 'arms_references')


# ---------------------------------------------------------------------------------------
# Figure 1b -- the same comparison as one number per arm. Figure 1 shows where the null
# partition sits, which is what makes "real - null" a credible quantity; once that is
# established the difference alone is easier to rank, and the arms are directly comparable
# down each column.
# ---------------------------------------------------------------------------------------
def fig_deltas():
    fig, axes = plt.subplots(2, 2, figsize=(ARMS_FIG_W, 6.4 * 0.82), dpi=300, sharex=True)
    for (i, j), metric, title, rowlabel in PANELS:
        ax = axes[i, j]
        for arm in ARMS:
            d = paired_deltas(metric, arm)
            if not len(d):
                continue
            x, beats = XPOS[arm], beats_ref(metric, arm)
            c = ARM_COLORS[arm] if beats else NS_EDGE
            ax.vlines(x, 0, d.mean(), color=c, alpha=0.45 if beats else 0.3, linewidth=3.2,
                      zorder=2)
            # Per-domain differences, which are the unit of evidence behind the mean and
            # behind the unanimity rule that decides the colour.
            for v in d:
                ax.scatter(x, v, s=7, color=c, alpha=0.28, edgecolors='none', zorder=2.5)
            ax.scatter(x, d.mean(), s=42, color=c if beats else NS_FILL,
                       edgecolors='black' if beats else NS_EDGE, linewidths=0.5, zorder=3.5)
        ax.axhline(0, color='0.35', lw=1.0, zorder=1.5)
        ax.set_title(title + ', real $-$ null', fontsize=10.5)
        ax.set_xlim(-0.8, XMAX - 0.2)
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
        style_spines(ax, drop_top_right=True)
        ax.tick_params(axis='both', which='major', labelsize=9)
        if rowlabel is not None:
            ax.set_ylabel(rowlabel, fontsize=9.5, fontweight='bold', labelpad=8)

    unit_divider(axes, axes[0])
    ring_best(axes[1, 1], 'fidelity_across_halves', lambda a: delta('fidelity_across_halves', a))

    arm_labels(axes[1])
    handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='black',
               markeredgecolor='black', markersize=6, label='real $-$ null partition'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='0.75',
               markeredgecolor='none', markersize=4, label='one prose domain (or pair)'),
        Line2D([0], [0], color=NS_EDGE, lw=3, alpha=0.35,
               label='does not beat the null in every domain'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.035),
               ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    place_brackets(fig, axes[1])
    save_fig(fig, 'arms_deltas')


# ---------------------------------------------------------------------------------------
# Figure 2 -- why the two rows of figure 1 disagree. Hubness buys within-domain performance
# and costs transfer, and the two fidelities are uncorrelated across arms.
# ---------------------------------------------------------------------------------------
def fig_hubness_tradeoff():
    fig, axes = plt.subplots(1, 3, figsize=(10.4 * 0.86, 3.3 * 0.86), dpi=300)
    # R^2 and r are not interchangeable here, and the difference IS the mechanism: R^2
    # rewards predicting the overall LEVEL of the connectivity, which a hubness partition
    # does well, while r scores only the pattern. Comparing a within-domain R^2 against an
    # across-domain r (which is forced, since R^2 is not defined across domains) therefore
    # manufactures a dissociation. Panels 1 and 2 are the same arms under the two measures.
    # Correlations are reported per unit type: on MLP neurons the block-model polish combines
    # high hubness with high transfer (Iteration 18), so one pooled r would mix two relations.
    specs = [
        (axes[0], 'fidelity_within', 'Within-domain fidelity R$^2$, real $-$ null',
         'Level: variance explained'),
        (axes[1], 'fidelity_within_r', 'Within-domain fidelity r, real $-$ null',
         'Pattern: within-domain r'),
        (axes[2], 'fidelity_across_halves', 'Across-domain fidelity r, real $-$ null',
         'Transfer: across-domain r'),
    ]
    for ax, my, ylabel, title in specs:
        stats = []
        for unit, marker, ls, name in ((False, 'o', '--', 'residual'), (True, 's', ':', 'MLP')):
            xs, ys = [], []
            for arm in ARMS:
                if is_mlp(arm) != unit:
                    continue
                x, y = mean('triviality_ami_hubness', 'real', arm), delta(my, arm)
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                xs.append(x)
                ys.append(y)
                ax.scatter(x, y, s=46 if unit else 52, marker=marker, color=ARM_COLORS[arm],
                           edgecolors='black', linewidths=0.5, zorder=3)
            if len(xs) < 3:
                continue
            r = float(np.corrcoef(xs, ys)[0, 1])
            stats.append('%s r = %+.2f' % (name, r))
            # Least-squares guide, drawn only where the relation is real.
            if abs(r) > 0.3:
                b, a = np.polyfit(xs, ys, 1)
                xx = np.linspace(min(xs), max(xs), 2)
                ax.plot(xx, a + b * xx, color='0.45', lw=1.1, ls=ls, zorder=1)
        ax.set_xlabel('AMI with hubness', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.text(0.97, 0.05, '\n'.join(stats), transform=ax.transAxes, fontsize=8, ha='right',
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))
        ax.grid(linestyle='--', alpha=0.4, zorder=0)
        style_spines(ax, drop_top_right=True)
        ax.tick_params(axis='both', which='major', labelsize=8.5)

    fig.legend(handles=group_handles() + unit_handles(),
               loc='lower center', bbox_to_anchor=(0.5, -0.17), ncol=5, frameon=False,
               fontsize=8)
    fig.tight_layout()
    save_fig(fig, 'hubness_tradeoff')


# ---------------------------------------------------------------------------------------
# Figure 3 -- the same dissociation arm by arm. Everything sits below the identity line:
# no arm transfers as well as it fits.
# ---------------------------------------------------------------------------------------
def fig_within_vs_across():
    fig, ax = plt.subplots(figsize=(5.6 * 0.9, 5.0 * 0.9), dpi=300)
    # Both axes are Pearson r: the across-domain measure cannot be R^2 (the two domains sit
    # on different scales), so the within-domain axis uses r as well. Against a within-domain
    # R^2 the relation looks orthogonal, which is a measure mismatch rather than a finding.
    xs, ys = [], []
    for arm in ARMS:
        x, y = delta('fidelity_within_r', arm), delta('fidelity_across_halves', arm)
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xs.append(x)
        ys.append(y)
        ax.scatter(x, y, s=54 if is_mlp(arm) else 62, marker='s' if is_mlp(arm) else 'o',
                   color=ARM_COLORS[arm], edgecolors='black', linewidths=0.5, zorder=3)
    lo, hi = 0, max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], color='0.6', lw=1.0, ls='--', zorder=1)
    ax.text(hi * 0.97, hi * 0.97, 'equal', fontsize=8, color='0.45', rotation=45,
            ha='right', va='bottom', rotation_mode='anchor')
    # Name the two ends of the story rather than all 24 points.
    # Upper left: above the identity line, where no arm can sit.
    ax.text(0.04, 0.96, 'r = %+.2f' % float(np.corrcoef(xs, ys)[0, 1]),
            transform=ax.transAxes, ha='left', va='top', fontsize=9,
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))
    # Label positions chosen to sit in empty space and never cross the identity line; the
    # two inside the crowded middle get a thin leader from a free spot below the cloud.
    labels = (
        ('vmf_ward200', 'Standardized\nWard k200', (9, -8), 'left', False),
        ('mlp:vmf_ward100_bm', 'MLP block\npolish, raw', (9, -8), 'left', False),
        ('ica', 'Spatial ICA k50', (0, -15), 'center', False),
        ('blockmodel100', 'Block-model k100', (0.30, 0.035), 'center', True),
    )
    for arm, text, off, ha, leader in labels:
        if arm not in XPOS:
            continue
        x, y = delta('fidelity_within_r', arm), delta('fidelity_across_halves', arm)
        if leader:
            ax.annotate(text, (x, y), xytext=off, textcoords='data', ha=ha, va='center',
                        fontsize=7.5, color='0.2',
                        arrowprops=dict(arrowstyle='-', color='0.45', lw=0.7,
                                        shrinkA=2, shrinkB=5))
        else:
            ax.annotate(text, (x, y), textcoords='offset points', xytext=off, ha=ha,
                        fontsize=7.5, color='0.2')
    ax.set_xlabel('Within-domain fidelity r, real $-$ null', fontsize=10)
    ax.set_ylabel('Across-domain fidelity r, real $-$ null', fontsize=10)
    # Room on the right for the labels of the two right-most arms.
    ax.set_xlim(0, max(xs) * 1.32)
    ax.set_ylim(bottom=0)
    ax.grid(linestyle='--', alpha=0.4, zorder=0)
    style_spines(ax, drop_top_right=True)
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.legend(handles=group_handles() + unit_handles(),
              loc='upper left', bbox_to_anchor=(1.04, 1.0), frameon=False, fontsize=8)
    fig.tight_layout()
    save_fig(fig, 'within_vs_across')


# ---------------------------------------------------------------------------------------
# Figure 4 -- ICA judged on the object it produces. Winner-take-all labels are the weakest
# in the set, while the maps themselves are among the most reproducible things measured.
# ---------------------------------------------------------------------------------------
def fig_maps_vs_labels():
    ica = [a for a in ARMS if vals('reliability_within_maps', 'real', a)]
    if not ica:
        return
    fig, ax = plt.subplots(figsize=(4.6 * 0.9, 3.1 * 0.9), dpi=300)
    x = np.arange(len(ica))
    for i, arm in enumerate(ica):
        lab = mean('reliability_within', 'real', arm)
        mp = mean('reliability_within_maps', 'real', arm)
        c = ARM_COLORS[arm]
        ax.vlines(i, lab, mp, color=c, alpha=0.45, linewidth=3.2, zorder=2)
        ax.scatter(i, lab, s=46, color=c, edgecolors='black', linewidths=0.5, zorder=3)
        ax.scatter(i, mp, s=46, marker='D', color=c, edgecolors='black', linewidths=0.5,
                   zorder=3)
    # The best label reliability anywhere in the set, for scale.
    best = max(ARMS, key=lambda a: mean('reliability_within', 'real', a))
    ax.axhline(mean('reliability_within', 'real', best), color='0.5', ls=':', lw=1.1)
    ax.text(len(ica) - 0.55, mean('reliability_within', 'real', best),
            'best labels of any arm\n(%s %s)' % (ARM_GROUP[best],
                                                 ARM_LABELS[best]),
            fontsize=7, color='0.35', ha='right', va='bottom')
    ax.set_xticks(x)
    ax.set_xticklabels(['ICA %s' % ARM_LABELS[a] for a in ica], fontsize=9)
    ax.set_ylabel('Reliability across halves', fontsize=10)
    ax.set_xlim(-0.6, len(ica) - 0.4)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
    style_spines(ax, drop_top_right=True)
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.legend(handles=[
        Line2D([0], [0], marker='D', color='none', markerfacecolor='black',
               markeredgecolor='black', markersize=6, label='component maps'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='black',
               markeredgecolor='black', markersize=6, label='winner-take-all labels'),
    ], loc='lower left', frameon=False, fontsize=8)
    fig.tight_layout()
    save_fig(fig, 'maps_vs_labels')


if __name__ == '__main__':
    os.makedirs('plots', exist_ok=True)
    fig_references()
    fig_deltas()
    fig_hubness_tradeoff()
    fig_within_vs_across()
    fig_maps_vs_labels()
    print('wrote plots/{arms_references,arms_deltas,hubness_tradeoff,within_vs_across,'
          'maps_vs_labels}'
          '.{svg,png} from %d arms' % len(ARMS))
