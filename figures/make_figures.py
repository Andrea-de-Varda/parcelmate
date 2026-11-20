"""Publication figures across every arm of every run (LOG.md Iterations 13-20).

    python figures/make_figures.py      ->  plots/*.svg + *.png

Combines eight score files into one comparison of 55 arms, plus the pooled-estimation scores
and the yardstick for the last round (scores_pooled_mlp.csv, yardstick_final_mlp.csv):

    scores_ladder.csv       the six-arm ladder, MiniBatch k-means, residual stream (Iter. 13)
    scores_yolo.csv         converged optimizer, Ward, block-model objective, k = 100 (Iter. 16)
    scores_yolo4.csv        spatial ICA and the sparsification rungs (Iter. 16)
    scores_yolo_mlp.csv     the same pipeline on MLP neurons instead of the residual stream
    scores_final_resid.csv  residual Ward at k = 150, 200 (final test T1, Iter. 18)
    scores_final_mlp.csv    final tests T1, T2, T5 on MLP neurons (Iter. 18)
    scores_last_mlp.csv     the last round: consensus polish, co-association, restarts, k (Iter. 20)
    scores_vmf_sparse_pca100_lloyd100_n200.csv
                            the confirmation arm: sparse profiles + 200 restarts (Iter. 21)

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
  5. final_candidates  -- every change to the base MLP pipeline, one row per arm, on the four
                          measures; the two changes kept are in bold
  6. pooled_estimation -- transfer and label agreement by what is fit and what is held out.
                          A record of T4 only: every clustering in the final design is
                          fitted within a single domain (Andrea, 2026-09-15)
  7. yardstick         -- consensus reliability against what steering from the other half
                          reaches: the headroom that belongs to the consensus
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
    'scores_last_mlp.csv': 'mlp:',   # the last round (Iterations 19-20)
    'scores_vmf_sparse_pca100_lloyd100_n200.csv': 'mlp:',  # the confirmation arm (Iteration 21)
}
# Rows not to load from a file. The last round's base arm reruns final_mlp's
# `vmf_pca100_lloyd100` with the same seeds and gives identical scores (Iteration 20), and
# both files carry the same MLP ceilings; loading them twice would double the per-domain dots.
# The one exception is the metric only the rerun has: co-association reliability needs the
# restart labels, which final_mlp did not store.
SKIP = {'scores_last_mlp.csv': {'vmf_pca100_lloyd100', '(ceiling)'},
        'scores_vmf_sparse_pca100_lloyd100_n200.csv': {'(ceiling)'}}
SKIP_EXCEPT_METRICS = {'reliability_within_coassoc'}
POOLED_FILE = 'scores_pooled_mlp.csv'
YARDSTICK_FILE = 'yardstick_final_mlp.csv'

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
    ('MLP Ward polish', '#f1b6da', '#8e0152', [
        ('mlp:vmf_ward100_bm', 'raw'),
        ('mlp:vmf_ward100_bm_double', 'double-centred'),
        ('mlp:vmf_ward100_bm_degree', 'degree-corrected'),
    ]),
    # The last round: variations on PCA-100 Lloyd consensus (the base arm is 'PCA100 Lloyd'
    # in MLP denoised), then the same with the block-model polish applied to the consensus.
    ('MLP Lloyd consensus', '#e3ea9f', '#4d5c0d', [
        ('mlp:vmf_sparse_pca100_lloyd100', 'sparse PCA100'),
        ('mlp:vmf_pca100_lloyd50', 'PCA100 k50'),
        ('mlp:vmf_pca100_lloyd200', 'PCA100 k200'),
        ('mlp:vmf_pca100_lloyd100_n200', '200 restarts'),
        ('mlp:vmf_pca100_lloyd100_coassoc', 'co-association'),
        ('mlp:vmf_pca100_lloyd100_n200_coassoc', '200 r., co-assoc.'),
        ('mlp:vmf_sparse_pca100_lloyd100_n200', 'sparse, 200 r. (final)'),
    ]),
    ('MLP consensus polish', '#dcc3a8', '#5a3417', [
        ('mlp:vmf_pca100_lloyd100_bm_raw', 'raw'),
        ('mlp:vmf_pca100_lloyd100_bm_double', 'double-centred'),
        ('mlp:vmf_pca100_lloyd100_bm_degree', 'degree-corrected'),
        ('mlp:vmf_sparse_pca100_lloyd100_bm_degree', 'sparse, degree'),
        ('mlp:vmf_pca100_lloyd50_bm_degree', 'k50, degree'),
        ('mlp:vmf_pca100_lloyd200_bm_degree', 'k200, degree'),
        ('mlp:vmf_pca100_lloyd100_coassoc_bm_degree', 'co-assoc., degree'),
    ]),
]


def load():
    rows = []
    for fname, prefix in SOURCES.items():
        path = os.path.join(HERE, fname)
        if not os.path.exists(path):
            continue
        skip = SKIP.get(fname, set())
        for r in csv.DictReader(open(path)):
            if r['variant'] in skip and r['metric'] not in SKIP_EXCEPT_METRICS:
                continue
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


ARM_COLORS, ARM_LABELS, ARM_GROUP, GROUP_DARK = {}, {}, {}, {}
for gname, c0, c1, arms in GROUPS:
    for (arm, label), colour in zip(arms, _ramp(c0, c1, len(arms))):
        ARM_COLORS[arm], ARM_LABELS[arm], ARM_GROUP[arm] = colour, label, gname
        GROUP_DARK[arm] = c1

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


FINAL = 'mlp:vmf_sparse_pca100_lloyd100_n200'   # the selected pipeline (METHOD_SELECTION.md)
ORIGINAL = 'legacy'   # the inherited pipeline as the ladder ran it (see fig_original_vs_final)


def ring_final(ax, y):
    """Ring the final pipeline's point on an arm axis."""
    if FINAL in XPOS and np.isfinite(y):
        ax.scatter(XPOS[FINAL], y, s=190, facecolors='none', edgecolors='0.15',
                   linewidths=1.3, zorder=4)


def final_handle():
    return Line2D([0], [0], marker='o', color='none', markerfacecolor='none',
                  markeredgecolor='0.15', markeredgewidth=1.3, markersize=11,
                  label='final pipeline')


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
    # The final pipeline is ringed in every panel, so its standing on each measure is read
    # off directly.
    for (i, j), metric, _, _ in PANELS:
        ring_final(axes[i, j], mean(metric, 'real', FINAL))

    arm_labels(axes[1])

    handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='white',
               markeredgecolor='black', markersize=6, label='null partition, real data'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='black',
               markeredgecolor='black', markersize=6, label='real'),
        Line2D([0], [0], color=NS_EDGE, lw=3, alpha=0.35,
               label='does not beat it in every domain'),
        final_handle(),
    ]
    # Below everything, including the bracket band, which hangs outside the axes; a
    # legend reserved inside the figure lands on top of it. bbox_inches='tight' at save
    # time keeps the negative offset in frame.
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.035),
               ncol=4, frameon=False, fontsize=8.5)
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
    for (i, j), metric, _, _ in PANELS:
        ring_final(axes[i, j], delta(metric, FINAL))

    arm_labels(axes[1])
    handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='black',
               markeredgecolor='black', markersize=6, label='real $-$ null partition'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='0.75',
               markeredgecolor='none', markersize=4, label='one prose domain (or pair)'),
        Line2D([0], [0], color=NS_EDGE, lw=3, alpha=0.35,
               label='does not beat the null in every domain'),
        final_handle(),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.035),
               ncol=4, frameon=False, fontsize=8.5)
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
def label_final(ax, x, y, off=(10, 8), leader=False):
    """Ring and name the final pipeline in a scatter of arms. With `leader`, `off` is a
    position in data coordinates and a thin line joins it to the ring."""
    ax.scatter(x, y, s=210, facecolors='none', edgecolors='0.15', linewidths=1.3, zorder=4)
    if leader:
        ax.annotate('final pipeline', (x, y), xytext=off, textcoords='data', ha='left',
                    va='center', fontsize=8, fontweight='bold', color='0.1',
                    arrowprops=dict(arrowstyle='-', color='0.3', lw=0.8, shrinkA=2, shrinkB=7))
    else:
        ax.annotate('final pipeline', (x, y), textcoords='offset points', xytext=off,
                    fontsize=8, fontweight='bold', color='0.1')


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
        # The Ward polish and the consensus polish, both raw, land on the same spot.
        ('mlp:vmf_ward100_bm', 'MLP raw polish\n(Ward, consensus)', (9, -8), 'left', False),
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
    if FINAL in XPOS:
        label_final(ax, delta('fidelity_within_r', FINAL), delta('fidelity_across_halves', FINAL),
                    off=(0.33, 0.07), leader=True)
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
# Figure 3b -- the two across-domain measures against each other: does a partition whose
# labels transfer also predict the other domain's connectivity?
# ---------------------------------------------------------------------------------------
SCOPE = {
    'across': ('reliability_across_halves', 'fidelity_across_halves', 'Across-domain',
               (0.62, 0.12)),
    'within': ('reliability_within', 'fidelity_within_r', 'Within-domain', (0.75, 0.56)),
}


def fig_across_reliability_vs_fidelity(mlp_only=False, scope='across'):
    """Reliability against fidelity for one scope (across or within domains).

    With `mlp_only`, only the MLP arms: residual-stream arms are confounded by dimension
    chains (Iteration 18), which inflate both axes, and the final choice was made among MLP
    arms."""
    m_rel, m_fid, name, label_at = SCOPE[scope]
    fig, ax = plt.subplots(figsize=(6.4 * 0.9, 4.6 * 0.9), dpi=300)
    xs, ys = [], []
    for arm in ARMS:
        if mlp_only and not is_mlp(arm):
            continue
        x, y = delta(m_rel, arm), delta(m_fid, arm)
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xs.append(x)
        ys.append(y)
        ax.scatter(x, y, s=54 if is_mlp(arm) else 62, marker='s' if is_mlp(arm) else 'o',
                   color=ARM_COLORS[arm], edgecolors='black', linewidths=0.5, zorder=3)
    ax.text(0.04, 0.96, 'r = %+.2f' % float(np.corrcoef(xs, ys)[0, 1]),
            transform=ax.transAxes, ha='left', va='top', fontsize=9,
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))
    # One scale on both axes, square, so distances along x and y compare directly.
    hi = 1.08 * max(max(xs), max(ys))
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect('equal')
    ax.plot([0, hi], [0, hi], color='0.6', lw=1.0, ls='--', zorder=1)
    if FINAL in XPOS:
        # Label from empty space (per scope), clear of the crowded consensus arms.
        label_final(ax, delta(m_rel, FINAL), delta(m_fid, FINAL),
                    off=(label_at[0] * hi, label_at[1] * hi), leader=True)
    ax.set_xlabel('%s reliability (ARI),\nreal $-$ null' % name, fontsize=10)
    ax.set_ylabel('%s fidelity r,\nreal $-$ null' % name, fontsize=10)
    ax.grid(linestyle='--', alpha=0.4, zorder=0)
    style_spines(ax, drop_top_right=True)
    ax.tick_params(axis='both', which='major', labelsize=9)
    if mlp_only:
        handles = [h for h, (_, _, _, arms) in zip(group_handles(), GROUPS) if is_mlp(arms[0][0])]
    else:
        handles = group_handles() + unit_handles()
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.04, 1.0), frameon=False,
              fontsize=8)
    fig.tight_layout()
    save_fig(fig, '%s_reliability_vs_fidelity%s' % (scope, '_mlp' if mlp_only else ''))


# ---------------------------------------------------------------------------------------
# Figure 3b' -- all four measures against each other on one scale, MLP arms only. Every
# measure is real minus the null partition, so 0 means "no better than per-unit properties"
# on every axis and the panels compare directly.
# ---------------------------------------------------------------------------------------
MATRIX_MEASURES = [
    ('reliability_within', 'Within-domain\nreliability (ARI)'),
    ('fidelity_within_r', 'Within-domain\nfidelity (r)'),
    ('reliability_across_halves', 'Across-domain\nreliability (ARI)'),
    ('fidelity_across_halves', 'Across-domain\nfidelity (r)'),
]


def fig_measures_matrix():
    arms = [a for a in ARMS if is_mlp(a)
            and all(np.isfinite(delta(m, a)) for m, _ in MATRIX_MEASURES)]
    if not arms:
        return
    D = {m: np.array([delta(m, a) for a in arms]) for m, _ in MATRIX_MEASURES}
    hi = 1.06 * max(v.max() for v in D.values())
    lo = min(0.0, min(v.min() for v in D.values()))
    n = len(MATRIX_MEASURES)
    fig, axes = plt.subplots(n, n, figsize=(8.4, 8.4), dpi=300)
    fi = arms.index(FINAL) if FINAL in arms else None
    for i, (my, ylab) in enumerate(MATRIX_MEASURES):
        for j, (mx, xlab) in enumerate(MATRIX_MEASURES):
            ax = axes[i, j]
            if j > i:
                ax.axis('off')
                continue
            ax.set_xlim(lo, hi)
            ax.set_aspect('equal')
            if i == j:
                # Diagonal: the measure's spread over arms as a strip along its own axis, in a
                # panel the same size and scale as the scatters.
                ax.set_ylim(lo, hi)
                band = lo + (hi - lo) * (0.06 + 0.10 * np.linspace(0, 1, len(arms)))
                ax.scatter(D[mx], band, s=12, marker='s', color=[ARM_COLORS[a] for a in arms],
                           edgecolors='black', linewidths=0.3, zorder=3)
                if fi is not None:
                    ax.axvline(D[mx][fi], color='0.15', lw=1.1, ls=(0, (3, 2)), zorder=2)
                    ax.text(D[mx][fi], lo + 0.22 * (hi - lo), ' final', fontsize=7,
                            color='0.15', ha='left', va='bottom')
                ax.set_yticks([])
                ax.spines['left'].set_visible(False)
                ax.text(0.5, 0.62, xlab, transform=ax.transAxes, ha='center', va='center',
                        fontsize=9, fontweight='bold', zorder=5,
                        bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
            else:
                ax.set_ylim(lo, hi)
                ax.plot([lo, hi], [lo, hi], color='0.7', lw=0.8, ls='--', zorder=1)
                ax.scatter(D[mx], D[my], s=16, marker='s', color=[ARM_COLORS[a] for a in arms],
                           edgecolors='black', linewidths=0.3, zorder=3)
                if fi is not None:
                    ax.scatter(D[mx][fi], D[my][fi], s=95, facecolors='none',
                               edgecolors='0.1', linewidths=1.2, zorder=4)
                r = float(np.corrcoef(D[mx], D[my])[0, 1])
                ax.text(0.05, 0.95, 'r = %+.2f' % r, transform=ax.transAxes, ha='left',
                        va='top', fontsize=7,
                        bbox=dict(facecolor='white', edgecolor='gray',
                                  boxstyle='round,pad=0.25'))
            ax.grid(linestyle='--', alpha=0.35, zorder=0)
            style_spines(ax, drop_top_right=True)
            ax.tick_params(axis='both', labelsize=7)
            if i == n - 1:
                ax.set_xlabel(xlab, fontsize=8.5)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(ylab, fontsize=8.5)
            elif i != j:
                ax.set_yticklabels([])
    handles = [h for h, (_, _, _, g) in zip(group_handles(), GROUPS) if is_mlp(g[0][0])]
    for h in handles:
        h.set_marker('s')
    handles.append(final_handle())
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.97, 0.93), frameon=False,
               fontsize=8.5, title='MLP arms, real $-$ null\n(same scale on every axis)',
               title_fontsize=8.5)
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    save_fig(fig, 'measures_matrix_mlp')


# ---------------------------------------------------------------------------------------
# Figure 3d -- the final pipeline against every other arm, one column per measure, nested
# within/across x reliability/fidelity. All four are real minus the null partition, so they
# share one axis. Adapted from the grouped dot-plot with hierarchical brackets.
# ---------------------------------------------------------------------------------------
FINAL_RED = '#c0392b'
FVA_COLUMNS = [   # (metric, column label, top-level group)
    ('reliability_within', 'Reliability\n(ARI)', 'Within domain'),
    ('fidelity_within_r', 'Fidelity\n(r)', 'Within domain'),
    ('reliability_across_halves', 'Reliability\n(ARI)', 'Across domains'),
    ('fidelity_across_halves', 'Fidelity\n(r)', 'Across domains'),
]


def _ordinal(n):
    return '%d%s' % (n, 'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th'))


def fig_final_vs_arms(mlp_only=False):
    from scipy.stats import gaussian_kde
    if FINAL not in XPOS:
        return
    others = [a for a in ARMS if a != FINAL and (is_mlp(a) or not mlp_only)]
    xs = [0.0, 1.0, 2.4, 3.4]   # a wider gap between the within and across pairs
    rng = np.random.default_rng(0)
    tops = []
    fig, ax = plt.subplots(figsize=(4.6 * 0.9, 3.5 * 0.9), dpi=300)
    for x, (metric, _, _) in zip(xs, FVA_COLUMNS):
        v = np.array([delta(metric, a) for a in others])
        v = v[np.isfinite(v)]
        f = delta(metric, FINAL)
        # Half-violin on the left: the distribution over arms, run out to three bandwidths
        # past the extreme arms so the density tapers to zero instead of being cut off.
        kde = gaussian_kde(v, bw_method=0.35)   # narrower than Scott's rule: tails stay near the data
        bw = 3 * kde.factor * v.std(ddof=1)
        grid = np.linspace(v.min() - bw, v.max() + bw, 300)
        dens = kde(grid)
        # Stop where the density falls under 2% of its peak: the curve tapers, but its tails
        # do not run past values no arm reaches.
        keep = dens >= 0.02 * dens.max()
        grid, dens = grid[keep], dens[keep]
        dens = 0.34 * dens / dens.max()
        ax.fill_betweenx(grid, x - 0.06 - dens, x - 0.06, color='0.55', alpha=0.28, lw=0,
                         zorder=2)
        ax.plot(x - 0.06 - dens, grid, color='0.45', lw=0.8, zorder=2.2)
        # Jittered dots on the right: one per arm.
        ax.scatter(x + 0.06 + rng.uniform(0, 0.22, len(v)), v, s=11, color='0.45', alpha=0.35,
                   edgecolors='none', zorder=3)
        # The final pipeline across both halves of the column, with its value and rank.
        rank = 1 + int((v > f).sum())
        ax.scatter(x, f, s=90, color=FINAL_RED, edgecolors='black', linewidths=0.7, zorder=5)
        # Value and rank above the column, clear of the violin and the dots.
        ax.text(x, grid.max() + 0.02, '%.2f\n%s / %d' % (f, _ordinal(rank), len(v) + 1),
                ha='center', va='bottom', fontsize=7, color=FINAL_RED, fontweight='bold',
                zorder=6)
        tops.append(grid.max())
    ax.axhline(0, color='0.35', lw=1.0, zorder=1.5)
    ax.set_ylim(top=max(tops) + 0.16)
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab, _ in FVA_COLUMNS], fontsize=8.5)
    ax.set_xlim(-0.55, xs[-1] + 0.45)
    ax.set_ylabel('Real $-$ null partition', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
    style_spines(ax, drop_top_right=True)
    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(axis='x', length=0, pad=4)
    # Top-level brackets under the column labels.
    tr = ax.get_xaxis_transform()
    for (x0, x1), name in (((xs[0], xs[1]), 'Within domain'), ((xs[2], xs[3]), 'Across domains')):
        y = -0.20
        ax.plot([x0 - 0.3, x0 - 0.3, x1 + 0.3, x1 + 0.3], [y + 0.025, y, y, y + 0.025],
                color='0.2', lw=1.2, transform=tr, clip_on=False)
        ax.text((x0 + x1) / 2, y - 0.03, name, ha='center', va='top', fontsize=9,
                fontweight='bold', transform=tr)
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='none', markerfacecolor=FINAL_RED,
               markeredgecolor='black', markersize=8, label='final pipeline'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='0.45', alpha=0.5,
               markeredgecolor='none', markersize=4,
               label='other arms (%s, n = %d)' % ('MLP' if mlp_only else 'all', len(others))),
    ], loc='upper right', frameon=False, fontsize=7.5)
    fig.tight_layout()
    save_fig(fig, 'final_vs_arms' + ('_mlp' if mlp_only else ''))


# ---------------------------------------------------------------------------------------
# Figure 3c -- the inherited pipeline against the final one, on every measure. The inherited
# pipeline is the ladder's `legacy` arm: residual stream, column-wise binarization (the S2
# bug), whitened PCA-200, MiniBatch k-means at k = 50. It ran 20 restarts, where the original
# code defaulted to 100, so it is the original method on a smaller restart budget.
# ---------------------------------------------------------------------------------------
ORIG_PANELS = [
    ('reliability_within', 'Reliability (ARI)\nwithin domain'),
    ('fidelity_within_r', 'Fidelity (r)\nwithin domain'),
    ('reliability_across_halves', 'Reliability (ARI)\nacross domains'),
    ('fidelity_across_halves', 'Fidelity (r)\nacross domains'),
    ('triviality_ami_hubness', 'Hubness (AMI)\nlower is better'),
]


def fig_original_vs_final():
    if ORIGINAL not in XPOS or FINAL not in XPOS:
        return
    pair = [(ORIGINAL, 'original', ARM_COLORS[ORIGINAL]), (FINAL, 'final', GROUP_DARK[FINAL])]
    fig, axes = plt.subplots(1, len(ORIG_PANELS), figsize=(11.0 * 0.82, 3.5 * 0.82), dpi=300)
    for ax, (metric, title) in zip(axes, ORIG_PANELS):
        with_null = not metric.startswith('triviality')
        for i, (arm, _, c) in enumerate(pair):
            real = mean(metric, 'real', arm)
            v = vals(metric, 'real', arm)
            ax.scatter(np.full(len(v), i) + np.linspace(-0.08, 0.08, len(v)), v, s=9, color=c,
                       alpha=0.35, edgecolors='none', zorder=2.5)
            if with_null:
                ref = mean(metric, 'pnull', arm)
                ax.vlines(i, min(real, ref), max(real, ref), color=c, alpha=0.45, lw=3.2,
                          zorder=2)
                ax.scatter(i, ref, s=40, facecolors='white', edgecolors=c, linewidths=1.2,
                           zorder=3)
            ax.scatter(i, real, s=50, color=c, edgecolors='black', linewidths=0.5, zorder=3.5)
            ax.annotate('%.2f' % real, (i, real), textcoords='offset points', xytext=(9, 0),
                        va='center', fontsize=7.5, color='0.2')
        ax.set_xticks([0, 1])
        ax.set_xticklabels([lab for _, lab, _ in pair], fontsize=9)
        ax.set_xlim(-0.5, 1.6)
        ax.set_ylim(bottom=0)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
        style_spines(ax, drop_top_right=True)
        ax.tick_params(axis='y', labelsize=8.5)
    fig.legend(handles=[
        Line2D([0], [0], marker='o', color='none', markerfacecolor='0.3',
               markeredgecolor='black', markersize=6, label='real'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='white',
               markeredgecolor='0.3', markersize=6, label='null partition, real data'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='0.6',
               markeredgecolor='none', markersize=4, alpha=0.6,
               label='one prose domain (or pair)'),
    ], loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(w_pad=1.0)
    save_fig(fig, 'original_vs_final')


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


# ---------------------------------------------------------------------------------------
# Figures 5-7 -- the last round and the decision (LOG.md Iteration 20). All on MLP neurons.
# ---------------------------------------------------------------------------------------
KEPT = ('mlp:vmf_sparse_pca100_lloyd100', 'mlp:vmf_pca100_lloyd100_n200',
        'mlp:vmf_sparse_pca100_lloyd100_n200')
BASE = 'mlp:vmf_pca100_lloyd100'
# Blocks of rows, top to bottom: algorithm, the two changes kept, the two not kept, k, polish.
FINAL_ROWS = [
    [('mlp:vmf_ward100', 'Ward, full profiles'),
     ('mlp:vmf_lloyd100', 'Lloyd, full profiles'),
     (BASE, 'Lloyd, PCA-100 (base)')],
    [('mlp:vmf_sparse_pca100_lloyd100', '+ sparse profiles'),
     ('mlp:vmf_pca100_lloyd100_n200', '+ 200 restarts'),
     ('mlp:vmf_sparse_pca100_lloyd100_n200', '+ both (final pipeline)')],
    [('mlp:vmf_pca100_lloyd100_coassoc', '+ co-association consensus'),
     ('mlp:vmf_pca100_lloyd100_n200_coassoc', '+ 200 restarts, co-association')],
    [('mlp:vmf_pca100_lloyd50', 'k = 50'),
     ('mlp:vmf_pca100_lloyd200', 'k = 200')],
    [('mlp:vmf_pca100_lloyd100_bm_raw', '+ polish, raw'),
     ('mlp:vmf_pca100_lloyd100_bm_double', '+ polish, double-centred'),
     ('mlp:vmf_pca100_lloyd100_bm_degree', '+ polish, degree-corrected')],
]
FINAL_PANELS = [
    ('reliability_within', False, 'Within-domain\nreliability (ARI)'),
    ('reliability_within_coassoc', False, 'Within-domain\nco-association r'),
    ('reliability_across_halves', False, 'Across-domain\nreliability (ARI)'),
    ('fidelity_within_r', True, 'Within-domain fidelity\nr, real $-$ null'),
    ('fidelity_across_halves', True, 'Across-domain fidelity\nr, real $-$ null'),
]


def fig_final_candidates():
    """Every change to the base pipeline on the four measures, one row per arm.

    Reliabilities are raw (their null references sit at 0.01-0.03); fidelities are real minus
    the null partition, paired by domain. The dashed line is the base arm, so a point right of
    it is a gain. Bold rows are the two changes kept and their combination, the final pipeline
    (confirmation arm, Iteration 21).
    """
    rows, ys, y = [], [], 0.0
    for block in FINAL_ROWS:
        for arm, label in block:
            if arm in PRESENT:
                rows.append((arm, label))
                ys.append(y)
                y += 1.0
        y += 0.6
    if BASE not in PRESENT or len(rows) < 3:
        return
    fig, axes = plt.subplots(1, len(FINAL_PANELS), figsize=(13.2 * 0.8, 4.9 * 0.8), dpi=300,
                             sharey=True)
    for ax, (metric, is_delta, title) in zip(axes, FINAL_PANELS):
        def values(arm):
            return paired_deltas(metric, arm) if is_delta else np.array(vals(metric, 'real', arm))
        base_v = values(BASE)
        if len(base_v):
            ax.axvline(base_v.mean(), color='0.45', lw=1.0, ls='--', zorder=1)
        for (arm, label), yy in zip(rows, ys):
            v = values(arm)
            if not len(v):
                continue
            # The dark end of the arm's group: one row per arm needs no within-group ramp, and
            # the light end of a ramp is unreadable on white.
            c = GROUP_DARK.get(arm, '0.3')
            ax.scatter(v, np.full(len(v), yy) + np.linspace(-0.18, 0.18, len(v)), s=8, color=c,
                       alpha=0.35, edgecolors='none', zorder=2)
            ax.scatter(v.mean(), yy, s=46, color=c, edgecolors='black', linewidths=0.5, zorder=3)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.5, zorder=0)
        style_spines(ax, drop_top_right=True)
        ax.tick_params(axis='both', which='major', labelsize=8)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([label for _, label in rows], fontsize=8.5)
    for tick, (arm, _) in zip(axes[0].get_yticklabels(), rows):
        if arm in KEPT:
            tick.set_fontweight('bold')
    axes[0].set_ylim(ys[-1] + 0.8, -0.8)
    fig.tight_layout(w_pad=0.8)
    save_fig(fig, 'final_candidates')


def load_plain(fname):
    path = os.path.join(HERE, fname)
    if not os.path.exists(path):
        return None
    out = list(csv.DictReader(open(path)))
    for r in out:
        r['value'] = float(r['value'])
    return out


POOL_TYPES = [
    ('domain\nto domain', lambda f, e: f in PROSE and e in PROSE),
    ('pool of 3\nto held-out', lambda f, e: f.startswith('lodo_') and e in PROSE),
    ('held-out\nto pool of 3', lambda f, e: f in PROSE and e.startswith('lodo_')),
    ('pair to\nother pair', lambda f, e: f.startswith('pair_')),
]
POOL_ARMS = [
    ('vmf_pca100_lloyd100', 'Lloyd PCA-100 consensus'),
    ('vmf_pca100_lloyd100_coassoc', 'co-association consensus'),
    ('vmf_pca100_lloyd100_bm_degree', 'degree-corrected polish'),
]


def fig_pooled_estimation():
    """Pooled estimation (T4): transfer and label agreement by what is fit and what is held out.

    Every comparison shares no data between the fitting and the evaluated connectome. The
    ceiling under each group is the uncompressed across-domain r for that comparison, which
    rises with the tokens on the fitting side; the last panel shows the leave-one-domain-out
    gain held-out domain by held-out domain.
    """
    rows = load_plain(POOLED_FILE)
    if rows is None:
        return
    idx = {}
    for r in rows:
        idx.setdefault((r['metric'], r['tree'], r['variant']), {})[(r['fit'], r['eval'])] = r['value']

    def pairs_of(metric, tree, variant, test):
        return {k: v for k, v in idx.get((metric, tree, variant), {}).items() if test(*k)}

    def deltas(metric, variant, test):
        real, ref = pairs_of(metric, 'real', variant, test), pairs_of(metric, 'pnull', variant, test)
        return np.array([real[k] - ref[k] for k in sorted(set(real) & set(ref))])

    fig, axes = plt.subplots(1, 3, figsize=(13.2 * 0.82, 3.8 * 0.82), dpi=300,
                             gridspec_kw=dict(width_ratios=[1.35, 1.35, 0.8]))
    offsets = np.linspace(-0.24, 0.24, len(POOL_ARMS))
    for ax, metric, is_delta, ylabel in (
            (axes[0], 'fidelity_across_halves', True, 'Across-domain fidelity r, real $-$ null'),
            (axes[1], 'reliability_across_halves', False, 'Across-domain reliability (ARI)')):
        for (variant, _), off in zip(POOL_ARMS, offsets):
            c = ARM_COLORS.get('mlp:' + variant, '0.3')
            for t, (_, test) in enumerate(POOL_TYPES):
                v = deltas(metric, variant, test) if is_delta else \
                    np.array(list(pairs_of(metric, 'real', variant, test).values()))
                if not len(v):
                    continue
                ax.scatter(np.full(len(v), t + off), v, s=8, color=c, alpha=0.35,
                           edgecolors='none', zorder=2)
                ax.scatter(t + off, v.mean(), s=44, color=c, edgecolors='black',
                           linewidths=0.5, zorder=3)
        labels = []
        for name, test in POOL_TYPES:
            ceil = pairs_of('fidelity_across_halves', 'real', '(ceiling)', test)
            labels.append('%s\nceiling %.2f' % (name, np.mean(list(ceil.values()))) if ceil else name)
        ax.set_xticks(range(len(POOL_TYPES)))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlim(-0.6, len(POOL_TYPES) - 0.4)
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        style_spines(ax, drop_top_right=True)
        ax.tick_params(axis='y', labelsize=8.5)

    ax = axes[2]
    variant = POOL_ARMS[0][0]
    c = ARM_COLORS.get('mlp:' + variant, '0.3')
    for d in PROSE:
        single = deltas('fidelity_across_halves', variant, lambda f, e, d=d: f in PROSE and e == d)
        lodo = deltas('fidelity_across_halves', variant, lambda f, e, d=d: f == 'lodo_' + d and e == d)
        if not (len(single) and len(lodo)):
            continue
        ax.plot([0, 1], [single.mean(), lodo.mean()], color=c, lw=1.2, zorder=2)
        ax.scatter([0, 1], [single.mean(), lodo.mean()], s=34, color=c, edgecolors='black',
                   linewidths=0.5, zorder=3)
        ax.text(1.08, lodo.mean(), d, fontsize=7, va='center', color='0.25')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['mean of the\n3 single domains', 'pool of the\n3 domains'], fontsize=7.5)
    ax.set_xlim(-0.35, 1.75)
    ax.set_ylabel('Fidelity r into the held-out domain,\nreal $-$ null', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    style_spines(ax, drop_top_right=True)
    ax.tick_params(axis='y', labelsize=8.5)

    fig.legend(handles=[Line2D([0], [0], marker='o', color='none',
                               markerfacecolor=ARM_COLORS.get('mlp:' + v, '0.3'),
                               markeredgecolor='black', markersize=6, label=lab)
                        for v, lab in POOL_ARMS],
               loc='lower center', bbox_to_anchor=(0.5, -0.09), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(w_pad=1.2)
    save_fig(fig, 'pooled_estimation')


POOL_SINGLE = ('vmf_pca100_lloyd100', 'Lloyd PCA-100 consensus')


def fig_pooled_estimation_single():
    """`pooled_estimation` for one arm, one bar per comparison.

    The final pipeline (sparse profiles, 200 restarts) was not run in T4. The nearest arm that
    was is its base, `vmf_pca100_lloyd100`: the same pipeline without sparsification and with
    40 restarts.
    """
    rows = load_plain(POOLED_FILE)
    if rows is None:
        return
    variant, name = POOL_SINGLE
    idx = {}
    for r in rows:
        idx.setdefault((r['metric'], r['tree'], r['variant']), {})[(r['fit'], r['eval'])] = r['value']

    def pairs_of(metric, tree, v, test):
        return {k: x for k, x in idx.get((metric, tree, v), {}).items() if test(*k)}

    def deltas(metric, test):
        real, ref = pairs_of(metric, 'real', variant, test), pairs_of(metric, 'pnull', variant, test)
        return np.array([real[k] - ref[k] for k in sorted(set(real) & set(ref))])

    c = GROUP_DARK.get('mlp:' + variant, '0.3')
    fig, axes = plt.subplots(1, 3, figsize=(12.0 * 0.82, 3.8 * 0.82), dpi=300,
                             gridspec_kw=dict(width_ratios=[1.2, 1.2, 0.8]))
    for ax, metric, is_delta, ylabel in (
            (axes[0], 'fidelity_across_halves', True, 'Across-domain fidelity r, real $-$ null'),
            (axes[1], 'reliability_across_halves', False, 'Across-domain reliability (ARI)')):
        labels = []
        for t, (cname, test) in enumerate(POOL_TYPES):
            v = deltas(metric, test) if is_delta else \
                np.array(list(pairs_of(metric, 'real', variant, test).values()))
            ceil = pairs_of('fidelity_across_halves', 'real', '(ceiling)', test)
            labels.append('%s\nceiling %.2f' % (cname, np.mean(list(ceil.values()))) if ceil else cname)
            if not len(v):
                continue
            ax.bar(t, v.mean(), width=0.6, color=c, alpha=0.75, edgecolor='black', lw=0.6,
                   zorder=2)
            ax.scatter(np.full(len(v), t) + np.linspace(-0.15, 0.15, len(v)), v, s=10,
                       color='0.15', alpha=0.6, edgecolors='none', zorder=3)
        ax.set_xticks(range(len(POOL_TYPES)))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlim(-0.6, len(POOL_TYPES) - 0.4)
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        style_spines(ax, drop_top_right=True)
        ax.tick_params(axis='y', labelsize=8.5)

    ax = axes[2]
    for d in PROSE:
        one = deltas('fidelity_across_halves', lambda f, e, d=d: f in PROSE and e == d)
        lodo = deltas('fidelity_across_halves', lambda f, e, d=d: f == 'lodo_' + d and e == d)
        if not (len(one) and len(lodo)):
            continue
        ax.plot([0, 1], [one.mean(), lodo.mean()], color=c, lw=1.2, zorder=2)
        ax.scatter([0, 1], [one.mean(), lodo.mean()], s=34, color=c, edgecolors='black',
                   linewidths=0.5, zorder=3)
        ax.text(1.08, lodo.mean(), d, fontsize=7, va='center', color='0.25')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['mean of the\n3 single domains', 'pool of the\n3 domains'], fontsize=7.5)
    ax.set_xlim(-0.35, 1.75)
    ax.set_ylabel('Fidelity r into the held-out domain,\nreal $-$ null', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    style_spines(ax, drop_top_right=True)
    ax.tick_params(axis='y', labelsize=8.5)
    fig.legend(handles=[
        mpl.patches.Patch(facecolor=c, alpha=0.75, edgecolor='black',
                          label=name + ' (final pipeline not run in T4)'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='0.15', alpha=0.6,
               markeredgecolor='none', markersize=4, label='one domain pair'),
    ], loc='lower center', bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(w_pad=1.2)
    save_fig(fig, 'pooled_estimation_single')


YARDSTICK_ARMS = [('vmf_ward100', 'Ward'), ('vmf_lloyd100', 'Lloyd,\nfull profiles'),
                  ('vmf_pca100_lloyd100', 'Lloyd,\nPCA-100')]


def fig_yardstick():
    """How far hard-label reliability is from what the data allow (T3).

    Circle: split-half ARI of the consensus (40 restarts). Diamond: the yardstick, Lloyd on
    one half started from the other half's partition. Square: 200 restarts, where run. Cross:
    the yardstick started from the null partition. The gap from circle to diamond is headroom
    that belongs to the consensus, not to the data.
    """
    rows = load_plain(YARDSTICK_FILE)
    if rows is None:
        return
    fig, ax = plt.subplots(figsize=(4.6 * 0.9, 3.4 * 0.9), dpi=300)
    for i, (variant, _) in enumerate(YARDSTICK_ARMS):
        c = ARM_COLORS.get('mlp:' + variant, '0.3')

        def pick(tree, metric):
            return np.array([r['value'] for r in rows if r['variant'] == variant
                             and r['tree'] == tree and r['metric'] == metric])
        rel, yard, null_yard = pick('real', 'reliability_ari'), pick('real', 'yardstick_ari'), \
            pick('pnull', 'yardstick_ari')
        if not len(yard):
            continue
        ax.vlines(i, rel.mean(), yard.mean(), color=c, alpha=0.45, lw=3.2, zorder=2)
        ax.scatter(np.full(len(yard), i + 0.12), yard, s=8, color=c, alpha=0.4, edgecolors='none', zorder=2)
        ax.scatter(i, rel.mean(), s=46, color=c, edgecolors='black', linewidths=0.5, zorder=3)
        ax.scatter(i, yard.mean(), s=46, marker='D', color=c, edgecolors='black', linewidths=0.5, zorder=3)
        ax.scatter(i, null_yard.mean(), s=40, marker='x', color='0.45', linewidths=1.2, zorder=3)
        more = 'mlp:%s_n200' % variant
        if more in PRESENT:
            ax.scatter(i, mean('reliability_within', 'real', more), s=42, marker='s', color=c,
                       edgecolors='black', linewidths=0.5, zorder=3.5)
    ax.set_xticks(range(len(YARDSTICK_ARMS)))
    ax.set_xticklabels([lab for _, lab in YARDSTICK_ARMS], fontsize=8.5)
    ax.set_xlim(-0.5, len(YARDSTICK_ARMS) - 0.5)
    ax.set_ylim(-0.03, 1.0)
    ax.set_ylabel('Agreement between halves (ARI), k = 100', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
    style_spines(ax, drop_top_right=True)
    ax.tick_params(axis='both', which='major', labelsize=8.5)
    ax.legend(handles=[
        Line2D([0], [0], marker='D', color='none', markerfacecolor='0.3', markeredgecolor='black',
               markersize=6, label='steered from the other half'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor='0.3', markeredgecolor='black',
               markersize=6, label='consensus, 200 restarts'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='0.3', markeredgecolor='black',
               markersize=6, label='consensus, 40 restarts'),
        Line2D([0], [0], marker='x', color='0.45', linestyle='none', markersize=6,
               label='steered, null partition'),
    ], loc='upper left', frameon=False, fontsize=7.5)
    fig.tight_layout()
    save_fig(fig, 'yardstick')


if __name__ == '__main__':
    os.makedirs('plots', exist_ok=True)
    fig_references()
    fig_deltas()
    fig_hubness_tradeoff()
    fig_within_vs_across()
    fig_across_reliability_vs_fidelity()
    fig_across_reliability_vs_fidelity(mlp_only=True)
    fig_across_reliability_vs_fidelity(mlp_only=True, scope='within')
    fig_measures_matrix()
    fig_final_vs_arms()
    fig_final_vs_arms(mlp_only=True)
    fig_original_vs_final()
    fig_maps_vs_labels()
    fig_final_candidates()
    fig_pooled_estimation()
    fig_pooled_estimation_single()
    fig_yardstick()
    print('wrote plots/{arms_references,arms_deltas,hubness_tradeoff,within_vs_across,'
          'across_reliability_vs_fidelity{,_mlp},within_reliability_vs_fidelity_mlp,measures_matrix_mlp,final_vs_arms{,_mlp},original_vs_final,maps_vs_labels,final_candidates,'
          'pooled_estimation,pooled_estimation_single,yardstick}'
          '.{svg,png} from %d arms' % len(ARMS))
