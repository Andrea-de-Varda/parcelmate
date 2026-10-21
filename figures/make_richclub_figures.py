"""Figures of the rich-club stage (LOG.md Iteration 45; plots/rich_club/README.md).

    python figures/make_richclub_figures.py

Reads results/richclub/<model>-resid-pool4-tiles/metrics/richclub/ for Pythia-70m, Qwen3.5-2B
and Qwen3.5-4B; a model whose results are absent gets a "pending" placeholder panel, so the
script can be rerun unchanged once 4B is done. Writes plots/rich_club/*.{svg,png}:

  concept        H2 (hubs anchor separate modules) against H1 (hubs form a club)
  method         cross-half selection, the N1 expectation, rho and c_i, the circular-shift null
  rho_curves     rho(n) for real and null trees, both directions, criterion range shaded
  layer_split    rho on same-layer and on other-layer hub pairs, real against null
  hub_level      every hub's c_i (all pairs, other-layer pairs) at n = 1% of live units
  composition    layers of the top 500 hubs; networks the club spans; hub-set reliability
  participation  participation of club members against the units ranked just below them
"""

import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['mathtext.fontset'] = 'dejavusans'

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
RES = os.path.join(ROOT, 'results', 'richclub')
OUT = os.path.join(ROOT, 'plots', 'rich_club')

MODELS = [('pythia-70m', 'Pythia-70m'), ('qwen3.5-2b', 'Qwen3.5-2B'), ('qwen3.5-4b', 'Qwen3.5-4B')]
COLOR = {'pythia-70m': '#6a51a3', 'qwen3.5-2b': '#2c5f8a', 'qwen3.5-4b': '#1b7f6e'}
NULL = '#9a9a9a'
GRAY_FILL, GRAY_EDGE = '#cccccc', '#999999'


# ------------------------------------------------------------------------------ helpers

def save(fig, stem):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, stem + '.svg'), format='svg', bbox_inches='tight')
    fig.savefig(os.path.join(OUT, stem + '.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def chrome(ax, lw=1.5, grid=True):
    ax.spines[['top', 'right']].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_linewidth(lw)
    if grid:
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
        ax.set_axisbelow(True)


def sci_ticks(val, _=None):
    if val <= 0:
        return ''
    e = int(np.floor(np.log10(val) + 1e-9))
    m = val / 10 ** e
    if abs(m - 1) < 1e-6:
        return r'$10^{%d}$' % e
    return r'$%d{\times}10^{%d}$' % (round(m), e)


def log_x(ax):
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10, subs=(1.0,)))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(sci_ticks))
    ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10, subs=(2.0, 5.0)))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def pending(ax, label):
    # axis('off') rather than clearing ticks: with sharey, clearing ticks clears every panel's
    ax.axis('off')
    ax.add_patch(Rectangle((0.04, 0.04), 0.92, 0.92, transform=ax.transAxes, fill=False,
                           ls='--', lw=1.2, ec='#bbbbbb'))
    ax.text(0.5, 0.5, '%s\nresults pending' % label, transform=ax.transAxes, ha='center', va='center',
            color='#999999', fontsize=10)


def load(model):
    d = os.path.join(RES, '%s-resid-pool4-tiles' % model, 'metrics', 'richclub')
    if not os.path.exists(os.path.join(d, 'verdict.json')):
        return None
    c = pd.read_csv(os.path.join(d, 'curves.csv'), keep_default_na=False, na_values=[''])
    for col in c.columns[4:]:
        c[col] = pd.to_numeric(c[col], errors='coerce')
    p = pd.read_csv(os.path.join(d, 'participation.csv'), keep_default_na=False, na_values=[''])
    for col in p.columns[4:]:
        p[col] = pd.to_numeric(p[col], errors='coerce')
    return dict(curves=c, part=p, verdict=json.load(open(os.path.join(d, 'verdict.json'))),
                units={t: np.load(os.path.join(d, 'units_%s.npz' % t)) for t in ('real', 'null')},
                hubs={t: np.load(os.path.join(d, 'hubs_%s.npz' % t)) for t in ('real', 'null')})


def one_percent(r):
    grid = np.asarray(r['verdict']['grid'])
    return int(grid[np.argmin(np.abs(np.log(grid / (0.01 * r['verdict']['n_live']))))])


def arrow(fig, xy0, xy1, **kw):
    fig.add_artist(FancyArrowPatch(xy0, xy1, transform=fig.transFigure, arrowstyle='-|>',
                                   mutation_scale=12, lw=1.4, color='#444444', **kw))


# ------------------------------------------------------------------------------ schematics

def toy_graph(ax, club, title):
    """Four modules of 9 units around one hub each; with `club`, the hubs couple to each other."""
    rng = np.random.RandomState(3)
    centres = np.array([[-1, 1], [1, 1], [-1, -1], [1, -1]]) * 1.25
    cols = ['#d4b9da', '#c6dbef', '#c7e9c0', '#fdd0a2']
    edge_c = '#bdbdbd'
    for m, (cx, cy) in enumerate(centres):
        ang = np.linspace(0, 2 * np.pi, 9, endpoint=False) + m
        pts = np.c_[cx + 0.62 * np.cos(ang), cy + 0.62 * np.sin(ang)] + rng.randn(9, 2) * 0.05
        for (x, y) in pts:
            ax.plot([cx, x], [cy, y], color=edge_c, lw=0.9, zorder=1)
        for i in range(9):
            j = (i + 1) % 9
            ax.plot(pts[[i, j], 0], pts[[i, j], 1], color=edge_c, lw=0.6, zorder=1)
        ax.scatter(pts[:, 0], pts[:, 1], s=34, color=cols[m], edgecolor='#666666', lw=0.6, zorder=3)
    if club:
        for a in range(4):
            for b in range(a + 1, 4):
                ax.plot(centres[[a, b], 0], centres[[a, b], 1], color='#333333', lw=2.4, zorder=2)
    else:   # one weak link per pair of neighbouring modules, through ordinary units
        for a, b in ((0, 1), (2, 3), (0, 2), (1, 3)):
            mid = (centres[a] + centres[b]) / 2
            ax.plot([centres[a][0] * 0.55 + mid[0] * 0.45, centres[b][0] * 0.55 + mid[0] * 0.45],
                    [centres[a][1] * 0.55 + mid[1] * 0.45, centres[b][1] * 0.55 + mid[1] * 0.45],
                    color=edge_c, lw=0.9, zorder=1)
    ax.scatter(centres[:, 0], centres[:, 1], s=190, color='#333333', edgecolor='white', lw=1.2, zorder=4)
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-2.3, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.text(0, 2.45, title, ha='center', va='bottom', fontsize=10.5, weight='bold')


def fig_concept():
    fig, axes = plt.subplots(1, 2, figsize=(6.4 * .8, 3.4 * .8))
    toy_graph(axes[0], False, 'H2: no rich club')
    toy_graph(axes[1], True, 'H1: rich club')
    axes[0].text(0, -2.45, 'hubs anchor separate modules', ha='center', va='top', fontsize=8.5, color='#555555')
    axes[1].text(0, -2.45, 'hubs couple to each other\nmore than their strength predicts', ha='center',
                 va='top', fontsize=8.5, color='#555555')
    for ax, lab in zip(axes, 'AB'):
        ax.text(-2.3, 2.45, lab, fontsize=16, weight='bold', va='bottom')
    fig.legend(handles=[Line2D([0], [0], marker='o', color='none', markerfacecolor='#333333', markersize=9,
                               label='hub (high strength)'),
                        Line2D([0], [0], marker='o', color='none', markerfacecolor='#c6dbef',
                               markeredgecolor='#666666', markersize=6, label='other unit')],
               loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=8.5)
    save(fig, 'concept')


def club_matrix(seed, n=40, hubs=8, strength=0.35):
    rng = np.random.RandomState(seed)
    W = np.abs(rng.randn(n, n)) * 0.12
    W[:hubs, :hubs] += strength
    W[:hubs, hubs:] += 0.08
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)
    return W


def fig_method():
    fig = plt.figure(figsize=(9.6 * .85, 6.2 * .85))
    Wa, Wb = club_matrix(1), club_matrix(2)
    perm = np.random.RandomState(0).permutation(40)
    cm = mpl.colormaps['Blues']

    # Row 1: half A -> ranking -> half B, hubs only, against N1
    axA = fig.add_axes([0.02, 0.56, 0.2, 0.31])
    axA.imshow(Wa[np.ix_(perm, perm)], cmap=cm, vmin=0, vmax=0.5)
    axA.set_xticks([]); axA.set_yticks([])
    axA.set_title('Half A connectome', fontsize=10, weight='bold')
    axA.text(0.5, -0.06, '$w_{ij}=|r_{ij}|$, Fisher mean of\nsample 1 of the four datasets',
             transform=axA.transAxes, ha='center', va='top', fontsize=8)

    axS = fig.add_axes([0.3, 0.56, 0.17, 0.31])
    s = Wa.sum(1)
    order = np.argsort(-s)
    cols = ['#333333' if i < 8 else GRAY_FILL for i in range(40)]
    axS.barh(np.arange(40), s[order], color=cols, height=0.85)
    axS.invert_yaxis()
    axS.set_yticks([])
    axS.set_xticks([])
    chrome(axS, grid=False)
    axS.spines['left'].set_visible(False)
    axS.set_title('Rank by strength', fontsize=10, weight='bold')
    axS.set_xlabel(r'$s_i=\Sigma_j\, w_{ij}$', fontsize=9.5)
    axS.text(s.max() * 1.02, 3.5, '$R_A(n)$:\ntop $n$', fontsize=8.5, va='center')

    axB = fig.add_axes([0.56, 0.56, 0.2, 0.31])
    hub = order[:8]
    sb = Wb.sum(1)
    E = np.outer(sb, sb) / sb.sum()
    np.fill_diagonal(E, 0)
    axB.imshow(Wb[np.ix_(hub, hub)], cmap=cm, vmin=0, vmax=0.5)
    axB.set_xticks([]); axB.set_yticks([])
    axB.set_title('Half B, hubs only', fontsize=10, weight='bold')
    axB.text(0.5, -0.06, 'observed $w^{B}_{ij}$', transform=axB.transAxes, ha='center', va='top', fontsize=8.5)
    axE = fig.add_axes([0.79, 0.56, 0.2, 0.31])
    axE.imshow(E[np.ix_(hub, hub)], cmap=cm, vmin=0, vmax=0.5)
    axE.set_xticks([]); axE.set_yticks([])
    axE.set_title('N1 expectation', fontsize=10, weight='bold')
    axE.text(0.5, -0.06, '$s_i s_j / 2W$  (same strengths,\npartners at random)', transform=axE.transAxes,
             ha='center', va='top', fontsize=8.5)
    arrow(fig, (0.225, 0.715), (0.295, 0.715))
    arrow(fig, (0.5, 0.715), (0.555, 0.715))
    fig.text(0.775, 0.715, 'vs', ha='center', va='center', fontsize=10)

    # Row 2: the statistics, and the circular-shift null
    fig.text(0.04, 0.37, 'Statistics for each club size $n$', fontsize=10, weight='bold')
    fig.text(0.04, 0.33, 'hubs selected on one half, evaluated on the other, both directions',
             fontsize=8.5, color='#555555')
    fig.text(0.05, 0.22,
             r'club:   $\rho(n)=\dfrac{\Sigma_{\,i<j\in R}\;w^{B}_{ij}}{\Sigma_{\,i<j\in R}\;s_i s_j/2W}$',
             fontsize=12)
    fig.text(0.05, 0.09,
             r'hub:    $c_i(n)=\dfrac{\Sigma_{\,j\in R,\,j\neq i}\;w^{B}_{ij}}{\Sigma_{\,j\in R,\,j\neq i}\;s_i s_j/2W}$',
             fontsize=12)
    fig.text(0.05, 0.03, r'$\rho, c_i>1$: hubs couple to each other beyond their strength', fontsize=8.5, color='#555555')
    fig.text(0.05, -0.01, r'$2W=\Sigma_i\, s_i$, strengths of the evaluation half', fontsize=8.5, color='#555555')

    axN = fig.add_axes([0.6, 0.0, 0.33, 0.25])
    t = np.arange(300)
    rng = np.random.RandomState(4)
    base = np.convolve(rng.randn(330), np.ones(25) / 5, 'same')[:300]
    x1 = base + 0.4 * rng.randn(300)
    x2 = base + 0.4 * rng.randn(300)
    axN.plot(t, x1 + 3.2, color='#2c5f8a', lw=1)
    axN.plot(t, x2 + 1.6, color='#2c5f8a', lw=1)
    axN.plot(t, np.roll(x2, 110) - 0.4, color=NULL, lw=1)
    axN.annotate('', xy=(110, -3.0), xytext=(0, -3.0), arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2))
    axN.text(116, -3.0, 'random circular shift', ha='left', va='center', fontsize=8, color='#555555')
    axN.text(305, 3.2, 'unit $i$', va='center', fontsize=8.5)
    axN.text(305, 1.6, 'unit $j$', va='center', fontsize=8.5)
    axN.text(305, -0.4, 'unit $j$, shifted', va='center', fontsize=8.5, color='#777777')
    axN.set_xlim(0, 300)
    axN.set_ylim(-3.6, 5)
    axN.axis('off')
    fig.text(0.6, 0.37, 'N3: circular-shift null', fontsize=10, weight='bold')
    fig.text(0.6, 0.33, "keeps each unit's mean, variance and autocorrelation; destroys\nco-activation. The whole analysis is repeated on this tree.",
             fontsize=8.5, color='#555555', va='top')
    fig.text(0.0, 0.93, 'A', fontsize=16, weight='bold')
    fig.text(0.0, 0.365, 'B', fontsize=16, weight='bold')
    fig.text(0.565, 0.365, 'C', fontsize=16, weight='bold')
    save(fig, 'method')


# ------------------------------------------------------------------------------ results

def criterion_span(ax, r, key, color):
    v = r['verdict'][key]['strength']
    if v['n_range']:
        ax.axvspan(v['n_range'][0], v['n_range'][1], color=color, alpha=0.08, zorder=0, lw=0)


def fig_rho(results):
    fig, axes = plt.subplots(1, 3, figsize=(9.6 * .8, 3.0 * .8), sharey=True)
    for ax, (m, lab) in zip(axes, MODELS):
        r = results[m]
        if r is None:
            pending(ax, lab)
            continue
        chrome(ax)
        c = r['curves']
        c = c[c.ranking == 'strength']
        for tree, col in (('real', COLOR[m]), ('null', NULL)):
            for d, ls in (('A->B', '-'), ('B->A', '--')):
                s = c[(c.tree == tree) & (c.direction == d)].sort_values('n')
                ax.plot(s.n, s.rho, ls=ls, color=col, lw=1.8, zorder=3)
        ax.axhline(1, color='black', lw=0.9, zorder=2)
        criterion_span(ax, r, 'rankings', COLOR[m])
        log_x(ax)
        ax.set_title(lab, fontsize=10.5, weight='bold')
        ax.text(0.97, 0.95, '%s live units' % format(r['verdict']['n_live'], ','), transform=ax.transAxes,
                ha='right', va='top', fontsize=8, color='#555555')
        ax.set_xlabel('Club size $n$', fontsize=11)
    axes[0].set_ylabel(r'$\rho(n)$', fontsize=12)
    axes[0].set_ylim(0.9, None)
    fig.legend(handles=[Line2D([0], [0], color=COLOR['qwen3.5-2b'], lw=1.8, label='real (model colour)'),
                        Line2D([0], [0], color=NULL, lw=1.8, label='circular-shift null'),
                        Line2D([0], [0], color='#444444', lw=1.8, ls='-', label='select A, evaluate B'),
                        Line2D([0], [0], color='#444444', lw=1.8, ls='--', label='select B, evaluate A'),
                        Rectangle((0, 0), 1, 1, color='#444444', alpha=0.12, label='criterion holds')],
               loc='lower center', bbox_to_anchor=(0.5, -0.17), ncol=5, frameon=False, fontsize=8.5)
    save(fig, 'rho_curves')


def mean_dir(c, tree, col):
    s = c[c.tree == tree].groupby('n')[col].mean()
    return s.index.values, s.values


def fig_layer_split(results):
    fig, axes = plt.subplots(1, 3, figsize=(9.6 * .8, 3.0 * .8), sharey=True)
    for ax, (m, lab) in zip(axes, MODELS):
        r = results[m]
        if r is None:
            pending(ax, lab)
            continue
        chrome(ax)
        c = r['curves']
        c = c[c.ranking == 'strength']
        for tree, col in (('real', COLOR[m]), ('null', NULL)):
            for key, ls, mk in (('rho_same', '-', 'o'), ('rho_other', '-', 's')):
                n, y = mean_dir(c, tree, key)
                ok = np.isfinite(y)
                ax.plot(n[ok], y[ok], ls=ls, marker=mk, ms=3.2, mfc='white' if key == 'rho_other' else col,
                        color=col, lw=1.6, zorder=3)
        ax.axhline(1, color='black', lw=0.9, zorder=2)
        criterion_span(ax, r, 'rankings_other_layers', COLOR[m])
        log_x(ax)
        ax.set_title(lab, fontsize=10.5, weight='bold')
        ax.set_xlabel('Club size $n$', fontsize=11)
    axes[0].set_ylabel(r'$\rho$ by hub pair', fontsize=12)
    fig.legend(handles=[Line2D([0], [0], color='#444444', marker='o', ms=4, lw=1.6, label='same-layer pairs'),
                        Line2D([0], [0], color='#444444', marker='s', ms=4, mfc='white', lw=1.6,
                               label='other-layer pairs'),
                        Line2D([0], [0], color=NULL, lw=1.6, label='circular-shift null'),
                        Rectangle((0, 0), 1, 1, color='#444444', alpha=0.12,
                                  label='criterion holds on other-layer pairs')],
               loc='lower center', bbox_to_anchor=(0.5, -0.17), ncol=4, frameon=False, fontsize=8.5)
    save(fig, 'layer_split')


def fig_hub_level(results):
    fig, axes = plt.subplots(1, 3, figsize=(8.4 * .8, 3.0 * .8), sharey=True)
    rng = np.random.RandomState(0)
    for ax, (m, lab) in zip(axes, MODELS):
        r = results[m]
        if r is None:
            pending(ax, lab)
            continue
        chrome(ax)
        n = one_percent(r)
        g = int(np.nonzero(np.asarray(r['verdict']['grid']) == n)[0][0])
        groups = []
        for key in ('c', 'c_other'):
            for tree in ('real', 'null'):
                v = np.concatenate([r['hubs'][tree]['%s_%s_strength' % (key, d)][:n, g] for d in ('AB', 'BA')])
                groups.append(np.log2(v[np.isfinite(v) & (v > 0)]))
        pos = [0, 0.9, 2.3, 3.2]
        for i, (x, v) in enumerate(zip(pos, groups)):
            real = i % 2 == 0
            col = COLOR[m] if real else NULL
            shown = v if len(v) <= 300 else rng.choice(v, 300, replace=False)   # display only; boxes use all
            ax.scatter(x + rng.uniform(-0.22, 0.22, len(shown)), shown, s=3, color=col, alpha=0.35 if real else 0.4,
                       lw=0, zorder=2)
            bp = ax.boxplot(v, positions=[x], widths=0.55, showfliers=False, patch_artist=True, zorder=3)
            for b in bp['boxes']:
                b.set(facecolor='none', edgecolor='black', lw=1.2)
            for k in ('whiskers', 'caps'):
                for w in bp[k]:
                    w.set(color='black', lw=1.1)
            for med in bp['medians']:
                med.set(color='black', lw=1.8)
        ax.axhline(0, color='black', lw=0.9, zorder=1)
        ax.set_xticks([0.45, 2.75])
        ax.set_xticklabels(['all pairs', 'other-layer\npairs'], fontsize=9)
        ax.set_title('%s\n$n$ = %s (1%% of units)' % (lab, format(n, ',')), fontsize=10, weight='bold')
    axes[0].set_ylabel(r'$\log_2\,c_i$', fontsize=12)
    fig.legend(handles=[Line2D([0], [0], marker='o', color='none', markerfacecolor='#555555', markersize=5,
                               label='real hubs (colour)'),
                        Line2D([0], [0], marker='o', color='none', markerfacecolor=NULL, markersize=5,
                               label='null hubs (gray)')],
               loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=8.5)
    save(fig, 'hub_level')


def fig_composition(results):
    fig = plt.figure(figsize=(9.6 * .8, 5.6 * .8))
    gs = fig.add_gridspec(2, 3, hspace=0.6, wspace=0.38)
    for i, (m, lab) in enumerate(MODELS):
        ax = fig.add_subplot(gs[0, i])
        r = results[m]
        if r is None:
            pending(ax, lab)
            continue
        chrome(ax)
        L = int(r['units']['real']['coordinates'][:, 0].max()) + 1
        for tree, col, off in (('real', COLOR[m], -0.2), ('null', NULL, 0.2)):
            u = r['units'][tree]
            lay = u['coordinates'][:, 0]
            o = u['order_strength_halfA'][:500]
            h = np.bincount(lay[o], minlength=L) / 500.0
            ax.bar(np.arange(L) + off, h, width=0.4, color=col, edgecolor='black' if tree == 'real' else GRAY_EDGE,
                   lw=0.5, zorder=3)
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_title(lab, fontsize=10.5, weight='bold')
        ax.set_xticks(np.arange(0, L, 4 if L > 12 else 1))
    fig.axes[0].set_ylabel('Share of top 500', fontsize=11)

    ax = fig.add_subplot(gs[1, 0:2])
    chrome(ax)
    ax2 = fig.add_subplot(gs[1, 2])
    chrome(ax2)
    for m, lab in MODELS:
        r = results[m]
        if r is None:
            continue
        p = r['part']
        p = p[(p.graph == 'weighted')]
        for tree, ls in (('real', '-'), ('null', '--')):
            s = p[p.tree == tree].groupby('n').networks_effective.mean()
            ax.plot(s.index / r['verdict']['n_live'], s.values, ls=ls, color=COLOR[m] if tree == 'real' else NULL,
                    lw=1.8, zorder=3)
        c = r['curves']
        s = c[(c.tree == 'real') & (c.ranking == 'strength')].groupby('n')[['jaccard_AB', 'jaccard_random']].mean()
        ax2.plot(s.index / r['verdict']['n_live'], s.jaccard_AB, color=COLOR[m], lw=1.8, zorder=3)
        ax2.plot(s.index / r['verdict']['n_live'], s.jaccard_random, color=COLOR[m], lw=1.0, ls=':', zorder=3)
    for a in (ax, ax2):
        log_x(a)
        a.set_xlabel('Club size $n$ / live units', fontsize=10)
    ax.set_ylabel('Networks spanned (eff.)', fontsize=10)
    ax2.set_ylabel('Jaccard, A vs B hubs', fontsize=10)
    ax2.set_ylim(0, 1.02)
    handles = [Line2D([0], [0], color=COLOR[m], lw=1.8, label=lab) for m, lab in MODELS if results[m] is not None]
    handles += [Line2D([0], [0], color=NULL, lw=1.8, ls='--', label='null (each model)'),
                Line2D([0], [0], color='#555555', lw=1.0, ls=':', label='random sets')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.06), ncol=5, frameon=False, fontsize=8.5)
    fig.text(0.07, 0.93, 'A', fontsize=16, weight='bold')
    fig.text(0.07, 0.45, 'B', fontsize=16, weight='bold')
    fig.text(0.675, 0.45, 'C', fontsize=16, weight='bold')
    save(fig, 'composition')


def fig_participation(results):
    graphs = [('top1%', 'top 1% graph'), ('top2%', 'top 2%'), ('top10%', 'top 10%'), ('weighted', 'weighted')]
    rows = []
    for m, lab in MODELS:
        for gk, glab in graphs:
            rows.append((m, lab, gk, glab))
    fig, ax = plt.subplots(figsize=(5.6 * .75, 4.4 * .75))
    ax.spines[['top', 'right']].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_linewidth(1.5)
    ax.grid(axis='x', linestyle='--', alpha=0.5, zorder=1)
    y = 0
    ticks, labels = [], []
    for m, lab in MODELS:
        r = results[m]
        for gk, glab in graphs:
            ticks.append(y)
            labels.append('%s  %s' % (lab if gk == 'top1%' else '', glab))
            if r is not None:
                n = one_percent(r)
                p = r['part']
                q = p[(p.tree == 'real') & (p.graph == gk) & (p.n == n)]
                club, below, live = q.P_club_mean.mean(), q.P_below_mean.mean(), q.P_live_median.mean()
                ax.plot([below, club], [y, y], color=COLOR[m], lw=2, zorder=2)
                ax.scatter([club], [y], s=36, color=COLOR[m], edgecolor='black', lw=0.8, zorder=4)
                ax.scatter([below], [y], s=36, color='white', edgecolor=COLOR[m], lw=1.4, zorder=4)
                ax.scatter([live], [y], marker='|', s=70, color='#555555', lw=1.4, zorder=3)
            else:
                ax.text(0.5, y, 'pending', transform=ax.get_yaxis_transform(), fontsize=8, color='#999999',
                        va='center', ha='center')
            y -= 1
        y -= 0.6
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel('Participation coefficient', fontsize=11)
    ax.legend(handles=[Line2D([0], [0], marker='o', color='none', markerfacecolor='#555555', markeredgecolor='black',
                              markersize=6, label='club (top 1% of units)'),
                       Line2D([0], [0], marker='o', color='none', markerfacecolor='white', markeredgecolor='#555555',
                              markersize=6, label='ranked just below'),
                       Line2D([0], [0], marker='|', color='#555555', markersize=8, lw=0, label='median unit')],
              loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8.5)
    save(fig, 'participation')


def main():
    results = {m: load(m) for m, _ in MODELS}
    fig_concept()
    fig_method()
    fig_rho(results)
    fig_layer_split(results)
    fig_hub_level(results)
    fig_composition(results)
    fig_participation(results)
    print('wrote', OUT, '; models:', ', '.join('%s %s' % (m, 'ok' if r else 'pending') for m, r in results.items()))


if __name__ == '__main__':
    main()
