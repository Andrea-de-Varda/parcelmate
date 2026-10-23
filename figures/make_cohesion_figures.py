"""Figures for plots/circuit_cohesion/README.md (LOG.md Iteration 46).

    python figures/make_cohesion_figures.py --model 2b

Reads figures/circuit_cohesion_<model>.csv (parcelmate.bin.circuit_cohesion), the coupling files
results/richclub/qwen3.5-<model>-resid-pool4-tiles{,_null}/metrics/coupling/coupling.npz, the
concentration scores figures/bank_concentration_<model>_resid_pool4.csv and the task labels.
Writes plots/circuit_cohesion/{concept,coupling,method,domains,controls}_<model>.{svg,png}.
"""

import argparse
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from parcelmate.cohesion import network_layers, normalized_coupling, outside_reference  # noqa: E402
from parcelmate.circuits import circuit_indices  # noqa: E402
from make_bank_concentration_figure import COLORS  # noqa: E402

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'DejaVu Sans'
OUT = os.path.join(ROOT, 'plots', 'circuit_cohesion')
GRAY_FILL, GRAY_EDGE = '#cccccc', '#999999'
ORDER = ['Syntax', 'Morphology & lexical semantics', 'Arithmetic', 'Logic, code & multiple choice', 'Factual knowledge',
         'Entity tracking & binding', 'In-context string tasks', 'Physical reasoning', 'Social reasoning']
TICK = {'Syntax': 'Syntax', 'Morphology & lexical semantics': 'Morph. &\nlexical', 'Arithmetic': 'Arith-\nmetic',
        'Logic, code & multiple choice': 'Logic,\ncode, MC', 'Factual knowledge': 'Factual',
        'Entity tracking & binding': 'Entity\ntracking', 'In-context string tasks': 'In-context\nstring',
        'Physical reasoning': 'Phys.', 'Social reasoning': 'Social'}
SHORT = {'Morphology & lexical semantics': 'Morphology &\nlexical', 'Logic, code & multiple choice': 'Logic, code\n& MC',
         'Factual knowledge': 'Factual', 'Entity tracking & binding': 'Entity tracking\n& binding',
         'In-context string tasks': 'In-context\nstring', 'Physical reasoning': 'Physical', 'Social reasoning': 'Social'}


def style(ax, grid='y'):
    ax.spines[['top', 'right']].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_linewidth(1.5)
    if grid:
        ax.grid(axis=grid, linestyle='--', alpha=0.5, zorder=0)
    ax.tick_params(labelsize=8)


def letter(ax, s, x=-0.14, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=13, weight='bold', va='bottom')


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + '.svg'), format='svg', bbox_inches='tight')
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('wrote plots/circuit_cohesion/%s.{svg,png}' % name)


# ---------------------------------------------------------------------------- data

def load(model):
    d = pd.read_csv(os.path.join(HERE, 'circuit_cohesion_%s.csv' % model), keep_default_na=False, na_values=['', 'nan', 'NaN'])
    lab = pd.read_csv(os.path.join(HERE, 'bank_task_labels.csv'))
    d = d.merge(lab[['group', 'task', 'name', 'domain']], on=['group', 'task'], validate='many_to_one')
    g = d.groupby(['tree', 'group', 'task', 'name', 'domain']).agg(
        C=('cohesion', 'mean'), R=('preference', 'mean'), Ru=('preference_unmatched', 'mean'),
        spread=('depth_spread', 'mean'), eff=('effective_networks', 'mean'),
        between=('median_between_coupling', 'mean')).reset_index()
    return d, g


# ---------------------------------------------------------------------------- figure 1

def concept():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), dpi=300)
    centers = [(0.2, 0.62), (0.5, 0.25), (0.8, 0.62)]
    rng = np.random.RandomState(3)
    for ax, title, strong in ((axes[0], 'A.  Several distinct networks', False),
                              (axes[1], 'B.  One larger network, cut in three', True)):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal'); ax.axis('off')
        if strong:
            ax.add_patch(FancyBboxPatch((0.04, 0.05), 0.92, 0.85, boxstyle='round,pad=0.02', fc='none',
                                        ec='#555555', lw=1.2, ls='--'))
        pts = []
        for (cx, cy) in centers:
            ax.add_patch(Circle((cx, cy), 0.14, fc='#f2f2f2', ec='#888888', lw=1.0))
            p = np.c_[cx + rng.uniform(-0.09, 0.09, 6), cy + rng.uniform(-0.09, 0.09, 6)]
            pts.append(p)
        for a in range(3):
            for b in range(a + 1, 3):
                n_links = 10 if strong else 1
                for _ in range(n_links):
                    i, j = rng.randint(6), rng.randint(6)
                    ax.plot([pts[a][i, 0], pts[b][j, 0]], [pts[a][i, 1], pts[b][j, 1]], color='#777777',
                            lw=0.8 if strong else 0.6, alpha=0.6 if strong else 0.5, zorder=1)
        for p in pts:
            for i in range(6):
                for j in range(i + 1, 6):
                    ax.plot(p[[i, j], 0], p[[i, j], 1], color='#777777', lw=0.6, alpha=0.5, zorder=1)
            ax.scatter(p[:, 0], p[:, 1], s=14, c='#bbbbbb', edgecolors='#666666', linewidths=0.4, zorder=2)
            ax.scatter(p[:2, 0], p[:2, 1], s=20, c=COLORS['Syntax'], edgecolors='black', linewidths=0.5, zorder=3)
        ax.text(0.5, 1.02, title, transform=ax.transAxes, ha='center', va='bottom', fontsize=9, weight='bold')
    fig.text(0.5, -0.02, 'red = circuit units; gray links = coupling between units', ha='center', fontsize=7.5, color='#555555')
    save(fig, 'concept')


# ---------------------------------------------------------------------------- figure 2

def coupling_fig(model):
    roots = {t: os.path.join(ROOT, 'results', 'richclub', 'qwen3.5-%s-resid-pool4-tiles%s' % (model, s), 'metrics', 'coupling',
                             'coupling.npz') for t, s in (('real', ''), ('null', '_null'))}
    fig = plt.figure(figsize=(7.2, 2.6), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1, 1], wspace=0.75)
    ax0, ax1, ax2 = (fig.add_subplot(gs[i]) for i in range(3))
    vals = {}
    for tree, path in roots.items():
        z = np.load(path)
        lab, co, M = z['labels_halfA'], z['coordinates'], z['M_halfA_on_halfB']
        k = M.shape[0]
        Bn = normalized_coupling(M)
        c = network_layers(lab, co, k)
        iu = np.triu_indices(k, 1)
        vals[tree] = (Bn, c, iu)
    Bn, c, iu = vals['real']
    order = np.argsort(c)
    im = ax0.imshow(Bn[np.ix_(order, order)], cmap='viridis', vmin=0.2, vmax=1.0, interpolation='nearest')
    ax0.set_xticks([]); ax0.set_yticks([])
    ax0.set_xlabel('network (by mean layer)', fontsize=8); ax0.set_ylabel('network (by mean layer)', fontsize=8)
    cb = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=7)
    cb.set_label('coupling', fontsize=7.5)
    letter(ax0, 'A', x=-0.3)
    # B: coupling against layer distance, real tree
    dl = np.abs(c[:, None] - c[None, :])[iu]
    b = Bn[iu]
    ax1.scatter(dl, b, s=2, color='#8a8a8a', alpha=0.25, lw=0, rasterized=True)
    edges = np.arange(0, np.ceil(dl.max()) + 2, 2)
    mid = (edges[:-1] + edges[1:]) / 2
    mean = [b[(dl >= lo) & (dl < hi)].mean() if ((dl >= lo) & (dl < hi)).any() else np.nan for lo, hi in zip(edges[:-1], edges[1:])]
    ax1.plot(mid, mean, color='black', lw=2, marker='o', ms=3)
    from scipy.stats import spearmanr
    rho = spearmanr(dl, b).correlation
    ax1.text(0.95, 0.95, 'ρ = %.2f' % rho, transform=ax1.transAxes, ha='right', va='top', fontsize=7.5,
             bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))
    ax1.set_xlabel('layer distance (mean layers)', fontsize=8); ax1.set_ylabel('normalised coupling', fontsize=8)
    ax1.set_ylim(0, 1.05)
    style(ax1)
    letter(ax1, 'B')
    # C: distribution, real vs null tree
    for tree, col in (('real', '#2b2b2b'), ('null', GRAY_EDGE)):
        Bn_, c_, iu_ = vals[tree]
        x = Bn_[iu_]
        ax2.hist(x, bins=np.linspace(0, 1.05, 43), color=col, alpha=0.75 if tree == 'real' else 0.6,
                 label='real (median %.2f)' % np.median(x) if tree == 'real' else 'null partition\n(median %.2f)' % np.median(x))
    ax2.axvline(1.0, color='black', lw=1, ls='--')
    ax2.text(0.97, ax2.get_ylim()[1] * 0.55, 'one\nnetwork', ha='right', va='center', fontsize=6.5)
    ax2.set_xlabel('coupling', fontsize=8); ax2.set_ylabel('network pairs', fontsize=8)
    ax2.legend(frameon=False, fontsize=6.3, loc='upper left', bbox_to_anchor=(-0.02, 1.0))
    style(ax2)
    letter(ax2, 'C')
    save(fig, 'coupling_%s' % model)


# ---------------------------------------------------------------------------- figure 3

def method(model, group='causalgym', task='filler_gap_pp', name='Filler-gap, PP gap'):
    """How the preference ratio is computed, on one task (half-A networks, half-B coupling)."""
    z = np.load(os.path.join(ROOT, 'results', 'richclub', 'qwen3.5-%s-resid-pool4-tiles' % model, 'metrics', 'coupling', 'coupling.npz'))
    lab, co, M = z['labels_halfA'], z['coordinates'], z['M_halfA_on_halfB']
    k = M.shape[0]
    Bn = normalized_coupling(M)
    depth = network_layers(lab, co, k)
    a = np.load(os.path.join(ROOT, 'results', 'patching_bank', 'Qwen_Qwen3-5-%s' % model.upper(), group, task, 'neuron_attribution.npy'))
    n_layers, width = a.shape
    flat = co[:, 0].astype(np.int64) * width + co[:, 1]
    unit_net = np.empty(n_layers * width, dtype=np.int64); unit_net[flat] = lab
    p = np.bincount(unit_net[circuit_indices(a, 0.1)], minlength=k) / 147.0
    O = outside_reference(p, Bn, depth, 3)
    inside = np.flatnonzero(p > 0); outside = np.flatnonzero(p == 0)
    col = COLORS['Syntax']
    fig = plt.figure(figsize=(7.2, 2.7), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=0.32)
    ax0, ax1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    # A: networks along depth; one pair of circuit networks and the outsiders it is compared with
    rng = np.random.RandomState(4)
    yj = rng.uniform(-1, 1, k)
    ax0.scatter(depth[outside], yj[outside], s=12, color=GRAY_FILL, edgecolors=GRAY_EDGE, linewidths=0.4, zorder=2)
    ax0.scatter(depth[inside], yj[inside], s=12 + 600 * p[inside], color=col, edgecolors='black', linewidths=0.4, zorder=3)
    big = inside[np.argsort(-p[inside])]
    i, j = big[0], big[1]
    near = outside[np.argsort(np.abs(depth[outside] - depth[j]), kind='stable')[:3]]
    ax0.plot([depth[i], depth[j]], [yj[i], yj[j]], color=col, lw=2.2, zorder=2.5)
    for m in near:
        ax0.plot([depth[i], depth[m]], [yj[i], yj[m]], color='#555555', lw=1.0, ls='--', zorder=1.5)
        ax0.scatter(depth[m], yj[m], s=22, color='white', edgecolors='#333333', linewidths=1.0, zorder=3)
    ax0.annotate('i', (depth[i], yj[i]), xytext=(-12, -4), textcoords='offset points', fontsize=8, weight='bold')
    ax0.annotate('j', (depth[j], yj[j]), xytext=(5, 5), textcoords='offset points', fontsize=8, weight='bold')
    ax0.set_yticks([]); ax0.spines['left'].set_visible(False)
    ax0.set_xlabel('network depth (mean layer of its neurons)', fontsize=8)
    ax0.legend(handles=[Line2D([0], [0], marker='o', color='none', markerfacecolor=col, markeredgecolor='black', markersize=6,
                               label="circuit's networks (size = share)"),
                        Line2D([0], [0], marker='o', color='none', markerfacecolor=GRAY_FILL, markeredgecolor=GRAY_EDGE, markersize=5,
                               label='networks outside the circuit'),
                        Line2D([0], [0], color=col, lw=2.2, label='coupling of the pair i-j'),
                        Line2D([0], [0], color='#555555', lw=1.0, ls='--', marker='o', markerfacecolor='white',
                               markeredgecolor='#333333', markersize=4.5, label="i to the 3 outsiders nearest j's depth")],
              frameon=False, fontsize=6.3, loc='upper center', bbox_to_anchor=(0.55, 1.24), ncol=2)
    style(ax0, grid=None)
    letter(ax0, 'A', x=-0.06, y=1.2)
    # B: every pair of circuit networks: its coupling against its comparison
    ii, jj = np.meshgrid(inside, inside, indexing='ij')
    mask = ii != jj
    x, y, w = O[ii, jj][mask], Bn[ii, jj][mask], (p[ii] * p[jj])[mask]
    ax1.scatter(x, y, s=4 + 3000 * w, color=col, alpha=0.55, edgecolors='black', linewidths=0.2, zorder=3)
    lo, hi = min(x.min(), y.min()) - 0.03, max(x.max(), y.max()) + 0.03
    ax1.plot([lo, hi], [lo, hi], color='black', lw=0.9, ls='--', zorder=2)
    R = (w * y).sum() / (w * x).sum()
    ax1.text(0.04, 0.96, 'ratio = %.2f' % R, transform=ax1.transAxes, ha='left', va='top', fontsize=7.5,
             bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))
    ax1.text(0.96, 0.04, 'above the line:\nthe pair is more coupled\nthan i to outsiders', transform=ax1.transAxes,
             ha='right', va='bottom', fontsize=6.2, color='#444444')
    ax1.set_xlim(lo, hi); ax1.set_ylim(lo, hi)
    ax1.set_xlabel('i to depth-matched outsiders', fontsize=8); ax1.set_ylabel('i to j (both in the circuit)', fontsize=8)
    style(ax1, grid='both')
    letter(ax1, 'B', x=-0.22, y=1.2)
    fig.text(0.5, -0.06, '%s, half-A networks on half-B coupling. B: every ordered pair of the circuit\'s networks (size = weight).' % name,
             ha='center', fontsize=7, color='#444444')
    save(fig, 'method_%s' % model)


# ---------------------------------------------------------------------------- figure 4

def domains_fig(g, model):
    from scipy.stats import wilcoxon
    r = g[g.tree == 'real'].set_index(['group', 'task'])
    n = g[g.tree == 'null'].set_index(['group', 'task'])
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=300)
    x0, ticks, tlabels = 0, [], []
    for d in ORDER:
        sub = r[r.domain == d].sort_values('R')
        xs = x0 + np.arange(len(sub))
        for (key, t), x in zip(sub.iterrows(), xs):
            up = t.R > 1
            c = COLORS[d] if up else GRAY_EDGE
            ax.plot([x, x], [1, t.R], color=c, lw=0.9, alpha=0.6, zorder=2)
            ax.scatter(x, n.loc[key].R, s=8, facecolors='white', edgecolors=GRAY_EDGE, linewidths=0.6, zorder=3)
            ax.scatter(x, t.R, s=11, facecolors=c if up else GRAY_FILL, edgecolors='black' if up else GRAY_EDGE,
                       linewidths=0.4, zorder=4)
        k = int((sub.R > 1).sum())
        pw = wilcoxon(sub.R - 1).pvalue if len(sub) >= 6 else np.nan
        ptxt = '' if np.isnan(pw) else ('\np < .001' if pw < 1e-3 else '\np = %.2g' % pw)
        ax.text(xs.mean(), 0.99, '%d/%d%s' % (k, len(sub), ptxt), transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=5.8, color=COLORS[d] if k > len(sub) / 2 else '#777777', linespacing=1.0)
        ticks.append(xs.mean()); tlabels.append(TICK[d] + '\n(%d)' % len(sub))
        x0 = xs[-1] + 7
    ax.axhline(1, color='#555555', lw=1, ls='--', zorder=1)
    ax.set_ylim(0.85, 1.42)
    ax.set_xticks(ticks); ax.set_xticklabels(tlabels, fontsize=6.4)
    ax.set_ylabel('preference ratio', fontsize=8.5)
    style(ax)
    handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor='#555555', markeredgecolor='black', markersize=4.5,
                      label='circuit, real networks (above 1)'),
               Line2D([0], [0], marker='o', color='none', markerfacecolor=GRAY_FILL, markeredgecolor=GRAY_EDGE, markersize=4.5,
                      label='circuit, real networks (below 1)'),
               Line2D([0], [0], marker='o', color='none', markerfacecolor='white', markeredgecolor=GRAY_EDGE, markersize=4.5,
                      label='same circuit, null partition')]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.53, 1.06))
    save(fig, 'domains_%s' % model)


# ---------------------------------------------------------------------------- figure 5

def controls(g, model):
    from scipy.stats import spearmanr
    r = g[g.tree == 'real']
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), dpi=300, gridspec_kw=dict(wspace=0.35))
    ax = axes[0]
    ax.scatter(r.spread, r.Ru, s=8, color=GRAY_EDGE, alpha=0.8, lw=0, zorder=3)
    for d in ORDER:
        s_ = r[r.domain == d]
        ax.scatter(s_.spread, s_.R, s=9, color=COLORS[d], edgecolors='black', linewidths=0.25, alpha=0.85, zorder=4)
    ax.axhline(1, color='black', lw=0.8, ls='--')
    ax.text(0.97, 0.97, 'all outsiders: ρ = %.2f\ndepth-matched: ρ = %.2f' % (spearmanr(r.spread, r.Ru).correlation,
                                                                          spearmanr(r.spread, r.R).correlation),
            transform=ax.transAxes, ha='right', va='top', fontsize=6.8,
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))
    ax.legend(handles=[Line2D([0], [0], marker='o', color='none', markerfacecolor=GRAY_EDGE, markersize=4,
                              label='compared with all outsiders'),
                       Line2D([0], [0], marker='o', color='none', markerfacecolor='#555555', markeredgecolor='black',
                              markersize=4, label='compared with depth-matched outsiders')],
              frameon=False, fontsize=6.3, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=1)
    ax.set_xlabel("depth spread of the circuit's networks (layers)", fontsize=8); ax.set_ylabel('preference ratio', fontsize=8)
    style(ax)
    letter(ax, 'A', x=-0.18, y=1.18)
    ax = axes[1]
    for d in ORDER:
        s_ = r[r.domain == d]
        ax.scatter(s_.eff, s_.R, s=9, color=COLORS[d], edgecolors='black', linewidths=0.25, alpha=0.85, zorder=3)
    ax.axhline(1, color='black', lw=0.8, ls='--')
    ax.text(0.97, 0.97, 'ρ = %.2f' % spearmanr(r.eff, r.R).correlation, transform=ax.transAxes, ha='right', va='top',
            fontsize=7.5, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))
    ax.set_xlabel('effective number of networks', fontsize=8); ax.set_ylabel('preference ratio', fontsize=8)
    style(ax, grid='both')
    letter(ax, 'B', x=-0.18, y=1.18)
    save(fig, 'controls_%s' % model)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='2b')
    args = ap.parse_args()
    d, g = load(args.model)
    concept()
    coupling_fig(args.model)
    method(args.model)
    domains_fig(g, args.model)
    controls(g, args.model)


if __name__ == '__main__':
    main()
