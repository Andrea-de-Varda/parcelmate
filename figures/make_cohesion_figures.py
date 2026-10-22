"""Figures for plots/circuit_cohesion/README.md (LOG.md Iteration 46).

    python figures/make_cohesion_figures.py --model 2b

Reads figures/circuit_cohesion_<model>.csv (parcelmate.bin.circuit_cohesion), the coupling files
results/richclub/qwen3.5-<model>-resid-pool4-tiles{,_null}/metrics/coupling/coupling.npz, the
concentration scores figures/bank_concentration_<model>_resid_pool4.csv and the task labels.
Writes plots/circuit_cohesion/{concept,coupling,example,domains,controls}_<model>.{svg,png}.
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
from parcelmate.cohesion import (cohesion_stats, layer_bins, network_layers, normalized_coupling,  # noqa: E402
                                 permute_within_bins)
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
        C=('cohesion', 'mean'), Cn=('null_cohesion_mean', 'mean'), S=('specificity', 'mean'),
        Sn=('null_specificity_mean', 'mean'), pC=('p_cohesion', 'max'), pS=('p_specificity', 'max'),
        eff=('effective_networks', 'mean'), between=('median_between_coupling', 'mean')).reset_index()
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

def example(model, group='causalgym', task='filler_gap_pp', name='Filler-gap, PP gap'):
    z = np.load(os.path.join(ROOT, 'results', 'richclub', 'qwen3.5-%s-resid-pool4-tiles' % model, 'metrics', 'coupling', 'coupling.npz'))
    lab, co, M = z['labels_halfA'], z['coordinates'], z['M_halfA_on_halfB']
    k = M.shape[0]
    Bn = normalized_coupling(M)
    c = network_layers(lab, co, k)
    bins = layer_bins(c, 10)
    roots = {'patching_bank': 'results/patching_bank/Qwen_Qwen3-5-%s' % model.upper(), 'patching': 'results/patching/Qwen_Qwen3-5-%s' % model.upper()}
    a = np.load(os.path.join(ROOT, roots['patching_bank'], group, task, 'neuron_attribution.npy'))
    n_layers, width = a.shape
    flat = co[:, 0].astype(np.int64) * width + co[:, 1]
    unit_net = np.empty(n_layers * width, dtype=np.int64); unit_net[flat] = lab
    circ = circuit_indices(a, 0.1)
    p = np.bincount(unit_net[circ], minlength=k) / len(circ)
    C, S = cohesion_stats(p, Bn)
    rng = np.random.RandomState(7)
    perms = permute_within_bins(bins, 1000, rng)
    Cn, Sn = cohesion_stats(p[perms], Bn)
    order = np.argsort(c, kind='stable')
    fig = plt.figure(figsize=(7.2, 2.5), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1], wspace=0.42)
    ax0, ax1, ax2 = (fig.add_subplot(gs[i]) for i in range(3))
    x = np.arange(k)
    ax0.bar(x, p[order], color=COLORS['Syntax'], width=0.9, label='circuit')
    ax0.bar(x, -p[perms[0]][order], color=GRAY_EDGE, width=0.9, label='one layer-matched permutation')
    for b_ in range(1, 10):
        ax0.axvline(b_ * 10 - 0.5, color='#bbbbbb', lw=0.6, ls=':')
    ax0.axhline(0, color='black', lw=0.8)
    ax0.set_xlabel('networks, by mean layer (dotted: the 10 layer bins)', fontsize=8)
    ax0.set_ylabel('share of circuit units', fontsize=8)
    yt = ax0.get_yticks(); ax0.set_yticks(yt); ax0.set_yticklabels(['%.2f' % abs(v) for v in yt])
    ax0.legend(frameon=False, fontsize=6.5, loc='upper left')
    style(ax0)
    letter(ax0, 'A', x=-0.1)
    for ax, null, obs, lab_, L in ((ax1, Cn, C, 'cohesion', 'B'), (ax2, Sn, S, 'specificity', 'C')):
        ax.hist(null, bins=30, color=GRAY_FILL, edgecolor=GRAY_EDGE, lw=0.4)
        ax.axvline(obs, color=COLORS['Syntax'], lw=2)
        pv = (1 + np.sum(null >= obs - 1e-12)) / 1001
        ax.text(0.03, 0.95, 'p = %.3f' % pv, transform=ax.transAxes, ha='left', va='top', fontsize=7.5,
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))
        ax.set_xlabel(lab_, fontsize=8); ax.set_ylabel('permutations', fontsize=8)
        style(ax)
        letter(ax, L)
    fig.text(0.5, 1.0, '%s (one direction: half-A networks, half-B coupling)' % name, ha='center', fontsize=8, color='#444444')
    save(fig, 'example_%s' % model)


# ---------------------------------------------------------------------------- figure 4

def domains_fig(g, model):
    r = g[g.tree == 'real'].copy()
    between = r.between.iloc[0]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), dpi=300, sharex=True)
    x0, ticks, tlabels = 0, [], []
    pos = {}
    for d in ORDER:
        sub = r[r.domain == d].sort_values('C')
        xs = x0 + np.arange(len(sub))
        for (_, t), x in zip(sub.iterrows(), xs):
            pos[(t.group, t.task)] = x
        ticks.append(xs.mean()); tlabels.append(TICK[d] + '\n(%d)' % len(sub))
        x0 = xs[-1] + 7
    for ax, val, nul, pcol, ylab, L in ((axes[0], 'C', 'Cn', 'pC', 'cohesion of the circuit\'s networks', 'A'),
                                        (axes[1], 'S', 'Sn', 'pS', 'specificity', 'B')):
        for _, t in r.iterrows():
            x = pos[(t.group, t.task)]
            sig = t[pcol] < 0.05
            col = COLORS[t.domain] if sig else GRAY_EDGE
            ax.plot([x, x], [t[val], t[nul]], color=col, lw=0.9, alpha=0.6, zorder=2)
            ax.scatter(x, t[nul], s=9, facecolors='white', edgecolors=GRAY_EDGE, linewidths=0.6, zorder=3)
            ax.scatter(x, t[val], s=11, facecolors=col if sig else GRAY_FILL, edgecolors='black' if sig else GRAY_EDGE,
                       linewidths=0.4, zorder=4)
        ax.set_ylabel(ylab, fontsize=8.5)
        style(ax)
        letter(ax, L, x=-0.09, y=0.97)
    axes[0].axhline(between, color='#555555', lw=1, ls='--', zorder=1)
    axes[0].text(0.995, (between - 0.35) / (0.72 - 0.35) - 0.015, 'typical pair of networks (%.2f)' % between, transform=axes[0].transAxes,
                 ha='right', va='top', fontsize=6.5, color='#555555', bbox=dict(facecolor='white', edgecolor='none', pad=1))
    axes[0].set_ylim(0.35, 0.72)
    axes[0].text(0.03, 0.98, 'one network = 1.0 (off scale)', transform=axes[0].transAxes, fontsize=6.5, color='#555555', va='top')
    axes[1].axhline(1.0, color='#555555', lw=1, ls='--', zorder=1)
    axes[1].set_xticks(ticks); axes[1].set_xticklabels(tlabels, fontsize=7)
    handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor='#555555', markeredgecolor='black', markersize=4.5,
                      label="circuit's networks (p < .05 in both directions)"),
               Line2D([0], [0], marker='o', color='none', markerfacecolor=GRAY_FILL, markeredgecolor=GRAY_EDGE, markersize=4.5,
                      label='not above the null in both directions'),
               Line2D([0], [0], marker='o', color='none', markerfacecolor='white', markeredgecolor=GRAY_EDGE, markersize=4.5,
                      label='layer-matched networks (mean of 1,000)')]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.53, 1.03))
    fig.subplots_adjust(hspace=0.12, top=0.95)
    save(fig, 'domains_%s' % model)


# ---------------------------------------------------------------------------- figure 5

def controls(g, model):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), dpi=300, gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.35))
    ax = axes[0]
    rng = np.random.RandomState(0)
    for i, d in enumerate(ORDER):
        for j, (tree, off) in enumerate((('real', -0.18), ('null', 0.18))):
            v = (g[(g.tree == tree) & (g.domain == d)].C - g[(g.tree == tree) & (g.domain == d)].Cn).values
            col = COLORS[d] if tree == 'real' else GRAY_EDGE
            ax.scatter(i + off + rng.uniform(-0.07, 0.07, len(v)), v, s=7, color=col, alpha=0.7, lw=0, zorder=3)
            ax.plot([i + off - 0.13, i + off + 0.13], [np.median(v)] * 2, color='black' if tree == 'real' else '#666666', lw=1.6, zorder=4)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([SHORT.get(d, d).replace('\n', ' ') for d in ORDER], rotation=40, ha='right', fontsize=6.8)
    ax.set_ylabel('cohesion − layer-matched null', fontsize=8)
    ax.legend(handles=[Line2D([0], [0], marker='o', color='none', markerfacecolor='#555555', markersize=4, label='real networks'),
                       Line2D([0], [0], marker='o', color='none', markerfacecolor=GRAY_EDGE, markersize=4, label='null partition')],
              frameon=False, fontsize=6.8, loc='upper left')
    style(ax)
    letter(ax, 'A', x=-0.12)
    ax = axes[1]
    r = g[g.tree == 'real']
    for d in ORDER:
        s = r[r.domain == d]
        ax.scatter(s.eff, s.C, s=10, color=COLORS[d], edgecolors='black', linewidths=0.3, alpha=0.85, zorder=3)
    from scipy.stats import spearmanr
    rho = spearmanr(r.eff, r.C).correlation
    ax.text(0.95, 0.05, 'ρ = %.2f' % rho, transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5,
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))
    ax.set_xlabel('effective number of networks', fontsize=8); ax.set_ylabel('cohesion', fontsize=8)
    style(ax, grid='both')
    letter(ax, 'B')
    save(fig, 'controls_%s' % model)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='2b')
    args = ap.parse_args()
    d, g = load(args.model)
    concept()
    coupling_fig(args.model)
    example(args.model)
    domains_fig(g, args.model)
    controls(g, args.model)


if __name__ == '__main__':
    main()
