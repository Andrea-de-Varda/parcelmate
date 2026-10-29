"""Figures for the intrinsic-timescales analysis (LOG.md Iteration 47), Qwen3.5-2B.

    python figures/make_timescales_figures.py

Reads results/timescales/qwen3.5-2b/{acf.h5, analysis_fit/} and figures/bank_task_labels.csv;
writes plots/timescales/*.{svg,png}:
  acf_examples      the ACF of three real units with their fits: decay (tau) and persistence (A x B)
  reliability       half A against half B, per unit, for decay and persistence
  depth             median and interquartile range per layer, intact and word-shuffled text
  networks          share of the within-layer variance carried by the networks: shuffled labels,
                    real networks, circular-shift null networks
  circuits_mean     per task: circuit mean against layer-matched random units (z), by domain
  circuits_var      per task: circuit variance of log tau (z), by domain
  circuits_var_accuracy  the same against the task's both-correct accuracy
"""

import os

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'DejaVu Sans'

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results', 'timescales', 'qwen3.5-2b')
OUT = os.path.join(ROOT, 'plots', 'timescales')

# Same domain colours as the circuits-vs-networks figures (make_bank_concentration_figure.py).
COLORS = {
    'Syntax': '#C0392B', 'Morphology & lexical semantics': '#D9776D',
    'Arithmetic': '#2471A3', 'Logic, code & multiple choice': '#5DADE2',
    'Factual knowledge': '#B7950B', 'Entity tracking & binding': '#148F77',
    'In-context string tasks': '#7D3C98', 'Physical reasoning': '#E67E22', 'Social reasoning': '#27AE60',
}
SHORT = {'Syntax': 'Syntax', 'Morphology & lexical semantics': 'Morph. &\nlex. sem.', 'Arithmetic': 'Arithmetic',
         'Logic, code & multiple choice': 'Logic,\ncode, MC', 'Factual knowledge': 'Factual',
         'Entity tracking & binding': 'Entity\ntracking', 'In-context string tasks': 'In-context\nstrings',
         'Physical reasoning': 'Physical', 'Social reasoning': 'Social'}
INTACT, SHUF = '#2C3E50', '#A0A0A0'
GRAY_FILL, GRAY_EDGE = '#cccccc', '#999999'


def chrome(ax, lw=1.5, grid='y'):
    ax.spines[['top', 'right']].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_linewidth(lw)
    if grid:
        ax.grid(axis=grid, linestyle='--', alpha=0.5, zorder=1)
    ax.set_axisbelow(True)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ('svg', 'png'):
        fig.savefig(os.path.join(OUT, '%s.%s' % (name, ext)), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('wrote plots/timescales/%s.{svg,png}' % name)


def stat_box(ax, text, loc='upper left', fontsize=8):
    x, ha = (0.04, 'left') if 'left' in loc else (0.96, 'right')
    y, va = (0.96, 'top') if 'upper' in loc else (0.04, 'bottom')
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize,
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))


def load():
    u = dict(np.load(os.path.join(RES, 'analysis_fit', 'units.npz')))
    u['layer'] = u['coordinates'][:, 0]
    return u


# ---------------------------------------------------------------------------- figures

def fig_acf_examples(u):
    """Three real units: fast, slow-decaying, persistent. ACF = mean over datasets and halves."""
    lay, tau = u['layer'], np.exp(u['intact_decay'])
    per, amp = u['intact_persistence'], u['intact_amplitude']
    ok = np.isfinite(tau) & np.isfinite(per)
    cands = {
        'fast': np.flatnonzero(ok & (lay == 3) & (tau > 1) & (tau < 2) & (per < 0.02)),        # short tau, no plateau
        'slow': np.flatnonzero(ok & (lay == 22) & (tau > 12) & (tau < 40) & (per < 0.08)),      # long tau
        'persistent': np.flatnonzero(ok & (lay == 22) & (tau < 4) & (per > 0.2)),               # large plateau
    }
    lag = np.arange(1, 257)
    picks, acf = {}, {}
    with h5py.File(os.path.join(RES, 'acf.h5'), 'r') as f:
        for k, c in cands.items():
            a = np.mean([f['intact/%s/%s/acf' % (d, h)][np.sort(c)] for d in ('wikitext', 'bookcorpus')
                         for h in ('halfA', 'halfB')], 0)[:, 1:]
            c = np.sort(c)
            fit = amp[c, None] * np.exp(-lag[None] / tau[c, None]) + per[c, None]
            rmse = np.sqrt(((a - fit) ** 2).mean(1))
            # a well-fitted unit (rmse in its candidates' best 10%) with a clearly visible ACF at lag 1
            good = rmse <= np.quantile(rmse, 0.1)
            j = np.flatnonzero(good)[np.argmax(a[good, 0])]
            picks[k], acf[k] = c[j], np.r_[1.0, a[j]]
    lag = np.arange(1, 257)
    style = {'fast': ('#5DADE2', 'fast unit'), 'slow': ('#C0392B', 'slow unit'),
             'persistent': ('#B7950B', 'persistent unit')}
    fig, ax = plt.subplots(figsize=(3.6 * .9, 2.6 * .9))
    for k, i in picks.items():
        col, lab = style[k]
        A, t, P = amp[i], tau[i], per[i]
        ax.plot(lag, acf[k][1:], color=col, lw=1.0, alpha=0.45, zorder=2)
        ax.plot(lag, A * np.exp(-lag / t) + P, color=col, lw=1.8, ls='--', zorder=3)
        print('  %s: layer %d, unit %d, tau %.1f tokens, persistence %.3f' % (k, lay[i], u['coordinates'][i, 1], t, P))
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax.set_xticklabels(['1', '2', '5', '10', '20', '50', '100', '200'])
    ax.set_xlim(1, 256)
    ax.set_ylim(-0.02, None)
    ax.set_xlabel('Lag (tokens)', fontsize=11)
    ax.set_ylabel('Autocorrelation', fontsize=11)
    ax.tick_params(labelsize=9)
    chrome(ax)
    h = [Line2D([0], [0], color=style[k][0], lw=2, label='%s (layer %d)' % (style[k][1], lay[picks[k]]))
         for k in ('fast', 'slow', 'persistent')]
    h += [Line2D([0], [0], color='gray', lw=1, alpha=0.5, label='data'),
          Line2D([0], [0], color='gray', lw=1.8, ls='--', label='fit')]
    ax.legend(handles=h, frameon=False, fontsize=7.5, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    save(fig, 'acf_examples')


def fig_reliability(u):
    fig, axes = plt.subplots(1, 2, figsize=(5.4 * .9, 2.5 * .9))
    for ax, (m, lab, logs) in zip(axes, (('intact_decay', 'Decay τ (tokens)', True),
                                         ('intact_persistence', 'Persistence', False))):
        a, b = u[m + '_halfA'], u[m + '_halfB']
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        if logs:
            a, b = np.exp(a), np.exp(b)
            lo, hi = np.quantile(np.r_[a, b], [0.005, 0.995])
            ax.set_xscale('log'); ax.set_yscale('log')
            bins = 'log'
            ax.hexbin(a, b, gridsize=45, xscale='log', yscale='log', bins=bins, cmap='Greys',
                      mincnt=1, extent=(np.log10(lo), np.log10(hi), np.log10(lo), np.log10(hi)), linewidths=0)
        else:
            lo, hi = np.quantile(np.r_[a, b], [0.005, 0.995])
            ax.hexbin(a, b, gridsize=45, bins='log', cmap='Greys', mincnt=1, extent=(lo, hi, lo, hi), linewidths=0)
        ax.plot([lo, hi], [lo, hi], color='#C0392B', lw=1, ls='--')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        if logs:
            t = [x for x in (0.5, 1, 2, 5, 10, 20, 50) if lo <= x <= hi]
            ax.set_xticks(t); ax.set_yticks(t)
            ax.set_xticklabels(['%g' % x for x in t]); ax.set_yticklabels(['%g' % x for x in t])
            ax.minorticks_off()
        ax.set_xlabel('Half A', fontsize=10); ax.set_ylabel('Half B', fontsize=10)
        ax.set_title(lab, fontsize=9, weight='bold')
        ax.tick_params(labelsize=8)
        chrome(ax, grid=None)
        stat_box(ax, 'ρ = %.2f' % spearmanr(a, b)[0])
    fig.tight_layout()
    save(fig, 'reliability')


def fig_depth(u):
    lay = u['layer']
    L = int(lay.max()) + 1
    fig, axes = plt.subplots(1, 2, figsize=(6.0 * .9, 2.5 * .9))
    for ax, (m, lab) in zip(axes, (('decay', 'Decay τ (tokens)'), ('persistence', 'Persistence'))):
        for cond, col, name in (('intact', INTACT, 'intact text'), ('shuffled', SHUF, 'words shuffled')):
            v = u['%s_%s' % (cond, m)]
            v = np.exp(v) if m == 'decay' else v
            layers = np.arange(1, L) if m == 'decay' else np.arange(L)   # layer 0: tau not identified
            q = np.array([np.nanquantile(v[lay == l], [0.25, 0.5, 0.75]) for l in layers])
            ax.fill_between(layers, q[:, 0], q[:, 2], color=col, alpha=0.2, lw=0, zorder=2)
            ax.plot(layers, q[:, 1], color=col, lw=2, marker='o', ms=2.5, zorder=3, label=name)
        if m == 'decay':
            ax.set_yscale('log')
            ax.set_yticks([0.5, 1, 2, 5, 10, 20]); ax.set_yticklabels(['0.5', '1', '2', '5', '10', '20'])
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_ylabel(lab, fontsize=10)
        ax.set_xlim(-0.5, L - 0.5)
        ax.tick_params(labelsize=8)
        chrome(ax)
    axes[1].legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    save(fig, 'depth')


def fig_networks():
    N = pd.read_csv(os.path.join(RES, 'analysis_fit', 'networks.csv'), keep_default_na=False)
    fig, ax = plt.subplots(figsize=(3.2 * .9, 2.5 * .9))
    groups = [('intact_decay', 'Decay'), ('intact_persistence', 'Persistence')]
    bars = [('perm', 'shuffled labels', 'white', GRAY_EDGE), ('real', 'real networks', INTACT, 'black'),
            ('null', 'circular-shift networks', GRAY_FILL, GRAY_EDGE)]
    w = 0.26
    for g, (m, _) in enumerate(groups):
        S = N[N.measure == m]
        vals = {'perm': S[S.tree == 'real'].perm_share.mean(), 'real': S[S.tree == 'real'].share.mean(),
                'null': S[S.tree == 'null'].share.mean()}
        for b, (k, lab, fc, ec) in enumerate(bars):
            x = g + (b - 1) * w
            ax.bar(x, vals[k], w, color=fc, edgecolor=ec, lw=1, zorder=3, label=lab if g == 0 else None)
            ax.text(x, vals[k] + 0.015, '%.2f' % vals[k] if vals[k] >= 0.01 else '<0.01', ha='center',
                    va='bottom', fontsize=7)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[1] for g in groups], fontsize=10)
    ax.set_ylabel('Variance explained', fontsize=10)
    ax.set_ylim(0, 0.9)
    ax.tick_params(labelsize=8)
    chrome(ax)
    ax.legend(frameon=False, fontsize=7.5, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    save(fig, 'networks')


def circuits():
    C = pd.read_csv(os.path.join(RES, 'analysis_fit', 'circuits.csv'))
    lab = pd.read_csv(os.path.join(ROOT, 'figures', 'bank_task_labels.csv'))
    C = C.merge(lab[['root', 'group', 'task', 'domain']], on=['root', 'group', 'task'], how='left')
    assert C.domain.notna().all(), 'tasks without a domain label'
    return C


def strip(ax, C, measure, stat, order, rng):
    D = C[C.measure == measure]
    for x, d in enumerate(order):
        z = D[D.domain == d]['%s_z' % stat].values
        jit = rng.uniform(-0.22, 0.22, len(z))
        sig = np.abs(z) > 1.96
        ax.scatter(x + jit[~sig], z[~sig], s=10, facecolors=GRAY_FILL, edgecolors=GRAY_EDGE, lw=0.4, zorder=3)
        ax.scatter(x + jit[sig], z[sig], s=12, facecolors=COLORS[d], edgecolors='black', lw=0.4, zorder=4, alpha=0.9)
        ax.plot([x - 0.3, x + 0.3], [np.median(z)] * 2, color='black', lw=2, zorder=5)
    ax.axhline(0, color='black', lw=1, zorder=2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(['%s\n(%d)' % (SHORT[d], (D.domain == d).sum()) for d in order], fontsize=7)
    for t, d in zip(ax.get_xticklabels(), order):
        t.set_color(COLORS[d])
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.tick_params(axis='y', labelsize=8)
    chrome(ax)


def fig_circuits_mean(C):
    order = (C[C.measure == 'intact_decay'].groupby('domain').mean_z.median().sort_values().index.tolist())
    fig, axes = plt.subplots(2, 1, figsize=(6.4 * .9, 4.6 * .9), sharex=True)
    rng = np.random.RandomState(0)
    for ax, (m, lab) in zip(axes, (('intact_decay', 'Decay τ (z)'),
                                   ('intact_persistence', 'Persistence (z)'))):
        strip(ax, C, m, 'mean', order, rng)
        ax.set_ylabel(lab, fontsize=10)
    for t in axes[0].get_xticklabels():
        t.set_visible(False)
    h = [Line2D([0], [0], marker='o', color='none', markerfacecolor='#555555', markeredgecolor='black', ms=4.5,
                label='task, |z| > 1.96'),
         Line2D([0], [0], marker='o', color='none', markerfacecolor=GRAY_FILL, markeredgecolor=GRAY_EDGE, ms=4.5,
                label='task, n.s.'),
         Line2D([0], [0], color='black', lw=2, label='domain median')]
    axes[0].legend(handles=h, frameon=False, fontsize=7, loc='upper left', ncol=3, bbox_to_anchor=(0, 1.18))
    fig.tight_layout()
    save(fig, 'circuits_mean')


def fig_circuits_var(C):
    D = C[C.measure == 'intact_decay']
    order = D.groupby('domain').mean_z.median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(6.4 * .9, 2.4 * .9))
    strip(ax, C, 'intact_decay', 'var', order, np.random.RandomState(1))
    ax.set_ylabel('Spread of τ (z)', fontsize=10)
    fig.tight_layout()
    save(fig, 'circuits_var')

    fig, ax = plt.subplots(figsize=(3.0 * .9, 2.5 * .9))
    for d in order:
        S = D[D.domain == d]
        ax.scatter(S.accuracy, S.var_z, s=12, color=COLORS[d], edgecolors='black', lw=0.3, alpha=0.85, zorder=3,
                   label=SHORT[d].replace('\n', ' '))
    ax.axhline(0, color='black', lw=1, zorder=2)
    r, p = spearmanr(D.accuracy, D.var_z)
    stat_box(ax, 'ρ = %.2f, p = %.2f' % (r, p), loc='upper left', fontsize=7.5)
    ax.set_xlabel('Task accuracy', fontsize=10)
    ax.set_ylabel('Spread of τ (z)', fontsize=10)
    ax.tick_params(labelsize=8)
    chrome(ax)
    ax.legend(frameon=False, fontsize=6.5, loc='upper left', bbox_to_anchor=(1.0, 1.0), handletextpad=0.1)
    save(fig, 'circuits_var_accuracy')


def main():
    u = load()
    fig_acf_examples(u)
    fig_reliability(u)
    fig_depth(u)
    fig_networks()
    C = circuits()
    fig_circuits_mean(C)
    fig_circuits_var(C)


if __name__ == '__main__':
    main()
