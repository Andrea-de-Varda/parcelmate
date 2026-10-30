"""Figures for the intrinsic-timescales analysis (LOG.md Iteration 47), Qwen3.5-2B.

    python figures/make_timescales_figures.py

Reads results/timescales/qwen3.5-2b/{acf.h5, analysis_lags/} and figures/bank_task_labels.csv;
writes plots/timescales/*.{svg,png}:
  acf_schematic     illustration (simulated unit): pairs of tokens k apart, and their correlation
  acf_examples      the ACF of three real units, read at lags 1, 10 and 100
  reliability       half A against half B, per unit, for ACF(10) and ACF(100)
  depth             median and interquartile range per layer of ACF(1), ACF(10), ACF(100),
                    intact and word-shuffled text
  networks          share of the within-layer variance carried by the networks: shuffled labels,
                    real networks, circular-shift null networks
  circuits_mean     per task: circuit mean against layer-matched random units (z), by domain
  circuits_var      per task: circuit spread of ACF(10) (z), by domain
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
TAG = 'analysis_lags'
LAGS = (1, 10, 100)
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
    u = dict(np.load(os.path.join(RES, TAG, 'units.npz')))
    u['layer'] = u['coordinates'][:, 0]
    return u


# ---------------------------------------------------------------------------- figures

def fig_acf_schematic():
    """Illustration with a simulated unit (AR(1), phi 0.85): the residual over tokens, two pairs of
    tokens k = 5 apart, and the scatter of all such pairs, whose correlation is ACF(5)."""
    rng = np.random.RandomState(3)
    n, k, phi = 4000, 5, 0.85
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + rng.normal(0, np.sqrt(1 - phi ** 2))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.2 * .9, 2.3 * .9), gridspec_kw=dict(width_ratios=[2.1, 1]))
    T = 40
    a1.plot(np.arange(T), r[:T], color=INTACT, lw=1.2, marker='o', ms=2.5, zorder=3)
    for t0, col in ((6, '#C0392B'), (24, '#2471A3')):
        a1.plot([t0, t0 + k], [r[t0], r[t0 + k]], 'o', ms=6, color=col, zorder=4)
        top = max(r[t0], r[t0 + k]) + 0.5
        a1.annotate('', xy=(t0 + k, top), xytext=(t0, top),
                    arrowprops=dict(arrowstyle='<->', color=col, lw=1.2))
        a1.text(t0 + k / 2, top + 0.12, 'k = %d' % k, color=col, ha='center', va='bottom', fontsize=7.5)
    a1.set_xlabel('Token position in the window', fontsize=10)
    a1.set_ylabel('Residual activation', fontsize=10)
    a1.set_xlim(-1, T)
    a1.tick_params(labelsize=8)
    chrome(a1)
    x, y = r[:-k], r[k:]
    a2.scatter(x, y, s=2, color='#888888', alpha=0.25, lw=0, zorder=2)
    for t0, col in ((6, '#C0392B'), (24, '#2471A3')):
        a2.scatter(r[t0], r[t0 + k], s=28, color=col, edgecolors='black', lw=0.5, zorder=4)
    a2.set_xlabel('Activation at t', fontsize=10)
    a2.set_ylabel('Activation at t + k', fontsize=10)
    a2.tick_params(labelsize=8)
    lim = np.quantile(np.abs(r), 0.999)
    a2.set_xlim(-lim, lim); a2.set_ylim(-lim, lim)
    chrome(a2, grid=None)
    stat_box(a2, 'ACF(%d) = r = %.2f' % (k, np.corrcoef(x, y)[0, 1]), fontsize=7.5)
    fig.tight_layout()
    save(fig, 'acf_schematic')


def fig_acf_examples(u):
    """Three real units, ACF = mean over datasets and halves: fast (next token only), slow (still
    correlated a sentence later) and persistent (correlated across the window)."""
    lay = u['layer']
    a1, a10, a100 = u['intact_acf1'], u['intact_acf10'], u['intact_acf100']
    cands = {
        'fast': (lay == 3) & (a10 < 0.02) & (a100 < 0.01),
        'slow': (lay == 22) & (a100 < 0.03),
        'persistent': (lay == 22) & (a100 > 0.25),
    }
    score = {'fast': a1, 'slow': a10, 'persistent': a1}
    picks = {k: int(np.flatnonzero(c)[np.argmax(score[k][c])]) for k, c in cands.items()}
    acf = {}
    with h5py.File(os.path.join(RES, 'acf.h5'), 'r') as f:
        for k, i in picks.items():
            acf[k] = np.mean([f['intact/%s/%s/acf' % (d, h)][i] for d in ('wikitext', 'bookcorpus')
                              for h in ('halfA', 'halfB')], 0)
    style = {'fast': ('#5DADE2', 'fast unit'), 'slow': ('#C0392B', 'slow unit'),
             'persistent': ('#B7950B', 'persistent unit')}
    lag = np.arange(1, 257)
    fig, ax = plt.subplots(figsize=(3.6 * .9, 2.6 * .9))
    for x in LAGS:
        ax.axvline(x, color='black', lw=0.8, ls=':', zorder=1)
    for k, i in picks.items():
        col, _ = style[k]
        ax.plot(lag, acf[k][1:], color=col, lw=1.6, zorder=2)
        ax.scatter(LAGS, acf[k][list(LAGS)], s=22, color=col, edgecolors='black', lw=0.6, zorder=4)
        print('  %s: layer %d, unit %d, ACF(1) %.3f, ACF(10) %.3f, ACF(100) %.3f'
              % (k, lay[i], u['coordinates'][i, 1], *acf[k][list(LAGS)]))
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax.set_xticklabels(['1', '2', '5', '10', '20', '50', '100', '200'])
    ax.minorticks_off()
    ax.set_xlim(1, 256)
    ax.set_ylim(-0.02, None)
    ax.set_xlabel('Lag (tokens)', fontsize=11)
    ax.set_ylabel('Autocorrelation', fontsize=11)
    ax.tick_params(labelsize=9)
    chrome(ax)
    h = [Line2D([0], [0], color=style[k][0], lw=2, label='%s (layer %d)' % (style[k][1], lay[picks[k]]))
         for k in ('fast', 'slow', 'persistent')]
    h += [Line2D([0], [0], color='black', lw=0.8, ls=':', label='lags 1, 10, 100')]
    ax.legend(handles=h, frameon=False, fontsize=7.5, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    save(fig, 'acf_examples')


def fig_reliability(u):
    fig, axes = plt.subplots(1, 2, figsize=(5.4 * .9, 2.5 * .9))
    for ax, k in zip(axes, (10, 100)):
        m = 'intact_acf%d' % k
        a, b = u[m + '_halfA'], u[m + '_halfB']
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        lo, hi = np.quantile(np.r_[a, b], [0.005, 0.995])
        ax.hexbin(a, b, gridsize=45, bins='log', cmap='Greys', mincnt=1, extent=(lo, hi, lo, hi), linewidths=0)
        ax.plot([lo, hi], [lo, hi], color='#C0392B', lw=1, ls='--')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel('Half A', fontsize=10); ax.set_ylabel('Half B', fontsize=10)
        ax.set_title('ACF(%d)' % k, fontsize=9, weight='bold')
        ax.tick_params(labelsize=8)
        chrome(ax, grid=None)
        stat_box(ax, 'ρ = %.2f' % spearmanr(a, b)[0])
    fig.tight_layout()
    save(fig, 'reliability')


def fig_depth(u):
    lay = u['layer']
    L = int(lay.max()) + 1
    layers = np.arange(L)
    fig, axes = plt.subplots(1, 3, figsize=(7.6 * .9, 2.4 * .9))
    for ax, k in zip(axes, LAGS):
        for cond, col, name in (('intact', INTACT, 'intact text'), ('shuffled', SHUF, 'words shuffled')):
            v = u['%s_acf%d' % (cond, k)]
            q = np.array([np.nanquantile(v[lay == l], [0.25, 0.5, 0.75]) for l in layers])
            ax.fill_between(layers, q[:, 0], q[:, 2], color=col, alpha=0.2, lw=0, zorder=2)
            ax.plot(layers, q[:, 1], color=col, lw=2, marker='o', ms=2.5, zorder=3, label=name)
        ax.set_title('ACF(%d)' % k, fontsize=9, weight='bold')
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_xlim(-0.5, L - 0.5)
        ax.set_ylim(0, None)
        ax.tick_params(labelsize=8)
        chrome(ax)
    axes[0].set_ylabel('Autocorrelation', fontsize=10)
    axes[0].legend(frameon=False, fontsize=7.5, loc='upper left')
    fig.tight_layout()
    save(fig, 'depth')


def fig_networks():
    N = pd.read_csv(os.path.join(RES, TAG, 'networks.csv'), keep_default_na=False)
    fig, ax = plt.subplots(figsize=(3.2 * .9, 2.5 * .9))
    groups = [('intact_acf10', 'ACF(10)'), ('intact_acf100', 'ACF(100)')]
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
    C = pd.read_csv(os.path.join(RES, TAG, 'circuits.csv'))
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
    order = (C[C.measure == 'intact_acf10'].groupby('domain').mean_z.median().sort_values().index.tolist())
    fig, axes = plt.subplots(2, 1, figsize=(6.4 * .9, 4.6 * .9), sharex=True)
    rng = np.random.RandomState(0)
    for ax, (m, lab) in zip(axes, (('intact_acf10', 'ACF(10) (z)'), ('intact_acf100', 'ACF(100) (z)'))):
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
    D = C[C.measure == 'intact_acf10']
    order = D.groupby('domain').mean_z.median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(6.4 * .9, 2.4 * .9))
    strip(ax, C, 'intact_acf10', 'var', order, np.random.RandomState(1))
    ax.set_ylabel('Spread of ACF(10) (z)', fontsize=10)
    fig.tight_layout()
    save(fig, 'circuits_var')

    fig, ax = plt.subplots(figsize=(3.0 * .9, 2.5 * .9))
    for d in order:
        S = D[D.domain == d]
        ax.scatter(S.accuracy, S.var_z, s=12, color=COLORS[d], edgecolors='black', lw=0.3, alpha=0.85, zorder=3,
                   label=SHORT[d].replace('\n', ' '))
    ax.axhline(0, color='black', lw=1, zorder=2)
    r, p = spearmanr(D.accuracy, D.var_z)
    stat_box(ax, 'ρ = %.2f, p = %.2g' % (r, p), loc='upper left', fontsize=7.5)
    ax.set_xlabel('Task accuracy', fontsize=10)
    ax.set_ylabel('Spread of ACF(10) (z)', fontsize=10)
    ax.tick_params(labelsize=8)
    chrome(ax)
    ax.legend(frameon=False, fontsize=6.5, loc='upper left', bbox_to_anchor=(1.0, 1.0), handletextpad=0.1)
    save(fig, 'circuits_var_accuracy')


def main():
    fig_acf_schematic()
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
