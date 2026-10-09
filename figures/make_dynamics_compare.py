"""Pythia-70m training dynamics, ordinary against residual connectome (LOG.md Iteration 39).

    python figures/make_dynamics_compare.py

Reads results/pythia/pythia-70m/dynamics (ordinary connectome, Iteration 28) and
results/pythia_resid/pythia-70m/dynamics (token type and position removed by a joint
least-squares fit before correlating, Iteration 39). Writes into plots/:

  dynamics_compare_70m.{svg,png}   seven measures, one pair of panels per measure (ordinary
                                   left, residual right, shared y axis), three pairs per row
  dynamics_timing_compare_70m.{svg,png}  the loss (thick black) against the structural
                                   measures, each rescaled to its progress from the
                                   initialisation (0) to the final checkpoint (1); ordinary
                                   left, residual right

Values are means over the four prose domains (faint dots: domains) and both halves. Dashed
lines: the null partition. The firing rate and the loss are computed on the raw activations
in both runs, so their two panels match by construction.
"""

import csv
import glob
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from make_pythia_figures import log_x, style, xpos

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'DejaVu Sans'

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
PLOTS = os.path.join(ROOT, 'plots')
TREES = (('ordinary connectome', os.path.join(ROOT, 'results', 'pythia', 'pythia-70m', 'dynamics')),
         ('residual connectome', os.path.join(ROOT, 'results', 'pythia_resid', 'pythia-70m', 'dynamics')))
COLORS = {'ordinary connectome': '#7f7f7f', 'residual connectome': '#2c5f8a'}
FINAL = 143000


def read(path):
    out = list(csv.DictReader(open(path))) if os.path.exists(path) else []
    for r in out:
        r['value'] = float(r['value'])
        r['step'] = int(r['step'])
    return out


def load(d):
    conn = []
    for p in sorted(glob.glob(os.path.join(d, 'connectome_step*.csv'))):
        conn.extend(read(p))
    return dict(conn=conn, part=read(os.path.join(d, 'partitions.csv')), sel=read(os.path.join(d, 'selectivity.csv')))


def by_domain(rows, **filt):
    acc = {}
    for r in rows:
        if all((r.get(k) == v) if not callable(v) else v(r) for k, v in filt.items()):
            acc.setdefault(r['step'], {}).setdefault(r['domain'], []).append(r['value'])
    return {s: np.array([np.mean(v) for v in dom.values()]) for s, dom in acc.items()}


OWN = dict(partition='-')
HELD = lambda r: r['key'] != r['eval_key']
HALVES = lambda r: r['key'] in ('halfA', 'halfB')

# (title, getter returning (real series, null series or None))
MEASURES = [
    ('Dimensionality (participation ratio)', lambda D: (by_domain(D['conn'], measure='dim_participation_ratio', **OWN), None)),
    ('Mean |r|', lambda D: (by_domain(D['conn'], measure='coupling_mean_absr', **OWN), None)),
    ('Concentration on top 1% of partners', lambda D: (by_domain(D['conn'], measure='coupling_concentration_top1pct', **OWN), None)),
    ('Modularity (held out)', lambda D: (by_domain(D['conn'], measure='segregation_modularity', partition='real', held=HELD),
                                         by_domain(D['conn'], measure='segregation_modularity', partition='null', held=HELD))),
    ('Layers per network (effective number)', lambda D: (by_domain(D['part'], measure='depth_median_effective_layers', tree='real', h=HALVES),
                                                         by_domain(D['part'], measure='depth_median_effective_layers', tree='null', h=HALVES))),
    ('Neuron firing rate (median)', lambda D: (by_domain(D['conn'], measure='act_firing_rate_median', **OWN), None)),
    ('Token-class coherence of the networks', lambda D: (by_domain(D['sel'], measure='selectivity_coherence_median', tree='real'),
                                                         by_domain(D['sel'], measure='selectivity_coherence_median', tree='null'))),
]


def draw(ax, data, color, ls='-', dots=True, lw=1.6, marker='o'):
    steps = sorted(data)
    if dots:
        for s in steps:
            ax.scatter(np.full(len(data[s]), xpos(s)), data[s], s=6, color=color, alpha=0.25, edgecolors='none', zorder=2)
    ax.plot([xpos(s) for s in steps], [np.nanmean(data[s]) for s in steps], color=color, lw=lw, ls=ls,
            marker=marker, markersize=3.5 if marker else 0, markeredgecolor='black', markeredgewidth=0.4, zorder=3)
    return steps


def fig_pairs(D):
    n = len(MEASURES)
    rows = int(np.ceil(n / 3))
    fig = plt.figure(figsize=(13 * .9, 3.0 * rows * .9), dpi=300)
    outer = fig.add_gridspec(rows, 3, wspace=0.32, hspace=0.75)
    for i, (title, get) in enumerate(MEASURES):
        inner = outer[i // 3, i % 3].subgridspec(1, 2, wspace=0.08)
        axes = [fig.add_subplot(inner[0, j]) for j in range(2)]
        lo, hi = np.inf, -np.inf
        for ax, (name, _) in zip(axes, TREES):
            real, null = get(D[name])
            steps = draw(ax, real, COLORS[name])
            if null:
                draw(ax, null, COLORS[name], ls='--', dots=False, lw=1.0, marker=None)
            vals = [v for d in (real, null) if d for a in d.values() for v in a]
            lo, hi = min(lo, np.nanmin(vals)), max(hi, np.nanmax(vals))
            style(ax)
            log_x(ax, steps)
            ax.tick_params(labelsize=6.5)
            ax.set_title(name.split()[0], fontsize=7.5, color=COLORS[name], pad=2)
        pad = 0.06 * (hi - lo)
        for j, ax in enumerate(axes):
            ax.set_ylim(lo - pad, hi + pad)
            if j:
                ax.set_yticklabels([])
        axes[0].text(1.04, 1.2, title, transform=axes[0].transAxes, ha='center', fontsize=9, fontweight='bold')
    handles = [Line2D([0], [0], color='0.3', lw=1.6, marker='o', markersize=3.5, markeredgecolor='black', label='real partition / measure'),
               Line2D([0], [0], color='0.3', lw=1.0, ls='--', label='null partition')]
    fig.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.9, 0.08), frameon=False, fontsize=8.5)
    fig.text(0.62, 0.2, 'Pythia-70m, means over four prose datasets (dots: datasets).\nLeft panel of each pair: ordinary connectome; right: token type\nand position removed (residual). Firing rate is computed on raw\nactivations in both runs.', fontsize=8, va='top', color='0.3')
    save(fig, 'dynamics_compare_70m')


TIMING = [
    ('loss', 'LM loss (drop made)', 'black'),
    ('firing', 'MLP sparsification (firing-rate fall)', '#e6550d'),
    ('modularity_raw', 'Modularity (raw)', '#3182bd'),
    ('modularity_gap', 'Modularity (real - null)', '#31a354'),
    ('concentration', 'Concentration on top 1% of partners', '#756bb1'),
    ('layers', 'Layers per network (fall made)', '#d6619e'),
]


def progress(D):
    mean = lambda d: {s: float(np.mean(v)) for s, v in d.items()}
    real = mean(by_domain(D['conn'], measure='segregation_modularity', partition='real', held=HELD))
    null = mean(by_domain(D['conn'], measure='segregation_modularity', partition='null', held=HELD))
    raw = {'loss': mean(by_domain(D['conn'], measure='lm_loss', **OWN)),
           'firing': mean(by_domain(D['conn'], measure='act_firing_rate_median', **OWN)),
           'modularity_raw': real, 'modularity_gap': {s: real[s] - null[s] for s in real},
           'concentration': mean(by_domain(D['conn'], measure='coupling_concentration_top1pct', **OWN)),
           'layers': mean(by_domain(D['part'], measure='depth_median_effective_layers', tree='real', h=HALVES))}
    return {k: {s: (d[s] - d[0]) / (d[FINAL] - d[0]) for s in sorted(d)} for k, d in raw.items()}, raw


def fig_timing(D):
    fig, axes = plt.subplots(1, 2, figsize=(7.4 * .95, 3.3 * .95), dpi=300, sharey=True)
    lo, hi = -0.12, 1.12
    Ps = {name: progress(D[name]) for name, _ in TREES}
    for P, _ in Ps.values():
        for k, _, _ in TIMING:
            lo, hi = min(lo, min(P[k].values()) - 0.06), max(hi, max(P[k].values()) + 0.06)
    for ax, (name, _) in zip(axes, TREES):
        P, raw = Ps[name]
        for k, _, c in TIMING:
            st = sorted(P[k])
            lw, z = (2.6, 4) if k == 'loss' else (1.3, 3)
            ax.plot([xpos(s) for s in st], [P[k][s] for s in st], color=c, lw=lw, marker='o', markersize=3,
                    markeredgecolor='black', markeredgewidth=0.3, zorder=z)
        style(ax)
        log_x(ax, sorted(P['loss']))
        ax.set_ylim(lo, hi)
        ax.set_xlabel('Training step', fontsize=8.5)
        ax.set_title('Pythia-70m, %s' % name, fontsize=9, fontweight='bold', color=COLORS[name])
    axes[0].set_ylabel('Progress from init (0) to final (1)', fontsize=8.5)
    handles = [Line2D([0], [0], color=c, lw=2.6 if k == 'loss' else 1.3, marker='o', markersize=3,
                      markeredgecolor='black', markeredgewidth=0.3, label=l) for k, l, c in TIMING]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, 'dynamics_timing_compare_70m')
    for name, (P, raw) in Ps.items():
        print(name)
        for k, _, _ in TIMING:
            print('  %-15s' % k + ' '.join('%7.3g' % raw[k][s] for s in sorted(raw[k])))


def save(fig, name):
    os.makedirs(PLOTS, exist_ok=True)
    fig.savefig(os.path.join(PLOTS, name + '.svg'), format='svg', bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS, name + '.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('wrote plots/%s' % name)


if __name__ == '__main__':
    D = {name: load(d) for name, d in TREES}
    fig_pairs(D)
    fig_timing(D)
