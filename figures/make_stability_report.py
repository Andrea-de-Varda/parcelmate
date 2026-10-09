"""Figures for plots/stability_and_residual/ (LOG.md Iterations 36-38).

    python figures/make_stability_report.py

Reads only small tables: figures/diag_partition_*.csv, results/qwen35/stability_2b/networks.csv,
figures/consensus_2b_prose_summary.csv, figures/explore_connectome_run*.csv (+ _partitions),
results/explore/run5_160m.csv, results/qwen35/{qwen3.5-2b,qwen3.5-2b-rest,consensus_2b_prose_tree}/circuits/.
Writes one SVG + PNG per figure.
"""

import collections
import csv
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'DejaVu Sans'

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, 'plots', 'stability_and_residual')
os.makedirs(OUT, exist_ok=True)
RES = os.path.join(ROOT, 'results')

REAL, NULL = '#2c5f8a', '#9a9a9a'
WITHIN, ACROSS = '#1f3b57', '#d07a2e'
UNTRAINED, TRAINED = '#b0b0b0', '#2c5f8a'
DOMAIN_COLORS = {'Lan': '#C0392B', 'MD': '#2471A3', 'ToM': '#27AE60', 'phys': '#E67E22'}
DOMAIN_LABELS = {'Lan': 'Language', 'MD': 'Formal', 'ToM': 'Social', 'phys': 'Physical'}
PROSE = ['wikitext', 'bookcorpus', 'agnews', 'tldr17']
SHORT = {'wikitext': 'wiki', 'bookcorpus': 'book', 'agnews': 'news', 'tldr17': 'tldr', 'codeparrot': 'code'}


def rd(p):
    return list(csv.DictReader(open(p)))


def style(ax, grid='y'):
    ax.spines[['top', 'right']].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_linewidth(1.5)
    if grid:
        ax.grid(axis=grid, linestyle='--', alpha=0.5, zorder=0)
    ax.tick_params(labelsize=8)


def letter(ax, t, x=-0.16):
    ax.text(x, 1.06, t, transform=ax.transAxes, fontsize=13, weight='bold', va='bottom')


def save(fig, name):
    for ext in ('svg', 'png'):
        fig.savefig(os.path.join(OUT, '%s.%s' % (name, ext)), format=ext, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


# ---------------------------------------------------------------- 1. diagnosis
def fig_diagnosis():
    fig, axes = plt.subplots(1, 3, figsize=(13 * .85, 3.6 * .85), gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
    plt.subplots_adjust(wspace=0.55)
    ax = axes[0]
    rows = rd(os.path.join(HERE, 'diag_partition_fragility.csv'))
    labels = ['connec-\ntome', 'minus\nstrength', 'profiles', 'top-10%\npartners', 'parti-\ntions']
    x = np.arange(len(rows))
    for i, r in enumerate(rows):
        w, a = float(r['within']), float(r['across'])
        ax.plot([i, i], [a, w], color='0.75', lw=3, zorder=2)
        ax.scatter(i, w, s=40, color=WITHIN, edgecolors='black', lw=0.5, zorder=3)
        ax.scatter(i, a, s=40, color=ACROSS, edgecolors='black', lw=0.5, zorder=3)
        ax.text(i + 0.12, a, '%.0f%%' % (100 * a / w), fontsize=7, va='center', color='0.3')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('similarity', fontsize=9)
    style(ax)
    letter(ax, 'A')
    ax.legend(handles=[Line2D([], [], marker='o', ls='', mfc=WITHIN, mec='black', label='within dataset (half A vs B)'),
                       Line2D([], [], marker='o', ls='', mfc=ACROSS, mec='black', label='across datasets')],
              frameon=False, fontsize=7, loc='lower left')
    ax.set_title('GPT-2: what survives', fontsize=9, weight='bold')

    ax = axes[1]
    nz = rd(os.path.join(HERE, 'diag_partition_noise.csv'))
    for kind, mk in (('iid', 'o'), ('lowrank', 's')):
        pts = sorted((float(r['connectome_r']), float(r['ari'])) for r in nz if r['noise'] == kind)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, color='0.35', lw=1.2, ms=5, label='%s noise' % ('random' if kind == 'iid' else 'low-rank'))
    seed = [float(r['ari']) for r in nz if r['noise'].startswith('same')][0]
    ax.scatter(1.0, seed, marker='D', s=30, color='0.35', zorder=3)
    ax.text(0.985, seed + 0.02, 'other seed', fontsize=7, ha='right')
    ax.scatter(0.513, 0.099, s=70, color=ACROSS, edgecolors='black', zorder=4)
    ax.text(0.53, 0.06, 'across\ndatasets', fontsize=7, color=ACROSS)
    ax.set_xlabel('connectome r with the original', fontsize=9)
    ax.set_ylabel('partition ARI', fontsize=9)
    ax.set_xlim(0.45, 1.03)
    ax.set_ylim(0, 0.5)
    style(ax)
    letter(ax, 'B')
    ax.legend(frameon=False, fontsize=7, loc='upper left')
    ax.set_title('Partitions are fragile', fontsize=9, weight='bold')

    ax = axes[2]
    net = [r for r in rd(os.path.join(RES, 'qwen35', 'stability_2b', 'networks.csv')) if r['tree'] == 'real']
    ds = PROSE + ['codeparrot']
    M = np.zeros((5, 5))
    for i, a in enumerate(ds):
        for j, b in enumerate(ds):
            M[i, j] = np.median([float(r['within'] if a == b else r['across_' + b]) for r in net if r['dataset'] == a])
    im = ax.imshow(M, cmap='Blues', vmin=0, vmax=0.9)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, '%.2f' % M[i, j], ha='center', va='center', fontsize=7, color='white' if M[i, j] > 0.5 else 'black')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([SHORT[d] for d in ds], fontsize=8)
    ax.set_yticklabels([SHORT[d] for d in ds], fontsize=8)
    ax.set_xlabel('found in', fontsize=9)
    ax.set_ylabel('network fitted on', fontsize=9)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('median best-match Dice', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    letter(ax, 'C', x=-0.45)
    ax.set_title('Qwen3.5-2B: same networks?', fontsize=9, weight='bold')
    save(fig, 'fig1_diagnosis')


# ---------------------------------------------------------------- 2. consensus (item 1)
def fig_consensus():
    r = {x['tree']: x for x in rd(os.path.join(HERE, 'consensus_2b_prose_summary.csv'))}
    fig, ax = plt.subplots(figsize=(5.2 * .85, 3.2 * .85))
    items = [('restarts 1-100\nvs 101-200', 'restarts_ari', 'single_dataset_restarts_ari_mean'),
             ('data half A\nvs half B', 'data_ari', 'single_dataset_halves_ari_mean')]
    for i, (lab, key, ref) in enumerate(items):
        for tr, dx, col in (('null', -0.15, NULL), ('real', 0.15, REAL)):
            v = float(r[tr][key])
            ax.bar(i + dx, v, width=0.28, color=col, edgecolor='black', lw=0.8, zorder=3)
            ax.text(i + dx, v + 0.015, '%.2f' % v, ha='center', fontsize=7)
        ax.hlines(float(r['real'][ref]), i - 0.32, i + 0.32, colors='black', linestyles=':', lw=1.3, zorder=4)
    ax.hlines(0.03, 1 - 0.32, 1 + 0.32, colors=ACROSS, linestyles='--', lw=1.3, zorder=4)
    ax.text(1.36, 0.03, 'single partitions\nacross datasets', fontsize=6.5, color=ACROSS, va='center')
    ax.text(1.36, float(r['real']['single_dataset_halves_ari_mean']), 'single dataset,\nwithin', fontsize=6.5, va='center')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([x[0] for x in items], fontsize=8)
    ax.set_xlim(-0.5, 1.9)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel('ARI between consensus partitions', fontsize=9)
    style(ax)
    ax.legend(handles=[mpl.patches.Patch(fc=REAL, ec='black', label='real'), mpl.patches.Patch(fc=NULL, ec='black', label='null partition')],
              frameon=False, fontsize=7, loc='upper right', ncol=1)
    save(fig, 'fig2_consensus')


# ---------------------------------------------------------------- 3. the residual connectome (item 3)
STEPS3 = (('0', 'untrained', UNTRAINED), ('143000', 'trained', TRAINED))
KEEP = (('base', 'ordinary\n(control)'), ('tokres', 'token removed\n(intermediate)'), ('ctxres+posres', 'residual\n(kept)'))


def fig_residual():
    """One run (Pythia-70m, run 4, 81,920 tokens per half) for both panels."""
    rows = rd(os.path.join(HERE, 'explore_connectome_run4.csv'))
    parts = rd(os.path.join(HERE, 'explore_connectome_run4_partitions.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(10 * .85, 3.5 * .85))
    plt.subplots_adjust(wspace=0.3)

    def bars(ax, get, ylab, ylim):
        for j, (v, _) in enumerate(KEEP):
            for k, (step, _, col) in enumerate(STEPS3):
                w, a = get(step, v)
                x0 = j + (k - 0.5) * 0.4
                for dx, val, hatch in ((-0.09, w, None), (0.09, a, '///')):
                    ax.bar(x0 + dx, val, width=0.17, color=col, edgecolor='black', lw=0.6, hatch=hatch, zorder=3)
                    ax.text(x0 + dx, val + 0.012, '%.2f' % val, ha='center', fontsize=6, rotation=90, va='bottom')
        ax.set_xticks(range(len(KEEP)))
        ax.set_xticklabels([lab for _, lab in KEEP], fontsize=8)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_ylim(*ylim)
        ax.axvspan(1.5, 2.5, color='#fff3d6', zorder=0)
        style(ax)

    bars(axes[0], lambda st, v: explore_means(rows, st, v), 'connectome similarity (r)', (0, 1.15))
    letter(axes[0], 'A')
    axes[0].set_title('Connectomes', fontsize=9, weight='bold')

    def part(st, v):
        r = [x for x in parts if x['step'] == st and x['variant'] == v][0]
        return float(r['ari_within']), float(r['ari_across'])
    bars(axes[1], part, 'partition agreement (ARI, k = 100)', (0, 0.72))
    letter(axes[1], 'B')
    axes[1].set_title('Networks (partitions)', fontsize=9, weight='bold')
    axes[1].legend(handles=[mpl.patches.Patch(fc=UNTRAINED, ec='black', label='untrained'),
                            mpl.patches.Patch(fc=TRAINED, ec='black', label='trained'),
                            mpl.patches.Patch(fc='white', ec='black', label='within dataset'),
                            mpl.patches.Patch(fc='white', ec='black', hatch='///', label='across datasets')],
                   frameon=False, fontsize=7, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    save(fig, 'fig3_residual')


def explore_means(rows, step, variant):
    R = [r for r in rows if r['step'] == step and r['variant'] == variant]
    w = np.mean([float(r['r']) for r in R if r['kind'] == 'within' and r['fit'] in PROSE])
    a = np.mean([float(r['r']) for r in R if r['kind'] == 'across' and r['fit'] in PROSE and r['eval'] in PROSE])
    return w, a


# ---------------------------------------------------------------- 5. circuits vs consensus (item 5)
def fig_consensus_circuits():
    from parcelmate.circuits import heldout_enrichment
    sets = [('qwen3.5-2b', 'wikitext', 'wiki'), ('qwen3.5-2b', 'bookcorpus', 'book'), ('qwen3.5-2b-rest', 'agnews', 'news'),
            ('qwen3.5-2b-rest', 'tldr17', 'tldr'), ('consensus_2b_prose_tree', 'consensus', 'consensus')]
    fig, axes = plt.subplots(1, 3, figsize=(13 * .85, 3.3 * .85))
    plt.subplots_adjust(wspace=0.4)
    vals = collections.defaultdict(dict)
    for tree, td, lab in sets:
        d = os.path.join(RES, 'qwen35', tree, 'circuits')
        st = collections.defaultdict(dict)
        for r in rd(os.path.join(d, 'structure.csv')):
            if r['text_domain'] == td and r['pct'] == '0.1':
                st[r['tree']].setdefault(r['measure'], []).append(float(r['value']))
        vals['E'][lab] = {t: np.mean(st[t]['spearman_overlap_excess']) for t in ('real', 'null')}
        g = [r for r in rd(os.path.join(d, 'graded_summary.csv')) if r['text_domain'] == td and r['task_group'] == 'all']
        vals['G'][lab] = {'real': np.mean([float(r['frac_p05_real']) for r in g]), 'null': np.mean([float(r['frac_p05_null']) for r in g])}
        rows = [r for r in rd(os.path.join(d, 'enrichment.csv')) if r['text_domain'] == td and r['pct'] == '0.1']
        he = {}
        for tr in ('real', 'null'):
            rs_ = []
            for h in ('halfA', 'halfB'):
                rr = [r for r in rows if r['tree'] == tr and r['half'] == h]
                tasks = sorted({r['task'] for r in rr})
                k = 1 + max(int(r['network']) for r in rr)
                O, E = np.zeros((len(tasks), k)), np.zeros((len(tasks), k))
                dom = {}
                for r in rr:
                    i = tasks.index(r['task'])
                    O[i, int(r['network'])] = float(r['observed'])
                    E[i, int(r['network'])] = float(r['expected'])
                    dom[r['task']] = r['task_domain']
                rs_ += [x['ratio'] for x in heldout_enrichment(O, E, [dom[t] for t in tasks], 5)]
            he[tr] = np.nanmean(rs_)
        vals['H'][lab] = he
    labs = [s[2] for s in sets]
    for ax, key, ylab, letter_ in ((axes[0], 'E', 'Spearman ρ, overlap vs similarity', 'A'),
                                   (axes[1], 'H', 'held-out enrichment (× chance)', 'B'),
                                   (axes[2], 'G', 'share of tasks explained beyond layer', 'C')):
        for i, lab in enumerate(labs):
            n, rl = vals[key][lab]['null'], vals[key][lab]['real']
            ax.plot([i - 0.12, i + 0.12], [n, rl], color='0.6', lw=1.2, zorder=2)
            ax.scatter(i - 0.12, n, s=38, color=NULL, edgecolors='black', lw=0.6, zorder=3)
            ax.scatter(i + 0.12, rl, s=38, color=REAL, edgecolors='black', lw=0.6, zorder=3)
        ax.axvspan(len(labs) - 1.5, len(labs) - 0.5, color='#fff3d6', zorder=0)
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, fontsize=8)
        ax.set_ylabel(ylab, fontsize=8.5)
        style(ax)
        letter(ax, letter_)
    axes[2].legend(handles=[Line2D([], [], marker='o', ls='', mfc=REAL, mec='black', label='real networks'),
                            Line2D([], [], marker='o', ls='', mfc=NULL, mec='black', label='null partition')],
                   frameon=False, fontsize=7, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    save(fig, 'fig4_consensus_circuits')


if __name__ == '__main__':
    fig_diagnosis()
    fig_consensus()
    fig_residual()
    fig_consensus_circuits()
