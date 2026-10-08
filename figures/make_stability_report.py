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


# ---------------------------------------------------------------- 3. components (item 3)
VARIANT_LABEL = {'base': 'ordinary', 'tokmean': 'token means only', 'freqmatch': 'frequency-matched',
                 'tokres': '− token', 'ctxres': '− token, prev. token', 'tokres+posres': '− token, position',
                 'ctxres+posres': '− token, prev. token, position', 'ctxres+posres+winres': '… − slow drift',
                 'sh:ctxres+posres': '… shared vocabulary', 'tokres+rank': 'rank residual',
                 'cm:tokres+rank': 'rank residual − layer gain'}


SHORT_V = {'base': 'ordinary', 'freqmatch': 'freq.-\nmatched', 'tokres': '− token', 'ctxres+posres': 'residual'}


def explore_means(rows, step, variant):
    R = [r for r in rows if r['step'] == step and r['variant'] == variant]
    w = np.mean([float(r['r']) for r in R if r['kind'] == 'within' and r['fit'] in PROSE])
    a = np.mean([float(r['r']) for r in R if r['kind'] == 'across' and r['fit'] in PROSE and r['eval'] in PROSE])
    return w, a


def fig_components():
    r2 = rd(os.path.join(HERE, 'explore_connectome_run2.csv'))
    r3 = rd(os.path.join(HERE, 'explore_connectome_run3.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(11 * .85, 4.2 * .85), gridspec_kw=dict(width_ratios=[1.25, 1]))
    plt.subplots_adjust(wspace=0.35)
    ax = axes[0]
    show = [('base', r2), ('tokmean', r2), ('freqmatch', r2), ('tokres', r2), ('ctxres', r2), ('ctxres+posres', r3),
            ('ctxres+posres+winres', r3), ('tokres+rank', r2), ('cm:tokres+rank', r2)]
    y = np.arange(len(show))[::-1]
    for yi, (v, rows) in zip(y, show):
        for step, col, off in (('0', UNTRAINED, 0.16), ('143000', TRAINED, -0.16)):
            w, a = explore_means(rows, step, v)
            ax.plot([a, w], [yi + off] * 2, color=col, lw=2.5, alpha=0.45, zorder=2)
            ax.scatter(w, yi + off, s=28, marker='o', color=col, edgecolors='black', lw=0.4, zorder=3)
            ax.scatter(a, yi + off, s=34, marker='D', color=col, edgecolors='black', lw=0.4, zorder=3)
    hl = [i for i, (v, _) in enumerate(show) if v == 'ctxres+posres'][0]
    ax.axhspan(y[hl] - 0.45, y[hl] + 0.45, color='#fff3d6', zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([VARIANT_LABEL[v] for v, _ in show], fontsize=8)
    ax.set_xlim(0.4, 1.01)
    ax.set_xlabel('connectome similarity (r)', fontsize=9)
    style(ax, grid='x')
    letter(ax, 'A', x=-0.42)
    ax.legend(handles=[Line2D([], [], marker='o', ls='', mfc='white', mec='black', label='within dataset'),
                       Line2D([], [], marker='D', ls='', mfc='white', mec='black', label='across datasets'),
                       Line2D([], [], color=UNTRAINED, lw=3, label='untrained (step 0)'),
                       Line2D([], [], color=TRAINED, lw=3, label='trained (step 143k)')],
              frameon=False, fontsize=7, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2)

    ax = axes[1]
    steps = ['0', '64', '1000', '16000', '143000']
    xs = np.arange(len(steps))
    for v, col, rows in (('base', '0.55', r2), ('freqmatch', '#7fa7c9', r2), ('tokres', '#c9a05a', r2), ('ctxres+posres', '#b03a2e', r2)):
        W = [explore_means(rows, s, v)[0] for s in steps]
        A = [explore_means(rows, s, v)[1] for s in steps]
        ax.plot(xs, A, color=col, lw=2, marker='D', ms=5, label=VARIANT_LABEL[v], zorder=3)
        ax.plot(xs, W, color=col, lw=1.2, ls=':', marker='o', ms=3.5, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(['init', '64', '1k', '16k', '143k'], fontsize=8)
    ax.set_xlabel('training step (Pythia-70m)', fontsize=9)
    ax.set_ylabel('connectome similarity (r)', fontsize=9)
    ax.set_ylim(0.4, 1.01)
    style(ax)
    letter(ax, 'B')
    ax.legend(frameon=False, fontsize=7, loc='lower right', title='across (solid), within (dotted)', title_fontsize=7)
    save(fig, 'fig3_components')


# ---------------------------------------------------------------- 4. residual: partitions and 160m
def fig_residual():
    p4 = os.path.join(HERE, 'explore_connectome_run4_partitions.csv')
    p5 = os.path.join(HERE, 'explore_connectome_run5_160m.csv')
    if not (os.path.exists(p4) and os.path.exists(p5)):
        print('skip fig4 (run4/run5 not pulled)')
        return
    fig, axes = plt.subplots(1, 2, figsize=(10 * .85, 3.4 * .85))
    plt.subplots_adjust(wspace=0.35)
    ax = axes[0]
    P = rd(p4)
    vs = ['base', 'tokres', 'ctxres+posres']
    for j, v in enumerate(vs):
        for step, dx, col in (('0', -0.18, UNTRAINED), ('143000', 0.18, TRAINED)):
            rr = [r for r in P if r['variant'] == v and r['step'] == step]
            if not rr:
                continue
            w, a = float(rr[0]['ari_within']), float(rr[0]['ari_across'])
            ax.bar(j + dx - 0.07, w, width=0.14, color=col, edgecolor='black', lw=0.6, zorder=3)
            ax.bar(j + dx + 0.07, a, width=0.14, color=col, edgecolor='black', lw=0.6, hatch='///', zorder=3)
    ax.set_xticks(range(len(vs)))
    ax.set_xticklabels([SHORT_V[v] for v in vs], fontsize=8)
    ax.set_ylabel('partition ARI (k = 100)', fontsize=9)
    ax.set_ylim(0, 0.75)
    style(ax)
    letter(ax, 'A')
    ax.legend(handles=[mpl.patches.Patch(fc='white', ec='black', label='within dataset'),
                       mpl.patches.Patch(fc='white', ec='black', hatch='///', label='across datasets'),
                       mpl.patches.Patch(fc=UNTRAINED, ec='black', label='untrained'),
                       mpl.patches.Patch(fc=TRAINED, ec='black', label='trained')],
              frameon=False, fontsize=7, loc='upper left', ncol=2)
    ax.set_title('Pythia-70m: partitions', fontsize=9, weight='bold')

    ax = axes[1]
    R5 = rd(p5)
    vs = ['base', 'freqmatch', 'tokres', 'ctxres+posres']
    for j, v in enumerate(vs):
        for step, dx, col in (('0', -0.18, UNTRAINED), ('143000', 0.18, TRAINED)):
            w, a = explore_means(R5, step, v)
            ax.plot([j + dx] * 2, [a, w], color=col, lw=3, alpha=0.5, zorder=2)
            ax.scatter(j + dx, w, s=28, marker='o', color=col, edgecolors='black', lw=0.4, zorder=3)
            ax.scatter(j + dx, a, s=34, marker='D', color=col, edgecolors='black', lw=0.4, zorder=3)
    ax.set_xticks(range(len(vs)))
    ax.set_xticklabels([SHORT_V[v] for v in vs], fontsize=8)
    ax.set_ylabel('connectome similarity (r)', fontsize=9)
    ax.set_ylim(0.2, 1.02)
    style(ax)
    letter(ax, 'B')
    ax.set_title('Pythia-160m replication', fontsize=9, weight='bold')
    ax.legend(handles=[Line2D([], [], marker='o', ls='', mfc='white', mec='black', label='within'),
                       Line2D([], [], marker='D', ls='', mfc='white', mec='black', label='across')],
              frameon=False, fontsize=7, loc='lower left')
    save(fig, 'fig4_residual_partitions')


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
    save(fig, 'fig5_consensus_circuits')


if __name__ == '__main__':
    fig_diagnosis()
    fig_consensus()
    fig_components()
    fig_residual()
    fig_consensus_circuits()
