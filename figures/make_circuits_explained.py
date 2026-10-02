"""Explanatory figures for plots/circuits_vs_networks/README.md (LOG.md Iteration 36).

    PYTHONPATH=. python figures/make_circuits_explained.py

Needs figures/circuits_example_2b_wikitext.npz (figures/export_circuit_figdata.py) and the
per-run tables under results/qwen35/. Writes into plots/circuits_vs_networks/:

  measures_diagram   how the three measures are computed (toy examples)
  fig1_patching_domains, fig2_worked_example, fig3_concentration,
  fig4_domains_reuse_networks, fig5_shared_structure
"""

import collections
import csv
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy import stats

from parcelmate.circuits import heldout_enrichment

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'DejaVu Sans'
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
RES = os.path.join(ROOT, 'results', 'qwen35')
OUT = os.path.join(ROOT, 'plots', 'circuits_vs_networks')
DC = {'Lan': '#C44E52', 'MD': '#4C72B0', 'ToM': '#55A868', 'phys': '#DD8452'}
DL = {'Lan': 'Language', 'MD': 'Formal', 'ToM': 'Social', 'phys': 'Physical'}
ORDER = ['Lan', 'MD', 'ToM', 'phys']
RAND = '#9a9a9a'
NET_PAL = ['#8dd3c7', '#fdb462', '#bebada', '#fb8072', '#80b1d3', '#b3de69', '#fccde5', '#d9d9d9']
SETS = [('qwen3.5-2b', 'bookcorpus', '2B book'), ('qwen3.5-2b', 'wikitext', '2B wiki'),
        ('qwen3.5-2b-rest', 'agnews', '2B news'), ('qwen3.5-2b-rest', 'tldr17', '2B tldr'),
        ('qwen3.5-2b-rest', 'codeparrot', '2B code'), ('qwen3.5-2b-pool5', 'pooled', '2B pool'),
        ('qwen3.5-4b', 'wikitext', '4B wiki'), ('qwen3.5-4b-pool5', 'pooled', '4B pool')]
X = np.load(os.path.join(HERE, 'circuits_example_2b_wikitext.npz'))
DOM = X['domain']


def style(ax, grid='y'):
    ax.spines[['top', 'right']].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_linewidth(1.5)
    if grid:
        ax.grid(axis=grid, linestyle='--', alpha=0.5, zorder=0)
    ax.tick_params(labelsize=8)


def save(fig, name):
    for ext in ('svg', 'png'):
        fig.savefig(os.path.join(OUT, '%s.%s' % (name, ext)), format=ext, dpi=300, bbox_inches='tight')
    plt.close(fig)


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'


def letter(ax, s):
    ax.text(-0.12, 1.06, s, transform=ax.transAxes, fontsize=12, weight='bold', va='bottom')


def rd(*p):
    return list(csv.DictReader(open(os.path.join(RES, *p))))


def domain_blocks(ax, n_ticks=True):
    """Coloured bars along the axes of a task x task matrix ordered by domain."""
    n = len(DOM)
    for i, d in enumerate(DOM):
        ax.add_patch(Rectangle((-2.2, i - 0.5), 1.5, 1, color=DC[d], clip_on=False))
        ax.add_patch(Rectangle((i - 0.5, n + 0.7), 1, 1.5, color=DC[d], clip_on=False))
    edges = [0] + [i for i in range(1, n) if DOM[i] != DOM[i - 1]] + [n]
    for e in edges[1:-1]:
        ax.axhline(e - 0.5, color='white', lw=1.5)
        ax.axvline(e - 0.5, color='white', lw=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    for s in ax.spines.values():
        s.set_visible(False)


# ============================================================ measures diagram
def measures_diagram():
    fig = plt.figure(figsize=(17 * .85, 5.0 * .85))
    gs = fig.add_gridspec(1, 3, wspace=0.12, width_ratios=[1, 1.25, 1.05])

    # (a) effective networks
    ax = fig.add_subplot(gs[0])
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(0, 9.7, 'a   Effective number of networks', fontsize=10, weight='bold', va='top')
    ax.text(0, 8.9, 'Colour each circuit neuron by its network and count\nthe networks, weighted by size: exp(entropy).',
            fontsize=7.5, color='0.3', va='top')
    def eff(labs):
        p = np.bincount(labs) / len(labs)
        p = p[p > 0]
        v = np.exp(-(p * np.log(p)).sum())
        return '%d' % round(v) if abs(v - round(v)) < 1e-9 else '%.1f' % v
    examples = [(l, eff(np.array(l))) for l in ([0] * 12, [0] * 6 + [1] * 6, [0] * 9 + [1, 2, 3], list(range(6)) * 2)]
    for r, (labs, val) in enumerate(examples):
        y = 6.9 - r * 1.75
        for j, l in enumerate(sorted(labs)):
            ax.add_patch(Circle((0.45 + j * 0.42, y), 0.17, color=NET_PAL[l], ec='0.25', lw=0.6))
        ax.text(5.7, y, '→', fontsize=13, va='center')
        ax.text(6.5, y, '%s effective\nnetwork%s' % (val, '' if val == '1' else 's'), fontsize=8.5, va='center')
    ax.text(0, 0.0, 'Fewer = more concentrated. The test: does a circuit\nuse fewer networks than random neurons drawn with the\nsame number per layer ("random sets", grey in the results)?',
            fontsize=7.3, color='0.3', va='bottom')

    # (b) held-out enrichment
    ax = fig.add_subplot(gs[1])
    ax.axis('off')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.text(0, 9.7, 'b   Do a domain\'s tasks reuse networks?', fontsize=10, weight='bold', va='top')
    ax.text(0, 8.9, 'Choose networks on the other tasks of a domain,\nthen score the task that was left out.',
            fontsize=7.5, color='0.3', va='top')
    rng = np.random.default_rng(3)
    nets = 10
    hot = [2, 5, 7]
    for r, name in enumerate(['task 2', 'task 3', 'task 4']):
        y = 7.3 - r * 0.9
        ax.text(0.0, y, name, fontsize=7.5, va='center')
        v = rng.uniform(0.1, 0.4, nets)
        v[hot] += rng.uniform(0.45, 0.6, len(hot))
        for j in range(nets):
            ax.add_patch(Rectangle((2.1 + j * 0.42, y - 0.3), 0.32, 0.6 * v[j], color='#4C72B0', alpha=0.85))
    ax.text(6.7, 6.4, '} sum the excess over\n   chance per network', fontsize=7.5, va='center')
    yb = 4.4
    for j in range(nets):
        ax.add_patch(Rectangle((2.1 + j * 0.42, yb - 0.25), 0.32, 0.5, color='#e9a23b' if j in hot else '#eeeeee', ec='0.4', lw=0.5))
    ax.text(0.0, yb, 'top networks', fontsize=7.5, va='center')
    ax.text(6.7, yb, '← keep the top 5\n    (here 3)', fontsize=7.5, va='center')
    yh = 2.6
    ax.text(0.0, yh, 'task 1\n(left out)', fontsize=7.5, va='center')
    v = rng.uniform(0.1, 0.4, nets)
    v[hot] += 0.45
    for j in range(nets):
        ax.add_patch(Rectangle((2.1 + j * 0.42, yh - 0.3), 0.32, 0.6 * v[j], color='#C44E52', alpha=0.85,
                               ec='#e9a23b' if j in hot else 'none', lw=1.5))
    ax.text(6.7, yh, '→ share of task 1\'s circuit inside them\n    ÷ share expected by chance', fontsize=7.5, va='center')
    ax.text(0, 0.3, 'Every task is left out in turn. Ratio > 1: tasks of\na domain put their circuits into the same networks.',
            fontsize=7.3, color='0.3', va='bottom')

    # (c) non-shared similarity
    ax = fig.add_subplot(gs[2])
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(0.4, 9.7, 'c   Do overlapping tasks share networks?', fontsize=10, weight='bold', va='top')
    ax.text(0.4, 8.9, 'Shared neurons are left out: they would be in\nthe same networks under any partition.', fontsize=7.5, color='0.3', va='top')
    ax.add_patch(Circle((2.6, 5.6), 1.7, color='#C44E52', alpha=0.35, ec='#C44E52', lw=1.5))
    ax.add_patch(Circle((4.6, 5.6), 1.7, color='#4C72B0', alpha=0.35, ec='#4C72B0', lw=1.5))
    ax.text(1.8, 5.6, 'A only', fontsize=8, ha='center', va='center')
    ax.text(5.4, 5.6, 'B only', fontsize=8, ha='center', va='center')
    ax.text(3.6, 5.6, 'shared\n(excluded)', fontsize=6.5, ha='center', va='center', color='0.25')
    ax.text(3.6, 7.6, 'circuit overlap  (patching)', fontsize=7.5, ha='center')
    for i, (x0, col) in enumerate(((0.6, '#C44E52'), (4.4, '#4C72B0'))):
        v = np.array([0.8, 0.2, 0.6, 0.1, 0.5, 0.15]) if i == 0 else np.array([0.7, 0.15, 0.5, 0.25, 0.6, 0.1])
        for j, h in enumerate(v):
            ax.add_patch(Rectangle((x0 + j * 0.42, 2.0), 0.32, 1.3 * h, color=col, alpha=0.85))
        ax.text(x0 + 1.2, 1.6, 'networks of\n%s only' % ('A' if i == 0 else 'B'), fontsize=7, ha='center', va='top')
    ax.text(3.65, 2.7, '≈ ?', fontsize=11, ha='center', va='center')
    ax.text(8.6, 2.7, 'cosine\nsimilarity,\nminus that of\nrandom sets', fontsize=7.3, ha='center', va='center')
    ax.text(0.4, 0.0, 'Across all task pairs: does more circuit overlap go\nwith more similar networks for the remaining neurons?',
            fontsize=7.3, color='0.3', va='bottom')
    save(fig, 'measures_diagram')


# ============================================================ fig 1: patching domains
def fig1():
    fig = plt.figure(figsize=(11.5 * .85, 4.2 * .85))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.5)
    ax = fig.add_subplot(gs[0])
    O = X['overlap'].copy()
    np.fill_diagonal(O, np.nan)
    im = ax.imshow(O, cmap='Greys', vmin=0, vmax=0.4)
    domain_blocks(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label('circuit overlap', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title('Circuit overlap between tasks (2B)', fontsize=9.5, weight='bold')
    ax.text(0.5, -0.17, 'tasks ordered by domain; dark = share many circuit neurons', transform=ax.transAxes,
            ha='center', fontsize=7.5, color='0.35')
    letter(ax, 'a')
    ax = fig.add_subplot(gs[1])
    x = 0
    ticks = []
    for d in ORDER:
        for m, tree in (('2B', 'qwen3.5-2b'), ('4B', 'qwen3.5-4b')):
            r = [r for r in rd(tree, 'circuits', 'domain_overlap.csv') if r['pct'] == '0.1' and r['task_domain'] == d][0]
            w, a, p = float(r['mean_within']), float(r['mean_across']), float(r['p'])
            ax.bar(x - 0.2, w, 0.38, color=DC[d], ec='black', lw=0.6, zorder=3)
            ax.bar(x + 0.2, a, 0.38, color=DC[d], alpha=0.3, ec=DC[d], lw=0.8, zorder=3)
            ax.text(x, max(w, a) + 0.008, stars(p), ha='center', fontsize=7.5)
            ticks.append((x, m))
            x += 1
        ax.text(x - 1.5, -0.075, DL[d], ha='center', va='top', fontsize=8.5, color=DC[d], weight='bold', transform=ax.get_xaxis_transform())
        x += 0.5
    ax.set_xticks([t[0] for t in ticks])
    ax.set_xticklabels([t[1] for t in ticks], fontsize=7.5)
    ax.tick_params(axis='x', pad=1)
    ax.set_ylabel('mean circuit overlap', fontsize=8.5)
    ax.set_ylim(0, 0.21)
    style(ax)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color='0.35', ec='black', lw=0.6, label='same-domain task pairs'),
                       Rectangle((0, 0), 1, 1, color='0.35', alpha=0.3, label='cross-domain task pairs')],
              frameon=False, fontsize=7.5, loc='upper right')
    ax.set_title('Same-domain tasks share circuits', fontsize=9.5, weight='bold')
    letter(ax, 'b')
    save(fig, 'fig1_patching_domains')


# ============================================================ fig 2: worked example
def fig2():
    eff_r, dr = X['real_eff'], X['real_eff_draws']
    # the Language task whose circuit is most concentrated relative to random
    cand = [i for i, d in enumerate(DOM) if d == 'Lan']
    i = max(cand, key=lambda i: (dr[i].mean() - eff_r[i]) / dr[i].std())
    fig = plt.figure(figsize=(13 * .85, 3.6 * .85))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    o, e = X['real_observed'][i].astype(float), X['real_expected'][i]
    order = np.argsort(-o, kind='stable')[:40]
    xs = np.arange(len(order))
    ax.bar(xs, o[order], 0.8, color=DC['Lan'], ec='black', lw=0.4, zorder=3, label='circuit (target)')
    ax.step(xs, e[order], where='mid', color='black', lw=1.4, zorder=4, label='random sets (mean)')
    ax.set_xlim(-0.8, len(order) - 0.2)
    ax.set_xticks([])
    ax.set_xlabel('networks, sorted by circuit count (40 of 100 shown)', fontsize=8)
    ax.set_ylabel('circuit neurons in network', fontsize=8.5)
    style(ax)
    ax.legend(frameon=False, fontsize=7.5)
    n = o.sum()
    top5 = o[order[:5]].sum() / n
    ax.text(0.98, 0.62, 'top 5 networks hold %.0f%% of the circuit\n(random sets: %.0f%%)' % (100 * top5, 100 * np.sort(e)[::-1][:5].sum() / n),
            transform=ax.transAxes, ha='right', fontsize=7.5, bbox=dict(fc='white', ec='gray', boxstyle='round,pad=0.3'))
    ax.set_title('One circuit across the networks\n(task: %s)' % X['task'][i].replace('_', ' '), fontsize=9.5, weight='bold')
    letter(ax, 'a')
    for j, (tree, ttl) in enumerate((('real', 'Real networks'), ('null', 'Null partition'))):
        ax = fig.add_subplot(gs[1 + j])
        d = X['%s_eff_draws' % tree][i]
        v = X['%s_eff' % tree][i]
        ax.hist(d, bins=30, color=RAND, alpha=0.8, zorder=3)
        ax.axvline(v, color=DC['Lan'] if tree == 'real' else 'black', lw=2.2, zorder=4)
        p = (1 + (d <= v).sum()) / (1 + len(d))
        ax.text(0.97, 0.95, 'circuit: %.0f\nrandom: %.0f ± %.0f\np = %.3f' % (v, d.mean(), d.std(), p),
                transform=ax.transAxes, va='top', ha='right', fontsize=7.5, bbox=dict(fc='white', ec='gray', boxstyle='round,pad=0.3'))
        ax.set_xlabel('effective number of networks', fontsize=8.5)
        if j == 0:
            ax.set_ylabel('random sets (of 1,000)', fontsize=8.5)
        ax.set_title(ttl, fontsize=9.5, weight='bold')
        ax.set_xlim(15, 70)
        style(ax)
        letter(ax, 'bc'[j])
    fig.text(0.5, -0.08, 'Qwen3.5-2B, networks fitted on wikitext (half A), circuit = top 0.1%% of 147,456 neurons (%d). Random sets: 1,000 draws '
             'with the circuit\'s number of neurons in every layer. Null partition: networks fitted on time-shifted activations.' % n,
             ha='center', fontsize=7, color='0.35')
    save(fig, 'fig2_worked_example')
    return i


# ============================================================ fig 3: concentration, all tasks
def fig3():
    fig = plt.figure(figsize=(13 * .85, 5.0 * .85))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.32)
    idx = [i for d in ORDER for i in np.flatnonzero(DOM == d)]
    for j, (tree, ttl) in enumerate((('real', 'Real networks'), ('null', 'Null partition'))):
        ax = fig.add_subplot(gs[j])
        for y, i in enumerate(idx):
            v, r = X['%s_eff' % tree][i], X['%s_eff_draws' % tree][i].mean()
            c = DC[DOM[i]]
            ax.hlines(y, v, r, color=c, alpha=0.4, lw=2.5, zorder=2)
            ax.scatter(r, y, s=22, facecolor='white', edgecolor=RAND, lw=1.3, zorder=3)
            ax.scatter(v, y, s=22, color=c, edgecolor='black', lw=0.4, zorder=4)
        ax.set_yticks([])
        ax.set_ylim(len(idx) - 0.5, -0.5)
        ax.set_xlim(20, 75)
        ax.set_xlabel('effective number of networks', fontsize=8.5)
        n_less = int((X['%s_eff' % tree] < X['%s_eff_draws' % tree].mean(1)).sum())
        ax.set_title('%s: %d of %d circuits\nin fewer networks than random' % (ttl, n_less, len(idx)), fontsize=9, weight='bold')
        style(ax, grid='x')
        letter(ax, 'ab'[j])
        if j == 0:
            for d in ORDER:
                ys = [y for y, i in enumerate(idx) if DOM[i] == d]
                ax.text(19, np.mean(ys), DL[d], color=DC[d], ha='right', va='center', fontsize=8, weight='bold')
            ax.legend(handles=[Line2D([], [], marker='o', ls='', mfc='0.3', mec='black', label='circuit'),
                               Line2D([], [], marker='o', ls='', mfc='white', mec=RAND, mew=1.3, label='random sets (mean)')],
                      frameon=False, fontsize=7.5, loc='upper left', bbox_to_anchor=(0.0, -0.12), ncol=2)
    # summary over network sets: circuit minus random, real vs null
    ax = fig.add_subplot(gs[2])
    for s, (tree, td, label) in enumerate(SETS):
        rows = [r for r in rd(tree, 'circuits', 'concentration.csv') if r['pct'] == '0.1' and r['text_domain'] == td]
        for tr, dx, col in (('null', -0.15, RAND), ('real', 0.15, '#2c5f8a')):
            per = collections.defaultdict(list)
            for r in rows:
                if r['tree'] == tr:
                    per[r['task']].append(float(r['effective_networks']) - float(r['null_effective_networks_mean']))
            v = np.array([np.mean(x) for x in per.values()])
            ax.errorbar(s + dx, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), fmt='o', color=col, mec='black', mew=0.5,
                        ms=6, capsize=2, zorder=4)
        per = collections.defaultdict(list)
        for r in rd(tree, 'circuits', 'paired_tasks.csv'):
            if r['pct'] == '0.1' and r['text_domain'] == td:
                per[r['task']].append(float(r['deficit_diff']))
        p = stats.wilcoxon([np.mean(v) for v in per.values()], alternative='less').pvalue
        ax.text(s, 6.3, stars(p), ha='center', fontsize=7.5)
    ax.axhline(0, color='#333333', lw=0.9, zorder=2)
    ax.set_xticks(range(len(SETS)))
    ax.set_xticklabels([l.replace(' ', '\n') for _, _, l in SETS], fontsize=7)
    ax.set_ylim(-17, 7.5)
    ax.set_ylabel('effective networks, circuit − random\n(mean ± SE over tasks)', fontsize=8.5)
    style(ax)
    ax.legend(handles=[Line2D([], [], marker='o', ls='', mfc='#2c5f8a', mec='black', mew=0.5, label='real networks'),
                       Line2D([], [], marker='o', ls='', mfc=RAND, mec='black', mew=0.5, label='null partition')],
              frameon=False, fontsize=7.5, loc='upper left', bbox_to_anchor=(0.0, -0.14), ncol=2)
    ax.set_title('All network sets\n(stars: real more concentrated than null, paired)', fontsize=9, weight='bold')
    letter(ax, 'c')
    fig.text(0.5, -0.1, 'a-b: every 2B task, networks fitted on wikitext (half A); line = how much fewer networks the circuit uses than random '
             'neurons from the same layers. c: the same difference averaged over tasks for every network set.', ha='center', fontsize=7, color='0.35')
    save(fig, 'fig3_concentration')


# ============================================================ fig 4: domains reuse networks
def fig4():
    shares = {tr: collections.defaultdict(list) for tr in ('real', 'null')}
    for tree, td, label in SETS:
        rows = [r for r in rd(tree, 'circuits', 'enrichment.csv') if r['text_domain'] == td and r['pct'] == '0.1']
        for tr in ('real', 'null'):
            for half in ('halfA', 'halfB'):
                rr = [r for r in rows if r['tree'] == tr and r['half'] == half]
                tasks = sorted({r['task'] for r in rr})
                k = 1 + max(int(r['network']) for r in rr)
                O, E = np.zeros((len(tasks), k)), np.zeros((len(tasks), k))
                dom = {}
                for r in rr:
                    a = tasks.index(r['task'])
                    O[a, int(r['network'])] = float(r['observed'])
                    E[a, int(r['network'])] = float(r['expected'])
                    dom[r['task']] = r['task_domain']
                res = heldout_enrichment(O, E, [dom[t] for t in tasks], 5)
                for x, t in zip(res, tasks):
                    if np.isfinite(x['ratio']):
                        shares[tr][dom[t]].append((label, half, x['share'], x['expected_share']))
    fig, axes = plt.subplots(1, 2, figsize=(11 * .85, 3.8 * .85), gridspec_kw=dict(width_ratios=[1.3, 1], wspace=0.35))
    ax = axes[0]
    for j, d in enumerate(ORDER):
        for tr, dx in (('null', -0.2), ('real', 0.2)):
            per = collections.defaultdict(list)
            for label, half, s, e in shares[tr][d]:
                per[(label, half)].append((s, e))
            ss = np.array([np.mean([a for a, b in v]) for v in per.values()])
            ee = np.array([np.mean([b for a, b in v]) for v in per.values()])
            col = DC[d] if tr == 'real' else RAND
            ax.bar(j + dx, 100 * ss.mean(), 0.36, color=col, ec='black', lw=0.6, zorder=3, alpha=0.9)
            ax.bar(j + dx, 100 * ee.mean(), 0.36, color='white', ec='black', lw=0.8, hatch='////', zorder=4, alpha=0.9)
            ax.scatter(j + dx + np.random.default_rng(j).uniform(-0.1, 0.1, len(ss)), 100 * ss, s=7, color='black', alpha=0.35, zorder=5)
    ax.set_xticks(range(4))
    ax.set_xticklabels([DL[d] for d in ORDER], fontsize=8.5)
    ax.set_ylabel('% of a left-out task\'s circuit in the\n5 networks its sibling tasks use most', fontsize=8.5)
    style(ax)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color='0.4', ec='black', lw=0.6, label='observed (left: null partition, right: real)'),
                       Rectangle((0, 0), 1, 1, fc='white', ec='black', hatch='////', label='expected by chance')],
              frameon=False, fontsize=7, loc='upper right')
    ax.set_title('A new task lands in its domain\'s networks', fontsize=9.5, weight='bold')
    letter(ax, 'a')
    ax = axes[1]
    for j, d in enumerate(ORDER):
        rat = {}
        for tr in ('real', 'null'):
            per = collections.defaultdict(list)
            for label, half, s, e in shares[tr][d]:
                per[(label, half)].append(s / e)
            rat[tr] = np.array([np.mean(v) for _, v in sorted(per.items())])
        for a, b in zip(rat['null'], rat['real']):
            ax.plot([j - 0.15, j + 0.15], [a, b], color='0.75', lw=0.7, zorder=2)
        ax.scatter([j - 0.15] * len(rat['null']), rat['null'], s=10, color=RAND, zorder=3)
        ax.scatter([j + 0.15] * len(rat['real']), rat['real'], s=10, color=DC[d], zorder=3)
        ax.scatter(j - 0.15, rat['null'].mean(), s=50, color=RAND, ec='black', zorder=4)
        ax.scatter(j + 0.15, rat['real'].mean(), s=50, color=DC[d], ec='black', zorder=4)
        ax.text(j + 0.15, rat['real'].mean(), '  %.1f×' % rat['real'].mean(), fontsize=7.5, va='center')
        p = stats.wilcoxon(rat['real'] - rat['null'], alternative='greater').pvalue
        ax.text(j, 4.35, stars(p), ha='center', fontsize=7.5)
    ax.axhline(1, color='#333333', ls='--', lw=0.9)
    ax.set_ylim(0.6, 4.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels([DL[d] for d in ORDER], fontsize=8.5)
    ax.set_ylabel('observed ÷ expected', fontsize=8.5)
    style(ax)
    ax.set_title('Ratio: null partition (grey) vs real', fontsize=9.5, weight='bold')
    letter(ax, 'b')
    fig.text(0.5, -0.06, 'Leave-one-task-out (see the measures diagram, b). Dots: the eight network sets x two halves; bars: their mean.',
             ha='center', fontsize=7, color='0.35')
    save(fig, 'fig4_domains_reuse_networks')


# ============================================================ fig 5: shared structure
def fig5():
    fig = plt.figure(figsize=(13.5 * .85, 7.6 * .85))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.9], hspace=0.45, wspace=0.35)
    def rel(M):
        return M - np.nanmean(M[np.triu_indices(len(M), 1)])
    mats = [(X['overlap'], 'Circuit overlap (patching)', 'Greys', (0, 0.4)),
            (rel(X['real_similarity']), 'Network similarity, real networks', 'RdBu_r', (-0.3, 0.3)),
            (rel(X['null_similarity']), 'Network similarity, null partition', 'RdBu_r', (-0.3, 0.3))]
    for j, (M, ttl, cmap, lim) in enumerate(mats):
        ax = fig.add_subplot(gs[0, j])
        M = M.copy()
        np.fill_diagonal(M, np.nan)
        im = ax.imshow(M, cmap=cmap, vmin=lim[0], vmax=lim[1])
        domain_blocks(ax)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cb.ax.tick_params(labelsize=7)
        ax.set_title(ttl, fontsize=9, weight='bold')
        letter(ax, 'abc'[j])
    fig.text(0.5, 0.49, 'b, c: similarity of the networks used by the neurons two tasks do NOT share (minus what random neurons from the same layers '
             'give), relative to the average task pair: red = more similar than the average pair. Tasks ordered by domain as in a.', ha='center', fontsize=7, color='0.35')
    iu = np.triu_indices(len(DOM), 1)
    o = X['overlap'][iu]
    same = DOM[iu[0]] == DOM[iu[1]]
    for j, (tree, ttl) in enumerate((('real', 'Real networks'), ('null', 'Null partition'))):
        ax = fig.add_subplot(gs[1, j])
        s = rel(X['%s_similarity' % tree])[iu]
        ax.scatter(o[~same], s[~same], s=9, color='#bbbbbb', zorder=3, label='cross-domain pair')
        ax.scatter(o[same], s[same], s=16, c=[DC[d] for d in DOM[iu[0]][same]], ec='black', lw=0.3, zorder=4, label='same-domain pair')
        ok = np.isfinite(s)
        rho = stats.spearmanr(o[ok], s[ok])[0]
        ax.text(0.97, 0.05, 'ρ = %.2f' % rho, transform=ax.transAxes, ha='right', fontsize=8,
                bbox=dict(fc='white', ec='gray', boxstyle='round,pad=0.3'))
        ax.axhline(0, color='#333333', lw=0.8)
        ax.set_xlabel('circuit overlap (patching)', fontsize=8.5)
        if j == 0:
            ax.set_ylabel('network similarity of non-shared\nneurons (relative to average pair)', fontsize=8.5)
        ax.set_ylim(-0.35, 0.65)
        ax.set_title(ttl + ': each dot a task pair', fontsize=9, weight='bold')
        style(ax)
        letter(ax, 'de'[j])
        if j == 0:
            ax.legend(frameon=False, fontsize=7, loc='upper left')
    ax = fig.add_subplot(gs[1, 2])
    for sidx, (tree, td, label) in enumerate(SETS):
        st = collections.defaultdict(dict)
        for r in rd(tree, 'circuits', 'structure.csv'):
            if r['pct'] == '0.1' and r['text_domain'] == td:
                st[(r['tree'], r['half'])][r['measure']] = float(r['value'])
        v = {tr: np.mean([st[(tr, h)]['spearman_overlap_excess'] for h in ('halfA', 'halfB')]) for tr in ('real', 'null')}
        ax.plot([sidx, sidx], [v['null'], v['real']], color='0.6', lw=1.2, zorder=2)
        ax.scatter(sidx, v['null'], s=36, color=RAND, ec='black', lw=0.5, zorder=3)
        ax.scatter(sidx, v['real'], s=36, color='#2c5f8a', ec='black', lw=0.5, zorder=3)
    ax.set_xticks(range(len(SETS)))
    ax.set_xticklabels([l.replace(' ', '\n') for _, _, l in SETS], fontsize=7)
    ax.set_ylabel('Spearman ρ (as in d, e)', fontsize=8.5)
    ax.set_ylim(0, 0.8)
    style(ax)
    ax.legend(handles=[Line2D([], [], marker='o', ls='', mfc='#2c5f8a', mec='black', mew=0.5, label='real networks'),
                       Line2D([], [], marker='o', ls='', mfc=RAND, mec='black', mew=0.5, label='null partition')],
              frameon=False, fontsize=7, loc='upper left')
    ax.set_title('All network sets', fontsize=9, weight='bold')
    letter(ax, 'f')
    save(fig, 'fig5_shared_structure')


os.makedirs(OUT, exist_ok=True)
measures_diagram()
fig1()
example = fig2()
fig3()
fig4()
fig5()
print('example task:', X['task'][example])
print('wrote plots/circuits_vs_networks/{measures_diagram,fig1..fig5}.{svg,png}')
