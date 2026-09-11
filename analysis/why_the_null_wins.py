"""The figure behind LOG.md Iteration 9: how |r| turns noise into clusterable structure.

    PYTHONPATH=. python analysis/why_the_null_wins.py   ->  plots/why_the_null_wins.{svg,png}

Four panels, left to right, on 300 simulated units:

  a  Shifted data, SIGNED r. Units have no relationship of any kind. The matrix looks like
     what it is: symmetric noise around zero.
  b  The SAME matrix after |r|. A smooth gradient appears. Nothing was added -- every
     negative value was reflected to positive, and the reflection has a systematic size,
     because how far a measured correlation scatters from zero depends on how many
     INDEPENDENT observations the two series really contain. Bartlett's formula gives
     Var(r_ij) ~ (1/T) sum_h rho_i(h) rho_j(h); for AR(1) that is (1+phi_i phi_j) /
     (1 - phi_i phi_j) / T. Note the PRODUCT: the inflation is multiplicative, not an
     additive per-row effect. A slow unit paired with a fast one gets no inflation at all
     (phi = 0.95 with phi = 0.00 has the full T_eff = 3000), which is why this panel shows a
     bright corner where both units are slow rather than a bright cross. Autocorrelation is a
     fixed property of the unit and a circular shift preserves it exactly, so the whole
     pattern survives the null. This is the artifact.
  c  Real-like data, |r|. Strong structure, but continuous and high-dimensional -- there is
     no partition of 20 blocks that describes it.
  d  Why the scores invert. The block model explains 3.5x MORE absolute variance on the real
     data. It scores 5x LOWER because R^2 is a fraction and the real denominator is 19x
     bigger.

Panels a and b are the answer to "how can noise help clustering": after |r| it is not noise,
it is a rank-one gradient -- the simplest and most reproducible thing a block model can be
handed. Units are sorted by mean connectivity in every panel, which is the order the
clusterer effectively discovers in b.
"""

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.expanduser('~/.claude/skills/scientific-figure-style/scripts'))
from figure_style import apply_style, save_fig, style_spines  # noqa: E402

apply_style()
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'DejaVu Sans'

N, T, K, F, SEED = 300, 3000, 20, 60, 0


def ar1(phis, rng):
    out = np.empty((len(phis), T))
    e = rng.standard_normal((len(phis), T))
    out[:, 0] = e[:, 0]
    for t in range(1, T):
        out[:, t] = phis * out[:, t - 1] + np.sqrt(1 - phis ** 2) * e[:, t]
    return out


def block_fit(R, lab):
    oh = np.eye(lab.max() + 1)[lab]
    n = oh.sum(0)
    pred = oh @ ((oh.T @ R @ oh) / np.outer(n, n)) @ oh.T
    ss_res = ((R - pred) ** 2).sum()
    ss_tot = ((R - R.mean()) ** 2).sum()
    return ss_tot, ss_tot - ss_res


def main():
    rng = np.random.default_rng(SEED)
    phi = rng.uniform(0.0, 0.95, N)          # per-unit autocorrelation: a fixed property
    W = rng.standard_normal((N, F)) * rng.random((N, F)) ** 2
    W /= np.linalg.norm(W, axis=1, keepdims=True)

    noise = ar1(phi, rng)
    real = 0.8 * (W @ ar1(np.full(F, 0.7), rng)) + 0.2 * noise

    def conn(X, absolute):
        R = np.corrcoef(X)
        R = np.abs(R) if absolute else R
        np.fill_diagonal(R, 0.0)
        return R

    signed, null, rl = conn(noise, False), conn(noise, True), conn(real, True)
    o_null = np.argsort(null.sum(1))         # sort by mean connectivity
    o_real = np.argsort(rl.sum(1))

    def coarse(M, order, k=K):
        """Average the matrix within k bins of units -- exactly what a block model does.

        Shown coarse-grained rather than raw because at 300x300 the per-entry sampling noise
        is an order of magnitude larger than the gradient, so the raw matrices all look like
        static and the figure would show nothing. Averaging within bins is not cosmetic: it
        is the operation the fidelity metric performs, so these panels are literally what the
        block model has to work with.
        """
        M = M[np.ix_(order, order)]
        edges = np.linspace(0, len(M), k + 1).astype(int)
        return np.array([[M[a:b, c:d].mean() for c, d in zip(edges[:-1], edges[1:])]
                         for a, b in zip(edges[:-1], edges[1:])])

    fig = plt.figure(figsize=(9.6 * 0.92, 3.0 * 0.92), dpi=300)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.25], wspace=0.55)

    panels = [
        (coarse(signed, o_null), 'a   shifted, signed r',
         'no relationships,\nand none visible', True),
        (coarse(null, o_null), 'b   same data, |r|',
         'still no relationships,\nbut a gradient appears', False),
        (coarse(rl, o_real), 'c   real-like, |r|',
         'strong structure, but\ncontinuous -- not blocks', False),
    ]
    for i, (M, title, sub, diverging) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        if diverging:
            lim = np.abs(M).max()
            ax.imshow(M, cmap='RdBu_r', vmin=-lim, vmax=lim, interpolation='nearest')
        else:
            ax.imshow(M, cmap='magma', vmin=M.min(), vmax=M.max(),
                      interpolation='nearest')
        ax.set_title(title, fontsize=8.5, fontweight='bold', loc='left', pad=5)
        ax.text(0.5, -0.10, sub, transform=ax.transAxes, fontsize=7.5,
                color='0.35', ha='center', va='top')
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel('units, sorted by\nmean connectivity', fontsize=8)

    # Panel d: the arithmetic of a ratio. Log axis because the two totals differ 19-fold.
    ax = fig.add_subplot(gs[0, 3])
    lab_n = KMeans(K, n_init=10, random_state=0).fit_predict(null)
    lab_r = KMeans(K, n_init=10, random_state=0).fit_predict(rl)
    tot_n, exp_n = block_fit(null, lab_n)
    tot_r, exp_r = block_fit(rl, lab_r)

    x = np.arange(2)
    w = 0.36
    ax.bar(x - w / 2, [tot_n, tot_r], width=w, color='#e0e0e0', edgecolor='#888888',
           label='variance present', zorder=3)
    ax.bar(x + w / 2, [exp_n, exp_r], width=w, color=['#999999', '#08306b'],
           edgecolor='black', linewidth=0.6, label='explained by 20 blocks', zorder=3)
    for xi, tot, exp in ((0, tot_n, exp_n), (1, tot_r, exp_r)):
        ax.text(xi - w / 2, tot * 1.12, '%.0f' % tot, ha='center', fontsize=7.5,
                color='0.35')
        ax.text(xi + w / 2, exp * 1.12, '%.0f' % exp, ha='center', fontsize=7.5,
                color='0.25')
        # Above the group, not inside it: at the bottom of a log axis the two labels
        # overlapped each other and the bars.
        ax.text(xi, tot * 1.9, 'R$^2$ = %.3f' % (exp / tot), ha='center', fontsize=8.5,
                fontweight='bold', color='#08306b' if xi else '#777777')
    ax.set_yscale('log')
    ax.set_ylim(1, tot_r * 5)
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks(x)
    ax.set_xticklabels(['shifted', 'real-like'], fontsize=8.5)
    ax.set_ylabel('variance (log)', fontsize=9)
    ax.set_title('d   why the scores invert', fontsize=8.5, fontweight='bold', loc='left',
                 pad=5)
    ax.text(0.5, -0.105, 'the real fit explains %.1fx MORE,\nyet scores %.0fx lower'
            % (exp_r / exp_n, (exp_n / tot_n) / (exp_r / tot_r)),
            transform=ax.transAxes, fontsize=7.5, color='0.35', ha='center', va='top')
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
    style_spines(ax, drop_top_right=True)
    ax.tick_params(axis='both', which='major', labelsize=8.5)
    # Below the panel: at the top it collided with the title, and inside it collided with
    # the R^2 labels, which sit above their groups.
    ax.legend(loc='upper center', frameon=False, fontsize=7.5,
              bbox_to_anchor=(0.5, -0.30), handlelength=1.2, ncol=1,
              borderaxespad=0, labelspacing=0.3)

    save_fig(fig, 'why_the_null_wins')


if __name__ == '__main__':
    os.makedirs('plots', exist_ok=True)
    main()
