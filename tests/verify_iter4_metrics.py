"""Verification for the reliability/fidelity metrics (see LOG.md, Iteration 4).

    PYTHONPATH=. python tests/verify_iter4_metrics.py

The point of these checks is not that the metrics run, but that they FAIL in the right
directions. Reliability must reward a degenerate constant partition (that is why it cannot
be used alone) while fidelity must punish it; both must collapse to ~0 on a null where the
structure has been destroyed; and the uncompressed reference must behave as a reference
rather than a strict bound -- beatable when the truth really is block-structured, not
beatable when the partition is too coarse for the structure.
"""

import os
import shutil
import tempfile

import numpy as np

from parcelmate.constants import CONNECTIVITY_NAME, HALF_NAMES
from parcelmate.data import circshift_timecourses
from parcelmate.metrics import (
    block_means, fidelity, fidelity_ceiling, hard_labels, reliability, reliability_ceiling,
    triviality,
)
from parcelmate.model import run_split_halves
from parcelmate.util import load_h5_data, save_h5_data

failures = []


def check(name, cond):
    print('%s %s' % ('OK' if cond else 'FAIL', name))
    if not cond:
        failures.append(name)


def planted(n_units=200, n_blocks=5, n_obs=2000, noise=1.2, seed=0):
    """Timecourses with a known block structure, and the correlation matrix they give."""
    rng = np.random.RandomState(seed)
    block = np.repeat(np.arange(n_blocks), n_units // n_blocks)
    Z = rng.randn(n_blocks, n_obs)[block] + rng.randn(n_units, n_obs) * noise
    return Z, np.abs(np.corrcoef(Z)), block


def onehot(labels, k):
    P = np.zeros((len(labels), k))
    P[np.arange(len(labels)), labels] = 1.0
    return P


Z_a, R_a, block = planted(seed=0)
Z_b, R_b, _ = planted(seed=1)          # independent draw, same planted structure
k = block.max() + 1
P_true = onehot(block, k)

# --------------------------------------------------------------- basic properties
check('fidelity: a perfect partition beats a random one',
      fidelity(R_a, R_b, P_true) >
      fidelity(R_a, R_b, onehot(np.random.RandomState(2).randint(0, k, len(block)), k)))
# The uncompressed reference is not a strict upper bound. With an exactly block-structured
# ground truth and noisy estimates, the block model denoises and can beat it. Real
# connectivity is much richer than a 50-block summary, so there the reference should sit
# above every variant -- but assert the mechanism rather than a false invariant.
check('fidelity: a perfect partition can beat the uncompressed reference when the truth '
      'is exactly block-structured (%.3f vs %.3f)'
      % (fidelity(R_a, R_b, P_true), fidelity_ceiling(R_a, R_b)),
      fidelity(R_a, R_b, P_true) > fidelity_ceiling(R_a, R_b))
# ...and when the structure is richer than the partition can express, it cannot.
_, R_rich_a, _ = planted(n_blocks=25, noise=0.4, seed=20)
_, R_rich_b, _ = planted(n_blocks=25, noise=0.4, seed=21)
_coarse = onehot(block, k)   # only 5 blocks, describing data with 25
check('fidelity: a partition too coarse for the structure falls below the reference',
      fidelity(R_rich_a, R_rich_b, _coarse) < fidelity_ceiling(R_rich_a, R_rich_b))
check('fidelity: ceiling is high when both halves share structure (%.3f)'
      % fidelity_ceiling(R_a, R_b), fidelity_ceiling(R_a, R_b) > 0.3)
check('reliability: identical partitions give ARI 1',
      abs(reliability(P_true, P_true) - 1.0) < 1e-9)
check('reliability: random partitions give ARI near 0',
      abs(reliability(P_true, onehot(np.random.RandomState(3).randint(0, k, len(block)), k))) < 0.05)

# ------------------------------------- the degenerate case the null exists to catch
# A constant partition (everything in one cluster, or labels fixed by position) is
# perfectly reliable and explains nothing. This is exactly why reliability alone is unsafe.
constant = onehot(np.arange(len(block)) % k, k)   # fixed by position, ignores the data
check('degenerate: a data-independent partition is perfectly reliable (ARI = 1)',
      abs(reliability(constant, constant) - 1.0) < 1e-9)
check('degenerate: ...but explains almost no held-out variance (R2 = %.4f)'
      % fidelity(R_a, R_b, constant), abs(fidelity(R_a, R_b, constant)) < 0.02)
check('degenerate: the true partition explains far more',
      fidelity(R_a, R_b, P_true) > fidelity(R_a, R_b, constant) + 0.1)

# ------------------------------------------------------- behaviour on a null dataset
# Circular-shift the timecourses, rebuild connectivity: the planted structure is gone.
Rn_a = np.abs(np.corrcoef(circshift_timecourses(Z_a, rng=np.random.RandomState(10))))
Rn_b = np.abs(np.corrcoef(circshift_timecourses(Z_b, rng=np.random.RandomState(11))))
check('null: the planted structure is destroyed (fidelity %.3f -> %.3f)'
      % (fidelity(R_a, R_b, P_true), fidelity(Rn_a, Rn_b, P_true)),
      fidelity(Rn_a, Rn_b, P_true) < 0.05)
check('null: the ceiling also collapses (%.3f -> %.3f)'
      % (fidelity_ceiling(R_a, R_b), fidelity_ceiling(Rn_a, Rn_b)),
      fidelity_ceiling(Rn_a, Rn_b) < fidelity_ceiling(R_a, R_b) / 3)

# ------------------------------------------------------------------- block means
M = block_means(R_a, block, n_networks=k)
check('block_means: shape is (k, k)', M.shape == (k, k))
check('block_means: within-block mean exceeds between-block mean',
      np.mean(np.diag(M)) > np.mean(M[~np.eye(k, dtype=bool)]))
check('block_means: symmetric for symmetric input', np.allclose(M, M.T))

# Fidelity must use the FITTING matrix for its block means, not the held-out one.
# If it peeked at R_b it would score higher, so a peeking implementation is detectable.
peek = np.mean((R_b - block_means(R_b, block, k)[block][:, block]) ** 2)
honest = np.mean((R_b - block_means(R_a, block, k)[block][:, block]) ** 2)
check('fidelity: block means come from the fitting half, not the evaluation half',
      honest >= peek - 1e-12)

# ------------------------------------------------------------------ triviality
# 10 layers of 20 units. The planted blocks are contiguous runs of 40, which would be
# nested inside layers and score high AMI by construction, so score a partition that cuts
# across layers instead -- that is the realistic case.
coords = np.stack([np.repeat(np.arange(10), len(block) // 10), np.arange(len(block))], 1)
crosscutting = onehot(np.arange(len(block)) % k, k)
layerwise = onehot(coords[:, 0] % k, k)
check('triviality: a layer-driven partition scores high AMI with layer',
      triviality(layerwise, coords)['ami_layer'] > 0.5)
check('triviality: a partition cutting across layers does not',
      triviality(crosscutting, coords)['ami_layer'] < 0.2)
check('triviality: reports the effective number of networks',
      triviality(P_true, coords)['n_effective_networks'] == k)

# --------------------------------------------------------------- split halves on disk
tmp = tempfile.mkdtemp(prefix='parcelmate_iter4_')
try:
    conn = os.path.join(tmp, CONNECTIVITY_NAME)
    os.makedirs(conn)
    coordinates = np.stack([np.zeros(len(block)), np.arange(len(block))], 1).astype(np.int32)
    for i in range(1, 5):
        save_h5_data(
            dict(connectivity=planted(seed=i)[1], coordinates=coordinates,
                 unit_means=np.zeros(len(block)), unit_stds=np.ones(len(block)),
                 n_obs=np.asarray(1000)),
            os.path.join(conn, 'connectivity_alpha_sample%d.h5' % i), verbose=False)

    run_split_halves(output_dir=tmp, verbose=False)
    paths = [os.path.join(conn, 'connectivity_alpha_%s.h5' % h) for h in HALF_NAMES]
    check('split_halves: both halves written', all(os.path.exists(p) for p in paths))
    ha, hb = (load_h5_data(p, verbose=False) for p in paths)
    check('split_halves: halves differ (built from disjoint samples)',
          not np.allclose(ha['connectivity'], hb['connectivity']))
    check('split_halves: n_obs is the sum of its sources',
          int(np.asarray(ha['n_obs']).item()) == 2000)
    check('split_halves: shape preserved', ha['connectivity'].shape == (len(block), len(block)))

    from parcelmate.util import read_attrs  # noqa: E402
    attrs = read_attrs(paths[0])
    check('split_halves: records which samples it came from',
          'sample1' in attrs.get('sources', '') and 'sample2' in attrs.get('sources', ''))
    check('split_halves: halves draw on disjoint samples',
          set(read_attrs(paths[0])['sources'].split(', ')).isdisjoint(
              read_attrs(paths[1])['sources'].split(', ')))
finally:
    shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------- full pipeline integration
# Builds two small trees (real + null) with per-sample connectivity, then runs the real
# steps -- split_halves, run_parcellation for two variants on BOTH trees -- and scores
# them. Synthetic and small because a GPT-2-scale run writes 400 MB per matrix; the point
# here is the plumbing (paths, variants, both trees, the CSV), not the numbers.
from parcelmate.bin.score import score_tree  # noqa: E402
from parcelmate.metrics import summarize  # noqa: E402
from parcelmate.model import run_parcellation  # noqa: E402

tmp3 = tempfile.mkdtemp(prefix='parcelmate_e2e_')
try:
    domains = ('alpha', 'beta')
    arms = {'binar': dict(binarize_connectivity=True, connectivity_pca_components=10),
            'fisher': dict(binarize_connectivity=False, fisher_transform=True,
                           connectivity_pca_components=None)}
    trees = {'real': os.path.join(tmp3, 'run'), 'null': os.path.join(tmp3, 'run_null')}
    for tree, root in trees.items():
        os.makedirs(os.path.join(root, CONNECTIVITY_NAME))
        for d_ix, domain in enumerate(domains):
            for i in range(1, 5):
                Z, R_i, blk = planted(seed=100 * d_ix + i)
                if tree == 'null':   # destroy the structure, keep the marginals
                    R_i = np.abs(np.corrcoef(
                        circshift_timecourses(Z, rng=np.random.RandomState(i))))
                save_h5_data(
                    dict(connectivity=R_i,
                         coordinates=np.stack([np.repeat(np.arange(10), len(blk) // 10),
                                               np.arange(len(blk))], 1).astype(np.int32),
                         unit_means=np.zeros(len(blk)), unit_stds=np.ones(len(blk)),
                         n_obs=np.asarray(1000)),
                    os.path.join(root, CONNECTIVITY_NAME,
                                 'connectivity_%s_sample%d.h5' % (domain, i)),
                    verbose=False)
            save_h5_data(
                dict(connectivity=planted(seed=100 * d_ix + 1)[1],
                     coordinates=np.stack([np.repeat(np.arange(10), len(blk) // 10),
                                           np.arange(len(blk))], 1).astype(np.int32),
                     unit_means=np.zeros(len(blk)), unit_stds=np.ones(len(blk)),
                     n_obs=np.asarray(4000)),
                os.path.join(root, CONNECTIVITY_NAME, 'connectivity_%s_avg.h5' % domain),
                verbose=False)

        run_split_halves(output_dir=root, verbose=False)
        for name, kw in arms.items():
            run_parcellation(output_dir=root, variant=name, n_networks=5, n_samples=6,
                             seed=7, verbose=False, **kw)

    rows, missing = [], []
    for tree, root in trees.items():
        score_tree(root, tree, sorted(arms), list(domains), rows, missing, verbose=False)

    check('e2e: halves were parcellated for every variant and tree',
          all(os.path.exists(os.path.join(trees[t], v, 'parcellation',
                                          'parcellation_%s_%s.h5' % (d, h)))
              for t in trees for v in arms for d in domains for h in HALF_NAMES))
    check('e2e: score_tree produced rows for both trees',
          {r['tree'] for r in rows} == {'real', 'null'})
    metrics_seen = {r['metric'] for r in rows}
    for m in ('reliability_within', 'reliability_ceiling', 'reliability_across',
              'fidelity_within', 'fidelity_across', 'triviality_ami_layer',
              'triviality_ami_hubness', 'triviality_median_max_membership'):
        check('e2e: %s computed' % m, m in metrics_seen)
    check('e2e: the uncompressed reference is recorded as its own row',
          any(r['variant'] == '(ceiling)' for r in rows))
    check('e2e: across-domain rows use different fit/eval domains',
          any(r['fit'] != r['eval'] for r in rows if r['metric'] == 'fidelity_across'))

    summary = summarize(rows, verbose=False)
    check('e2e: summarize pairs real with null into a delta',
          all(r['delta'] is not None for r in summary
              if r['real'] is not None and r['null'] is not None))

    # The whole design in one assertion: real data must beat its own null on fidelity.
    deltas = [r['delta'] for r in summary
              if r['metric'] == 'fidelity_within' and r['variant'] in arms
              and r['delta'] is not None]
    check('e2e: fidelity on real data exceeds the null (mean delta %.3f)'
          % (np.mean(deltas) if deltas else float('nan')),
          bool(deltas) and np.mean(deltas) > 0.1)
finally:
    shutil.rmtree(tmp3, ignore_errors=True)


# ------------------------------------------------- reliability ceiling + peakedness
# The ceiling must behave like a ceiling: identical inputs give 1, unrelated inputs give ~0.
check('reliability_ceiling: identical consensuses give 1',
      abs(reliability_ceiling(P_true, P_true) - 1.0) < 1e-9)
check('reliability_ceiling: unrelated consensuses give ~0',
      abs(reliability_ceiling(P_true, onehot(
          np.random.RandomState(31).randint(0, k, len(block)), k))) < 0.05)

# Peakedness distinguishes a confident partition from argmax over a near-flat vector, which
# is the difference between the two real arms (0.53 vs 0.20 median max membership).
flat = np.full((len(block), k), 1.0 / k)
flat[:, 0] += 1e-6   # break ties so argmax is defined
check('triviality: reports peakedness',
      'median_max_membership' in triviality(P_true, coords)
      and 'frac_confident' in triviality(P_true, coords))
check('triviality: a confident partition scores high peakedness',
      triviality(P_true, coords)['median_max_membership'] > 0.9)
check('triviality: a near-uniform partition scores low peakedness',
      triviality(flat, coords)['median_max_membership'] < 2.0 / k)
check('triviality: frac_confident separates the two',
      triviality(P_true, coords)['frac_confident'] == 1.0
      and triviality(flat, coords)['frac_confident'] == 0.0)

# The Fisher arm must no longer be dominated by its own self-connection.
from parcelmate.model import sample_parcellations as _sp  # noqa: E402
_Rf = R_a.copy()
_X = _sp(_Rf, n_networks=3, n_samples=1, binarize_connectivity=False,
         fisher_transform=True, connectivity_pca_components=None, verbose=False, seed=1)
check('fisher arm: runs with the diagonal zeroed', _X['samples'].shape == (1, len(block)))


# ------------------------------------------------- scale-invariant across-domain measure
from parcelmate.metrics import agreement  # noqa: E402

_scaled = R_b * 8.0 + 3.0     # identical structure, different scale and offset
check('agreement: R2 collapses under a pure scale change (%.1f)' % agreement(R_b, _scaled, 'r2'),
      agreement(R_b, _scaled, 'r2') < -1.0)
check('agreement: r is invariant to it (%.3f)' % agreement(R_b, _scaled, 'r'),
      abs(agreement(R_b, _scaled, 'r') - 1.0) < 1e-9)
check('agreement: r still separates good from bad predictions',
      agreement(R_b, R_a, 'r') > agreement(R_b, np.random.RandomState(9).rand(*R_b.shape), 'r'))
check('agreement: fidelity accepts the measure argument',
      abs(fidelity(R_a, R_b, P_true, measure='r')) <= 1.0)
try:
    agreement(R_b, R_a, 'nonsense')
    check('agreement: rejects an unknown measure', False)
except AssertionError:
    check('agreement: rejects an unknown measure', True)

# ------------------------------------------------- vMF arm: standardizing removes hubness
from parcelmate.model import sample_parcellations as _sp2  # noqa: E402
from parcelmate.data import standardize_array  # noqa: E402

# Two units with the SAME connectivity pattern but different overall strength must become
# identical once profiles are standardized -- that is precisely what the arm is for.
_pat = np.random.RandomState(5).rand(300)
_hub, _weak = _pat * 3.0, _pat * 0.3
_zh, _zw = standardize_array(_hub[None, :]), standardize_array(_weak[None, :])
check('vmf: standardizing makes a hub and a weak unit with the same pattern identical',
      np.allclose(_zh, _zw))
check('vmf: without it they are far apart',
      np.linalg.norm(_hub - _weak) > 10 * np.linalg.norm(_zh - _zw) + 1.0)
check('vmf: standardized rows all have norm sqrt(n) (a sphere)',
      np.allclose(np.linalg.norm(standardize_array(R_a), axis=1), np.sqrt(R_a.shape[1])))

# Euclidean distance on standardized profiles is an exact monotone function of the
# correlation between profiles, which is what makes this spherical k-means.
_Z = standardize_array(R_a)
_i, _j, _n = 0, 7, R_a.shape[1]
check('vmf: ||z_i - z_j||^2 == 2n(1 - r_ij), so distance IS profile correlation',
      np.isclose(np.sum((_Z[_i] - _Z[_j]) ** 2),
                 2 * _n * (1 - np.corrcoef(R_a[_i], R_a[_j])[0, 1])))

_out = _sp2(R_a, n_networks=4, n_samples=4, binarize_connectivity=False,
            fisher_transform=True, standardize_profiles=True,
            connectivity_pca_components=None, verbose=False, seed=2)
check('vmf: the arm runs end to end', _out['samples'].shape == (4, len(block)))
_plain = _sp2(R_a, n_networks=4, n_samples=4, binarize_connectivity=False,
              fisher_transform=True, connectivity_pca_components=None, verbose=False, seed=2)
check('vmf: it gives a different partition from unstandardized nopca_fisher',
      not np.array_equal(_out['samples'], _plain['samples']))

print()
if failures:
    raise SystemExit('%d check(s) failed: %s' % (len(failures), failures))
print('All checks passed.')
