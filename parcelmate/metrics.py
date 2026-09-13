"""Reliability and fidelity scoring for parcellations.

Two metrics, deliberately degenerate in opposite directions, so that a change which
improves both is a real improvement rather than a move along a tradeoff curve:

  reliability  Do two parcellations of the same domain, fit on disjoint tokens, agree?
               Compares a partition to *another partition* (adjusted Rand index).
               Maximized by trivial solutions: an algorithm that always returns the same
               labels scores 1.0 without describing anything.

  fidelity     Does a partition fit on half A explain the held-out connectivity of half B?
               Compares a partition to *a data matrix* (variance explained by its block
               structure). Maximized by overfit solutions: more blocks always explain more.

Both are reported relative to a circular-shift null, which is what stops the degenerate
cases from scoring well. The null destroys cross-unit correlation while preserving each
unit's own marginal and autocorrelation, so an algorithm whose apparent stability comes
from its own inductive bias scores the same on the null as on real data, and the
difference is zero.

Each metric also gets a ceiling, and the two ceilings partial out different nuisances:

  fidelity ceiling     Predict R_B from R_A directly, no parcellation. Holds the data
                       split fixed and removes the *model* limitation, so what is left is
                       sampling noise between halves. NOT a strict upper bound -- see
                       `fidelity_ceiling`.
  reliability ceiling  Compare two consensuses built from disjoint halves of the SAME
                       restarts on the SAME data. Holds the model fixed and removes the
                       *data* difference, so what is left is algorithmic instability.

The second matters more than it looks. Arms differ enormously in how stable their
clustering is -- measured on real wikitext, `current` scores ARI 0.096 across seeds on
identical data while `nopca_fisher` scores 0.741 -- so raw reliability is not comparable
across arms without dividing through by each arm's own noise floor.

Raw components are always reported alongside any derived quantity: the caller gets real,
null and ceiling separately, and computes deltas or ratios downstream. Nothing is stored
pre-normalized.
"""

import numpy as np
from scipy import optimize
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from parcelmate.util import stderr


def hard_labels(parcellation):
    """Soft memberships (n_units x n_networks) -> one label per unit."""
    return np.asarray(parcellation).argmax(axis=1)


def upper_triangle(R):
    """Off-diagonal entries of a symmetric matrix, flattened.

    The diagonal is excluded throughout: it is 1.0 by construction, carries no
    information, and would otherwise inflate every variance-explained figure.
    """
    iu = np.triu_indices(R.shape[0], k=1)
    return R[iu]


def block_means(R, labels, n_networks=None):
    """Mean connectivity within each (network, network) block.

    Estimated on the *fitting* matrix and applied unchanged to the held-out one, so both
    the partition and the values it predicts with are out of sample.
    """
    R = np.array(R, dtype=np.float64, copy=True)
    labels = np.asarray(labels)
    k = int(n_networks or labels.max() + 1)
    onehot = np.zeros((len(labels), k), dtype=np.float64)
    onehot[np.arange(len(labels)), labels] = 1.0
    sizes = onehot.sum(axis=0)

    # The self-pairs R[i, i] are excluded, to match `variance_explained`, which scores the
    # strict upper triangle. Leaving them in would pull every within-block mean toward 1.
    diag = np.diag(R).copy()
    np.fill_diagonal(R, 0.0)
    sums = onehot.T @ R @ onehot

    # Pair counts are the OUTER product of the block sizes: block (a, b) contains
    # n_a * n_b ordered pairs. `onehot.T @ onehot` gives diag(n) instead, which zeroed
    # every off-diagonal block and made the predictions nonsense.
    counts = np.outer(sizes, sizes) - np.diag(sizes)  # drop the n_a self-pairs on a == b
    with np.errstate(invalid='ignore', divide='ignore'):
        M = np.where(counts > 0, sums / np.maximum(counts, 1.0), 0.0)
    np.fill_diagonal(R, diag)  # leave the caller's view of R untouched in spirit

    return M


def agreement(R_eval, prediction, measure='r2'):
    """How well a prediction matches the off-diagonal of R_eval.

    `measure='r2'` gives variance explained, which is the stricter quantity: it punishes
    getting the scale wrong as well as the pattern. Not clipped at zero, because a negative
    value is informative -- it means the prediction is worse than the grand mean.

    `measure='r'` gives the Pearson correlation between prediction and target, which is
    invariant to scale and offset. Required whenever the two matrices live on different
    scales. Domains differ enormously in overall correlation magnitude -- whitespace sits at
    mean |r| 0.37 against wikitext at 0.047 -- so an R^2 computed across domains measures
    the scale mismatch rather than whether the structure transfers, and comes out at -118
    for whitespace against agnews. Within a domain the two halves share a scale, so R^2 is
    meaningful there and is reported alongside r.
    """
    y = upper_triangle(np.asarray(R_eval, dtype=np.float64))
    yhat = upper_triangle(np.asarray(prediction, dtype=np.float64)) \
        if np.ndim(prediction) == 2 else np.asarray(prediction, dtype=np.float64)
    if measure == 'r':
        if y.std() == 0 or yhat.std() == 0:
            return float('nan')
        return float(np.corrcoef(y, yhat)[0, 1])
    assert measure == 'r2', 'measure must be "r2" or "r", got %r' % measure
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        return float('nan')

    return float(1.0 - np.sum((y - yhat) ** 2) / ss_tot)


def variance_explained(R_eval, prediction):
    """Backwards-compatible alias for `agreement(..., measure='r2')`."""
    return agreement(R_eval, prediction, measure='r2')


def fidelity(R_fit, R_eval, parcellation_fit, measure='r2'):
    """Variance of held-out connectivity explained by a partition fit elsewhere.

    `R_fit`/`parcellation_fit` come from the fitting data; `R_eval` is held out. Block
    means are estimated on R_fit, so both the partition and the values it predicts with
    are out of sample.

    Hard argmax labels, deliberately. The deliverable is a discrete parcellation, and
    reliability (ARI) requires hard labels anyway, so both metrics then describe the same
    object. An earlier `soft=True` path was removed: it took block means from the argmax
    labels and only softened the *prediction*, which corresponds to no clean model. A
    genuinely soft version would need soft block means too, at which point it is a
    mixed-membership block model -- a different question, not a variant of this one.

    Note that argmax is only meaningful when memberships are actually peaked. See
    `triviality`, which reports peakedness alongside, because the two arms differ sharply:
    `nopca_fisher` puts half its units above 0.5 membership, `current` puts 1.6% there.
    """
    P = np.asarray(parcellation_fit, dtype=np.float64)
    labels = hard_labels(P)
    M = block_means(R_fit, labels, n_networks=P.shape[1])

    return agreement(R_eval, M[labels][:, labels], measure=measure)


def fidelity_insample(R_fit, parcellation_fit, measure='r2'):
    """How well the partition describes the data it was fit to -- the block-model ceiling.

    `fidelity_ceiling` is a very loose bound: it removes compression entirely, so it says
    what a predictor with 50 million free parameters achieves. A 50-cluster block model has
    about 1,275, a compression of roughly 40,000 to 1, and cannot approach that however
    good the partition is. Scoring the fitted partition on its own data gives the tighter
    reference -- the best this model class can do here.

    The ratio of held-out to in-sample fidelity then separates two explanations that the
    uncompressed ceiling alone conflates: a low held-out score with a high in-sample score
    means the partition overfits its half, while both being low means block models are
    simply a poor description of this connectome regardless of the partition.

    Recomputes the block means rather than sharing them with the held-out call. That is
    about 0.2 s per call against a ~34 min scoring job, and keeping the two paths
    independent means neither can silently contaminate the other.
    """
    return fidelity(R_fit, R_fit, parcellation_fit, measure=measure)


def fidelity_ceiling(R_fit, R_eval, measure='r2'):
    """Predict held-out connectivity from the fitting matrix directly, uncompressed.

    No parcellation, no clustering: every unit keeps its own identity, so whatever is left
    unexplained is the sampling noise between the two halves. This is the reference a
    parcellation is working against: 0.30 against a reference of 0.35 is doing well, the
    same 0.30 against 0.95 is not.

    It is NOT a strict upper bound, and the exception is informative. The uncompressed
    predictor carries all of the fitting half's noise into the prediction, whereas a block
    model averages that noise away. So when the underlying structure genuinely is
    block-like and the estimates are noisy, a parcellation can *beat* this reference --
    denoising more than it discards. Verified in the test suite on synthetic data with an
    exactly-block-structured ground truth. On real connectivity, whose structure is far
    richer than any 50-block summary, expect the reference to sit comfortably above every
    variant; a variant that exceeds it would be a substantive finding, not a bug.
    """
    return agreement(R_eval, R_fit, measure=measure)


def reliability(parcellation_a, parcellation_b):
    """Adjusted Rand index between two parcellations of the same units.

    Adjusted, not raw: the correction for chance is what makes the number comparable
    across different numbers of networks and different cluster size distributions.
    """
    return float(adjusted_rand_score(hard_labels(parcellation_a), hard_labels(parcellation_b)))


def map_reliability(maps_a, maps_b):
    """Mean correlation between Hungarian-matched component maps from two halves.

    The soft counterpart of `reliability` for an arm whose native output is a set of
    continuous maps over units (spatial ICA, LOG.md Iteration 15). Maps are matched by
    maximising total correlation, so a permutation of components costs nothing; signs are
    expected to have been fixed upstream (positive skew), so plain rather than absolute
    correlation is used and a sign flip does show up as a loss.
    """
    A = np.asarray(maps_a, dtype=np.float64)
    B = np.asarray(maps_b, dtype=np.float64)
    assert A.shape == B.shape, 'map shapes differ: %s vs %s' % (A.shape, B.shape)
    A = (A - A.mean(0)) / (A.std(0) + 1e-12)
    B = (B - B.mean(0)) / (B.std(0) + 1e-12)
    C = A.T @ B / A.shape[0]
    r_ix, c_ix = optimize.linear_sum_assignment(C, maximize=True)

    return float(C[r_ix, c_ix].mean())


def reliability_ceiling(parcellation_split1, parcellation_split2):
    """Agreement between two consensuses built on the SAME data with disjoint restarts.

    The reliability analogue of `fidelity_ceiling`, and it partials out the complementary
    nuisance: the fidelity ceiling removes the model limitation to expose data noise, this
    removes the data difference to expose algorithmic noise. There is no literal analogue
    of "no parcellation" here, because reliability compares two partitions and removing the
    parcellation leaves nothing to compare.

    Why it is necessary rather than nice to have: seed variability cannot be neutralized by
    fixing the seed, because with different data the same RNG stream produces an unrelated
    trajectory. Measuring it is the only option. And since arms differ by nearly an order of
    magnitude in this floor, raw cross-half ARI is not comparable across arms without it.

    Free to compute: the consensus already averages many restarts, so aligning two disjoint
    halves of them costs two extra Hungarian alignments -- measured at 0.1% of the k-means
    time. It is slightly pessimistic, since each half uses half the restarts and a consensus
    over more restarts is more stable.
    """
    return reliability(parcellation_split1, parcellation_split2)


def triviality(parcellation, coordinates, connectivity=None, noise_scale=None, n_bins=10):
    """How much of a parcellation is explained by properties that are not connectivity.

    Directly targets the "a dumb algorithm could score well" objection. If a parcellation
    is largely recoverable from the layer index, it is telling us nothing a `for layer in
    layers` loop would not. Hubness is included because the pre-S2 transposed binarization
    made clustering partly a function of degree, so it is a known failure mode here.

    `noise_scale` is the per-unit mean null standard deviation from a surrogate-normalized
    tree, and `ami_noise_scale` is the diagnostic for the confound that normalization
    INTRODUCES on the real tree. Dividing by sigma turns each entry into an effect size, so a
    unit whose correlations are hard to measure -- high autocorrelation, near-constant,
    massive-activation dimensions -- has uniformly small |z|. Measured on wikitext sample 1,
    the correlation between a unit's noise scale and its total strength goes from -0.32 in
    |r| to -0.80 in |z|, and the relative spread of unit strength from 0.21 to 0.36: the
    magnitude field is not removed on real data, it is inverted and strengthened.

    That is the correct behaviour for an effect size, and on the NULL tree the field does
    vanish (relative spread 1.160 -> 0.018), which is what the normalization was for. But it
    means a real parcellation can now score well by sorting units on detectability, which is
    a fixed property of the unit and therefore reproducible, while the null -- having no
    field left -- cannot match it. That would be a positive real-minus-null for a reason that
    has nothing to do with connectivity. This metric is how we see it rather than report it.
    """
    labels = hard_labels(parcellation)
    out = {'ami_layer': float(adjusted_mutual_info_score(coordinates[:, 0], labels))}
    # The second coordinate is the unit's index within its layer. For residual-stream units
    # that is the residual dimension, and dimension d at layer l is dimension d at layer
    # l+1 plus one block's update, so 768 chains of 13 units are an architectural given
    # of this connectome. A parcellation that follows the chains has low layer AMI (each
    # cluster spans all layers) and HIGH dimension AMI; this metric is what tells the two
    # apart. For MLP units the index is the neuron index and no chain exists, so this
    # should sit near zero there (LOG.md Iteration 14).
    out['ami_dimension'] = float(adjusted_mutual_info_score(coordinates[:, 1], labels))
    if connectivity is not None:
        strength = np.abs(np.nan_to_num(connectivity)).sum(axis=1)
        # Rank-based bins, so the measure does not depend on the scale of |r|.
        ranks = np.argsort(np.argsort(strength))
        out['ami_hubness'] = float(adjusted_mutual_info_score(
            (ranks * n_bins // len(ranks)), labels))
    if noise_scale is not None:
        ns = np.nan_to_num(np.asarray(noise_scale, dtype=np.float64)).ravel()
        assert len(ns) == len(labels), \
            'noise_scale has %d entries for %d units' % (len(ns), len(labels))
        ranks = np.argsort(np.argsort(ns))
        out['ami_noise_scale'] = float(adjusted_mutual_info_score(
            (ranks * n_bins // len(ranks)), labels))
    out['n_effective_networks'] = int(len(np.unique(labels)))
    # Peakedness of the soft memberships. Reported because argmax is only meaningful when
    # the membership vector is actually peaked; a median max of 0.20 at k=20 (uniform is
    # 0.05) means the label is the top of a nearly flat noisy vector.
    top = np.asarray(parcellation, dtype=np.float64).max(axis=1)
    out['median_max_membership'] = float(np.median(top))
    out['frac_confident'] = float((top > 0.5).mean())

    return out


def _block_stats(R0, labels, k):
    """Block pair sums, pair counts and means for a matrix whose diagonal is already zero.

    One n x n x k product; everything else is k x k. `R0` must have a zero diagonal so the
    self-pairs drop out of the sums, matching `block_means`.
    """
    n = R0.shape[0]
    onehot = np.zeros((n, k), dtype=R0.dtype)
    onehot[np.arange(n), labels] = 1.0
    S = R0 @ onehot                                  # n x k: each unit's sum into each block
    sizes = onehot.sum(axis=0)
    sums = onehot.T @ S                              # k x k
    counts = np.outer(sizes, sizes) - np.diag(sizes)
    with np.errstate(invalid='ignore', divide='ignore'):
        M = np.where(counts > 0, sums / np.maximum(counts, 1.0), 0.0)
    return S, onehot, sizes, sums, counts, M


def block_sse(R, labels, n_networks=None, _sumsq=None, _R0=None):
    """Sum of squared errors of the block-mean model on the strict upper triangle of R.

    `fidelity(..., measure='r2')` is 1 - SSE / SS_tot. Computed from the block sums rather
    than an n x n prediction: over all ordered off-diagonal pairs,
    SSE = sum r^2 - 2 sum_ab M_ab sums_ab + sum_ab counts_ab M_ab^2, and the upper triangle
    is half of that. `_sumsq`/`_R0` let a caller that already holds the zero-diagonal copy
    and its sum of squares skip recomputing them each sweep.
    """
    labels = np.asarray(labels).astype(int)
    k = int(n_networks or labels.max() + 1)
    if _R0 is None:
        _R0 = np.array(R, dtype=np.float64, copy=True)
        np.fill_diagonal(_R0, 0.0)
    if _sumsq is None:
        _sumsq = float(np.einsum('ij,ij->', _R0, _R0))
    _, _, _, sums, counts, M = _block_stats(_R0, labels, k)
    full = _sumsq - 2.0 * float(np.sum(M * sums)) + float(np.sum(counts * M ** 2))

    return 0.5 * full


def blockmodel_refine(R, labels, n_networks, max_iter=50, verbose=False):
    """Refine a partition to directly minimize the block-model error that fidelity scores.

    k-means on connectivity profiles fits r_ij ~ mu_{c(i), j}: one free value per (cluster,
    unit), n of them per cluster. Fidelity scores r_ij ~ m_{c(i) c(j)}: k^2 free values in
    total. Those are different objectives, and the partition that minimizes the first need
    not do well on the second. This is coordinate descent on the second: alternate the
    block means given the labels with moving every unit to the block whose row of means
    predicts its connections best (LOG.md Iteration 14, YOLO 2).

    For unit i and candidate block c the error is sum_{j != i} (r_ij - m_{c, c(j)})^2, which
    expands over the blocks b of the partners as sum_b [Q_ib - 2 m_cb S_ib + n_b^(i) m_cb^2]
    with S_ib = sum_{j in b, j != i} r_ij, Q_ib the matching sum of squares (independent of c,
    dropped), and n_b^(i) the number of partners of i in block b (n_b, less one if i is in
    b itself). So one sweep is one n x n x k product plus k x k arithmetic; the first
    version of this routine also copied the matrix twice and built an n x n prediction per
    sweep, which cost 2.9 h per matrix on the cluster (jobs 17394669/74).

    All units move at once. Batch updates on a symmetric objective can overshoot, so the
    SSE is tracked every sweep and the best labeling seen is returned, whether or not the
    final sweep was it. Empty blocks are allowed (they predict nothing and cost nothing),
    and are visible downstream as `n_effective_networks`.
    """
    R0 = np.array(R, dtype=np.float64, copy=True)
    np.fill_diagonal(R0, 0.0)
    sumsq = float(np.einsum('ij,ij->', R0, R0))
    n = R0.shape[0]
    k = int(n_networks)
    labels = np.asarray(labels).astype(int).copy()

    def sse_from(sums, counts, M):
        return 0.5 * (sumsq - 2.0 * float(np.sum(M * sums)) + float(np.sum(counts * M ** 2)))

    S, onehot, sizes, sums, counts, M = _block_stats(R0, labels, k)
    best_labels, best_sse = labels.copy(), sse_from(sums, counts, M)
    for it in range(max_iter):
        # cost_ic = -2 (S M^T)_ic + sum_b n_b^(i) m_cb^2 ;  n_b^(i) = n_b - onehot_ib
        M2 = M ** 2
        cost = -2.0 * (S @ M.T) + (sizes @ M2.T)[None, :] - onehot @ M2.T
        new_labels = cost.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        n_moved = int((new_labels != labels).sum())
        labels = new_labels
        S, onehot, sizes, sums, counts, M = _block_stats(R0, labels, k)
        sse = sse_from(sums, counts, M)
        if verbose:
            stderr('    blockmodel sweep %d: sse %.6g (%d moved)\n' % (it + 1, sse, n_moved))
        if sse < best_sse:
            best_sse, best_labels = sse, labels.copy()
        elif sse > best_sse:
            # Overshoot; the previous best is the answer. One more sweep from here would
            # typically oscillate rather than improve.
            break

    return best_labels, best_sse


SUMMARY_TREES = ('real', 'null', 'pnull', 'rand')


def summarize(rows, verbose=True, indent=0):
    """Collapse per-domain rows into reference-calibrated, domain-equal summaries.

    Four trees. `real` is the pipeline on real data. `null` is the pipeline on circularly
    shifted data, scored on shifted data -- the original design, kept for the record, but
    its R^2 sits on a different denominator from the real one and is not directly comparable
    (LOG.md Iteration 9). `pnull` is the null PARTITION evaluated on REAL data: same target
    matrix, same denominator, so real - pnull is the credit the clustering earns beyond
    what a partition carrying only per-unit properties earns. `rand` is a seeded random
    partition of the same k on real data: what "any 50 blocks" gets. `delta` is real - null
    (historical); `delta_pnull` and `delta_rand` are the ones to read.

    Domains are weighted equally rather than by token count: the question is whether a
    method works across domains, and letting a large corpus dominate would answer a
    different one. Every domain contributes the same number of tokens here anyway, so the
    two weightings coincide for now; stating it explicitly keeps that from becoming an
    unexamined assumption if the budgets ever differ.
    """
    import collections
    grouped = collections.defaultdict(dict)
    for r in rows:
        grouped[(r['variant'], r['metric'], r['fit'], r['eval'])][r['tree']] = r['value']

    def diff(a, b):
        return None if a is None or b is None else a - b

    out = []
    for (variant, metric, fit, ev), by_tree in sorted(grouped.items()):
        vals = {t: by_tree.get(t) for t in SUMMARY_TREES}
        out.append(dict(
            variant=variant, metric=metric, fit=fit, eval=ev, **vals,
            delta=diff(vals['real'], vals['null']),
            delta_pnull=diff(vals['real'], vals['pnull']),
            delta_rand=diff(vals['real'], vals['rand']),
        ))
    if verbose:
        stderr('%sSummarized %d comparisons\n' % (' ' * indent, len(out)))

    return out


def domain_average(summary, metric, variant, field='delta'):
    """Mean over domains of a calibrated value, weighting every domain equally."""
    vals = [r[field] for r in summary
            if r['metric'] == metric and r['variant'] == variant
            and r[field] is not None and np.isfinite(r[field])]

    return float(np.mean(vals)) if vals else float('nan')
