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

Fidelity additionally gets a ceiling: predicting R_B from R_A directly, with no
parcellation at all. That is the best any compression could do given the sampling noise
between halves, so it converts fidelity from an arbitrary number into a fraction of what
was achievable.
"""

import numpy as np
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


def variance_explained(R_eval, prediction):
    """R^2 of a prediction against the off-diagonal of R_eval.

    Not clipped at zero: a genuinely bad predictor should be allowed to score negative,
    which is informative (it means the block means are worse than the grand mean).
    """
    y = upper_triangle(np.asarray(R_eval, dtype=np.float64))
    yhat = upper_triangle(np.asarray(prediction, dtype=np.float64)) \
        if np.ndim(prediction) == 2 else np.asarray(prediction, dtype=np.float64)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        return float('nan')

    return float(1.0 - np.sum((y - yhat) ** 2) / ss_tot)


def fidelity(R_fit, R_eval, parcellation_fit, soft=False):
    """Variance of held-out connectivity explained by a partition fit elsewhere.

    `R_fit`/`parcellation_fit` come from the fitting data; `R_eval` is held out. Block
    means are estimated on R_fit (out-of-sample in both structure and values).

    With `soft=True` the prediction is P M P^T using the membership probabilities rather
    than argmax labels, which uses the information the consensus averaging produced
    instead of discarding it. Reported alongside the hard version rather than instead of
    it, since the two answer slightly different questions and the choice was never settled.
    """
    P = np.asarray(parcellation_fit, dtype=np.float64)
    labels = hard_labels(P)
    M = block_means(R_fit, labels, n_networks=P.shape[1])
    if soft:
        rows = P.sum(axis=1, keepdims=True)
        Pn = np.divide(P, rows, out=np.zeros_like(P), where=rows > 0)
        prediction = Pn @ M @ Pn.T
    else:
        prediction = M[labels][:, labels]

    return variance_explained(R_eval, prediction)


def fidelity_ceiling(R_fit, R_eval):
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
    return variance_explained(R_eval, R_fit)


def reliability(parcellation_a, parcellation_b):
    """Adjusted Rand index between two parcellations of the same units.

    Adjusted, not raw: the correction for chance is what makes the number comparable
    across different numbers of networks and different cluster size distributions.
    """
    return float(adjusted_rand_score(hard_labels(parcellation_a), hard_labels(parcellation_b)))


def triviality(parcellation, coordinates, connectivity=None, n_bins=10):
    """How much of a parcellation is explained by properties that are not connectivity.

    Directly targets the "a dumb algorithm could score well" objection. If a parcellation
    is largely recoverable from the layer index, it is telling us nothing a `for layer in
    layers` loop would not. Hubness is included because the pre-S2 transposed binarization
    made clustering partly a function of degree, so it is a known failure mode here.
    """
    labels = hard_labels(parcellation)
    out = {'ami_layer': float(adjusted_mutual_info_score(coordinates[:, 0], labels))}
    if connectivity is not None:
        strength = np.abs(np.nan_to_num(connectivity)).sum(axis=1)
        # Rank-based bins, so the measure does not depend on the scale of |r|.
        ranks = np.argsort(np.argsort(strength))
        out['ami_hubness'] = float(adjusted_mutual_info_score(
            (ranks * n_bins // len(ranks)), labels))
    out['n_effective_networks'] = int(len(np.unique(labels)))

    return out


def summarize(rows, verbose=True, indent=0):
    """Collapse per-domain rows into null-calibrated, domain-equal summaries.

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

    out = []
    for (variant, metric, fit, ev), by_tree in sorted(grouped.items()):
        real, null = by_tree.get('real'), by_tree.get('null')
        out.append(dict(
            variant=variant, metric=metric, fit=fit, eval=ev,
            real=real, null=null,
            delta=(None if real is None or null is None else real - null),
        ))
    if verbose:
        stderr('%sSummarized %d comparisons\n' % (' ' * indent, len(out)))

    return out


def domain_average(summary, metric, variant):
    """Mean over domains of the null-calibrated value, weighting every domain equally."""
    vals = [r['delta'] for r in summary
            if r['metric'] == metric and r['variant'] == variant
            and r['delta'] is not None and np.isfinite(r['delta'])]

    return float(np.mean(vals)) if vals else float('nan')
