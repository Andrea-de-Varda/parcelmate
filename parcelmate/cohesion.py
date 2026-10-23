"""How strongly the networks a circuit spans prefer each other (LOG.md Iteration 46).

A circuit (the top units of a task's attribution) falls into several networks of a partition,
with shares p (sum 1). Coupling between networks comes from the block means of |r|
(`metrics.stream_block_means`): B[i, j] = mean |r| between the units of networks i and j
(self-pairs excluded on the diagonal), computed on the half the partition was NOT fitted on.
It is normalised by the within-network coupling, Bn[i, j] = B[i, j] / sqrt(B[i, i] B[j, j]),
so 1 means "as tight as a single network".

Two quantities per circuit:

    cohesion    C = sum_{i != j} p_i p_j Bn_ij / sum_{i != j} p_i p_j
                how coupled the circuit's networks are to each other, on the 0-1 scale of Bn
                (1 = one network). Descriptive only.

    preference  R = sum_{i != j} p_i p_j Bn_ij / sum_{i != j} p_i p_j O_ij
                O_ij = mean Bn between network i and the `n_near` networks outside the circuit
                whose depth (mean layer) is closest to network j's.
                How coupled the circuit's networks are to each other, relative to how coupled
                they are to networks outside the circuit at the same depths. 1 = no preference.

The preference ratio is the test statistic (Andrea, 2026-10-22). Depth is matched pair by pair
because coupling falls with the depth distance between networks: against all outside networks,
any layer-concentrated circuit would look grouped (the unmatched ratio correlated -0.72 with a
circuit's depth spread, and the null partition exceeded 1 in 192 of 193 tasks). Network i
appears in the numerator and the denominator of every pair, so a hub coupled at the same level
to every network scores 1. Networks that are each more coupled to everyone, and therefore
doubly coupled to each other, score above 1: they are coupled to each other more than to
outsiders, which is what the ratio asks. This replaces the earlier layer-binned permutation reference and the
specificity ratio (first version of Iteration 46).
"""

import numpy as np


def normalized_coupling(B):
    """B / sqrt(diag diag^T), diagonal 1. Networks with zero within-coupling get 0 rows."""
    B = np.asarray(B, dtype=np.float64)
    d = np.sqrt(np.clip(np.diag(B), 0.0, None))
    with np.errstate(invalid='ignore', divide='ignore'):
        Bn = B / np.outer(d, d)
    Bn = np.where(np.isfinite(Bn), Bn, 0.0)
    np.fill_diagonal(Bn, 1.0)
    return Bn


def network_layers(labels, coordinates, k):
    """Mean layer of each network's units (NaN for an empty network)."""
    layer = np.asarray(coordinates)[:, 0].astype(np.float64)
    sums = np.bincount(labels, weights=layer, minlength=k)
    counts = np.bincount(labels, minlength=k)
    with np.errstate(invalid='ignore', divide='ignore'):
        return sums / counts


def _pair_weights(p):
    W = np.outer(p, p)
    np.fill_diagonal(W, 0.0)
    return W


def cohesion(p, Bn):
    """Share-weighted mean coupling over pairs of distinct networks; NaN with one network."""
    W = _pair_weights(np.asarray(p, dtype=np.float64))
    den = W.sum()
    return float((W * Bn).sum() / den) if den > 0 else float('nan')


def outside_reference(p, Bn, depth, n_near=3):
    """O[i, j]: mean coupling of network i to the n_near outside networks nearest j in depth.

    Outside = networks holding none of the circuit (p == 0). Columns of circuit networks only
    are meaningful; others are 0. Ties in depth are broken by network index.
    """
    p = np.asarray(p, dtype=np.float64)
    k = len(p)
    inside = np.flatnonzero(p > 0)
    outside = np.flatnonzero(p == 0)
    O = np.zeros((k, k))
    if len(outside) == 0:
        return O * np.nan
    m = min(n_near, len(outside))
    N = np.zeros((len(outside), k))
    for j in inside:
        near = np.argsort(np.abs(depth[outside] - depth[j]), kind='stable')[:m]
        N[near, j] = 1.0 / m
    return Bn[:, outside] @ N


def preference(p, Bn, depth, n_near=3):
    """Coupling among the circuit's networks / their coupling to depth-matched outsiders."""
    p = np.asarray(p, dtype=np.float64)
    W = _pair_weights(p)
    if W.sum() == 0:
        return float('nan')
    O = outside_reference(p, Bn, depth, n_near)
    den = (W * O).sum()
    return float((W * Bn).sum() / den) if den > 0 else float('nan')


def unmatched_preference(p, Bn):
    """Diagnostic only: coupling among the circuit's networks / their coupling to ALL outside
    networks, regardless of depth. Inflated for layer-concentrated circuits (why depth is matched)."""
    p = np.asarray(p, dtype=np.float64)
    out = p == 0
    if not out.any():
        return float('nan')
    ref = (p @ Bn[:, out]).sum() / out.sum() / p.sum()
    return cohesion(p, Bn) / ref


def depth_spread(p, depth):
    """Share-weighted standard deviation of the depth of the circuit's networks (layers)."""
    p = np.asarray(p, dtype=np.float64)
    mu = np.sum(p * depth)
    return float(np.sqrt(np.sum(p * (depth - mu) ** 2)))
