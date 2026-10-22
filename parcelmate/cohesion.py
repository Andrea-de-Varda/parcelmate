"""Cohesion of the networks a circuit spans (LOG.md Iteration 46).

A circuit (the top units of a task's attribution) falls into several networks of a partition,
with shares p (sum 1). The question is whether those networks are more of one thing than
other networks would be: are they coupled to each other more strongly than networks matched
on layer position?

Coupling between networks comes from the block means of |r| (`metrics.stream_block_means`):
B[i, j] = mean |r| between the units of networks i and j (self-pairs excluded on the
diagonal), computed on the half the partition was NOT fitted on. It is normalised by the
within-network coupling, Bn[i, j] = B[i, j] / sqrt(B[i, i] B[j, j]), so 1 means "as tight as a
single network".

    cohesion     C = sum_{i != j} p_i p_j Bn_ij / sum_{i != j} p_i p_j
    specificity  S = C / sum_i p_i E_i,   E_i = mean_{j != i} Bn_ij

S > 1: the circuit's networks are coupled to each other more than they are to networks in
general, which a set of hub networks (strongly coupled to everyone) would not be.

Null (Andrea, 2026-10-21): the circuit's share profile, given to other networks. Network
identities are permuted within bins of layer position (each network's mean layer), so a
replacement network sits at a similar depth; the profile itself (how many networks, with
which weights) is unchanged.
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


def layer_bins(centres, n_bins):
    """Bin index of every network by layer position (quantile bins of equal size)."""
    order = np.argsort(centres, kind='stable')
    bins = np.empty(len(centres), dtype=np.int64)
    for b, idx in enumerate(np.array_split(order, n_bins)):
        bins[idx] = b
    return bins


def permute_within_bins(bins, n_draws, rng):
    """(n_draws, k) permutations of network indices that keep every network in its bin."""
    k = len(bins)
    out = np.tile(np.arange(k), (n_draws, 1))
    groups = [np.flatnonzero(bins == b) for b in np.unique(bins)]
    for d in range(n_draws):
        for g in groups:
            out[d, g] = rng.permutation(g)
    return out


def cohesion_stats(P, Bn):
    """Cohesion and specificity for one or many share vectors.

    P  (k,) or (n, k) shares summing to 1 per row.
    Returns (C, S) as floats or (n,) arrays. A vector with a single network has no pairs:
    C and S are NaN.
    """
    P = np.atleast_2d(np.asarray(P, dtype=np.float64))
    k = Bn.shape[0]
    off = Bn - np.diag(np.diag(Bn))
    num = np.einsum('ni,ij,nj->n', P, off, P)
    sq = (P ** 2).sum(1)
    den = P.sum(1) ** 2 - sq
    E = off.sum(1) / (k - 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        C = np.where(den > 0, num / den, np.nan)
        S = C / (P @ E)
    if C.shape[0] == 1:
        return float(C[0]), float(S[0])
    return C, S


def cohesion_test(shares, Bn, bins, n_draws, rng):
    """Observed C and S against the layer-binned network-permutation null.

    p = (1 + #{null >= observed}) / (1 + n_draws), one-sided (more cohesive / specific).
    """
    C, S = cohesion_stats(shares, Bn)
    perms = permute_within_bins(bins, n_draws, rng)
    # network j of a null set receives the share the circuit gave network perms[d, j]
    Pn = shares[perms]
    Cn, Sn = cohesion_stats(Pn, Bn)
    out = dict(cohesion=C, specificity=S,
               null_cohesion_mean=float(np.nanmean(Cn)), null_cohesion_sd=float(np.nanstd(Cn, ddof=1)),
               null_specificity_mean=float(np.nanmean(Sn)), null_specificity_sd=float(np.nanstd(Sn, ddof=1)),
               p_cohesion=float((1 + np.sum(Cn >= C - 1e-12)) / (1 + n_draws)),
               p_specificity=float((1 + np.sum(Sn >= S - 1e-12)) / (1 + n_draws)),
               n_networks_spanned=int((shares > 0).sum()),
               effective_networks=float(np.exp(-np.sum(shares[shares > 0] * np.log(shares[shares > 0])))))
    return out
