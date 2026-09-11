"""Why does a circular-shift null often score HIGHER than real data?

    PYTHONPATH=. python analysis/null_mechanism.py

This reproduces the inversion from first principles, on 300 simulated units, with no access
to the real connectomes and no bug anywhere. The short version is that both metrics reward
STRUCTURE THAT IS SIMPLE AND STABLE, and the shifted data has less structure but far simpler
and more stable structure than the real data does.

Three facts compound.

1. A circular shift preserves each unit's autocorrelation. Two independent series with
   autocorrelation tau_i, tau_j have sampling correlation of size ~sqrt(tau_i tau_j / T).
   That is a separable, rank-one pattern across the matrix.

2. The pipeline takes |r|. For genuinely uncorrelated units the correlations are zero-mean
   noise, and E|r_ij| = sqrt(2/pi) * sigma_ij. Taking the absolute value converts zero-mean
   noise into a POSITIVE MEAN FIELD with exactly the rank-one shape from (1). So the null
   connectome is not structureless: it is dominated by a clean outer product of per-unit
   connection strength. See LOG.md S2/Iteration 8 -- this is the same hubness that the
   `triviality_ami_hubness` diagnostic measures.

3. A rank-one matrix is the single most block-friendly object there is: sort units by their
   strength and a block model reproduces it almost exactly. Meanwhile the real connectome
   carries strong but CONTINUOUS structure -- graded, overlapping, high-dimensional -- which
   a 50-block model cannot express. R^2 is a ratio, so what matters is not how much
   structure there is but what fraction of it the model class can reach. The null has little
   variance and nearly all of it is reachable; the real data has a great deal and almost
   none of it is.

The same three facts explain the reliability inversion. Per-unit autocorrelation is a fixed
property of the unit, identical in both halves and across domains, so "sort units by
strength" is a highly reproducible partition -- the null gets a high ARI for re-describing a
constant. Real continuous structure gives k-means no stable basin, so it gets a low one.

Standardizing each unit's profile divides the magnitude field out, which is why the
`vmf_profile` arm is the only one whose scores are not dominated by the artifact.

Two refinements from the ablations (second table). Both ingredients are necessary: signed r
instead of |r| kills the null's block R2 and ARI, and so does homogeneous autocorrelation.
And "rank-one" is approximate: Bartlett gives Var(r_ij) ~ <rho_i, rho_j> / T, a Gram matrix
of autocorrelation functions, which is rank-one only when the ACFs form a one-parameter
family. Units sharing a distinctive spectral signature (a periodic component keeps its period
under a circular shift, and |cos| of a random phase difference averages 2/pi) stay mutually
correlated in |r| whatever their phases. That is a pattern over units, not a row scale, so
z-scoring does not remove it and it reproduces across halves. It is the candidate for the
0.289 null reliability that `vmf_profile` retains on real data with AMI-hubness 0.07 and
AMI-layer 0.05 -- and it is exactly the per-unit temporal structure the circular shift is
specified to preserve, so subtracting it is the null doing its job.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari

N, T, K, F = 300, 3000, 20, 60
SEED = 0


def ar1_bank(phis, T, rng):
    """Independent AR(1) series, one per entry of `phis`. No cross-unit structure at all."""
    out = np.empty((len(phis), T))
    e = rng.standard_normal((len(phis), T))
    out[:, 0] = e[:, 0]
    for t in range(1, T):
        out[:, t] = phis * out[:, t - 1] + np.sqrt(1 - phis ** 2) * e[:, t]
    return out


def conn(X):
    """|r| with a zeroed diagonal -- what the pipeline feeds to the clusterer."""
    R = np.abs(np.corrcoef(X))
    np.fill_diagonal(R, 0.0)
    return R


def block_r2(R, labels):
    """In-sample R^2 of the block-mean model, the same form as parcelmate.metrics."""
    oh = np.eye(labels.max() + 1)[labels]
    n = oh.sum(0)
    pred = oh @ ((oh.T @ R @ oh) / np.outer(n, n)) @ oh.T
    return 1 - ((R - pred) ** 2).sum() / ((R - R.mean()) ** 2).sum()


def fit(R, standardize):
    P = R
    if standardize:
        P = (R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-12)
    return KMeans(K, n_init=10, random_state=0).fit_predict(P)


def main():
    rng = np.random.default_rng(SEED)
    # Per-unit autocorrelation and factor loadings are properties of the UNIT, so they are
    # fixed once and shared by both simulated halves -- exactly as in the real pipeline,
    # where a unit is the same unit in half A and half B.
    phi = rng.uniform(0.0, 0.95, N)
    W = rng.standard_normal((N, F)) * rng.random((N, F)) ** 2
    W /= np.linalg.norm(W, axis=1, keepdims=True)

    def draw(kind):
        noise = ar1_bank(phi, T, rng)
        if kind == 'null':
            return noise                                    # zero true structure
        return 0.8 * (W @ ar1_bank(np.full(F, 0.7), T, rng)) + 0.2 * noise

    print('%-6s %-12s %8s %8s %9s %9s %9s' % (
        'data', 'standardize', 'mean|r|', 'SS_tot', 'block R2', 'ARI', 'AMI hub'))
    print('-' * 70)
    for kind in ('null', 'real'):
        Ra, Rb = conn(draw(kind)), conn(draw(kind))
        strength = Ra.sum(1)
        decile = (np.argsort(np.argsort(strength)) * 10) // N
        for std in (False, True):
            a, b = fit(Ra, std), fit(Rb, std)
            print('%-6s %-12s %8.3f %8.1f %9.3f %9.3f %9.3f' % (
                kind, str(std), Ra[np.triu_indices(N, 1)].mean(),
                ((Ra - Ra.mean()) ** 2).sum(), block_r2(Ra, a), ari(a, b),
                ami(decile, a)))

    ablations(rng)


def ablations(rng):
    """Which ingredient matters, and what survives standardizing."""
    phi = rng.uniform(0.0, 0.95, N)
    family = rng.integers(0, 3, N)     # 0: plain AR(1); 1: +period 64; 2: +period 200

    def spectral():
        x = ar1_bank(phi, T, rng)
        t = np.arange(T)
        for f, period in ((1, 64), (2, 200)):
            idx = family == f
            ph = rng.uniform(0, 2 * np.pi, idx.sum())[:, None]
            x[idx] += 0.8 * np.sin(2 * np.pi * t / period + ph)
        return x

    conds = (
        ('|r|, heterogeneous AR(1)  [pipeline]', lambda: ar1_bank(phi, T, rng), True),
        ('signed r, heterogeneous AR(1)', lambda: ar1_bank(phi, T, rng), False),
        ('|r|, homogeneous AR(1)', lambda: ar1_bank(np.full(N, 0.6), T, rng), True),
        ('|r|, + spectral families', spectral, True),
    )
    print('\n%-38s %-5s %9s %9s %9s' % ('null-like data', 'std', 'block R2', 'ARI', 'AMI fam'))
    print('-' * 74)
    for name, draw, absolute in conds:
        mats = []
        for _ in range(2):
            R = np.corrcoef(draw())
            R = np.abs(R) if absolute else R
            np.fill_diagonal(R, 0.0)
            mats.append(R)
        Ra, Rb = mats
        for std in (False, True):
            a, b = fit(Ra, std), fit(Rb, std)
            print('%-38s %-5s %9.3f %9.3f %9.3f' % (
                name, str(std), block_r2(Ra, a), ari(a, b), ami(family, a)))


if __name__ == '__main__':
    main()
