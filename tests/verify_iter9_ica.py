"""Verification for YOLO 4 (LOG.md Iteration 15): spatial ICA and the sparsification rungs.

    PYTHONPATH=. python tests/verify_iter9_ica.py

Covers: the identity between spatial ICA computed from the signed correlation matrix and
from the activations (same whitened spatial subspace), recovery of planted spatial sources,
sign fixing and z-scoring of maps, winner-take-all labels, the 3-D (soft map) branch of the
consensus alignment, `map_reliability`, the `ica_maps` output and provenance of an ICA arm,
the misconfiguration guards, the global-vs-row binarization scope, `sparsify_fisher`, and
`reliability_within_maps` reaching the score table.
"""

import os
import sys
import tempfile

import numpy as np
from scipy import linalg
from sklearn.metrics import adjusted_rand_score

from parcelmate.bin.score import score_config
from parcelmate.constants import CONNECTIVITY_NAME, HALF_NAMES
from parcelmate.data import fisher
from parcelmate.metrics import map_reliability
from parcelmate.model import _align_samples, run_parcellation, sample_parcellations
from parcelmate.util import load_h5_data, read_attrs, save_h5_data

failures = []
n_checks = [0]


def check(name, cond):
    n_checks[0] += 1
    print('%s %s' % ('OK' if cond else 'FAIL', name))
    if not cond:
        failures.append(name)


def planted_spatial(T=4000, V=300, k=5, seed=0, noise=3.0, loadings=None):
    """Token x unit data with k INDEPENDENT sparse spatial sources; returns X, R, labels, L.

    Spatial ICA assumes the component maps are independent over units, and it recovers the
    maps of the VARIANCE-NORMALIZED data (every correlation matrix normalizes each unit by
    its own variance; MELODIC does the same). If a unit's variance is dominated by one
    source, normalizing it makes the k maps dependent again -- checked: with heavy-tailed
    loadings and weak noise FastICA recovers the raw maps at r = 0.99 and the normalized
    ones at only r ~ 0.7, whatever the contrast function. That is a property of spatial
    ICA on correlation matrices (LOG.md Iteration 15), not something to test the code
    against; the generator here keeps per-unit variance noise-dominated (sparse 0/1
    loadings, noise sd 3) so the normalization is nearly constant. `loadings` lets a second
    half share the planted maps with new sources and new noise.
    """
    rng = np.random.RandomState(seed)
    # Sparse AND heavy-tailed: 30% of units per source, exponential weights. Checked on
    # the exact mixture, the normalized mixture and the finite-T eigenvectors (all r > 0.97).
    # Binary 0/1 maps fail already on the exact mixture (near-Gaussian kurtosis; FastICA's
    # contrasts are kurtosis-driven), and dense exponential maps fail after normalization.
    L = loadings if loadings is not None else \
        (rng.rand(V, k) < 0.3) * rng.exponential(size=(V, k))
    sources = rng.randn(T, k)
    X = sources @ L.T + noise * rng.randn(T, V)
    Xs = (X - X.mean(0)) / X.std(0)
    # The maps the standardized data actually contains are L scaled by each unit's 1/sd.
    L_norm = L / X.std(0)[:, None]
    Lz = (L_norm - L_norm.mean(0)) / L_norm.std(0)
    labels = Lz.argmax(1)
    # A winner-take-all label is only defined where a unit has a clear winner: 70% of
    # loadings are exactly zero, so many units have none. Units with a margin of one z unit
    # between best and second-best source are the ones a WTA label can be checked on.
    top2 = np.sort(Lz, axis=1)[:, -2:]
    clear = (top2[:, 1] - top2[:, 0]) > 1.0
    return Xs, Xs.T @ Xs / T, labels, L_norm, clear, L


X, R, truth, L_true, clear, L_raw = planted_spatial()
V, k = R.shape[0], truth.max() + 1

# ------------------------------------------------------ identity with the timecourse route
_, evecs = linalg.eigh(R, subset_by_index=[V - k, V - 1])
Z1 = evecs * np.sqrt(V)
_, _, Vt = np.linalg.svd(X, full_matrices=False)
Z2 = Vt[:k].T * np.sqrt(V)
check('whitened spatial PCs from R span the same subspace as the SVD of the activations',
      np.allclose(Z1 @ Z1.T, Z2 @ Z2.T, atol=1e-8))
check('they have unit RMS over units (FastICA centers and re-whitens them itself)',
      np.allclose((Z1 ** 2).mean(0), 1, atol=1e-9))

# ------------------------------------------------------ the ICA path
out = sample_parcellations(np.abs(R), n_networks=k, n_samples=3, clustering='ica',
                           signed_connectivity=R, binarize_connectivity=False,
                           fisher_transform=False, connectivity_pca_components=None,
                           seed=0, verbose=False)
check('ica path returns hard labels, unweighted scores and z-scored maps',
      out['samples'].shape == (3, V) and 'maps' in out and out['maps'].shape == (3, V, k)
      and np.allclose(out['scores'], 0))
maps = out['maps'][0].astype(np.float64)
check('maps are z-scored over units',
      np.allclose(maps.mean(0), 0, atol=1e-5) and np.allclose(maps.std(0), 1, atol=1e-3))
check('every map has positive skew after the sign fix', np.all((maps ** 3).mean(0) > 0))
check('labels are the winner-take-all of the maps',
      np.array_equal(out['samples'][0], maps.argmax(1)))
C = np.abs(np.corrcoef(maps.T, L_true.T)[:k, k:])
check('recovered maps match the planted maps (min matched |r| %.3f)' % C.max(0).min(),
      C.max(0).min() > 0.9)
ari = adjusted_rand_score(truth[clear], out['samples'][0][clear])
check('winner-take-all ICA recovers the planted labels on the %d units with a clear winner '
      '(ARI %.3f)' % (clear.sum(), ari), ari > 0.8)

# ------------------------------------------------------ consensus of soft maps
perm = np.random.RandomState(3).permutation(k)
stack = np.stack([maps, maps[:, perm], maps[:, perm[::-1]]])
cons = _align_samples(stack, verbose=False).T
C = np.corrcoef(cons.T, maps.T)[:k, k:]
check('3-D alignment undoes a column permutation (matched corr %.3f)' % C.max(1).min(),
      C.max(1).min() > 0.999)

# ------------------------------------------------------ map reliability
check('map_reliability of a map with itself is 1', abs(map_reliability(maps, maps) - 1) < 1e-9)
check('map_reliability is invariant to a component permutation',
      abs(map_reliability(maps, maps[:, perm]) - 1) < 1e-9)
rnd = np.random.RandomState(1).randn(V, k)
check('map_reliability of unrelated maps is ~0 (%.3f)' % map_reliability(maps, rnd),
      abs(map_reliability(maps, rnd)) < 0.2)
# With signs flipped the best Hungarian match avoids the -1 diagonal and lands near 0.
check('a sign flip is a loss, not a free relabelling (%.3f)' % map_reliability(maps, -maps),
      map_reliability(maps, -maps) < 0.3)

# ------------------------------------------------------ misconfiguration guards
for bad in (dict(binarize_connectivity=True), dict(fisher_transform=True),
            dict(standardize_profiles=True), dict(connectivity_pca_components=10)):
    try:
        sample_parcellations(np.abs(R), n_networks=k, n_samples=1, clustering='ica',
                             signed_connectivity=R, verbose=False, **bad)
        check('ica rejects %s' % bad, False)
    except AssertionError:
        check('ica rejects %s' % bad, True)
try:
    sample_parcellations(np.abs(R), n_networks=k, n_samples=1, clustering='ica',
                         binarize_connectivity=False, verbose=False)
    check('ica without a signed matrix is rejected', False)
except AssertionError:
    check('ica without a signed matrix is rejected', True)

# ------------------------------------------------------ binarization scope, sparsify
Ra = np.abs(R)
rng = np.random.RandomState(0)
hub = rng.rand(V) < 0.2
Rh = Ra.copy()
Rh[hub] *= 3
Rh[:, hub] *= 3
Rh = np.clip(Rh, 0.0, 0.95)   # still a valid |r| matrix, with hubs
np.fill_diagonal(Rh, 1.0)
row = (Ra > np.quantile(Ra, 0.9, axis=1, keepdims=True)).astype(int)
glob = (Rh > np.quantile(Rh, 0.9)).astype(int)
check('row-wise binarization gives every row the same density',
      row.sum(1).std() < 1)
check('global binarization gives hubs more partners than weak units',
      glob[hub].sum(1).mean() > 2 * glob[~hub].sum(1).mean())
outg = sample_parcellations(Rh, n_networks=k, n_samples=1, binarize_connectivity=True,
                            binarize_scope='global', clustering='kmeans', seed=0, verbose=False)
check('binarize_scope=global runs through sample_parcellations', outg['samples'].shape == (1, V))
# On a matrix without ties (the hub matrix saturates at the clip and ties reduce the count).
X_f = fisher(np.clip(Ra.copy(), -1, 1))
np.fill_diagonal(X_f, 0.0)
thr = np.quantile(X_f, 0.9, axis=1, keepdims=True)
kept = np.where(X_f > thr, X_f, 0.0)
check('sparsify_fisher keeps ~10%% of each row with Fisher magnitudes (%.3f nonzero)'
      % (kept != 0).mean(), abs((kept != 0).mean() - 0.1) < 0.01
      and np.allclose(kept[kept != 0], X_f[kept != 0]))
outs = sample_parcellations(Rh, n_networks=k, n_samples=1, binarize_connectivity=False,
                            fisher_transform=True, sparsify_fisher=True, clustering='kmeans',
                            seed=0, verbose=False)
check('sparsify_fisher runs through sample_parcellations', outs['samples'].shape == (1, V))
try:
    sample_parcellations(Rh, n_networks=k, n_samples=1, binarize_connectivity=False,
                         fisher_transform=False, sparsify_fisher=True, verbose=False)
    check('sparsify_fisher without fisher_transform is rejected', False)
except AssertionError:
    check('sparsify_fisher without fisher_transform is rejected', True)

# ------------------------------------------------------ an ICA arm end to end, then scored
tmp = tempfile.mkdtemp()
X_b, R_b, _, _, _, _ = planted_spatial(seed=1, loadings=L_raw)   # same maps, new tokens
coords = np.stack([np.repeat(np.arange(3), V // 3), np.tile(np.arange(V // 3), 3)], 1).astype(np.int32)
for tree in ('run', 'run_null'):
    d = os.path.join(tmp, tree, CONNECTIVITY_NAME)
    os.makedirs(d)
    for key, Rk in (('avg', R), (HALF_NAMES[0], R), (HALF_NAMES[1], R_b)):
        Rw = Rk if tree == 'run' else np.random.RandomState(7).permutation(Rk)
        save_h5_data(dict(connectivity=Rw, coordinates=coords),
                     os.path.join(d, '%s_dom_%s.h5' % (CONNECTIVITY_NAME, key)), verbose=False)
for tree in ('run', 'run_null'):
    run_parcellation(output_dir=os.path.join(tmp, tree), n_networks=k, n_samples=3,
                     clustering='ica', binarize_connectivity=False, fisher_transform=False,
                     connectivity_pca_components=None, seed=0, variant='ica', verbose=False)
pa = os.path.join(tmp, 'run', 'ica', 'parcellation', 'parcellation_dom_halfA.h5')
da = load_h5_data(pa, verbose=False)
attrs = read_attrs(pa)
check('ica arm writes ica_maps of shape (n_units, k) beside the parcellation',
      'ica_maps' in da and da['ica_maps'].shape == (V, k))
check('ica arm provenance records clustering=ica', attrs.get('clustering') == 'ica')
check('consensus map agrees with the winner-take-all consensus (ARI %.3f)'
      % adjusted_rand_score(da['parcellation'].argmax(1), da['ica_maps'].argmax(1)),
      adjusted_rand_score(da['parcellation'].argmax(1), da['ica_maps'].argmax(1)) > 0.8)
cfg = dict(output_dir=os.path.join(tmp, 'run'), seed=1, connectivity=dict(domains=['dom']),
           parcellation_variants=dict(ica={}))
rows, _ = score_config(cfg, cross_domain=False, verbose=False, allow_partial=True)
mr = [r['value'] for r in rows if r['metric'] == 'reliability_within_maps' and r['tree'] == 'real']
check('reliability_within_maps reaches the score table for the real tree (%.3f)'
      % (mr[0] if mr else float('nan')), len(mr) == 1 and 0.5 < mr[0] <= 1.0)

print('\n%d check(s), %d failure(s)' % (n_checks[0], len(failures)))
if failures:
    for f in failures:
        print('  - %s' % f)
    sys.exit(1)
