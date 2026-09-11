"""Verification for surrogate normalization as an explicit per-arm choice (LOG.md Iterations
10 and 12).

    PYTHONPATH=. python tests/verify_iter6_surrogate_norm.py

Two claims. (1) Dividing each connectivity entry by its own null standard deviation removes
the positive mean field that `|r|` manufactures out of noise. (2) Whether that happens is an
explicit argument recorded in the parcellation's provenance -- not a property of the file,
not a config flag -- so one tree can hold |r| arms and |z| arms side by side and the scorer
reads the choice back from the parcellation rather than being told separately.

The checks are built so the realistic failure modes are loud: normalization applied in the
parcellation but not the scorer, presence of `surrogate_var` silently switching an |r| arm
to |z|, a |z| arm run on a tree without variances, or the Fisher transform saturating on |z|.
"""

import os
import shutil
import tempfile

import numpy as np

from parcelmate.bin.score import load_connectivity
from parcelmate.constants import CONNECTIVITY_NAME, HALF_NAMES
from parcelmate.data import circshift_timecourses
from parcelmate.metrics import fidelity, triviality
from parcelmate.model import run_parcellation, run_split_halves, sample_parcellations
from parcelmate.util import (
    average_surrogate_var, connectivity_matrix, load_h5_data, read_attrs, save_h5_data,
)

failures = []


def check(name, cond):
    print('%s %s' % ('OK' if cond else 'FAIL', name))
    if not cond:
        failures.append(name)


def _raises(fn, exc=AssertionError):
    try:
        fn()
    except exc:
        return True
    return False


# ---------------------------------------------------------------------------------------
# The mechanism, on data where the truth is known: heterogeneous autocorrelation, NO
# cross-unit structure.
# ---------------------------------------------------------------------------------------
rng = np.random.RandomState(0)
N, T, K = 120, 4000, 24
phi = rng.uniform(0.0, 0.95, N)
e = rng.randn(N, T)
X = np.empty((N, T))
X[:, 0] = e[:, 0]
for t in range(1, T):
    X[:, t] = phi * X[:, t - 1] + np.sqrt(1 - phi ** 2) * e[:, t]

R = np.corrcoef(X)
np.fill_diagonal(R, 0.0)
var = np.zeros((N, N))
for k in range(K):
    r_k = np.corrcoef(circshift_timecourses(X, rng=np.random.RandomState(100 + k)))
    np.fill_diagonal(r_k, 0.0)
    var += r_k ** 2
var /= K

absr = connectivity_matrix(dict(connectivity=R))
absz = connectivity_matrix(dict(connectivity=R, surrogate_var=var), 'surrogate')

check('normalize=None is exactly |r|', np.allclose(absr, np.abs(np.nan_to_num(R))))
check('normalize=None stays |r| even when surrogate_var is present -- presence does not '
      'trigger anything',
      np.allclose(connectivity_matrix(dict(connectivity=R, surrogate_var=var)), absr))
check('normalize="surrogate" returns something different', not np.allclose(absr, absz))
check('normalize="surrogate" without surrogate_var raises',
      _raises(lambda: connectivity_matrix(dict(connectivity=R), 'surrogate')))
check('an unknown normalize value raises',
      _raises(lambda: connectivity_matrix(dict(connectivity=R, surrogate_var=var), 'zscore')))
check('both modes are non-negative', absr.min() >= 0 and absz.min() >= 0)


def strength_spread(M):
    s = M.sum(1)
    return float(s.std() / s.mean())


check('|r| has a per-unit magnitude field on structureless data (rel. spread %.3f)'
      % strength_spread(absr), strength_spread(absr) > 0.10)
check('|z| flattens it (rel. spread %.3f, at least 2x smaller)' % strength_spread(absz),
      strength_spread(absz) < strength_spread(absr) / 2)
c_r = abs(np.corrcoef(phi, absr.mean(1))[0, 1])
c_z = abs(np.corrcoef(phi, absz.mean(1))[0, 1])
check('|r| strength tracks autocorrelation (r = %.2f) and |z| does not (r = %.2f)'
      % (c_r, c_z), c_r > 0.5 and c_z < c_r / 2)


def strength_blocks(M, k=10):
    ranks = np.argsort(np.argsort(M.sum(1)))
    return np.eye(k)[(ranks * k) // len(ranks)]


r2_r = fidelity(absr, absr, strength_blocks(absr), measure='r2')
r2_z = fidelity(absz, absz, strength_blocks(absz), measure='r2')
check('a hubness-sorted block model fits |r| noise (R2 = %.3f) but not |z| (R2 = %.3f)'
      % (r2_r, r2_z), r2_r > 0.05 and r2_z < r2_r / 3)

v1, v2 = np.full((3, 3), 0.4), np.full((3, 3), 0.8)
check('Var(mean of m independent) = sum/m^2',
      np.allclose(average_surrogate_var([v1, v2]), (v1 + v2) / 4))
check('a half (2 samples) has a larger null variance than the avg (4 samples)',
      average_surrogate_var([v1, v1]).mean() > average_surrogate_var([v1] * 4).mean())

# ---------------------------------------------------------------------------------------
# The confound |z| introduces on real data, and its diagnostic.
# ---------------------------------------------------------------------------------------
coords_t = np.stack([np.repeat(np.arange(10), N // 10), np.arange(N)], 1)
noise_scale = np.sqrt(var).mean(axis=1)
rand_P = np.eye(10)[np.random.RandomState(1).randint(0, 10, N)]
check('ami_noise_scale is absent unless a noise scale is supplied',
      'ami_noise_scale' not in triviality(rand_P, coords_t))
check('ami_noise_scale ~ 0 for a partition unrelated to detectability',
      abs(triviality(rand_P, coords_t, noise_scale=noise_scale)['ami_noise_scale']) < 0.1)
ns_lab = (np.argsort(np.argsort(noise_scale)) * 10) // N
check('ami_noise_scale = 1 for a partition that IS the detectability decile',
      triviality(np.eye(10)[ns_lab], coords_t, noise_scale=noise_scale)['ami_noise_scale'] > 0.99)
check('a mismatched noise_scale length raises rather than scoring nonsense',
      _raises(lambda: triviality(rand_P, coords_t, noise_scale=noise_scale[:5])))

# ---------------------------------------------------------------------------------------
# The Fisher arms. fisher() maps anything above 1 to arctanh(0.999) = 3.8 with no error, so
# handing it |z| would silently binarize the arm at |z| > 1 (S11).
# ---------------------------------------------------------------------------------------
check('sample_parcellations(fisher_transform=True) refuses a |z| matrix loudly',
      _raises(lambda: sample_parcellations(absz, n_networks=3, n_samples=2,
                                           binarize_connectivity=False, fisher_transform=True,
                                           verbose=False, seed=0)))

# ---------------------------------------------------------------------------------------
# Plumbing. One tree, two arms: an |r| arm and a |z| arm on the SAME files, with the
# scorer reading each arm's choice from its parcellation rather than being told.
# ---------------------------------------------------------------------------------------
tmp = tempfile.mkdtemp(prefix='parcelmate_iter6_')
try:
    conn_dir = os.path.join(tmp, CONNECTIVITY_NAME)
    os.makedirs(conn_dir)
    coords = coords_t.astype(np.int32)
    for i in range(1, 5):
        save_h5_data(dict(connectivity=R + 0.001 * i, coordinates=coords,
                          unit_means=np.zeros(N), unit_stds=np.ones(N),
                          n_obs=np.asarray(1000), surrogate_var=var),
                     os.path.join(conn_dir, 'connectivity_alpha_sample%d.h5' % i),
                     verbose=False)
    save_h5_data(dict(connectivity=R, coordinates=coords, unit_means=np.zeros(N),
                      unit_stds=np.ones(N), n_obs=np.asarray(4000), surrogate_var=var),
                 os.path.join(conn_dir, 'connectivity_alpha_avg.h5'), verbose=False)
    run_split_halves(output_dir=tmp, verbose=False)

    half_path = os.path.join(conn_dir, 'connectivity_alpha_%s.h5' % HALF_NAMES[0])
    half = load_h5_data(half_path, verbose=False)
    check('split_halves propagates surrogate_var to the halves', 'surrogate_var' in half)
    check('the half variance is the 2-sample average, not a copy of the per-sample one',
          np.allclose(half['surrogate_var'], average_surrogate_var([var, var])))
    check('score.load_connectivity(path) is raw |r|',
          np.allclose(load_connectivity(half_path), np.abs(np.nan_to_num(half['connectivity']))))
    check('score.load_connectivity(path, "surrogate") == util.connectivity_matrix(..., "surrogate")',
          np.allclose(load_connectivity(half_path, 'surrogate'),
                      connectivity_matrix(half, 'surrogate')))

    common = dict(output_dir=tmp, n_networks=3, n_samples=2, seed=1,
                  binarize_connectivity=False, fisher_transform=True,
                  connectivity_pca_components=None, verbose=False)
    run_parcellation(variant='fisher', **common)
    run_parcellation(variant='fisher_z', normalize='surrogate', **common)
    a_r = read_attrs(os.path.join(tmp, 'fisher', 'parcellation', 'parcellation_alpha_avg.h5'))
    a_z = read_attrs(os.path.join(tmp, 'fisher_z', 'parcellation', 'parcellation_alpha_avg.h5'))
    check('the |r| arm records normalize=None, input_normalized=False, transform applied',
          a_r['normalize'] == 'None' and not a_r['input_normalized']
          and a_r['fisher_transform_applied'])
    check('the |z| arm on the SAME file records normalize=surrogate, transform bypassed',
          a_z['normalize'] == 'surrogate' and a_z['input_normalized']
          and not a_z['fisher_transform_applied'])
    check('both arms record the configured fisher_transform=True',
          a_r['fisher_transform'] and a_z['fisher_transform'])
    P_r = load_h5_data(os.path.join(tmp, 'fisher', 'parcellation',
                                    'parcellation_alpha_avg.h5'), verbose=False)['parcellation']
    P_z = load_h5_data(os.path.join(tmp, 'fisher_z', 'parcellation',
                                    'parcellation_alpha_avg.h5'), verbose=False)['parcellation']
    check('the two arms produce different parcellations from the same file',
          not np.allclose(P_r, P_z))

    # A partial set of variances must be refused, not silently averaged with the wrong
    # denominator.
    d = load_h5_data(os.path.join(conn_dir, 'connectivity_alpha_sample1.h5'), verbose=False)
    del d['surrogate_var']
    save_h5_data(d, os.path.join(conn_dir, 'connectivity_alpha_sample1.h5'), verbose=False)
    for name in HALF_NAMES:
        os.remove(os.path.join(conn_dir, 'connectivity_alpha_%s.h5' % name))
    check('a partial set of surrogate variances raises rather than averaging wrongly',
          _raises(lambda: run_split_halves(output_dir=tmp, verbose=False)))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# A |z| arm on a tree that has no variances must fail, not fall back to |r|.
tmp2 = tempfile.mkdtemp(prefix='parcelmate_iter6b_')
try:
    conn_dir = os.path.join(tmp2, CONNECTIVITY_NAME)
    os.makedirs(conn_dir)
    save_h5_data(dict(connectivity=R, coordinates=coords_t.astype(np.int32),
                      unit_means=np.zeros(N), unit_stds=np.ones(N), n_obs=np.asarray(4000)),
                 os.path.join(conn_dir, 'connectivity_alpha_avg.h5'), verbose=False)
    check('normalize="surrogate" on a tree without surrogate_var raises',
          _raises(lambda: run_parcellation(output_dir=tmp2, variant='z', n_networks=3,
                                           n_samples=2, seed=1, binarize_connectivity=False,
                                           fisher_transform=True, normalize='surrogate',
                                           connectivity_pca_components=None, verbose=False)))
finally:
    shutil.rmtree(tmp2, ignore_errors=True)

print('\n%s' % ('All checks passed.' if not failures
                else 'FAILURES:\n  ' + '\n  '.join(failures)))
raise SystemExit(1 if failures else 0)
