"""Verification for surrogate normalization (see LOG.md, Iteration 10).

    PYTHONPATH=. python tests/verify_iter6_surrogate_norm.py

The claim being tested is specific: dividing each connectivity entry by its own null
standard deviation removes the positive mean field that `|r|` manufactures out of noise, and
therefore removes the null's ability to out-score real data on both metrics.

The checks are built so that the realistic failure modes are loud. Normalization applied in
the parcellation but not the scorer, or to the real tree but not the null, or with the
scored null's own shift folded into the variance estimate, would all leave a pipeline that
runs clean and produces plausible numbers.
"""

import os
import shutil
import tempfile

import numpy as np

from parcelmate.bin.score import load_connectivity
from parcelmate.constants import CONNECTIVITY_NAME, HALF_NAMES
from parcelmate.data import circshift_timecourses
from parcelmate.metrics import fidelity, hard_labels
from parcelmate.model import run_split_halves
from parcelmate.util import average_surrogate_var, load_h5_data, save_h5_data, \
    surrogate_normalized

failures = []


def check(name, cond):
    print('%s %s' % ('OK' if cond else 'FAIL', name))
    if not cond:
        failures.append(name)


def _raises(fn):
    try:
        fn()
    except AssertionError:
        return True
    return False


# ---------------------------------------------------------------------------------------
# The mechanism itself, on data where the truth is known: units with heterogeneous
# autocorrelation and NO cross-unit structure.
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

absr = surrogate_normalized(dict(connectivity=R))
absz = surrogate_normalized(dict(connectivity=R, surrogate_var=var))

check('without surrogate_var the helper is exactly |r|, so old trees are unchanged',
      np.allclose(absr, np.abs(np.nan_to_num(R))))
check('with surrogate_var the helper returns something different', not np.allclose(absr, absz))
check('the helper is non-negative in both modes', absr.min() >= 0 and absz.min() >= 0)

# The field: strength varies by unit under |r| and should not under |z|.
def strength_spread(M):
    s = M.sum(1)
    return float(s.std() / s.mean())


check('|r| has a per-unit magnitude field on structureless data (rel. spread %.3f)'
      % strength_spread(absr), strength_spread(absr) > 0.10)
check('|z| flattens it (rel. spread %.3f, at least 2x smaller)' % strength_spread(absz),
      strength_spread(absz) < strength_spread(absr) / 2)

# Correlation between a unit's mean |r| and its autocorrelation is the artifact itself.
c_r = abs(np.corrcoef(phi, absr.mean(1))[0, 1])
c_z = abs(np.corrcoef(phi, absz.mean(1))[0, 1])
check('|r| strength tracks autocorrelation (r = %.2f) and |z| does not (r = %.2f)'
      % (c_r, c_z), c_r > 0.5 and c_z < c_r / 2)

# The consequence that matters: a block model must no longer fit structureless data.
# The partition must be the one the clusterer actually finds on |r| noise -- units grouped
# by connection strength. Blocking by unit INDEX instead tests nothing: the units are in no
# particular order, so no block model fits either matrix and the check passes vacuously.
def strength_blocks(M, k=10):
    ranks = np.argsort(np.argsort(M.sum(1)))
    return np.eye(k)[(ranks * k) // len(ranks)]


r2_r = fidelity(absr, absr, strength_blocks(absr), measure='r2')
r2_z = fidelity(absz, absz, strength_blocks(absz), measure='r2')
check('a hubness-sorted block model fits |r| noise (R2 = %.3f) but not |z| (R2 = %.3f)'
      % (r2_r, r2_z), r2_r > 0.05 and r2_z < r2_r / 3)

# ---------------------------------------------------------------------------------------
# Averaging rule.
# ---------------------------------------------------------------------------------------
v1, v2 = np.full((3, 3), 0.4), np.full((3, 3), 0.8)
check('Var(mean of m independent) = sum/m^2',
      np.allclose(average_surrogate_var([v1, v2]), (v1 + v2) / 4))
check('a half (2 samples) has a larger null variance than the avg (4 samples)',
      average_surrogate_var([v1, v1]).mean() > average_surrogate_var([v1] * 4).mean())

# ---------------------------------------------------------------------------------------
# Plumbing: the scorer and the parcellation must read the SAME matrix, and split_halves
# must propagate the variances rather than dropping them.
# ---------------------------------------------------------------------------------------
tmp = tempfile.mkdtemp(prefix='parcelmate_iter6_')
try:
    conn_dir = os.path.join(tmp, CONNECTIVITY_NAME)
    os.makedirs(conn_dir)
    coords = np.stack([np.repeat(np.arange(10), N // 10), np.arange(N)], 1).astype(np.int32)
    for i in range(1, 5):
        save_h5_data(dict(connectivity=R + 0.001 * i, coordinates=coords,
                          unit_means=np.zeros(N), unit_stds=np.ones(N),
                          n_obs=np.asarray(1000), surrogate_var=var),
                     os.path.join(conn_dir, 'connectivity_alpha_sample%d.h5' % i),
                     verbose=False)
    run_split_halves(output_dir=tmp, verbose=False)

    half = load_h5_data(os.path.join(conn_dir, 'connectivity_alpha_%s.h5' % HALF_NAMES[0]),
                        verbose=False)
    check('split_halves propagates surrogate_var to the halves', 'surrogate_var' in half)
    check('the half variance is the 2-sample average, not a copy of the per-sample one',
          np.allclose(half['surrogate_var'], average_surrogate_var([var, var])))

    # The scorer must return exactly what the parcellation clusters.
    path = os.path.join(conn_dir, 'connectivity_alpha_%s.h5' % HALF_NAMES[0])
    check('score.load_connectivity == util.surrogate_normalized on the same file',
          np.allclose(load_connectivity(path),
                      surrogate_normalized(load_h5_data(path, verbose=False))))
    check('and it is normalized, not raw |r|',
          not np.allclose(load_connectivity(path),
                          np.abs(np.nan_to_num(half['connectivity']))))

    # A partial set of variances must be refused, not silently averaged with the wrong
    # denominator -- the failure a re-run with a changed n_surrogates would produce.
    d = load_h5_data(os.path.join(conn_dir, 'connectivity_alpha_sample1.h5'), verbose=False)
    del d['surrogate_var']
    save_h5_data(d, os.path.join(conn_dir, 'connectivity_alpha_sample1.h5'), verbose=False)
    for name in HALF_NAMES:
        os.remove(os.path.join(conn_dir, 'connectivity_alpha_%s.h5' % name))
    try:
        run_split_halves(output_dir=tmp, verbose=False)
        raised = False
    except AssertionError:
        raised = True
    check('a partial set of surrogate variances raises rather than averaging wrongly', raised)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------------------------------
# The confound normalization INTRODUCES on the real tree, and the diagnostic for it.
# ---------------------------------------------------------------------------------------
from parcelmate.metrics import triviality

n_units = absz.shape[0]
coords_t = np.stack([np.repeat(np.arange(10), n_units // 10), np.arange(n_units)], 1)
noise_scale = np.sqrt(var).mean(axis=1)
rand_P = np.eye(10)[np.random.RandomState(1).randint(0, 10, n_units)]
check('ami_noise_scale is absent unless a noise scale is supplied',
      'ami_noise_scale' not in triviality(rand_P, coords_t))
check('ami_noise_scale ~ 0 for a partition unrelated to detectability',
      abs(triviality(rand_P, coords_t, noise_scale=noise_scale)['ami_noise_scale']) < 0.1)
ns_lab = (np.argsort(np.argsort(noise_scale)) * 10) // n_units
check('ami_noise_scale = 1 for a partition that IS the detectability decile',
      triviality(np.eye(10)[ns_lab], coords_t,
                 noise_scale=noise_scale)['ami_noise_scale'] > 0.99)
check('a mismatched noise_scale length raises rather than scoring nonsense',
      _raises(lambda: triviality(rand_P, coords_t, noise_scale=noise_scale[:5])))

# ---------------------------------------------------------------------------------------
# The Fisher arms. fisher() maps anything above 1 to arctanh(0.999) = 3.8 with no error, so
# handing it |z| would silently binarize three of the five arms at |z| > 1. That must be
# loud in sample_parcellations, and run_parcellation must route normalized input around it.
# ---------------------------------------------------------------------------------------
from parcelmate.model import run_parcellation, sample_parcellations

try:
    sample_parcellations(absz, n_networks=3, n_samples=2, binarize_connectivity=False,
                         fisher_transform=True, verbose=False, seed=0)
    raised = False
except AssertionError as err:
    raised = 'not a correlation matrix' in str(err)
check('sample_parcellations(fisher_transform=True) refuses a |z| matrix loudly', raised)

tmp = tempfile.mkdtemp(prefix='parcelmate_iter6b_')
try:
    conn_dir = os.path.join(tmp, CONNECTIVITY_NAME)
    os.makedirs(conn_dir)
    coords = np.stack([np.repeat(np.arange(10), N // 10), np.arange(N)], 1).astype(np.int32)
    for key, extra in (('avg', dict(surrogate_var=var)), ('raw', {})):
        d = dict(connectivity=R, coordinates=coords, unit_means=np.zeros(N),
                 unit_stds=np.ones(N), n_obs=np.asarray(1000), **extra)
        save_h5_data(d, os.path.join(conn_dir, 'connectivity_%s_avg.h5' % key), verbose=False)
    # One config, two trees: the same "Fisher" arm must run on both.
    run_parcellation(output_dir=tmp, variant='fisher', n_networks=3, n_samples=2, seed=1,
                     binarize_connectivity=False, fisher_transform=True,
                     connectivity_pca_components=None, verbose=False)
    import h5py
    attrs = {}
    for key in ('avg', 'raw'):
        with h5py.File(os.path.join(tmp, 'fisher', 'parcellation',
                                    'parcellation_%s_avg.h5' % key), 'r') as f:
            attrs[key] = dict(f.attrs)
    check('a normalized tree records input_normalized=True and fisher_transform_applied=False',
          attrs['avg']['input_normalized'] and not attrs['avg']['fisher_transform_applied'])
    check('an unnormalized tree records input_normalized=False and the transform applied',
          not attrs['raw']['input_normalized'] and attrs['raw']['fisher_transform_applied'])
    check('the configured arm setting is still recorded as fisher_transform=True on both',
          attrs['avg']['fisher_transform'] and attrs['raw']['fisher_transform'])
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print('\n%s' % ('All checks passed.' if not failures
                else 'FAILURES:\n  ' + '\n  '.join(failures)))
raise SystemExit(1 if failures else 0)
