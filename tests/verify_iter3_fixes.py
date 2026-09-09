"""Verification for the reliability/fidelity experiment machinery (see LOG.md, Iteration 3).

Run from the repo root with the `analysis` conda environment:
    PYTHONPATH=. python tests/verify_iter3_fixes.py

Covers:
  circshift null  - shifts are per-unit, distinct, marginal-preserving, and actually
                    destroy cross-unit correlation
  three arms      - legacy (reproduces the pre-S2 transpose), current, and the no-PCA
                    Fisher-transformed ablation, all reachable and mutually distinct
  run_connectivity - `null_model='circshift'` writes a parallel tree with identical filenames,
                    caching notices a missing null, and the guards fire
"""

import os
import shutil
import tempfile

import numpy as np

from parcelmate.constants import CONNECTIVITY_NAME
from parcelmate.data import circshift_timecourses, correlate
from parcelmate.model import run_parcellation, sample_parcellations
from parcelmate.util import array_fingerprint, load_h5_data, save_h5_data

failures = []


def check(name, cond):
    print('%s %s' % ('OK' if cond else 'FAIL', name))
    if not cond:
        failures.append(name)


# ----------------------------------------------------------------- circshift null
rng = np.random.RandomState(0)
n_units, n_tokens = 40, 500
# Correlated data: two blocks of units sharing a latent signal.
latent = rng.randn(2, n_tokens)
X = np.repeat(latent, n_units // 2, axis=0) + rng.randn(n_units, n_tokens) * 0.5

S = circshift_timecourses(X, rng=np.random.RandomState(1))
check('circshift: shape preserved', S.shape == X.shape)
check('circshift: every row is a rotation of the original',
      all(np.allclose(np.sort(S[i]), np.sort(X[i])) for i in range(n_units)))
check('circshift: per-unit mean and std preserved exactly',
      np.allclose(S.mean(axis=1), X.mean(axis=1)) and np.allclose(S.std(axis=1), X.std(axis=1)))
check('circshift: no row left unshifted', not any(np.allclose(S[i], X[i]) for i in range(n_units)))

R_real = correlate(X.copy(), rowvar=True, use_gpu=False)
R_null = correlate(S.copy(), rowvar=True, use_gpu=False)
iu = np.triu_indices(n_units, k=1)
check('circshift: destroys cross-unit correlation (%.3f -> %.3f mean |r|)'
      % (np.abs(R_real[iu]).mean(), np.abs(R_null[iu]).mean()),
      np.abs(R_null[iu]).mean() < np.abs(R_real[iu]).mean() / 3)

check('circshift: reproducible for a given rng seed',
      np.array_equal(circshift_timecourses(X, rng=np.random.RandomState(7)),
                     circshift_timecourses(X, rng=np.random.RandomState(7))))
check('circshift: varies with seed',
      not np.array_equal(circshift_timecourses(X, rng=np.random.RandomState(7)),
                         circshift_timecourses(X, rng=np.random.RandomState(8))))
try:
    circshift_timecourses(np.zeros((10, 5)), rng=np.random.RandomState(0))
    check('circshift: refuses when tokens <= units', False)
except AssertionError:
    check('circshift: refuses when tokens <= units', True)


# ----------------------------------------------------------------- the three arms
def make_connectivity(n=120, n_blocks=4, n_obs=400, seed=0):
    """A genuine correlation matrix (|r| <= 1) with planted block structure.

    Built by correlating synthetic timecourses rather than by perturbing a block indicator,
    so the result satisfies the constraints real connectivity does -- bounded, symmetric,
    positive semi-definite. The earlier version of this helper produced entries above 1,
    which is not a correlation matrix and made the Fisher arm fail spuriously.
    """
    r = np.random.RandomState(seed)
    block = np.repeat(np.arange(n_blocks), n // n_blocks)
    latent = r.randn(n_blocks, n_obs)
    Z = latent[block] + r.randn(n, n_obs) * 1.2
    R = np.abs(np.corrcoef(Z))
    return R, block


R, true_block = make_connectivity()
common = dict(n_networks=4, n_samples=8, verbose=False, seed=3)

arm_current = sample_parcellations(R, binarize_connectivity=True, legacy_binarize=False,
                                   connectivity_pca_components=10, **common)
arm_legacy = sample_parcellations(R, binarize_connectivity=True, legacy_binarize=True,
                                  connectivity_pca_components=10, **common)
arm_ablation = sample_parcellations(R, binarize_connectivity=False, fisher_transform=True,
                                    connectivity_pca_components=None, **common)
for name, arm in [('current', arm_current), ('legacy', arm_legacy), ('ablation', arm_ablation)]:
    check('arms: %s returns one label per unit' % name, arm['samples'].shape == (8, R.shape[0]))
check('arms: legacy differs from current (the S2 transpose changes results)',
      not np.array_equal(arm_legacy['samples'], arm_current['samples']))
check('arms: ablation differs from current',
      not np.array_equal(arm_ablation['samples'], arm_current['samples']))
check('arms: connectivity matrix is not mutated by any arm', np.isfinite(R).all() and R.max() <= 1.0)

# The legacy binarization is the transpose of the intended one, so its row densities vary
# while the corrected one gives every unit the same number of partners.
B_cur = (R > np.quantile(R, 0.9, axis=1, keepdims=True))
B_leg = (R > np.quantile(R, 0.9, axis=1))
check('arms: corrected binarization gives uniform row density',
      len(set(B_cur.sum(axis=1))) == 1)
check('arms: legacy binarization gives varying row density (hubness leaks in)',
      len(set(B_leg.sum(axis=1))) > 1)

try:
    sample_parcellations(R, binarize_connectivity=True, fisher_transform=True, **common)
    check('arms: binarize + fisher_transform is rejected', False)
except AssertionError:
    check('arms: binarize + fisher_transform is rejected', True)


# --------------------------------------------------- arms reach run_parcellation
# (the on-disk layout itself is checked in the variant section below)
tmp = tempfile.mkdtemp(prefix='parcelmate_iter3_')
try:
    conn = os.path.join(tmp, 'real', CONNECTIVITY_NAME)
    os.makedirs(conn)
    coords = np.stack([np.zeros(R.shape[0]), np.arange(R.shape[0])], axis=1).astype(np.int32)
    save_h5_data(dict(connectivity=R, coordinates=coords),
                 os.path.join(conn, 'connectivity_alpha_avg.h5'), verbose=False)

    run_parcellation(output_dir=os.path.join(tmp, 'real'), variant='abl', n_networks=4,
                     n_samples=6, binarize_connectivity=False, fisher_transform=True,
                     connectivity_pca_components=None, seed=5, verbose=False)
    abl = load_h5_data(os.path.join(tmp, 'real', 'abl', 'parcellation',
                                    'parcellation_alpha_avg.h5'), verbose=False)
    check('plumbing: run_parcellation forwards the ablation arm',
          'parcellation' in abl and abl['parcellation'].shape == (R.shape[0], 4))

    run_parcellation(output_dir=os.path.join(tmp, 'real'), variant='leg', n_networks=4,
                     n_samples=6, binarize_connectivity=True, legacy_binarize=True,
                     connectivity_pca_components=10, seed=5, verbose=False)
    leg = load_h5_data(os.path.join(tmp, 'real', 'leg', 'parcellation',
                                    'parcellation_alpha_avg.h5'), verbose=False)
    check('plumbing: run_parcellation forwards the legacy arm',
          not np.allclose(abl['parcellation'], leg['parcellation']))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# run_connectivity's null guards (cheap to check without running a model)
import inspect  # noqa: E402

from parcelmate.model import run_connectivity  # noqa: E402

params = inspect.signature(run_connectivity).parameters
check('plumbing: run_connectivity exposes null_model and null_output_dir',
      'null_model' in params and 'null_output_dir' in params)
check('plumbing: null_model defaults to off', params['null_model'].default is None)
# Named null_model, not null: YAML parses a bare `null` key as None, which arrives as a
# non-string keyword and fails with "keywords must be strings".
check('plumbing: no bare `null` parameter (YAML reserves it)', 'null' not in params)
for bad, why in [(dict(null_model='shuffle', null_output_dir='/tmp/x'), 'an unknown null_model'),
                 (dict(null_model='circshift'), 'a missing null_output_dir')]:
    try:
        run_connectivity(domains=('whitespace',), verbose=False, **bad)
        check('plumbing: rejects %s' % why, False)
    except AssertionError:
        check('plumbing: rejects %s' % why, True)
    except Exception as e:  # noqa: BLE001
        check('plumbing: rejects %s (got %s)' % (why, type(e).__name__), False)


# ------------------------------------------------- variant layout and provenance
from parcelmate.constants import PARCELLATION_NAME, RESERVED_VARIANT_NAMES  # noqa: E402
from parcelmate.model import run_subnetwork_extraction  # noqa: E402
from parcelmate.util import read_attrs  # noqa: E402

tmp2 = tempfile.mkdtemp(prefix='parcelmate_variants_')
try:
    run = os.path.join(tmp2, 'run')
    conn = os.path.join(run, CONNECTIVITY_NAME)
    os.makedirs(conn)
    coords = np.stack([np.zeros(R.shape[0]), np.arange(R.shape[0])], axis=1).astype(np.int32)
    for domain in ('alpha', 'beta'):
        save_h5_data(dict(connectivity=R, coordinates=coords),
                     os.path.join(conn, 'connectivity_%s_avg.h5' % domain), verbose=False)

    arms = {
        'legacy': dict(binarize_connectivity=True, legacy_binarize=True,
                       connectivity_pca_components=10),
        'current': dict(binarize_connectivity=True, legacy_binarize=False,
                        connectivity_pca_components=10),
        'nopca_fisher': dict(binarize_connectivity=False, fisher_transform=True,
                             connectivity_pca_components=None),
    }
    for name, kw in arms.items():
        run_parcellation(output_dir=run, variant=name, n_networks=4, n_samples=6,
                         seed=5, verbose=False, **kw)

    got = {}
    for name in arms:
        f = os.path.join(run, name, PARCELLATION_NAME, 'parcellation_alpha_avg.h5')
        check('variants: %s wrote its own parcellation file' % name, os.path.exists(f))
        got[name] = load_h5_data(f, verbose=False)['parcellation']

    check('variants: arms coexist and differ',
          all(not np.allclose(got[a], got[b])
              for a, b in [('legacy', 'current'), ('current', 'nopca_fisher')]))
    check('variants: the connectivity file is never written to',
          'parcellation' not in load_h5_data(
              os.path.join(conn, 'connectivity_alpha_avg.h5'), verbose=False))

    attrs = read_attrs(os.path.join(run, 'current', PARCELLATION_NAME, 'parcellation_alpha_avg.h5'))
    for key in ('variant', 'domain', 'n_networks', 'legacy_binarize', 'fisher_transform',
                'seed', 'source_path', 'source_fingerprint', 'git_commit', 'created'):
        check('provenance: %s recorded' % key, key in attrs)
    check('provenance: variant name is correct', attrs.get('variant') == 'current')
    check('provenance: source fingerprint matches the connectivity actually used',
          attrs.get('source_fingerprint') == array_fingerprint(R))

    # A recomputed connectivity changes the fingerprint, so staleness is detectable --
    # this is what the parcellation-inside-connectivity layout could not express (M8).
    check('provenance: a different connectivity yields a different fingerprint',
          array_fingerprint(R) != array_fingerprint(make_connectivity(seed=99)[0]))

    for name in arms:
        run_subnetwork_extraction(output_dir=run, variant=name, verbose=False)
        sub = os.path.join(run, name, 'subnetwork', 'parcellation_shared_avg.h5')
        check('variants: subnetwork extraction writes under %s' % name, os.path.exists(sub))

    try:
        run_parcellation(output_dir=run, variant=CONNECTIVITY_NAME, n_networks=4,
                         n_samples=4, verbose=False)
        check('variants: a reserved variant name is rejected', False)
    except AssertionError:
        check('variants: a reserved variant name is rejected', True)
    check('variants: reserved list covers the shared dirs',
          CONNECTIVITY_NAME in RESERVED_VARIANT_NAMES and 'metrics' in RESERVED_VARIANT_NAMES)
finally:
    shutil.rmtree(tmp2, ignore_errors=True)

print()
if failures:
    raise SystemExit('%d check(s) failed: %s' % (len(failures), failures))
print('All checks passed.')
