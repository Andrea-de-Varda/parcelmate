"""Verification for the iteration-1 changes (see LOG.md).

Run from the repo root with the `analysis` conda environment:
    PYTHONPATH=. python tests/verify_iter1_fixes.py

Covers:
  S7  - seeding: derive_seed determinism, reproducible parcellations, and the requirement
        that the 100 clustering restarts stay DIFFERENT from one another
  M4  - run_parcellation parcellates only `*_avg.h5` unless parcellate_samples=True
  M6  - subnetwork extraction requires a full reciprocal-best-match clique, is
        independent of domain ordering, and no longer crashes when nothing survives
  M1  - correlate() auto-detects the absence of a GPU instead of crashing
"""

import os
import shutil
import tempfile

import numpy as np

from parcelmate.constants import CONNECTIVITY_NAME, SUBNETWORK_NAME
from parcelmate.data import correlate
from parcelmate.model import (
    _is_reciprocal_clique,
    align_samples,
    run_parcellation,
    run_subnetwork_extraction,
    sample_parcellations,
)
from parcelmate.util import derive_seed, load_h5_data, save_h5_data

failures = []


def check(name, cond):
    print('%s %s' % ('OK' if cond else 'FAIL', name))
    if not cond:
        failures.append(name)


def make_connectivity(n_units=120, n_blocks=4, seed=0):
    """Block-structured symmetric connectivity, so a real cluster structure exists."""
    rng = np.random.RandomState(seed)
    block = np.repeat(np.arange(n_blocks), n_units // n_blocks)
    R = (block[:, None] == block[None, :]).astype(float)
    R = R + rng.randn(n_units, n_units) * 0.35
    R = (R + R.T) / 2
    np.fill_diagonal(R, 1.0)
    return R, block


# ---------------------------------------------------------------- S7: derive_seed
check('S7: derive_seed is deterministic', derive_seed(42, 'wikitext') == derive_seed(42, 'wikitext'))
check('S7: derive_seed separates keys', derive_seed(42, 'wikitext') != derive_seed(42, 'bookcorpus'))
check('S7: derive_seed separates base seeds', derive_seed(1, 'x') != derive_seed(2, 'x'))
check('S7: derive_seed passes None through', derive_seed(None, 'x') is None)
check('S7: derive_seed stays in uint32 range', 0 <= derive_seed(42, 'x') < 2 ** 32)

# ------------------------------------------------- S7: reproducibility of parcellation
R, true_block = make_connectivity()
kw = dict(n_networks=4, n_samples=12, binarize_connectivity=True,
          connectivity_pca_components=10)
a = sample_parcellations(R, seed=123, verbose=False, **kw)
b = sample_parcellations(R, seed=123, verbose=False, **kw)
c = sample_parcellations(R, seed=456, verbose=False, **kw)
check('S7: same seed reproduces clustering restarts', np.array_equal(a['samples'], b['samples']))
check('S7: same seed reproduces inertias', np.allclose(a['scores'], b['scores']))
check('S7: different seed gives different restarts', not np.array_equal(a['samples'], c['samples']))

# The consensus is meaningless if every restart is identical, so confirm the shared RNG
# stream still varies the initializations within a single call.
n_distinct = len({tuple(row) for row in a['samples']})
check('S7: restarts within a run are not all identical (%d distinct of 12)' % n_distinct, n_distinct > 1)

p1 = align_samples(a['samples'], a['scores'], seed=7, verbose=False)
p2 = align_samples(b['samples'], b['scores'], seed=7, verbose=False)
check('S7: alignment is reproducible', np.allclose(p1, p2))
check('S7: parcellation rows still sum to 1', np.allclose(p1.sum(axis=1), 1))

# ------------------------------------------- S7: `random` baseline vocab order (S8)
# tokenizer.get_vocab() returns a dict whose iteration order is randomized per process,
# so the token list must be sorted or the same seed picks different ids on each run.
# Cannot be observed within one process, so assert the invariant that fixes it.
from transformers import AutoTokenizer  # noqa: E402

from parcelmate.data import BaselineDataset  # noqa: E402

_tok = AutoTokenizer.from_pretrained('gpt2')
_ds = BaselineDataset('random', 16, tokenizer=_tok, seed=5)
check('S8: random-baseline token list is deterministically ordered',
      list(_ds.tokens) == sorted(_ds.tokens))
check('S8: random baseline is reproducible for a given seed',
      next(BaselineDataset('random', 16, tokenizer=_tok, seed=5))
      == next(BaselineDataset('random', 16, tokenizer=_tok, seed=5)))
check('S8: random baseline varies with seed',
      next(BaselineDataset('random', 16, tokenizer=_tok, seed=5))
      != next(BaselineDataset('random', 16, tokenizer=_tok, seed=6)))

# ---------------------------------------------------------------- M1: CPU fallback
X = np.random.RandomState(0).randn(40, 15)
Rc = correlate(X.copy(), rowvar=True, use_gpu=False)
Rn = np.corrcoef(X, rowvar=True)
check('M1: correlate(use_gpu=False) matches np.corrcoef', np.allclose(Rc, Rn, atol=1e-6))
check('M1: correlate defaults to auto-detect (no crash)', correlate(X.copy(), rowvar=True).shape == (40, 40))

# ---------------------------------------------------------------- M4 and M6 on disk
tmp = tempfile.mkdtemp(prefix='parcelmate_iter1_')
try:
    conn_dir = os.path.join(tmp, CONNECTIVITY_NAME)
    os.makedirs(conn_dir)
    coords = np.stack([np.zeros(R.shape[0]), np.arange(R.shape[0])], axis=1).astype(np.int32)
    for domain in ('alpha', 'beta', 'gamma'):
        for key in ('avg', 'sample1', 'sample2'):
            save_h5_data(
                dict(connectivity=R, coordinates=coords),
                os.path.join(conn_dir, 'connectivity_%s_%s.h5' % (domain, key)),
                verbose=False
            )

    run_parcellation(output_dir=tmp, seed=11, verbose=False, **kw)
    has = {f: 'parcellation' in load_h5_data(os.path.join(conn_dir, f), verbose=False)
           for f in sorted(os.listdir(conn_dir))}
    check('M4: avg files are parcellated', all(v for f, v in has.items() if f.endswith('avg.h5')))
    check('M4: per-sample files are skipped by default',
          not any(v for f, v in has.items() if 'sample' in f))

    run_parcellation(output_dir=tmp, seed=11, parcellate_samples=True, verbose=False, **kw)
    has2 = {f: 'parcellation' in load_h5_data(os.path.join(conn_dir, f), verbose=False)
            for f in sorted(os.listdir(conn_dir))}
    check('M4: parcellate_samples=True recovers per-sample files', all(has2.values()))

    # --- M6: clique extraction
    run_subnetwork_extraction(output_dir=tmp, verbose=False)
    out = load_h5_data(
        os.path.join(tmp, SUBNETWORK_NAME, 'parcellation_shared_avg.h5'), verbose=False
    )
    n_all = out['parcellation'].shape[1]
    check('M6: extraction produces a (n_units, n_networks) parcellation',
          out['parcellation'].shape[0] == R.shape[0])
    check('M6: at least one shared network found across identical domains', n_all > 0)

    # Order independence: the surviving set must not depend on which domain is first.
    run_subnetwork_extraction(output_dir=tmp, domains=['gamma', 'beta', 'alpha'], verbose=False)
    out_rev = load_h5_data(
        os.path.join(tmp, SUBNETWORK_NAME, 'parcellation_shared_avg.h5'), verbose=False
    )
    check('M6: result is independent of the order domains are supplied in',
          np.allclose(np.sort(out['parcellation'], axis=None),
                      np.sort(out_rev['parcellation'], axis=None)))

    # Domain subsetting (the mechanism for excluding baselines).
    run_subnetwork_extraction(output_dir=tmp, domains=['alpha', 'beta'], verbose=False)
    out_sub = load_h5_data(
        os.path.join(tmp, SUBNETWORK_NAME, 'parcellation_shared_avg.h5'), verbose=False
    )
    check('M6: subsetting domains is at least as permissive as using all',
          out_sub['parcellation'].shape[1] >= n_all)

    # Empty result must not raise (previously np.stack([]) crashed).
    empty_dir = os.path.join(tmp, 'empty')
    os.makedirs(os.path.join(empty_dir, CONNECTIVITY_NAME))
    rng = np.random.RandomState(3)
    for domain in ('alpha', 'beta'):
        # Orthogonal random parcellations: reciprocal best matches are unlikely to be
        # consistent, and any that survive must still not crash the save path.
        parc = rng.rand(R.shape[0], 4)
        save_h5_data(
            dict(connectivity=R, coordinates=coords, parcellation=parc),
            os.path.join(empty_dir, CONNECTIVITY_NAME, 'connectivity_%s_avg.h5' % domain),
            verbose=False
        )
    try:
        run_subnetwork_extraction(output_dir=empty_dir, verbose=False)
        check('M6: degenerate input does not raise', True)
    except Exception as e:  # noqa: BLE001
        check('M6: degenerate input does not raise (%s: %s)' % (type(e).__name__, e), False)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------- M6: clique predicate
shared = {
    'a': {'b': {0: 5}, 'c': {0: 9}},
    'b': {'a': {5: 0}, 'c': {5: 9}},
    'c': {'a': {9: 0}, 'b': {9: 5}},
}
check('M6: consistent triangle is a clique',
      _is_reciprocal_clique({'a': 0, 'b': 5, 'c': 9}, ['a', 'b', 'c'], shared))

# The case the old chaining accepted and the clique test rejects: a->b and b->c hold,
# but a and c disagree, so identity drifted across the chain.
shared_broken = {
    'a': {'b': {0: 5}, 'c': {0: 3}},
    'b': {'a': {5: 0}, 'c': {5: 9}},
    'c': {'a': {3: 0}, 'b': {9: 5}},
}
check('M6: chain that drifts is rejected',
      not _is_reciprocal_clique({'a': 0, 'b': 5, 'c': 9}, ['a', 'b', 'c'], shared_broken))

# ---------------------------------------------------------------- M8: merge semantics
import io                                               # noqa: E402
import contextlib                                       # noqa: E402

from parcelmate.util import h5_keys, warn_dropped_keys  # noqa: E402

tmp8 = tempfile.mkdtemp(prefix='parcelmate_m8_')
try:
    path = os.path.join(tmp8, 'f.h5')
    conn = np.arange(12, dtype=float).reshape(3, 4)
    save_h5_data(dict(connectivity=conn, coordinates=np.zeros((3, 2))), path, verbose=False)

    check('M8: h5_keys reads keys without loading data',
          sorted(h5_keys(path)) == ['connectivity', 'coordinates'])
    check('M8: h5_keys on a missing file returns empty',
          h5_keys(os.path.join(tmp8, 'nope.h5')) == [])

    save_h5_data(dict(parcellation=np.ones((3, 5))), path, merge=True, verbose=False)
    after = load_h5_data(path, verbose=False)
    check('M8: merge preserves existing keys',
          sorted(after) == ['connectivity', 'coordinates', 'parcellation'])
    check('M8: merge leaves existing data unchanged', np.array_equal(after['connectivity'], conn))
    check('M8: merge writes the new key', after['parcellation'].shape == (3, 5))

    save_h5_data(dict(parcellation=np.ones((3, 9))), path, merge=True, verbose=False)
    check('M8: merge replaces a key of different shape',
          load_h5_data(path, verbose=False)['parcellation'].shape == (3, 9))

    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        dropped = warn_dropped_keys(path, dict(connectivity=conn, coordinates=np.zeros((3, 2))))
    check('M8: truncating write reports the derived key it will discard', dropped == ['parcellation'])
    check('M8: the report names the file and the key',
          'parcellation' in buf.getvalue() and 'f.h5' in buf.getvalue())

    save_h5_data(dict(connectivity=conn, coordinates=np.zeros((3, 2))), path, verbose=False)
    check('M8: truncating write still drops the stale parcellation (correct invalidation)',
          'parcellation' not in h5_keys(path))

    buf2 = io.StringIO()
    with contextlib.redirect_stderr(buf2):
        none_dropped = warn_dropped_keys(path, dict(connectivity=conn))
    check('M8: no report when no derived key is present', none_dropped == [])

    fresh = os.path.join(tmp8, 'fresh.h5')
    save_h5_data(dict(a=np.zeros(3)), fresh, merge=True, verbose=False)
    check('M8: merge on a new file creates it', h5_keys(fresh) == ['a'])
finally:
    shutil.rmtree(tmp8, ignore_errors=True)


print()
if failures:
    raise SystemExit('%d check(s) failed: %s' % (len(failures), failures))
print('All checks passed.')
