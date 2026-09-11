import sys
import os
import hashlib
import numpy as np
import h5py

def stderr(s):
    sys.stderr.write(s)
    sys.stderr.flush()


def derive_seed(seed, *keys):
    """Deterministically derive a sub-seed from a base seed and one or more keys.

    Used so that each unit of work (a domain's data draw, one file's parcellation) is
    reproducible *in isolation*, not merely as part of a whole-pipeline run. Because the
    pipeline caches to HDF5 and skips completed work, re-running a single domain must
    reproduce what the full run produced for that domain; a single global RNG stream
    would make each domain depend on how many draws preceded it.

    Uses hashlib rather than the builtin hash(), which is salted per process for strings
    and would therefore not be reproducible across runs.
    """
    if seed is None:
        return None
    key = '|'.join(str(k) for k in keys)
    digest = hashlib.sha256(key.encode('utf-8')).hexdigest()

    return int((int(seed) + int(digest[:8], 16)) % (2 ** 32))


def set_seed(seed):
    """Seed the global RNGs (numpy, torch, python's random). No-op if seed is None.

    Note that this does not make GPU results bit-exact: the tiled matmul in
    `correlate` accumulates in a nondeterministic order on CUDA. Seeding pins the
    algorithmic choices (data order, cluster initializations, randomized SVD), which
    is what reproducibility of the parcellation requires.
    """
    if seed is None:
        return
    import random
    import torch
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DERIVED_KEYS = ('parcellation',)  # Computed *from* connectivity, so invalidated when it changes


def h5_keys(path):
    """Keys present in an HDF5 file, without reading any array data. Empty if absent/unreadable."""
    if not os.path.exists(path):
        return []
    try:
        with h5py.File(path, 'r') as f:
            return list(f.keys())
    except (OSError, KeyError):
        return []


def save_h5_data(
        data,
        path,
        merge=False,
        attrs=None,
        verbose=True,
        indent=0
):
    """Write arrays to an HDF5 file.

    By default the file is truncated, so keys absent from `data` are dropped. Pass
    `merge=True` to update only the given keys and leave the rest of the file intact --
    use that when adding a derived product to a file whose other contents are still
    valid, which also avoids rewriting the (large) connectivity matrix to append a
    (small) parcellation.
    """
    dirpath = os.path.dirname(path)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath)
    if verbose:
        stderr('%sSaving to %s%s\n' % (' ' * indent, path, ' (merge)' if merge else ''))
    mode = 'a' if (merge and os.path.exists(path)) else 'w'
    with h5py.File(path, mode) as f:
        for key in data:
            if key in f:  # h5py cannot resize in place, so replace
                del f[key]
            f.create_dataset(key, data=data[key])
        for key, value in (attrs or {}).items():
            f.attrs[key] = '' if value is None else value


def warn_dropped_keys(path, keys_to_write, verbose=True, indent=0):
    """Announce derived keys that a truncating write is about to discard.

    Discarding them is correct -- a parcellation computed from the previous connectivity
    does not describe the new one -- but it must not be silent, because it throws away the
    output of the pipeline's longest stage and nothing else in the log would say so.
    """
    dropped = [k for k in h5_keys(path) if k in DERIVED_KEYS and k not in keys_to_write]
    if dropped and verbose:
        stderr('%sNOTE: discarding stale %s in %s (recomputed connectivity invalidates it; re-run the parcellation step)\n' % (
            ' ' * indent, ', '.join(dropped), os.path.basename(path)
        ))

    return dropped


def load_h5_array(path, key):
    """Load a single dataset from an HDF5 file, without reading the rest of it.

    `load_h5_data` reads every key, which for a connectivity file means pulling ~400 MB
    into memory to get at a 9984-element vector.
    """
    with h5py.File(path, 'r') as f:
        # `[()]` rather than `[:]`: it reads datasets of any shape, including the scalar
        # ones (e.g. n_obs), where `[:]` raises "Illegal slicing argument for scalar
        # dataspace".
        return f[key][()]


def load_h5_data(path, verbose=True, indent=0):
    if verbose:
        stderr('%sLoading from %s\n' % (' ' * indent, path))
    out = {}
    with h5py.File(path, 'r') as f:
        for key in f.keys():
            out[key] = f[key][()]

    return out

def git_commit():
    """Short hash of the current commit, with a `-dirty` suffix if the tree has changes.

    Recorded in every derived file so a result can be traced back to the code that made it.
    Returns 'unknown' outside a git checkout rather than raising.
    """
    import subprocess
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        rev = subprocess.run(['git', '-C', root, 'rev-parse', '--short', 'HEAD'],
                             capture_output=True, text=True, timeout=10)
        if rev.returncode != 0:
            return 'unknown'
        dirty = subprocess.run(['git', '-C', root, 'status', '--porcelain'],
                               capture_output=True, text=True, timeout=10)
        return rev.stdout.strip() + ('-dirty' if dirty.stdout.strip() else '')
    except Exception:  # noqa: BLE001 -- provenance must never break a pipeline run
        return 'unknown'


def array_fingerprint(arr):
    """Content hash of an array, for detecting that a derived product is stale.

    A parcellation is computed *from* a connectivity matrix. Storing the source's
    fingerprint alongside it makes "this was built from a matrix that no longer exists"
    a detectable condition rather than a silent inconsistency (the failure mode behind
    M8). Hashes the full buffer -- ~0.3 s for a 400 MB matrix, once per parcellation.
    """
    return hashlib.sha256(np.ascontiguousarray(arr)).hexdigest()[:16]


def read_attrs(path):
    """Attributes of an HDF5 file as a plain dict. Empty if the file is absent/unreadable."""
    if not os.path.exists(path):
        return {}
    try:
        with h5py.File(path, 'r') as f:
            return {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
    except (OSError, KeyError):
        return {}

def surrogate_normalized(data):
    """Form the non-negative connectivity matrix the clusterer and the scorer both consume.

    THE ONLY PLACE `|r|` IS FORMED. Both `run_parcellation` and `score.load_connectivity`
    call this, so the parcellation and the metric can never disagree about what matrix they
    are talking about -- a class of bug this project has already paid for once (LOG.md S10).

    If the file carries `surrogate_var` -- the per-pair variance of the correlation under
    circular shifts, written by `run_connectivity(n_surrogates=K)` -- each entry is divided
    by its own null standard deviation before the absolute value is taken, giving |z| rather
    than |r|. Otherwise the matrix is returned as |r| and the behaviour is exactly what it
    was, so old trees score unchanged.

    Why this matters (LOG.md Iteration 9). For a pair of units with no true correlation, r is
    zero-mean noise of size sigma_ij, so E|r_ij| = sqrt(2/pi) * sigma_ij: the absolute value
    turns noise into a POSITIVE MEAN FIELD whose shape is sigma_ij, and sigma_ij varies by
    pair because it depends on the two units' autocorrelations, which a circular shift
    preserves exactly. That field is separable-ish, hence maximally block-fittable and
    maximally reproducible, which is why a structureless null out-scored real data on both
    metrics. Dividing by sigma_ij makes E|z_ij| the SAME CONSTANT for every pair, so the
    field is gone: the null becomes exchangeable, a block model has nothing to fit, and
    k-means has no per-unit magnitude to sort by.

    Detection is by presence, not by a flag. A boolean that had to be set identically in the
    parcellation config and the scoring config is exactly the kind of thing that gets set in
    one and forgotten in the other; here a tree either carries the variances or it does not.
    """
    R = np.nan_to_num(np.asarray(data['connectivity']))
    if 'surrogate_var' in data:
        sd = np.sqrt(np.maximum(np.nan_to_num(np.asarray(data['surrogate_var'])), 0.0))
        # Pairs whose null variance underflowed to zero carry no information about effect
        # size; zeroing them is the conservative reading and keeps the matrix finite.
        R = np.divide(R, sd, out=np.zeros_like(R, dtype=np.float64), where=sd > 0)
    return np.abs(R)


def average_surrogate_var(variances):
    """Null variance of a Fisher average of independent connectivity estimates.

    The halves and the `avg` file are Fisher averages of per-sample matrices. Under the null
    the correlations are near zero, where arctanh(r) = r + O(r^3), so the Fisher average is a
    plain mean to the accuracy that matters here and Var(mean of m independent) = sum/m^2.
    The approximation is used ONLY on null variances, which are small by construction; it is
    never applied to the real correlations, which are averaged exactly as before.
    """
    v = [np.nan_to_num(np.asarray(x, dtype=np.float64)) for x in variances]
    return sum(v) / float(len(v) ** 2)
