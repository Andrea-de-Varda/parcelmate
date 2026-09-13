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


def variants_tag(variants, max_len=100):
    """Name for a restricted set of arms (`-V`), used in job names and score file names.

    The arm names joined by '_' while that stays short, as before. Past `max_len` it becomes
    '<n>arms_<digest>', the digest taken over the sorted names, so two different subsets
    never share a name and no path runs into the 255-byte file-name limit: the 13-arm early
    score of configs/final_mlp.yml is 215 characters joined, before the job prefix and the
    SLURM log suffix are added. `max_len` sits above the longest list used before (72
    characters, the YOLO 1 partial score), so earlier job and file names are unchanged.
    """
    joined = '_'.join(variants)
    if len(joined) <= max_len:
        return joined
    digest = hashlib.sha1(' '.join(sorted(variants)).encode('utf-8')).hexdigest()[:8]

    return '%darms_%s' % (len(variants), digest)


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

def connectivity_matrix(data, normalize=None):
    """Form the non-negative connectivity matrix the clusterer and the scorer both consume.

    THE ONLY PLACE `|r|` IS FORMED. `run_parcellation` and `score.load_connectivity` both
    call this, so the parcellation and the metric can never disagree about what matrix they
    are talking about -- a class of bug this project has already paid for once (LOG.md S10).

    `normalize` is an explicit per-arm choice, recorded in the parcellation's provenance and
    read back from there by the scorer, so one experiment tree can hold |r| arms and |z| arms
    side by side and the normalization becomes a rung of the ladder rather than a property
    of the whole experiment:

      None         |r|. The quantity of scientific interest: how strongly two units couple.
      'surrogate'  |r / sigma|, sigma_ij being the per-pair null standard deviation written by
                   `run_connectivity(n_surrogates=K)`. An effect size: how confidently the
                   coupling is non-zero. Removes the positive mean field that |r| manufactures
                   from noise (E|r| = sqrt(2/pi) sigma for an uncorrelated pair), at the price of
                   reweighting every profile by partner precision. See LOG.md Iterations 9-12.

    Earlier this was triggered by the mere presence of `surrogate_var` in the file, which
    forced a second experiment for one design choice. An explicit argument is the cleanup.
    """
    R = np.nan_to_num(np.asarray(data['connectivity']))
    if normalize is None:
        return np.abs(R)
    assert normalize == 'surrogate', 'Unknown normalize=%r (None or "surrogate")' % (normalize,)
    assert 'surrogate_var' in data, (
        'normalize="surrogate" needs `surrogate_var` in the connectivity file; this tree was '
        'built without `n_surrogates`. Re-run connectivity with n_surrogates > 0.')
    sd = np.sqrt(np.maximum(np.nan_to_num(np.asarray(data['surrogate_var'])), 0.0))
    # Pairs whose null variance underflowed to zero carry no information about effect
    # size; zeroing them is the conservative reading and keeps the matrix finite.
    return np.abs(np.divide(R, sd, out=np.zeros_like(R, dtype=np.float64), where=sd > 0))


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
