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
