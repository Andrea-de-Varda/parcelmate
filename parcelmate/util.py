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
    if merge and os.path.exists(path):
        with h5py.File(path, 'a') as f:
            for key in data:
                if key in f:  # h5py cannot resize in place, so replace
                    del f[key]
                f.create_dataset(key, data=data[key])
    else:
        with h5py.File(path, 'w') as f:
            for key in data:
                f.create_dataset(key, data=data[key])


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


def load_h5_data(path, verbose=True, indent=0):
    if verbose:
        stderr('%sLoading from %s\n' % (' ' * indent, path))
    out = {}
    with h5py.File(path, 'r') as f:
        for key in f.keys():
            out[key] = f[key][:]

    return out