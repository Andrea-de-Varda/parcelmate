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


def save_h5_data(
        data,
        path,
        verbose=True,
        indent=0
):
    dirpath = os.path.dirname(path)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    if verbose:
        stderr('%sSaving to %s\n' % (' ' * indent, path))
    with h5py.File(path, 'w') as f:
        for key in data:
            f.create_dataset(key, data=data[key])


def load_h5_data(path, verbose=True, indent=0):
    if verbose:
        stderr('%sLoading from %s\n' % (' ' * indent, path))
    out = {}
    with h5py.File(path, 'r') as f:
        for key in f.keys():
            out[key] = f[key][:]

    return out