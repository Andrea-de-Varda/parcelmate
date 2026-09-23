"""Out-of-core connectivity for models whose unit count makes the dense matrix impossible
(LOG.md Iteration 25). Qwen3.5-4B has 294,912 MLP neurons: a float32 half is 350 GB.

What changes against the in-memory path, and what does not:

  storage    each half's |r|-free (signed) correlation is written to HDF5 as float16 row
             tiles (`connectivity`, chunked by rows), with attrs `storage = tiled_fp16`.
             Nothing downstream ever loads it whole: `TiledMatrix` serves row blocks as
             float32 |r| exactly as `util.connectivity_matrix` would. Quantisation error
             of fp16 on r in [-1, 1] is about 5e-4 absolute at |r| = 1, 5e-5 at 0.1.
  timecourses two forward passes per sample instead of one: the first accumulates each
             unit's exact mean and standard deviation (float64 Welford), the second stores
             the z-scored activations as float16 (unit variance, so the quantisation is
             uniform and overflow impossible). r_ij is then the mean over tokens of
             z_i z_j, the Pearson correlation up to that quantisation.
  correlation on the GPU in row x column tiles with float32 accumulation (TF32 tensor
             cores where available); the column blocks stream from host RAM, the row block
             stays resident.
  the null   circular shifts applied on the GPU to each tile's rows on the fly, so no
             shifted copy of the timecourses exists. Offsets are drawn as in
             `data.circshift_timecourses` (seeded per domain and sample, distinct) when
             the units are fewer than the tokens, and with replacement otherwise: at
             294,912 units and 98,304 tokens, a fraction 1/T of pairs share an offset
             and keep their true lag-zero correlation, about 1e-5 of the matrix.
  the half   the Fisher mean of the two samples' correlations, formed tile by tile
             (arctanh of each, averaged, tanh), exactly `data.fisher_average`.
  profiles   the Fisher / standardize / top-10% / re-standardize transform is row-local,
             so a row block of profiles is computed from its own tile on the fly
             (`stream_profiles`); nothing is cached.
  PCA        randomized range finder with one power iteration on the streamed profiles
             (`randomized_pca_features`): four passes over the half, features = the
             projection onto the top components, unwhitened, as sklearn's PCA gives them
             up to sign and the approximation.
  clustering unchanged: Lloyd restarts on the N x 100 features, Hungarian consensus.
  scoring    `bin/score_big.py`: block means and pair moments streamed from the tiles;
             every partition and both measures in one pass per matrix.

Also written per half: `unit_strength`, the row sums of |r| (the hubness measure), so the
scorer's triviality metric needs no pass of its own.
"""

import os
import time

import h5py
import numpy as np
import torch

from parcelmate.constants import CONNECTIVITY_NAME, EXTENSION, HALF_NAMES
from parcelmate.util import derive_seed, git_commit, stderr

STORAGE_TILED = 'tiled_fp16'
ROW_CHUNK = 256          # HDF5 chunk: 256 rows x N columns (150 MB at N = 295k)
DEFAULT_BLOCK = 8192     # row / column block for the GPU tiles


def is_tiled(path):
    """Whether a connectivity file was written by this module."""
    with h5py.File(path, 'r') as f:
        return f.attrs.get('storage', '') == STORAGE_TILED


# ---------------------------------------------------------------------------- reading

class TiledMatrix:
    """Row-block access to a tiled half as float32 |r|, never loading it whole.

    `M[s:e]` returns rows s..e-1 with NaN -> 0 and the absolute value taken, which is what
    `util.connectivity_matrix(data, normalize=None)` returns for the dense path. `shape`,
    `strength` (row sums of |r|) and the unit statistics come from the file.
    """

    def __init__(self, path):
        self.path = path
        with h5py.File(path, 'r') as f:
            assert f.attrs.get('storage', '') == STORAGE_TILED, '%s is not a tiled half' % path
            self.shape = tuple(f['connectivity'].shape)
            self.strength = np.asarray(f['unit_strength'], dtype=np.float64)
            self.coordinates = np.asarray(f['coordinates'])
            self.attrs = dict(f.attrs)
        self._f = None

    def _file(self):
        if self._f is None:
            self._f = h5py.File(self.path, 'r')
        return self._f

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, slice):
            rows = self._file()['connectivity'][key]
        else:
            rows = self._file()['connectivity'][key]
        out = np.asarray(rows, dtype=np.float32)
        np.nan_to_num(out, copy=False)
        return np.abs(out, out=out)

    def signed_rows(self, s, e):
        out = np.asarray(self._file()['connectivity'][s:e], dtype=np.float32)
        return np.nan_to_num(out, copy=False)

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None


def open_connectivity(path):
    """A TiledMatrix for a tiled half, or the dense |r| for an ordinary file."""
    if is_tiled(path):
        return TiledMatrix(path)
    from parcelmate.util import connectivity_matrix, load_h5_data
    return connectivity_matrix(load_h5_data(path, verbose=False), None, inplace=True)


# ---------------------------------------------------------------------------- timecourses

def _mlp_hooks(model, captured):
    from parcelmate.model import mlp_projections

    def make_hook(layer):
        def hook(module, inputs):
            captured[layer] = inputs[0]
        return hook
    projections = mlp_projections(model)
    return [p.register_forward_pre_hook(make_hook(l)) for l, p in projections], len(projections)


def zscored_timecourses(model, input_ids, attention_mask, batch_size=8, verbose=True, indent=0):
    """Two passes over the inputs: exact per-unit mean and std, then z-scored fp16 rows.

    Returns dict(z=(N, T) float16, unit_means, unit_stds, n_obs, coordinates). Units are
    all MLP neurons of every block in layer order, coordinates (layer, neuron index).
    """
    device = next(model.parameters()).device
    T = int(attention_mask.sum())
    captured = {}
    hooks, n_layers = _mlp_hooks(model, captured)
    widths = None
    # Pass 1: Welford over tokens, per unit, in float64, ONE LAYER AT A TIME: a tokens x N
    # float64 tensor of all layers at once was 9 GB per batch at 147k units (out of memory
    # on a 48 GB card next to the model); per layer it is tokens x width, under 1 GB.
    count = 0
    mean = m2 = None
    offsets = None
    t0 = time.time()
    n_batches = int(np.ceil(input_ids.size(0) / batch_size))
    for i in range(0, input_ids.size(0), batch_size):
        ids = input_ids[i:i + batch_size].to(device)
        mask = attention_mask[i:i + batch_size].to(device)
        with torch.no_grad():
            captured.clear()
            model(input_ids=ids, attention_mask=mask)
            m = mask.bool()
            if widths is None:
                widths = [int(captured[l].shape[-1]) for l in range(n_layers)]
                N = sum(widths)
                offsets = np.concatenate([[0], np.cumsum(widths)])
                mean = torch.zeros(N, dtype=torch.float64, device=device)
                m2 = torch.zeros(N, dtype=torch.float64, device=device)
            n_b = int(m.sum())
            tot = count + n_b
            for l in range(n_layers):
                x = captured[l][m].double()          # tokens x width
                sl = slice(int(offsets[l]), int(offsets[l + 1]))
                b_mean = x.mean(0)
                b_m2 = ((x - b_mean) ** 2).sum(0)
                delta = b_mean - mean[sl]
                mean[sl] += delta * (n_b / tot)
                m2[sl] += b_m2 + delta ** 2 * (count * n_b / tot)
                del x, b_mean, b_m2, delta
            count = tot
            captured.clear()
        if verbose:
            stderr('\r%sstats batch %d/%d' % (' ' * indent, i // batch_size + 1, n_batches))
    # Population variance, so the stored diagonal is exactly 1 as in the dense path
    # (`correlate` normalises by the row norm). A unit whose spread is below 1e-6 of its
    # mean, or below 1e-6 absolutely, is constant to float precision: its z-scores are set
    # to 0 and its correlations are 0, where the dense path would correlate float noise.
    # Real post-nonlinearity activations have spreads of 1e-2 to 1e1, far above this.
    std = torch.sqrt(m2 / max(count, 1))
    ok = std > 1e-6 * torch.clamp(mean.abs(), min=1.0)
    inv = torch.where(ok, 1.0 / torch.where(ok, std, torch.ones_like(std)), torch.zeros_like(std))
    std = torch.where(ok, std, torch.zeros_like(std))
    if verbose:
        stderr('  (%.0f s)\n' % (time.time() - t0))
    # Pass 2: z-scores as float16 into host memory, again layer by layer.
    Z = np.empty((N, T), dtype=np.float16)
    t = 0
    t0 = time.time()
    for i in range(0, input_ids.size(0), batch_size):
        ids = input_ids[i:i + batch_size].to(device)
        mask = attention_mask[i:i + batch_size].to(device)
        with torch.no_grad():
            captured.clear()
            model(input_ids=ids, attention_mask=mask)
            m = mask.bool()
            n_b = int(m.sum())
            for l in range(n_layers):
                sl = slice(int(offsets[l]), int(offsets[l + 1]))
                z = ((captured[l][m].double() - mean[sl]) * inv[sl]).T.to(torch.float16).cpu().numpy()
                Z[sl, t:t + n_b] = z
                del z
            captured.clear()
        t += n_b
        if verbose:
            stderr('\r%sz-score batch %d/%d' % (' ' * indent, i // batch_size + 1, n_batches))
    assert t == T, 'wrote %d of %d tokens' % (t, T)
    if verbose:
        stderr('  (%.0f s)\n' % (time.time() - t0))
    for h in hooks:
        h.remove()
    coordinates = np.zeros((N, 2), dtype=np.int32)
    h = 0
    for layer, w in enumerate(widths):
        coordinates[h:h + w, 0] = layer
        coordinates[h:h + w, 1] = np.arange(w)
        h += w
    return dict(z=Z, unit_means=mean.cpu().numpy(), unit_stds=std.cpu().numpy(),
                n_obs=np.asarray(T), coordinates=coordinates)


def null_offsets(n_units, n_tokens, seed):
    """Per-unit circular-shift offsets, as `data.circshift_timecourses` draws them where it
    can (distinct offsets), with replacement where the units outnumber the tokens."""
    rng = np.random.RandomState(seed % (2 ** 32))
    if n_units < n_tokens - 1:
        return rng.choice(np.arange(1, n_tokens), size=n_units, replace=False)
    return rng.randint(1, n_tokens, size=n_units)


def roll_rows(x, offsets, sub=1024):
    """Circularly shift each row of `x` (rows x T, on device) by its own offset.

    Vectorised gather in sub-blocks, so the int64 index never exceeds sub x T entries.
    x_out[i, t] = x[i, (t - offset_i) mod T], i.e. numpy.roll(x[i], offset_i).
    """
    rows, T = x.shape
    out = torch.empty_like(x)
    ar = torch.arange(T, device=x.device)
    off = torch.as_tensor(offsets, device=x.device, dtype=torch.long)
    for s in range(0, rows, sub):
        e = min(s + sub, rows)
        idx = (ar[None, :] - off[s:e, None]) % T
        out[s:e] = torch.gather(x[s:e], 1, idx)
    return out


# ---------------------------------------------------------------------------- correlation

def write_tiled_half(zs, offsets, out_real, out_null, stats, provenance, eps=1e-3,
                     block=DEFAULT_BLOCK, device=None, verbose=True, indent=0):
    """Fisher-mean correlation of the samples in `zs`, real and circularly shifted, as
    float16 tiles.

    zs        list of (N, T) float16 arrays (host), one per sample of this half
    offsets   list of per-unit shift vectors, one per sample (None -> no null file)
    out_real  path of the real half; out_null path of the null half (or None)
    stats     dict(unit_means, unit_stds, n_obs, coordinates) pooled for this half
    """
    device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    N, T = zs[0].shape
    n_s = len(zs)
    block = int(min(block, N))
    write_null = out_null is not None
    assert not write_null or len(offsets) == n_s
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    files = {}
    for tree, path in (('real', out_real), ('null', out_null)):
        if path is None:
            continue
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d)
        f = h5py.File(path + '.partial', 'w')
        f.create_dataset('connectivity', shape=(N, N), dtype=np.float16,
                         chunks=(min(ROW_CHUNK, N), N))
        files[tree] = f
    strength = {tree: np.zeros(N, dtype=np.float64) for tree in files}
    offs = [torch.as_tensor(o, device=device, dtype=torch.long) if write_null else None
            for o in (offsets if write_null else [None] * n_s)]
    scale = 1.0 - eps
    t0 = time.time()
    n_blocks = int(np.ceil(N / block))
    for bi, s in enumerate(range(0, N, block)):
        e = min(s + block, N)
        # Row block of every sample, real and (rolled) null, resident on the GPU.
        rows_real = [torch.as_tensor(z[s:e]).to(device).float() for z in zs]
        rows_null = [roll_rows(r, offs[k][s:e]) for k, r in enumerate(rows_real)] if write_null else None
        acc = {tree: torch.zeros(e - s, N, dtype=torch.float32, device=device) for tree in files}
        for cs in range(0, N, block):
            ce = min(cs + block, N)
            for k, z in enumerate(zs):
                col = torch.as_tensor(z[cs:ce]).to(device).float()
                r = (rows_real[k] @ col.T) / float(T)
                acc['real'][:, cs:ce] += torch.atanh(torch.clamp(r * scale, -scale, scale))
                if write_null:
                    col_n = roll_rows(col, offs[k][cs:ce])
                    r = (rows_null[k] @ col_n.T) / float(T)
                    acc['null'][:, cs:ce] += torch.atanh(torch.clamp(r * scale, -scale, scale))
                    del col_n
                del col, r
        for tree, f in files.items():
            half = torch.tanh(acc[tree] / float(n_s))
            strength[tree][s:e] = half.abs().sum(1).double().cpu().numpy()
            f['connectivity'][s:e] = half.to(torch.float16).cpu().numpy()
        del acc, rows_real, rows_null
        if verbose:
            stderr('\r%srow block %d/%d (%.0f s)' % (' ' * indent, bi + 1, n_blocks, time.time() - t0))
    if verbose:
        stderr('\n')
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = tf32
    for tree, f in files.items():
        f.create_dataset('unit_strength', data=strength[tree])
        for k in ('unit_means', 'unit_stds', 'n_obs', 'coordinates'):
            f.create_dataset(k, data=np.asarray(stats[k]))
        for k, v in provenance.items():
            f.attrs[k] = '' if v is None else v
        f.attrs['storage'] = STORAGE_TILED
        f.attrs['n_units'] = int(N)
        f.attrs['dtype'] = 'float16'
        f.attrs['tf32'] = True
        f.attrs['null_offsets_distinct'] = bool(N < T - 1)
        f.close()
        path = out_real if tree == 'real' else out_null
        os.replace(path + '.partial', path)


def write_tiled_domain(model, input_ids, attention_mask, n_samples, domain, connectivity_dir,
                       null_connectivity_dir, seed, null_model='circshift', batch_size=8, eps=1e-3,
                       block=DEFAULT_BLOCK, provenance=None, verbose=True, indent=0):
    """The `outputs: [halves]` layout for one domain, out of core.

    Samples 1..n/2 form half A and n/2+1..n half B, as `run_split_halves` defines them.
    Each half needs its samples' z-scored timecourses resident at once (2 x N x T fp16:
    116 GB for Qwen3.5-4B), and nothing else of that size.
    """
    from parcelmate.model import pool_unit_stats
    assert n_samples >= 2 and n_samples % 2 == 0, 'tiled halves need an even n_samples'
    device = next(model.parameters()).device
    n = int(np.ceil(len(input_ids) / n_samples))
    provenance = dict(provenance or {})
    for h, name in enumerate(HALF_NAMES):
        sample_ids = list(range(h * (n_samples // 2), (h + 1) * (n_samples // 2)))
        zs, offsets, means, stds, counts, coords = [], [], [], [], [], None
        for k in sample_ids:
            if verbose:
                stderr('%sSample %d/%d (%s)\n' % (' ' * indent, k + 1, n_samples, name))
            tc = zscored_timecourses(model, input_ids[k * n:(k + 1) * n],
                                     attention_mask[k * n:(k + 1) * n], batch_size=batch_size,
                                     verbose=verbose, indent=indent + 2)
            zs.append(tc['z'])
            means.append(tc['unit_means'])
            stds.append(tc['unit_stds'])
            counts.append(int(tc['n_obs']))
            coords = tc['coordinates']
            if null_model:
                # Same seed keys as the dense path (`derive_seed(seed, 'null', domain, sample)`).
                N, T = tc['z'].shape
                offsets.append(null_offsets(N, T, derive_seed(seed, 'null', domain, k + 1)))
        pooled_means, pooled_stds = pool_unit_stats(means, stds, counts)
        stats = dict(unit_means=pooled_means, unit_stds=pooled_stds, n_obs=np.asarray(sum(counts)),
                     coordinates=coords)
        prov = dict(provenance, domain=domain, key=name,
                    sources=', '.join('sample%d' % (k + 1) for k in sample_ids),
                    n_obs=int(sum(counts)), git_commit=git_commit(),
                    created=time.strftime('%Y-%m-%dT%H:%M:%S'))
        out_real = os.path.join(connectivity_dir, '%s_%s_%s%s' % (CONNECTIVITY_NAME, domain, name, EXTENSION))
        out_null = None if not null_model else os.path.join(
            null_connectivity_dir, '%s_%s_%s%s' % (CONNECTIVITY_NAME, domain, name, EXTENSION))
        if verbose:
            stderr('%sCorrelating %s (%d units, %d tokens per sample)\n' % (' ' * indent, name, zs[0].shape[0], zs[0].shape[1]))
        # The model is not needed while correlating and its weights are the largest single
        # allocation on the card (16 GB for a 4B model in float32), so move it out of the
        # way and bring it back for the next half's forward passes.
        model.to('cpu')
        torch.cuda.empty_cache()
        write_tiled_half(zs, offsets if null_model else None, out_real, out_null, stats, prov,
                         eps=eps, block=block, verbose=verbose, indent=indent + 2)
        del zs
        model.to(device)


# ---------------------------------------------------------------------------- profiles

def profile_block(R_abs, s, e, eps=1e-3, q=0.9):
    """The confirmed pipeline's profile transform for rows s..e-1 of |r| (float32 in place):
    Fisher (arctanh of (1 - eps) |r|), zero diagonal, z-score, keep the top 10 % of each
    row, re-standardize. Identical to `model.sample_parcellations` with fisher_transform,
    standardize_profiles and sparsify_profiles, row by row.
    """
    from parcelmate.model import sparsify_rows_inplace, standardize_rows_inplace
    X = R_abs
    np.clip(X, 0.0, 1.0, out=X)
    X *= (1.0 - eps)
    np.arctanh(X, out=X)
    X[np.arange(e - s), np.arange(s, e)] = 0.0
    standardize_rows_inplace(X)
    sparsify_rows_inplace(X, q=q)
    standardize_rows_inplace(X)
    return X


def stream_profiles(tiled, block=2048, eps=1e-3):
    """Yield (s, e, P) with P the float32 profiles of rows s..e-1."""
    N = tiled.shape[0]
    for s in range(0, N, block):
        e = min(s + block, N)
        yield s, e, profile_block(tiled[s:e], s, e, eps=eps)


def randomized_pca_features(tiled, n_components=100, oversample=100, n_iter=1, seed=0,
                            block=2048, eps=1e-3, verbose=True, indent=0):
    """Unwhitened PCA scores of the streamed profiles by a randomized range finder.

    Passes over the half: one for the column means and Y = X Omega, two per power
    iteration (Z = X^T Y, Y = X Z), one for B = Q^T X. Features are X V_k = Q (U S)_k, so
    no further pass. With `n_iter=1` that is four passes. Mean-centring is applied through
    the rank-one identity (X - 1 mu^T) M = X M - 1 (mu^T M).
    """
    N = tiled.shape[0]
    rng = np.random.RandomState(seed % (2 ** 32))
    k = min(n_components + oversample, N)   # at k = N the range finder is exact
    assert n_components <= N, 'more components (%d) than units (%d)' % (n_components, N)
    omega = rng.standard_normal((N, k)).astype(np.float32)
    mu = np.zeros(N, dtype=np.float64)
    Y = np.zeros((N, k), dtype=np.float64)
    t0 = time.time()
    # Pass 1: column means and X Omega.
    for s, e, P in stream_profiles(tiled, block=block, eps=eps):
        mu += P.sum(0, dtype=np.float64)
        Y[s:e] = P.astype(np.float64) @ omega
        if verbose:
            stderr('\r%sPCA pass 1: rows %d/%d (%.0f s)' % (' ' * indent, e, N, time.time() - t0))
    mu /= N
    Y -= mu @ omega.astype(np.float64)       # centring: subtract 1 (mu^T Omega)
    Q, _ = np.linalg.qr(Y)
    for it in range(n_iter):
        # Z = (X - 1 mu^T)^T Q = X^T Q - mu (1^T Q)
        Z = np.zeros((N, k), dtype=np.float64)
        for s, e, P in stream_profiles(tiled, block=block, eps=eps):
            Z += P.T.astype(np.float64) @ Q[s:e]
            if verbose:
                stderr('\r%sPCA pass %d: rows %d/%d (%.0f s)' % (' ' * indent, 2 + 2 * it, e, N, time.time() - t0))
        Z -= np.outer(mu, Q.sum(0))
        Z, _ = np.linalg.qr(Z)
        Y = np.zeros((N, k), dtype=np.float64)
        for s, e, P in stream_profiles(tiled, block=block, eps=eps):
            Y[s:e] = P.astype(np.float64) @ Z
            if verbose:
                stderr('\r%sPCA pass %d: rows %d/%d (%.0f s)' % (' ' * indent, 3 + 2 * it, e, N, time.time() - t0))
        Y -= mu @ Z
        Q, _ = np.linalg.qr(Y)
    # B = Q^T (X - 1 mu^T) = Q^T X - (Q^T 1) mu^T, k x N
    B = np.zeros((k, N), dtype=np.float64)
    for s, e, P in stream_profiles(tiled, block=block, eps=eps):
        B += Q[s:e].T @ P.astype(np.float64)
        if verbose:
            stderr('\r%sPCA pass %d: rows %d/%d (%.0f s)' % (' ' * indent, 2 + 2 * n_iter, e, N, time.time() - t0))
    B -= np.outer(Q.sum(0), mu)
    U, S, _ = np.linalg.svd(B, full_matrices=False)
    features = (Q @ (U[:, :n_components] * S[:n_components])).astype(np.float32)
    if verbose:
        stderr('\n%sPCA done: %d components, top singular values %s (%.0f s)\n' % (
            ' ' * indent, n_components, np.round(S[:3], 1), time.time() - t0))
    return features, S[:n_components]
