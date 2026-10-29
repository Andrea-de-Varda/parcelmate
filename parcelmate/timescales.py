"""Intrinsic timescales of MLP units (LOG.md Iteration 47, info/TIMESCALES_PLAN.md).

For every unit, the autocorrelation of its activation over tokens, within windows of contiguous
text, on the residuals of the token-and-position fit (Iterations 38-41):

    ACF(k) = sum over within-window pairs (t, t + k) of (x_t - mu)(x_{t+k} - mu) / (n_k sigma^2)

mu and sigma^2 are the unit's mean and variance over all its tokens in the sample (one dataset,
one half), not per window, so a difference in level between windows stays in the ACF as a slow
component. Tokens whose type the fit dropped (fewer than `min_count` reference occurrences) are
masked: every pair that contains one is left out, and n_k counts the pairs kept. The ACF is
accumulated with FFTs over each window, zero-padded so that no pair wraps around, as four
within-window sums per lag: sum x_t x_{t+k}, sum x_t m_{t+k}, sum m_t x_{t+k} (m the mask, x
zero where masked) and n_k = sum m_t m_{t+k}; the centred covariance is

    C(k) = sum x_t x_{t+k} - mu (sum x_t m_{t+k} + sum m_t x_{t+k}) + n_k mu^2.

tau_int is the sum of ACF(1..L): in tokens, about how long the activation stays correlated with
itself (an exponential decay with time constant tau gives about tau).

Windows (Iteration 47): wikitext, the first `seq_len` tokens of randomly chosen articles at least
that long (articles are regrouped from the dataset's lines at the level-1 ` = Title = ` headings);
bookcorpus, `seq_len` contiguous tokens of consecutive sentences joined by a space, from randomly
chosen, non-overlapping blocks of rows (the rows are in book order). The connectome samples cannot
be reused: `get_dataset` shuffles rows before packing them.
"""

import re

import numpy as np
import torch

from parcelmate.util import derive_seed, stderr

WIKITEXT_TITLE = re.compile(r'^ = [^=].* = \n$')


# ---------------------------------------------------------------------------- windows

def wikitext_articles(lines):
    """Group wikitext lines into articles at the level-1 headings (` = Title = `); the text
    before the first heading is dropped. Section headings (` = = X = = `) stay inside."""
    articles, cur = [], None
    for t in lines:
        if WIKITEXT_TITLE.match(t):
            if cur:
                articles.append(''.join(cur))
            cur = [t]
        elif cur is not None:
            cur.append(t)
    if cur:
        articles.append(''.join(cur))
    return articles


def article_windows(articles, tokenizer, n, seq_len, rng):
    """The first `seq_len` tokens of `n` randomly chosen articles at least that long, in the
    order drawn. Returns (windows (n, seq_len) int64, article indices)."""
    out, idx = [], []
    for i in rng.permutation(len(articles)):
        ids = tokenizer(articles[i])['input_ids']
        if len(ids) >= seq_len:
            out.append(ids[:seq_len])
            idx.append(int(i))
            if len(out) == n:
                break
    assert len(out) == n, 'only %d articles reach %d tokens' % (len(out), seq_len)
    return np.asarray(out, dtype=np.int64), np.asarray(idx)


def block_windows(get_rows, n_rows, tokenizer, n, seq_len, rng, block=200):
    """`seq_len` contiguous tokens from each of `n` randomly chosen blocks of `block` consecutive
    rows (rows joined by a space), in the order drawn. Blocks lie on a fixed grid, so no two
    windows share a row. get_rows(start, stop) -> list of str. Returns (windows, block starts)."""
    out, starts = [], []
    for b in rng.permutation(n_rows // block):
        s = int(b) * block
        ids = tokenizer(' '.join(get_rows(s, s + block)))['input_ids']
        if len(ids) >= seq_len:
            out.append(ids[:seq_len])
            starts.append(s)
            if len(out) == n:
                break
    assert len(out) == n, 'only %d blocks reach %d tokens' % (len(out), seq_len)
    return np.asarray(out, dtype=np.int64), np.asarray(starts)


def load_windows(dataset, tokenizer, n, seq_len, seed):
    """`n` windows of `dataset` (wikitext or bookcorpus) and where each came from."""
    import datasets
    rng = np.random.RandomState(derive_seed(seed, 'timescales', dataset) % (2 ** 32))
    if dataset == 'wikitext':
        ds = datasets.load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='train')
        return article_windows(wikitext_articles(ds['text']), tokenizer, n, seq_len, rng)
    if dataset == 'bookcorpus':
        ds = datasets.load_dataset('rojagtap/bookcorpus', split='train')
        return block_windows(lambda s, e: ds[s:e]['text'], len(ds), tokenizer, n, seq_len, rng)
    raise ValueError('timescales windows are defined for wikitext and bookcorpus, got %r' % (dataset,))


# ---------------------------------------------------------------------------- the ACF

def lagged_sums(R, m, max_lag):
    """Within-window lagged sums for one batch, summed over windows.

    R  (B, T, U) values, zero where masked;  m  (B, T) float mask.
    Returns xx, xm, mx  (max_lag + 1, U) and nk  (max_lag + 1,):
    xx[k] = sum_t R_t R_{t+k},  xm[k] = sum_t R_t m_{t+k},  mx[k] = sum_t m_t R_{t+k},
    nk[k] = sum_t m_t m_{t+k}, every sum over pairs inside a window.
    """
    T = R.shape[1]
    n = 1 << int(np.ceil(np.log2(T + max_lag + 1)))   # no circular wrap for lags <= max_lag
    FR = torch.fft.rfft(R, n=n, dim=1)
    Fm = torch.fft.rfft(m, n=n, dim=1)[..., None]
    # irfft(conj(Fa) Fb)[k] = sum_t a_t b_{t+k}
    xx = torch.fft.irfft(FR.conj() * FR, n=n, dim=1)[:, :max_lag + 1].sum(0)
    xm = torch.fft.irfft(FR.conj() * Fm, n=n, dim=1)[:, :max_lag + 1].sum(0)
    mx = torch.fft.irfft(Fm.conj() * FR, n=n, dim=1)[:, :max_lag + 1].sum(0)
    nk = torch.fft.irfft(Fm.conj() * Fm, n=n, dim=1)[:, :max_lag + 1, 0].sum(0)
    return xx, xm, mx, nk


class ACFAccumulator:
    """Sums over batches of windows, per unit, in float64; `finalize` gives the ACF."""

    def __init__(self, n_units, max_lag, device):
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)
        self.max_lag = max_lag
        self.xx, self.xm, self.mx = z(max_lag + 1, n_units), z(max_lag + 1, n_units), z(max_lag + 1, n_units)
        self.nk = z(max_lag + 1)
        self.s1, self.s2, self.pos = z(n_units), z(n_units), z(n_units)
        self.n = 0.0

    def add(self, R, m, cols, positive=None):
        """R (B, T, len(cols)) residuals zeroed where masked, m (B, T); cols a slice of units.
        `positive` (len(cols),): count of kept tokens whose raw activation is above zero."""
        xx, xm, mx, nk = lagged_sums(R, m, self.max_lag)
        self.xx[:, cols] += xx.double()
        self.xm[:, cols] += xm.double()
        self.mx[:, cols] += mx.double()
        self.s1[cols] += R.sum((0, 1)).double()
        self.s2[cols] += (R.double() ** 2).sum((0, 1))
        if positive is not None:
            self.pos[cols] += positive.double()
        if cols.start == 0:                       # the mask is shared by every unit: count it once
            self.nk += torch.round(nk.double())
            self.n += float(m.sum())

    def finalize(self, min_var=1e-12):
        """ACF (units, max_lag + 1), mean, variance, share of tokens above zero, token count.
        Units with variance below `min_var` (constant after the fit) get NaN."""
        mu = self.s1 / self.n
        var = self.s2 / self.n - mu ** 2
        C = self.xx - mu[None] * (self.xm + self.mx) + self.nk[:, None] * mu[None] ** 2
        acf = C / (self.nk[:, None] * var[None])
        acf[:, var < min_var] = float('nan')
        return dict(acf=acf.T.float().cpu().numpy(), mean=mu.cpu().numpy(), var=var.cpu().numpy(),
                    positive_share=(self.pos / self.n).cpu().numpy(), n_tokens=self.n,
                    n_pairs=self.nk.cpu().numpy())


@torch.no_grad()
def unit_acf(model, input_ids, fit, max_lag, batch_size=8, device=None, verbose=True, indent=0):
    """ACF of every MLP unit's token-and-position residual over `input_ids` (windows, seq_len),
    all windows full length. `fit` is a `residual.TokenPositionFit` on reference windows."""
    from parcelmate.residual import _mlp_capture
    device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    input_ids = torch.as_tensor(input_ids, dtype=torch.long)
    seq_len = input_ids.shape[1]
    assert fit.b.shape[0] >= seq_len, 'the fit has fewer positions than the windows'
    look = fit.look.to(device)
    a = fit.a.to(device)
    b = fit.b[:seq_len].to(device)
    hooks, captured = _mlp_capture(model)
    acc = None
    nb = int(np.ceil(input_ids.shape[0] / batch_size))
    try:
        for i in range(0, input_ids.shape[0], batch_size):
            if verbose:
                stderr('\r%sbatch %d/%d' % (' ' * indent, i // batch_size + 1, nb))
            ids = input_ids[i:i + batch_size].to(device)
            captured.clear()
            model(input_ids=ids, attention_mask=torch.ones_like(ids))
            rows = look[ids]
            keep = rows >= 0
            m = keep.float()
            rows = rows.clamp(min=0)
            widths = [int(captured[l].shape[-1]) for l in range(len(captured))]
            offsets = np.concatenate([[0], np.cumsum(widths)])
            if acc is None:
                acc = ACFAccumulator(int(offsets[-1]), max_lag, device)
            for l in range(len(widths)):
                cols = slice(int(offsets[l]), int(offsets[l + 1]))
                X = captured[l].float()
                positive = ((X > 0) & keep[..., None]).sum((0, 1))
                # Gather from the layer's columns only (a[rows] would build all units).
                R = (X - a[:, cols][rows] - b[:, cols][None]) * m[..., None]
                acc.add(R, m, cols, positive)
                del X, R
            captured.clear()
    finally:
        for h in hooks:
            h.remove()
    if verbose:
        stderr('\n')
    out = acc.finalize()
    del a, b, acc
    model.to('cpu')
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------- summaries

def tau_int(acf, max_lag=None):
    """Sum of the ACF over lags 1..max_lag (all stored lags by default), per unit."""
    max_lag = acf.shape[1] - 1 if max_lag is None else max_lag
    return acf[:, 1:max_lag + 1].sum(1)


def exp_fit(acf, max_lag=None):
    """Fit ACF(k) = A (exp(-k / tau) + B) over lags 1..max_lag for every unit (the primate
    form, Murray et al. 2014). Returns tau, A, B and a success flag; a fit fails when it does
    not converge or tau ends on a bound (0.1 or 10,000 tokens)."""
    from scipy.optimize import curve_fit
    max_lag = acf.shape[1] - 1 if max_lag is None else max_lag
    k = np.arange(1, max_lag + 1, dtype=np.float64)
    f = lambda k, A, tau, B: A * (np.exp(-k / tau) + B)
    out = np.full((acf.shape[0], 3), np.nan)
    ok = np.zeros(acf.shape[0], dtype=bool)
    for i in range(acf.shape[0]):
        y = acf[i, 1:max_lag + 1].astype(np.float64)
        if not np.all(np.isfinite(y)):
            continue
        try:
            p, _ = curve_fit(f, k, y, p0=(max(y[0], 1e-3), 5.0, 0.0),
                             bounds=([0.0, 0.1, -1.0], [2.0, 1e4, 1.0]), maxfev=2000)
        except (RuntimeError, ValueError):
            continue
        out[i] = p
        ok[i] = 0.1 * 1.001 < p[1] < 1e4 * 0.999
    return dict(tau=out[:, 1], A=out[:, 0], B=out[:, 2], ok=ok)


# ---------------------------------------------------------------------------- networks

def layer_means(tau, layers, n_layers):
    """Mean of tau per layer over finite units."""
    ok = np.isfinite(tau)
    s = np.bincount(layers[ok], weights=tau[ok], minlength=n_layers)
    c = np.bincount(layers[ok], minlength=n_layers)
    with np.errstate(invalid='ignore', divide='ignore'):
        return s / c


def layer_adjusted_network_tau(tau, labels, layers, k, n_layers):
    """Per network: its mean tau minus the mean its layer composition predicts,
    sum_l (share of the network's units in layer l) x (mean tau of layer l). Equal to the mean of
    tau - layer mean over the network's units. Returns (adjusted (k,), mean (k,), units (k,));
    networks without finite units get NaN."""
    ok = np.isfinite(tau)
    dev = tau - layer_means(tau, layers, n_layers)[layers]
    c = np.bincount(labels[ok], minlength=k).astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        adj = np.bincount(labels[ok], weights=dev[ok], minlength=k) / c
        mean = np.bincount(labels[ok], weights=tau[ok], minlength=k) / c
    return adj, mean, c


def network_spread(tau, labels, layers, k, n_layers):
    """The test statistic: variance across networks of the layer-adjusted tau."""
    adj, _, c = layer_adjusted_network_tau(tau, labels, layers, k, n_layers)
    return float(np.var(adj[c > 0]))


def permute_within_layer(labels, layers, rng):
    """Labels permuted among the units of each layer: every network keeps its size and its
    count of units in every layer."""
    out = labels.copy()
    for l in np.unique(layers):
        ix = np.flatnonzero(layers == l)
        out[ix] = labels[ix[rng.permutation(len(ix))]]
    return out


def network_test(tau, labels, layers, k, n_layers, n_perm, rng):
    """Spread of layer-adjusted network tau against within-layer permutations of the labels:
    observed, permutation mean and sd, z, and p = (1 + #{perm >= obs}) / (1 + n_perm)."""
    obs = network_spread(tau, labels, layers, k, n_layers)
    perm = np.array([network_spread(tau, permute_within_layer(labels, layers, rng), layers, k, n_layers)
                     for _ in range(n_perm)])
    sd = perm.std(ddof=1)
    return dict(spread=obs, perm_mean=float(perm.mean()), perm_sd=float(sd),
                z=float((obs - perm.mean()) / sd) if sd > 0 else float('nan'),
                p=float((1 + (perm >= obs - 1e-15).sum()) / (1 + n_perm)), n_perm=int(n_perm))


def residualize_within_layer(tau, covariates, layers):
    """tau minus its least-squares prediction from `covariates` (units x c) and an intercept,
    fitted separately in each layer, plus the layer mean back (so layer means are kept)."""
    out = np.full_like(tau, np.nan, dtype=np.float64)
    for l in np.unique(layers):
        ix = np.flatnonzero((layers == l) & np.isfinite(tau) & np.all(np.isfinite(covariates), 1))
        if len(ix) == 0:
            continue
        Z = np.column_stack([np.ones(len(ix)), covariates[ix]])
        beta, *_ = np.linalg.lstsq(Z, tau[ix], rcond=None)
        out[ix] = tau[ix] - Z @ beta + tau[ix].mean()
    return out


# ---------------------------------------------------------------------------- circuits

def circuit_tau_test(tau, circuit, width, n_layers, n_draws, rng):
    """Mean and variance of tau over the circuit's units against layer-matched random sets
    (`circuits.layer_matched_draws`): observed, null mean and sd, z, and the share of draws at
    or above the observed value (one-sided upper p, (1 + #) / (1 + n_draws)). Units with
    non-finite tau are left out of every set."""
    from parcelmate.circuits import layer_matched_draws
    draws = layer_matched_draws(circuit, width, n_layers, n_draws, rng)
    T = tau[draws]
    obs_t = tau[np.asarray(circuit)]
    out = dict(n_units=int(np.isfinite(obs_t).sum()))
    for name, fn in (('mean', lambda x, ax: np.nanmean(x, axis=ax)),
                     ('var', lambda x, ax: np.nanvar(x, axis=ax, ddof=1))):
        obs = float(fn(obs_t, None))
        null = fn(T, 1)
        sd = null.std(ddof=1)
        out.update({name: obs, name + '_null_mean': float(null.mean()), name + '_null_sd': float(sd),
                    name + '_z': float((obs - null.mean()) / sd) if sd > 0 else float('nan'),
                    name + '_p_upper': float((1 + (null >= obs - 1e-15).sum()) / (1 + n_draws)),
                    name + '_p_lower': float((1 + (null <= obs + 1e-15).sum()) / (1 + n_draws))})
    return out
