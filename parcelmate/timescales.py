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

With one mean per sample, a level that differs between windows makes pairs inside a window
correlate even when the unit has no memory. The within-window ACF (`WindowACFAccumulator`,
third revision, 2026-10-29) centres and scales each window on its own and takes the Fisher mean
of the per-window correlations over windows.

tau_int is the sum of ACF(1..L): in tokens, about how long the activation stays correlated with
itself (an exponential decay with time constant tau gives about tau).

Windows (Iteration 47): wikitext, the first `seq_len` tokens of randomly chosen articles at least
that long (articles are regrouped from the dataset's lines at the level-1 ` = Title = ` headings);
bookcorpus, `seq_len` contiguous tokens of consecutive sentences joined by a space, from randomly
chosen, non-overlapping blocks of rows (the rows are in book order). The connectome samples cannot
be reused: `get_dataset` shuffles rows before packing them.
"""

import numpy as np
import torch

from parcelmate.data import article_windows, block_windows, wikitext_articles  # noqa: F401 (moved to data, Iteration 48)
from parcelmate.util import derive_seed, stderr


# ---------------------------------------------------------------------------- windows

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


def window_lagged_sums(R, m, max_lag):
    """`lagged_sums` kept per window: xx (B, max_lag + 1, U) and nk (B, max_lag + 1), with
    xx[w, k] = sum_t R_t R_{t+k} and nk[w, k] = sum_t m_t m_{t+k} over the pairs of window w."""
    T = R.shape[1]
    n = 1 << int(np.ceil(np.log2(T + max_lag + 1)))
    FR = torch.fft.rfft(R, n=n, dim=1)
    Fm = torch.fft.rfft(m, n=n, dim=1)
    xx = torch.fft.irfft(FR.conj() * FR, n=n, dim=1)[:, :max_lag + 1]
    nk = torch.fft.irfft(Fm.conj() * Fm, n=n, dim=1)[:, :max_lag + 1]
    return xx, nk


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


class WindowACFAccumulator:
    """The within-window ACF (Iteration 47, third revision). Per window w and unit, the residual
    is centred on the window's own mean and scaled by the window's own variance,

        r_w(k) = sum over the window's pairs (t, t + k) of (x_t - mu_w)(x_{t+k} - mu_w) / (n_wk sigma_w^2),

    and the ACF is the Fisher mean over windows, tanh(mean_w atanh r_w(k)). A level that differs
    between windows is no longer correlation, as it is in `ACFAccumulator` (one mean per sample).

    A window in which the unit is constant has no r_w and is left out of that unit's mean; such
    windows are counted (`n_undefined`). Constant means up to rounding: sd below 1e-5 of the
    window mean's magnitude (`CONST`), since a constant window's variance is not exactly 0 in
    floating point. |r_w| is clipped to 1 - 1e-7 before
    atanh, and the clipped values are counted (`n_clipped`). Also per unit: the total variance and
    its token-weighted split into the mean within-window variance and the variance of the window
    means (`var_within` + `var_between` = `var`)."""

    CLIP = 1 - 1e-7
    CONST = 1e-5

    def __init__(self, n_units, max_lag, device):
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)
        self.max_lag = max_lag
        self.zsum, self.zcnt = z(max_lag, n_units), z(max_lag, n_units)
        self.nk = z(max_lag + 1)
        self.s1, self.s2, self.sw, self.pos = z(n_units), z(n_units), z(n_units), z(n_units)
        self.undefined, self.clipped = z(n_units), z(n_units)
        self.n = 0.0

    def add(self, R, m, cols, positive=None):
        """As `ACFAccumulator.add`: R (B, T, len(cols)) residuals zeroed where masked, m (B, T)."""
        nw = m.sum(1)                                              # (B,) kept tokens per window
        mu_w = R.sum(1) / nw.clamp(min=1)[:, None]                 # (B, U)
        Rc = (R - mu_w[:, None]) * m[..., None]
        var_w = (Rc.double() ** 2).sum(1) / nw.double().clamp(min=1)[:, None]
        xx, nk = window_lagged_sums(Rc, m, self.max_lag)
        xx, nk = xx[:, 1:].double(), torch.round(nk.double())
        # constant in the window, up to rounding: sd below CONST x |window mean| (float32 rounding
        # of the mean leaves about 1e-7 of it); not a cut on small but real fluctuations
        ok_w = (var_w > (self.CONST * mu_w.double()) ** 2 + 1e-30) & (nw > 0)[:, None]   # (B, U)
        valid = ok_w[:, None, :] & (nk[:, 1:] > 0)[..., None]      # (B, K, U)
        r = xx / (nk[:, 1:, None].clamp(min=1) * torch.where(ok_w, var_w, torch.ones_like(var_w))[:, None])
        self.clipped[cols] += ((r.abs() > self.CLIP) & valid).sum((0, 1)).double()
        z = torch.atanh(r.clamp(-self.CLIP, self.CLIP))
        self.zsum[:, cols] += torch.where(valid, z, torch.zeros_like(z)).sum(0)
        self.zcnt[:, cols] += valid.sum(0).double()
        self.undefined[cols] += (~ok_w & (nw > 0)[:, None]).sum(0).double()
        self.s1[cols] += R.sum((0, 1)).double()
        self.s2[cols] += (R.double() ** 2).sum((0, 1))
        self.sw[cols] += (var_w * nw.double()[:, None]).sum(0)
        if positive is not None:
            self.pos[cols] += positive.double()
        if cols.start == 0:                       # the mask is shared by every unit: count it once
            self.nk += nk.sum(0)
            self.n += float(m.sum())

    def finalize(self):
        """ACF (units, max_lag + 1) with ACF(0) = 1, the variances, counts, as `ACFAccumulator`."""
        mu = self.s1 / self.n
        var = self.s2 / self.n - mu ** 2
        within = self.sw / self.n
        zbar = self.zsum / self.zcnt                                # NaN where no window had r
        acf = torch.cat([torch.ones_like(zbar[:1]), torch.tanh(zbar)], 0)
        acf[0, self.zcnt[0] == 0] = float('nan')
        return dict(acf=acf.T.float().cpu().numpy(), mean=mu.cpu().numpy(), var=var.cpu().numpy(),
                    var_within=within.cpu().numpy(), var_between=(var - within).cpu().numpy(),
                    positive_share=(self.pos / self.n).cpu().numpy(), n_tokens=self.n,
                    n_pairs=self.nk.cpu().numpy(), n_windows=self.zcnt[0].cpu().numpy(),
                    n_undefined=self.undefined.cpu().numpy(), n_clipped=self.clipped.cpu().numpy())


@torch.no_grad()
def unit_acf(model, input_ids, fit, max_lag, batch_size=8, device=None, verbose=True, indent=0,
             centring='global'):
    """ACF of every MLP unit's token-and-position residual over `input_ids` (windows, seq_len),
    all windows full length. `fit` is a `residual.TokenPositionFit` on reference windows.
    `centring`: 'global' (one mean and variance per sample, `ACFAccumulator`) or 'window' (per
    window, Fisher mean over windows, `WindowACFAccumulator`)."""
    assert centring in ('global', 'window'), centring
    Acc = ACFAccumulator if centring == 'global' else WindowACFAccumulator
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
                acc = Acc(int(offsets[-1]), max_lag, device)
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
