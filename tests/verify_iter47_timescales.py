"""Verification for Iteration 47 (intrinsic timescales of MLP units).

    PYTHONPATH=. python tests/verify_iter47_timescales.py

CPU only. Covers: the FFT lagged sums equal brute-force sums over within-window pairs, with
masked tokens, and no pair crosses a window; the ACF of AR(1) series recovers phi^k and tau_int
its analytic value, with and without masked tokens; tau_int and the exponential fit recover the
time constant of an exponential ACF; `unit_acf` on a tiny random GPT-2 equals a direct computation
from hooked activations minus the fitted a(type) + b(position); wikitext article grouping; block
windows do not overlap; the shuffled condition keeps each window's bag of tokens; within-layer
permutation keeps every network's layer counts; the network test is null when tau depends only on
layer and detects a planted slow network; covariate residualization removes a covariate effect;
the circuit test keeps layer counts and detects a slow circuit (mean) and a mixed one (variance).
Within-window ACF (third revision): per-window lagged sums equal brute force; the accumulator
equals a direct per-window Fisher mean, with masked tokens; a unit with per-window offsets and no
memory has a global ACF at the between-window share and a within-window ACF near 0; AR(1) gives
phi^k less the window-mean bias; within + between variance equals the total; a window with zero
variance is counted and left out, not the unit; `unit_acf(centring='window')` equals a direct
computation; `analyze --acf` reads the named file.
"""

import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch

from parcelmate.timescales import (ACFAccumulator, WindowACFAccumulator, article_windows, block_windows,
                                   circuit_tau_test, exp_fit, lagged_sums, layer_adjusted_network_tau, network_test,
                                   permute_within_layer, residualize_within_layer, tau_int, unit_acf,
                                   window_lagged_sums, wikitext_articles)

failures = []
n_checks = [0]


def check(name, cond):
    n_checks[0] += 1
    print('%s %s' % ('OK' if cond else 'FAIL', name))
    if not cond:
        failures.append(name)


rs = np.random.RandomState(0)

# ---------------------------------------------------------------- lagged sums
B, T, U, K = 3, 40, 5, 12
R = rs.normal(size=(B, T, U))
m = (rs.rand(B, T) > 0.15).astype(float)
R = R * m[..., None]
xx, xm, mx, nk = [x.numpy() for x in lagged_sums(torch.tensor(R), torch.tensor(m), K)]
bxx = np.zeros((K + 1, U)); bxm = np.zeros((K + 1, U)); bmx = np.zeros((K + 1, U)); bnk = np.zeros(K + 1)
for b in range(B):
    for k in range(K + 1):
        for t in range(T - k):
            bxx[k] += R[b, t] * R[b, t + k]
            bxm[k] += R[b, t] * m[b, t + k]
            bmx[k] += m[b, t] * R[b, t + k]
            bnk[k] += m[b, t] * m[b, t + k]
check('lagged sums: FFT equals brute force over within-window pairs (xx, xm, mx, nk)',
      np.allclose(xx, bxx, atol=1e-8) and np.allclose(xm, bxm, atol=1e-8) and np.allclose(mx, bmx, atol=1e-8)
      and np.allclose(nk, bnk, atol=1e-8))
# a pair across windows would correlate the end of one window with the start of the next
R2 = np.zeros((2, 10, 1)); R2[0, -1, 0] = 1; R2[1, 0, 0] = 1
xx2 = lagged_sums(torch.tensor(R2), torch.ones(2, 10, dtype=torch.float64), 5)[0].numpy()
check('lagged sums: no pair crosses a window boundary', np.allclose(xx2[1:], 0))


# ---------------------------------------------------------------- AR(1)
def ar1(phi, n_win, T, U, rng, mu=0.3):
    x = np.zeros((n_win, T, U))
    x[:, 0] = rng.normal(size=(n_win, U)) / np.sqrt(1 - phi ** 2)
    for t in range(1, T):
        x[:, t] = phi * x[:, t - 1] + rng.normal(size=(n_win, U))
    return x + mu


phi, K = 0.8, 30
X = ar1(phi, 200, 256, 4, rs)
for name, mask in (('no mask', np.ones(X.shape[:2])), ('5% masked', (rs.rand(*X.shape[:2]) > 0.05).astype(float))):
    acc = ACFAccumulator(4, K, 'cpu')
    for i in range(0, 200, 50):
        acc.add(torch.tensor(X[i:i + 50] * mask[i:i + 50, :, None]), torch.tensor(mask[i:i + 50]), slice(0, 4))
    res = acc.finalize()
    a = res['acf']
    check('AR(1), %s: ACF(0) = 1 and ACF(k) = phi^k (max error %.3f)' % (name, np.abs(a - phi ** np.arange(K + 1)).max()),
          np.allclose(a[:, 0], 1, atol=1e-6) and np.abs(a - phi ** np.arange(K + 1)).max() < 0.03)
    true = phi * (1 - phi ** K) / (1 - phi)
    check('AR(1), %s: tau_int %.2f against analytic %.2f' % (name, tau_int(a).mean(), true),
          abs(tau_int(a).mean() - true) < 0.15 * true)
check('accumulator: mean is the series mean (0.3)', np.allclose(res['mean'], 0.3, atol=0.1))

# ---------------------------------------------------------------- within-window ACF
def direct_window_acf(X, mask, K):
    """Per window: centre and scale on the window's kept tokens, r_w(k) over its pairs; Fisher mean.
    A window whose sd is below 1e-5 of its mean's magnitude is constant and left out."""
    nwin, T_, U_ = X.shape
    zs = np.zeros((K, U_)); zc = np.zeros((K, U_))
    for w in range(nwin):
        kp = mask[w] > 0
        mu_w = X[w][kp].mean(0); v_w = X[w][kp].var(0)
        v_w = np.where(v_w > (1e-5 * mu_w) ** 2 + 1e-30, v_w, 0)
        Z_ = (X[w] - mu_w) * mask[w][:, None]
        for kk in range(1, K + 1):
            num = (Z_[:T_ - kk] * Z_[kk:]).sum(0)
            cnt = (mask[w][:T_ - kk] * mask[w][kk:]).sum()
            ok = (v_w > 0) & (cnt > 0)
            r_ = np.where(ok, num / (max(cnt, 1) * np.where(v_w > 0, v_w, 1)), 0)
            zs[kk - 1] += np.where(ok, np.arctanh(np.clip(r_, -1 + 1e-7, 1 - 1e-7)), 0); zc[kk - 1] += ok
    return np.concatenate([np.ones((1, U_)), np.tanh(zs / zc)]).T


Bw, Tw_, Uw, Kw = 4, 50, 3, 8
Xw = rs.normal(size=(Bw, Tw_, Uw)) + rs.normal(size=(Bw, 1, Uw)) * 3
mw = (rs.rand(Bw, Tw_) > 0.15).astype(float)
xxw, nkw = [x.numpy() for x in window_lagged_sums(torch.tensor(Xw * mw[..., None]), torch.tensor(mw), Kw)]
bf = np.array([[(Xw[w, :Tw_ - k] * mw[w, :Tw_ - k, None] * Xw[w, k:] * mw[w, k:, None]).sum(0) for k in range(Kw + 1)]
               for w in range(Bw)])
check('window lagged sums: FFT equals brute force, per window',
      np.allclose(xxw, bf, atol=1e-8) and np.allclose(nkw[:, 3], [(mw[w, :-3] * mw[w, 3:]).sum() for w in range(Bw)]))
acc = WindowACFAccumulator(Uw, Kw, 'cpu')
acc.add(torch.tensor(Xw[:2] * mw[:2, :, None]), torch.tensor(mw[:2]), slice(0, Uw))
acc.add(torch.tensor(Xw[2:] * mw[2:, :, None]), torch.tensor(mw[2:]), slice(0, Uw))
resw = acc.finalize()
check('within-window accumulator equals a direct per-window Fisher mean, with masked tokens, over two batches '
      '(max error %.1e)' % np.abs(resw['acf'] - direct_window_acf(Xw, mw, Kw)).max(),
      np.allclose(resw['acf'], direct_window_acf(Xw, mw, Kw), atol=1e-6))

# offsets per window, no memory: global ACF = between share at every lag, within-window ACF ~ 0
nwin, Tl, Kl = 200, 512, 100
off = rs.normal(size=(nwin, 1, 2)) * np.array([1.0, 2.0])     # between variance 1 and 4, noise variance 1
Xo = off + rs.normal(size=(nwin, Tl, 2))
ones = np.ones((nwin, Tl))
g_acc, w_acc = ACFAccumulator(2, Kl, 'cpu'), WindowACFAccumulator(2, Kl, 'cpu')
for i in range(0, nwin, 50):
    g_acc.add(torch.tensor(Xo[i:i + 50]), torch.tensor(ones[i:i + 50]), slice(0, 2))
    w_acc.add(torch.tensor(Xo[i:i + 50]), torch.tensor(ones[i:i + 50]), slice(0, 2))
g, w_ = g_acc.finalize(), w_acc.finalize()
share = w_['var_between'] / w_['var']
check('offsets, no memory: global ACF(1..100) sits at the between-window share (%s against %s)'
      % (np.round(g['acf'][:, 1:].mean(1), 3), np.round(share, 3)),
      np.allclose(g['acf'][:, 1:].mean(1), share, atol=0.03) and share[1] > share[0] > 0.3)
check('offsets, no memory: within-window ACF(1..100) is about 0 (max |mean| %.4f)' % np.abs(w_['acf'][:, 1:].mean(1)).max(),
      np.abs(w_['acf'][:, 1:].mean(1)).max() < 0.01)
check('within + between variance = total; between is the offset variance (%s)' % np.round(w_['var_between'], 2),
      np.allclose(w_['var_within'] + w_['var_between'], w_['var']) and np.allclose(w_['var_between'], [1, 4], rtol=0.3))
check('within-window: no window undefined or clipped on real-valued noise',
      w_['n_undefined'].sum() == 0 and w_['n_clipped'].sum() == 0 and np.all(w_['n_windows'] == nwin))

# AR(1): phi^k less the bias of a window mean estimated from the same window
phi, Ka, Ta = 0.8, 30, 1024
Xa = ar1(phi, 100, Ta, 2, rs, mu=0) + rs.normal(size=(100, 1, 2)) * 2
wa = WindowACFAccumulator(2, Ka, 'cpu')
for i in range(0, 100, 25):
    wa.add(torch.tensor(Xa[i:i + 25]), torch.ones(25, Ta, dtype=torch.float64), slice(0, 2))
aa = wa.finalize()['acf']
bias = (1 + phi) / ((1 - phi) * Ta)          # the window mean absorbs about this much of the ACF
check('AR(1) with window offsets: within-window ACF(k) = phi^k - %.3f (max error %.3f)'
      % (bias, np.abs(aa - (phi ** np.arange(Ka + 1) - bias * (np.arange(Ka + 1) > 0))).max()),
      np.abs(aa - (phi ** np.arange(Ka + 1) - bias * (np.arange(Ka + 1) > 0))).max() < 0.03)

# a window in which a unit is constant: counted, left out of that unit only
Xc = rs.normal(size=(5, 40, 2)); Xc[2, :, 0] = 0.7
cacc = WindowACFAccumulator(2, 5, 'cpu')
cacc.add(torch.tensor(Xc), torch.ones(5, 40, dtype=torch.float64), slice(0, 2))
cres = cacc.finalize()
keep_w = [0, 1, 3, 4]
check('a constant window is counted (n_undefined %s) and left out; the unit keeps a finite ACF'
      % cres['n_undefined'].astype(int),
      list(cres['n_undefined']) == [1, 0] and list(cres['n_windows']) == [4, 5] and np.isfinite(cres['acf']).all()
      and np.allclose(cres['acf'][0], direct_window_acf(Xc[keep_w], np.ones((4, 40)), 5)[0], atol=1e-6))

# ---------------------------------------------------------------- exponential ACF
k = np.arange(257)
taus = np.array([2.0, 10.0, 40.0])
A = np.exp(-k[None] / taus[:, None])
check('tau_int of exp(-k / tau) is about tau - 0.5: %s' % np.round(tau_int(A), 2),
      np.allclose(tau_int(A), taus - 0.5, atol=0.1 * taus))
A2 = 0.7 * (np.exp(-k[None] / taus[:, None]) + 0.05)
f = exp_fit(A2)
check('exponential fit recovers tau, A and B: %s' % np.round(f['tau'], 2),
      f['ok'].all() and np.allclose(f['tau'], taus, rtol=1e-3) and np.allclose(f['A'], 0.7, rtol=1e-3)
      and np.allclose(f['B'], 0.05, atol=1e-3))
A3 = A2.copy(); A3[0, 5] = np.nan
check('exponential fit: a unit with NaN fails, not crashes', not exp_fit(A3)['ok'][0])

# ---------------------------------------------------------------- unit_acf end to end
from transformers import GPT2Config, GPT2Model
from parcelmate.residual import fit_token_position_large, _mlp_capture

torch.manual_seed(0)
model = GPT2Model(GPT2Config(n_layer=2, n_embd=16, n_head=2, n_positions=64, vocab_size=50)).float().eval()
V, Tw = 50, 64
ref = torch.randint(0, 40, (40, Tw))
win = torch.randint(0, 45, (9, Tw))             # ids 40-44 are absent from the reference: masked
fit = fit_token_position_large(model, ref, torch.ones_like(ref), V, batch_size=8, min_count=3, verbose=False)
out = unit_acf(model, win, fit, 10, batch_size=4, verbose=False)
hooks, cap = _mlp_capture(model)
with torch.no_grad():
    model(input_ids=win)
X = torch.cat([cap[l] for l in range(len(cap))], -1).double().numpy()
for h in hooks:
    h.remove()
rows = fit.look[win].numpy()
keep = rows >= 0
Rr = X - fit.a.double().numpy()[np.clip(rows, 0, None)] - fit.b.double().numpy()[None, :Tw]
mu = Rr[keep].mean(0); var = Rr[keep].var(0)
Z = (Rr - mu) * keep[..., None]
direct = np.zeros((X.shape[-1], 11))
for kk in range(11):
    num = (Z[:, :Tw - kk] * Z[:, kk:]).sum((0, 1))
    cnt = (keep[:, :Tw - kk] & keep[:, kk:]).sum()
    direct[:, kk] = num / (cnt * var)
check('unit_acf: some tokens are masked (%d of %d)' % ((~keep).sum(), keep.size), (~keep).sum() > 0)
check('unit_acf equals a direct computation from hooked activations (max error %.1e)'
      % np.abs(out['acf'] - direct).max(), np.allclose(out['acf'], direct, atol=1e-4))
check('unit_acf: positive share counts kept tokens above zero',
      np.allclose(out['positive_share'], ((X > 0) & keep[..., None]).sum((0, 1)) / keep.sum()))

outw = unit_acf(model, win, fit, 10, batch_size=4, verbose=False, centring='window')
check('unit_acf(centring=window) equals a direct per-window Fisher mean of the residuals (max error %.1e)'
      % np.abs(outw['acf'] - direct_window_acf(Rr, keep.astype(float), 10)).max(),
      np.allclose(outw['acf'], direct_window_acf(Rr, keep.astype(float), 10), atol=1e-4))
check('unit_acf(centring=window): the total variance is the global one', np.allclose(outw['var'], out['var'], rtol=1e-4))

# ---------------------------------------------------------------- windows
lines = ['', ' = A = \n', '', ' x y \n', ' = = Sec = = \n', ' z \n', ' = B = \n', ' w \n']
arts = wikitext_articles(lines)
check('wikitext articles split at level-1 headings only, text before the first dropped',
      arts == [' = A = \n x y \n = = Sec = = \n z \n', ' = B = \n w \n'])


class FakeTok:
    def __call__(self, text):
        return dict(input_ids=[len(w) for w in text.split()])


rows_text = ['a bb ccc'] * 1000
w, starts = block_windows(lambda s, e: rows_text[s:e], len(rows_text), FakeTok(), 4, 50, np.random.RandomState(1), block=20)
check('block windows: requested shape, distinct non-overlapping blocks',
      w.shape == (4, 50) and len(set(starts)) == 4 and np.all(np.diff(np.sort(starts)) >= 20))
w, idx = article_windows(['a ' * 10, 'b ' * 60, 'c ' * 70, 'd ' * 5], FakeTok(), 2, 50, np.random.RandomState(0))
check('article windows: only articles reaching seq_len are used', set(idx) == {1, 2} and w.shape == (2, 50))

from parcelmate.data import shuffle_words_within
ids = rs.randint(0, 30, (3, 50))
starts_tab = rs.rand(30) > 0.5
sh = np.asarray(shuffle_words_within(ids.tolist(), starts_tab, 7))
check('shuffled condition keeps every window\'s bag of tokens and changes the order',
      all(np.array_equal(np.sort(a), np.sort(b)) for a, b in zip(ids, sh)) and not np.array_equal(ids, sh))

# ---------------------------------------------------------------- networks
n_layers, width, k = 6, 400, 20
layers = np.repeat(np.arange(n_layers), width)
labels = rs.randint(0, k, n_layers * width)
p = permute_within_layer(labels, layers, rs)
same = all(np.array_equal(np.bincount(labels[layers == l], minlength=k), np.bincount(p[layers == l], minlength=k))
           for l in range(n_layers))
check('within-layer permutation keeps every network\'s count in every layer', same and not np.array_equal(p, labels))
# networks concentrated in layers: raw means differ, layer-adjusted ones should not
labels_l = (layers * 3 + rs.randint(0, 3, len(layers))) % k
tau = 2.0 + 3.0 * layers + rs.normal(0, 1, len(layers))
adj, mean, c = layer_adjusted_network_tau(tau, labels_l, layers, k, n_layers)
check('layer-adjusted network tau removes depth (sd of raw means %.2f, adjusted %.2f)'
      % (np.nanstd(mean), np.nanstd(adj)), np.nanstd(adj) < 0.2 and np.nanstd(mean) > 3)
res0 = network_test(tau, labels_l, layers, k, n_layers, 200, np.random.RandomState(3))
check('network test: tau depending only on layer gives no effect (z %.2f, p %.2f)' % (res0['z'], res0['p']),
      abs(res0['z']) < 3 and res0['p'] > 0.01)
tau_p = tau.copy(); tau_p[labels_l == 4] += 1.5
res1 = network_test(tau_p, labels_l, layers, k, n_layers, 200, np.random.RandomState(3))
check('network test: a planted slow network is detected (z %.1f, p %.3f)' % (res1['z'], res1['p']),
      res1['z'] > 5 and res1['p'] < 0.01)
check('network test: NaN units are ignored',
      np.isfinite(network_test(np.where(rs.rand(len(tau)) < 0.05, np.nan, tau), labels_l, layers, k, n_layers, 20,
                               np.random.RandomState(0))['spread']))
cv = rs.normal(size=(len(tau), 1))
tau_c = tau + 2.0 * cv[:, 0]
r = residualize_within_layer(tau_c, cv, layers)
check('covariate residualization removes the covariate and keeps layer means',
      abs(np.corrcoef(r, cv[:, 0])[0, 1]) < 0.02
      and np.allclose([r[layers == l].mean() for l in range(n_layers)], [tau_c[layers == l].mean() for l in range(n_layers)]))

# ---------------------------------------------------------------- circuits
from parcelmate.circuits import layer_matched_draws
circ = np.concatenate([2 * width + rs.choice(width, 30, replace=False), 4 * width + rs.choice(width, 20, replace=False)])
d = layer_matched_draws(circ, width, n_layers, 50, np.random.RandomState(0))
check('layer-matched draws reproduce the circuit\'s layer counts',
      all(np.array_equal(np.bincount(row // width, minlength=n_layers), np.bincount(circ // width, minlength=n_layers))
          for row in d))
tau_s = tau.copy(); tau_s[circ] += 2.0
r_slow = circuit_tau_test(tau_s, circ, width, n_layers, 500, np.random.RandomState(1))
check('circuit test: a slow circuit has mean z %.1f > 3, p_upper %.3f' % (r_slow['mean_z'], r_slow['mean_p_upper']),
      r_slow['mean_z'] > 3 and r_slow['mean_p_upper'] < 0.01)
tau_m = tau.copy(); half = circ[::2]; other = circ[1::2]
tau_m[half] += 2.5; tau_m[other] -= 2.5
r_mix = circuit_tau_test(tau_m, circ, width, n_layers, 500, np.random.RandomState(1))
check('circuit test: a mixed fast/slow circuit has variance z %.1f > 3 and |mean z| %.1f < 3'
      % (r_mix['var_z'], abs(r_mix['mean_z'])), r_mix['var_z'] > 3 and abs(r_mix['mean_z']) < 3)
r_null = circuit_tau_test(tau, circ, width, n_layers, 500, np.random.RandomState(1))
check('circuit test: a random circuit is unremarkable (mean z %.1f, var z %.1f)' % (r_null['mean_z'], r_null['var_z']),
      abs(r_null['mean_z']) < 3 and abs(r_null['var_z']) < 3)

# ---------------------------------------------------------------- analyze end to end
import csv
import json
import shutil
import sys
import tempfile

import h5py

tmp = tempfile.mkdtemp()
try:
    nl, wd, kk, L = 3, 40, 4, 128
    N = nl * wd
    lag = np.arange(L + 1)
    unit_tau = 1.0 + np.repeat(np.arange(nl), wd) + rs.uniform(0, 1, N)
    unit_B = rs.uniform(0.02, 0.3, N)          # plateau, relative to the decaying amplitude
    unit_A = rs.uniform(0.3, 0.8, N)
    with h5py.File(os.path.join(tmp, 'acf.h5'), 'w') as f:
        f.attrs['max_lag'] = L
        f['coordinates'] = np.stack(np.divmod(np.arange(N), wd), 1).astype(np.int32)
        f['layer_types'] = np.array(['linear_attention', 'linear_attention', 'full_attention'], dtype='S')
        for c, scale in (('intact', 1.0), ('shuffled', 2.0)):
            for dname in ('wikitext', 'bookcorpus'):
                for h in ('halfA', 'halfB'):
                    g = f.require_group('%s/%s/%s' % (c, dname, h))
                    a_ = unit_A[:, None] * (np.exp(-lag[None] / (scale * unit_tau[:, None])) + unit_B[:, None])
                    a_[:, 0] = 1.0
                    g['acf'] = a_ + rs.normal(0, 0.0005, (N, L + 1))
                    g['mean'] = np.zeros(N); g['var'] = rs.uniform(0.5, 2, N)
                    g['positive_share'] = rs.uniform(0.2, 0.8, N); g['n_pairs'] = np.ones(L + 1)
    coords = np.stack(np.divmod(np.arange(N), wd), 1)
    for tree in ('nets', 'nets_null'):
        os.makedirs(os.path.join(tmp, tree, 'final', 'parcellation'))
        for h in ('halfA', 'halfB'):
            P = np.eye(kk)[rs.randint(0, kk, N)]
            with h5py.File(os.path.join(tmp, tree, 'final', 'parcellation', 'parcellation_pooled_%s.h5' % h), 'w') as f:
                f['parcellation'] = P; f['coordinates'] = coords
    proot = os.path.join(tmp, 'patch', 'Model')
    with open(os.path.join(os.makedirs(proot) or proot, 'accuracy.csv'), 'w', newline='') as f:
        w_ = csv.writer(f); w_.writerow(['domain', 'task', 'both_correct_accuracy', 'n_both_correct'])
        for i in range(3):
            os.makedirs(os.path.join(proot, 'grp', 't%d' % i))
            np.save(os.path.join(proot, 'grp', 't%d' % i, 'neuron_attribution.npy'), rs.normal(size=(nl, wd)))
            w_.writerow(['grp', 't%d' % i, 0.7 + 0.1 * i, 200])
    from parcelmate.bin import timescales as bt
    sys.argv = ['x', 'analyze', '--out', tmp, '--networks', os.path.join(tmp, 'nets'), '--patching', proot,
                '--pct', '10', '--draws', '50', '--perm', '20']
    bt.main()
    o = os.path.join(tmp, 'analysis_lags')
    s = json.load(open(os.path.join(o, 'summary.json')))
    circ = list(csv.DictReader(open(os.path.join(o, 'circuits.csv'))))
    nets = list(csv.DictReader(open(os.path.join(o, 'networks.csv'))))
    u = np.load(os.path.join(o, 'units.npz'))
    true = {(c, k): unit_A * (np.exp(-k / (sc * unit_tau)) + unit_B) for c, sc in (('intact', 1.0), ('shuffled', 2.0))
            for k in (1, 10, 100)}
    check('analyze: acf1, acf10, acf100 are the ACF at those lags, both conditions',
          all(np.allclose(u['%s_acf%d' % ck], v, atol=0.002) for ck, v in true.items()))
    check('analyze: per-half values kept', np.allclose(u['intact_acf10_halfA'], true[('intact', 10)], atol=0.002))
    check('analyze: tau_int kept as a reference', np.allclose(
        u['intact_tau_int_halfA'], (unit_A[:, None] * (np.exp(-lag[1:] / unit_tau[:, None]) + unit_B[:, None])).sum(1), atol=0.1))
    check('analyze: split-half reliability near 1 on near-noise-free ACFs, gate passed',
          min(s['split_half'][m] for m in bt.TESTED) > 0.95 and s['gate_passed'])
    check('analyze: no fit in the summary', 'fit_ok' not in s and s['lags'] == [1, 10, 100])
    check('analyze: one circuit row per task and measure (4), accuracy attached',
          len(circ) == 12 and np.allclose(sorted({round(float(r['accuracy']), 6) for r in circ}), [0.7, 0.8, 0.9]))
    check('analyze: one network row per partition (2 halves x real/null) and measure (6)', len(nets) == 24)
    from scipy.stats import spearmanr
    check('analyze: depth Spearman of acf10 matches a direct computation', np.isclose(
        s['depth_spearman']['intact_acf10'], spearmanr(np.repeat(np.arange(nl), wd), u['intact_acf10'])[0]))
    shutil.copy(os.path.join(tmp, 'acf.h5'), os.path.join(tmp, 'acf_within.h5'))
    with h5py.File(os.path.join(tmp, 'acf.h5'), 'a') as f:
        for c in ('intact', 'shuffled'):
            for dname in ('wikitext', 'bookcorpus'):
                for h in ('halfA', 'halfB'):
                    f['%s/%s/%s/acf' % (c, dname, h)][...] = 0.0     # acf.h5 no longer holds the values
    sys.argv = ['x', 'analyze', '--out', tmp, '--acf', 'acf_within.h5', '--tag', 'analysis_within', '--networks',
                os.path.join(tmp, 'nets'), '--patching', proot, '--pct', '10', '--draws', '50', '--perm', '20',
                '--ignore-gate']
    bt.main()
    u2 = np.load(os.path.join(tmp, 'analysis_within', 'units.npz'))
    check('analyze --acf acf_within.h5 reads that file, not acf.h5', np.allclose(u2['intact_acf10'], u['intact_acf10']))
    lay_rows = list(csv.DictReader(open(os.path.join(o, 'layers.csv'))))
    check('analyze: layers.csv has a median for every measure and layer',
          len(lay_rows) == nl and all('%s_median' % m in lay_rows[0] for m in bt.DESCRIBED))
finally:
    shutil.rmtree(tmp)

print('\n%d checks, %d failures' % (n_checks[0], len(failures)))
if failures:
    for f_ in failures:
        print('  FAIL', f_)
    raise SystemExit(1)
