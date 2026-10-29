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
"""

import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch

from parcelmate.timescales import (ACFAccumulator, article_windows, block_windows, circuit_tau_test, exp_fit,
                                   lagged_sums, layer_adjusted_network_tau, network_test, permute_within_layer,
                                   residualize_within_layer, tau_int, unit_acf, wikitext_articles)

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
    nl, wd, kk, L = 3, 40, 4, 16
    N = nl * wd
    lag = np.arange(L + 1)
    unit_tau = 1.0 + np.repeat(np.arange(nl), wd) + rs.uniform(0, 1, N)
    with h5py.File(os.path.join(tmp, 'acf.h5'), 'w') as f:
        f.attrs['max_lag'] = L
        f['coordinates'] = np.stack(np.divmod(np.arange(N), wd), 1).astype(np.int32)
        f['layer_types'] = np.array(['linear_attention', 'linear_attention', 'full_attention'], dtype='S')
        for c, scale in (('intact', 1.0), ('shuffled', 0.5)):
            for dname in ('wikitext', 'bookcorpus'):
                for h in ('halfA', 'halfB'):
                    g = f.require_group('%s/%s/%s' % (c, dname, h))
                    g['acf'] = np.exp(-lag[None] / (scale * unit_tau[:, None])) + rs.normal(0, 0.002, (N, L + 1))
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
                '--pct', '10', '--draws', '50', '--perm', '20', '--cores', '2']
    bt.main()
    s = json.load(open(os.path.join(tmp, 'analysis', 'summary.json')))
    circ = list(csv.DictReader(open(os.path.join(tmp, 'analysis', 'circuits.csv'))))
    nets = list(csv.DictReader(open(os.path.join(tmp, 'analysis', 'networks.csv'))))
    u = np.load(os.path.join(tmp, 'analysis', 'units.npz'))
    check('analyze: tau_int of the synthetic ACF is about tau - 0.5 (intact)',
          np.allclose(u['tau_intact'], np.exp(-lag[1:, None] / unit_tau[None]).sum(0), atol=0.05))
    check('analyze: diff = intact - shuffled', np.allclose(u['tau_diff'], u['tau_intact'] - u['tau_shuffled']))
    check('analyze: split-half reliability near 1 on noise-free ACFs, gate passed',
          s['split_half']['intact'] > 0.95 and s['gate_passed'])
    check('analyze: exponential fits succeed on exponential ACFs', s['exp_fit_ok']['intact'] > 0.95)
    check('analyze: one circuit row per task and measure, accuracy attached',
          len(circ) == 9 and np.allclose(sorted({round(float(r['accuracy']), 6) for r in circ}), [0.7, 0.8, 0.9]))
    check('analyze: one network row per partition (2 halves x real/null) and measure (4)', len(nets) == 16)
    check('analyze: depth correlation positive', s['depth_spearman']['intact'] > 0.5)
finally:
    shutil.rmtree(tmp)

print('\n%d checks, %d failures' % (n_checks[0], len(failures)))
if failures:
    for f_ in failures:
        print('  FAIL', f_)
    raise SystemExit(1)
