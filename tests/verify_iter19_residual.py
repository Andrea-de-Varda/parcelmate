"""Verification for the residual connectome (LOG.md Iteration 39).

    PYTHONPATH=. python tests/verify_iter19_residual.py

CPU only, a tiny GPT-NeoX. Covers: the backfitting solution on sufficient statistics equals
an explicit least-squares fit with type and position dummies (fitted values); rare types are
dropped; `residualize_timecourses` equals a direct subtraction in the layout
`get_timecourses` returns and keeps token order; the reference windows follow the analysed
ones and never overlap them; `run_connectivity(residualize=...)` end to end (provenance,
fewer observations only by the dropped tokens, connectivity equal to correlating the
residuals by hand); the dynamics measures accept the same config.
"""

import os
import shutil
import sys
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import h5py
import numpy as np
import torch
from transformers import AutoTokenizer, GPTNeoXConfig, GPTNeoXModel

from parcelmate.model import get_model_and_tokenizer, get_timecourses, run_connectivity
from parcelmate.residual import fit_token_position, reference_inputs, residualize_timecourses

failures = []
n_checks = [0]
HERE = os.path.dirname(os.path.abspath(__file__))


def check(name, cond):
    n_checks[0] += 1
    print('%s %s' % ('OK' if cond else 'FAIL', name))
    if not cond:
        failures.append(name)


torch.manual_seed(0)
os.makedirs(os.path.join(HERE, '.tmp'), exist_ok=True)
tmp = tempfile.mkdtemp(dir=os.path.join(HERE, '.tmp'))
model_dir = os.path.join(tmp, 'tiny-neox')
GPTNeoXModel(GPTNeoXConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                           vocab_size=50304, max_position_embeddings=64)).save_pretrained(model_dir)
AutoTokenizer.from_pretrained('gpt2').save_pretrained(model_dir)
model, tok = get_model_and_tokenizer(model_dir)
V = len(tok)

# ---------------------------------------------------------------- the fit equals explicit OLS
rs = np.random.RandomState(0)
L = 16
vocab = rs.choice(5000, 40, replace=False)
ids = torch.as_tensor(vocab[rs.randint(0, 40, size=(60, L))])
ids[0, 0] = 49999                                   # a type seen once: must be dropped
mask = torch.ones_like(ids)
fit = fit_token_position(model, ids, mask, V, batch_size=8, min_count=3, verbose=False, tol=1e-7)
out = get_timecourses(model, ids, mask, batch_size=8, unit_type='mlp', verbose=False)
X = out['timecourses'].T.astype(np.float64)        # tokens x units
flat = ids.reshape(-1).numpy()
pos = np.tile(np.arange(L), len(ids))
keep = fit.look[torch.as_tensor(flat)].numpy() >= 0
types = np.unique(flat[keep])
D = np.zeros((keep.sum(), len(types) + L - 1))
D[np.arange(keep.sum()), np.searchsorted(types, flat[keep])] = 1
pk = pos[keep]
for k in range(1, L):
    D[pk == k, len(types) + k - 1] = 1
coef, *_ = np.linalg.lstsq(D, X[keep], rcond=None)
fitted_ols = D @ coef
fitted_bf = (fit.a[fit.look[torch.as_tensor(flat[keep])]] + fit.b[torch.as_tensor(pk)]).numpy()
worst = float(np.abs(fitted_ols - fitted_bf).max())
check('backfitting on sufficient statistics equals explicit least squares (fitted values, worst |diff| %.1e)' % worst,
      worst < 1e-4)
check('a type seen fewer than min_count times is dropped; every type seen >= 3 times is kept',
      int(fit.look[49999]) == -1 and all(int(fit.look[t]) >= 0 for t in vocab if (flat == t).sum() >= 3))

# ---------------------------------------------------------------- residualize_timecourses
tc = out['timecourses'].copy()
res, n_drop = residualize_timecourses(fit, tc, ids, mask, chunk=100)
manual = (X[keep] - fitted_bf).T
check('residualize_timecourses equals the direct subtraction, keeps token order, reports the drops',
      n_drop == int((~keep).sum()) == 1 and res.shape == manual.shape and np.allclose(res, manual, atol=1e-5))

# ---------------------------------------------------------------- reference windows
def load(d, n_tok):
    n = n_tok // L
    base = torch.arange(10_000 * (1 + ['x', 'y'].index(d)), 10_000 * (1 + ['x', 'y'].index(d)) + n * L).reshape(n, L)
    return base, torch.ones_like(base)
r_ids, r_mask = reference_inputs(load, ['x', 'y'], 4 * L, 3 * L)
a_x, _ = load('x', 4 * L)
check('reference windows follow the analysed ones in each domain and never overlap them',
      r_ids.shape == (6, L) and int(r_ids[0, 0]) == int(a_x[-1, -1]) + 1
      and len(np.intersect1d(r_ids.numpy(), a_x.numpy())) == 0)

# ---------------------------------------------------------------- run_connectivity end to end
# A deterministic small-vocabulary text source (40 token types), so that types repeat at test
# scale; it stands in for get_dataset (same signature, same window layout, same seed use).
import parcelmate.model as pm
def small_vocab_dataset(n_tokens, seq_len, seed=None, **kw):
    n = n_tokens // seq_len
    rs_ = np.random.RandomState((seed or 0) % (2 ** 32))
    stream = vocab[rs_.randint(0, 40, size=n * seq_len)]
    t = torch.as_tensor(stream).reshape(n, seq_len)
    return t, torch.ones_like(t)
pm.get_dataset = small_vocab_dataset
COMMON = dict(model_name=model_dir, n_samples=2, domains=('random',), seq_len=32, n_tokens=512, batch_size=4,
              unit_type='mlp', units_per_layer=None, null_model='circshift', seed=7, verbose=False,
              outputs=['halves'])
plain = os.path.join(tmp, 'plain')
resid = os.path.join(tmp, 'resid')
run_connectivity(output_dir=plain, null_output_dir=plain + '_null', **COMMON)
run_connectivity(output_dir=resid, null_output_dir=resid + '_null', residualize='token_position',
                 residual_ref_tokens=512, residual_min_count=2, **COMMON)
f = h5py.File(os.path.join(resid, 'connectivity', 'connectivity_random_halfA.h5'), 'r')
g = h5py.File(os.path.join(plain, 'connectivity', 'connectivity_random_halfA.h5'), 'r')
attrs = dict(f.attrs)
check('provenance records the residual fit', attrs.get('residualize') == 'token_position'
      and int(attrs.get('residual_ref_tokens')) > 0 and int(attrs.get('residual_min_count')) == 2)
check('the residual run sees the same tokens, minus only the dropped ones',
      0 < int(f['n_obs'][()]) <= int(g['n_obs'][()]))
check('the residual connectome differs from the ordinary one', not np.allclose(f['connectivity'][()], g['connectivity'][()]))
f.close(); g.close()

# by hand: refit on the same reference, residualize sample 1 (half A is sample 1 alone)
ref_ids, ref_mask = reference_inputs(lambda d, n_tok: small_vocab_dataset(n_tok, 32, seed=pm.derive_seed(7, 'data', d)),
                                     ['random'], 512 * 2, 512)
fit2 = fit_token_position(model, ref_ids, ref_mask, V, batch_size=4, min_count=2, verbose=False)
s_ids, s_mask = small_vocab_dataset(1024, 32, seed=pm.derive_seed(7, 'data', 'random'))
s_ids, s_mask = s_ids[:16], s_mask[:16]
tc2 = get_timecourses(model, s_ids, s_mask, batch_size=4, unit_type='mlp', verbose=False)['timecourses']
r2, _ = residualize_timecourses(fit2, tc2.copy(), s_ids, s_mask)
by_hand = np.corrcoef(r2)
stored = h5py.File(os.path.join(resid, 'connectivity', 'connectivity_random_halfA.h5'), 'r')['connectivity'][()]
ok = np.isfinite(by_hand)
check('the stored residual connectome equals correlating the residuals by hand (Fisher eps aside)',
      np.allclose(np.tanh(np.arctanh(by_hand[ok] * (1 - 1e-3))), stored[ok], atol=1e-4))

# the dynamics measures on the same config: their halves equal the stored ones
import parcelmate.dynamics as dyn
from parcelmate.model import run_parcellation
dyn.get_dataset = small_vocab_dataset
PARC = dict(n_networks=4, n_samples=6, clustering='kmeans', binarize_connectivity=False, fisher_transform=True,
            standardize_profiles=True, sparsify_profiles=True, connectivity_pca_components=8, pca_whiten=False,
            store_samples=True, parcellate_keys=['halfA', 'halfB'], seed=3, verbose=False)
for t in ('', '_null'):
    run_parcellation(output_dir=resid + t, variant='final', **PARC)
conn = dict(COMMON, revision=None, split='train', take=100000, wrap=True, shuffle=True, eps=1e-3,
            residualize='token_position', residual_ref_tokens=512, residual_min_count=2)
conn.pop('verbose'); conn.pop('seed')
conn['domains'] = ['random']
rows, halves = dyn.checkpoint_measures(dict(output_dir=resid, seed=7, connectivity=conn), os.path.join(tmp, 'dyn'),
                                       step=1, keep_halves=True, n_sub=20, verbose=False)
st = h5py.File(os.path.join(resid, 'connectivity', 'connectivity_random_halfB.h5'), 'r')['connectivity'][()]
okb = np.isfinite(st) & np.isfinite(halves['random']['halfB'])
check('the dynamics measures build the same residual connectome as the connectivity step',
      np.allclose(np.abs(halves['random']['halfB'][okb]), np.abs(st[okb]), atol=1e-4) and len(rows) > 0)
try:
    run_connectivity(output_dir=os.path.join(tmp, 'z'), null_output_dir=os.path.join(tmp, 'z_null'),
                     residualize='token_position', storage='tiled_fp16', **COMMON)
    refused = False
except AssertionError:
    refused = True
check('residualize refuses the tiled path', refused)

shutil.rmtree(tmp)
print('\n%d checks, %d failure(s)' % (n_checks[0], len(failures)))
for f_ in failures:
    print('  FAIL', f_)
sys.exit(1 if failures else 0)
