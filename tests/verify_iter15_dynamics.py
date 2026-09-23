"""Verification for the Iteration 28 additions (descriptive training-dynamics measures).

    PYTHONPATH=. python tests/verify_iter15_dynamics.py

CPU only; a tiny GPT-NeoX causal LM with the GPT-2 tokenizer and the `random` baseline
domain. Covers: token classes on hand-built strings, including the context rule for
word-initial tokens; gini, the spectrum summaries, coupling and segregation measures against
direct dense computations; `checkpoint_measures` end to end, with its recomputed halves equal
to the connectivity step's, its firing rates and class means equal to direct computations on
the same timecourses, and its loss equal to the model's own loss; the partition measures,
cross-domain generality, split and merge counts and birth steps on constructed partitions
with known answers; and the `partitions` and `combine` subcommands end to end.
"""

import csv
import os
import shutil
import subprocess
import sys
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import h5py
import numpy as np
import torch
import yaml
from transformers import AutoTokenizer, GPTNeoXConfig, GPTNeoXForCausalLM

from parcelmate.constants import CONNECTIVITY_NAME, HALF_NAMES
from parcelmate.dynamics import (
    TOKEN_CLASSES, birth_steps, checkpoint_measures, coupling_measures, cross_domain_generality,
    gini, jaccard_matrix, partition_measures, segregation, spectrum_measures, token_class_table,
    token_classes, transition_measures,
)
from parcelmate.model import domain_data_kwargs, get_timecourses, run_connectivity, run_parcellation
from parcelmate.util import load_h5_data

failures = []
n_checks = [0]
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
ENV = dict(os.environ, PYTHONPATH='.')


def check(name, cond):
    n_checks[0] += 1
    print('%s %s' % ('OK' if cond else 'FAIL', name))
    if not cond:
        failures.append(name)


torch.manual_seed(0)
os.makedirs(os.path.join(HERE, '.tmp'), exist_ok=True)
tmp = tempfile.mkdtemp(dir=os.path.join(HERE, '.tmp'))
tok = AutoTokenizer.from_pretrained('gpt2')

# ---------------------------------------------------------------- token classes
table = token_class_table(tok)
cls_name = lambda text: [TOKEN_CLASSES[c] for c in token_classes(np.array([tok(text)['input_ids']]), table)[0]]
check('token classes: function word, content word, continuation, punctuation, digit',
      cls_name(' the cat') == ['function_word', 'content_word']
      and cls_name(' walking,')[-1] == 'punctuation'
      and cls_name(' 42') == ['digit'])
ids = tok(' antidisestablishment')['input_ids']   # ' ant', 'idis', 'establishment' in GPT-2
check('token classes: a multi-piece word is one word start then continuations',
      len(ids) == 3 and cls_name(' antidisestablishment')[0] == 'content_word'
      and all(c == 'continuation' for c in cls_name(' antidisestablishment')[1:]))
check('token classes: whitespace, and a word after a newline or at the start is word-initial',
      cls_name('The\nThe') == ['function_word', 'whitespace', 'function_word']
      and cls_name('"Cat')[-1] == 'content_word')
check('function-word list is lower-cased closed-class English',
      table['func'][tok(' The')['input_ids'][0]] and not table['func'][tok(' cat')['input_ids'][0]])

# ---------------------------------------------------------------- small measures
check('gini: 0 for equal values, (n-1)/n for one holder, known value',
      abs(gini([1, 1, 1, 1])) < 1e-12 and abs(gini([0, 0, 0, 1]) - 0.75) < 1e-12
      and abs(gini([1, 2, 3, 4]) - 0.25) < 1e-12)
rng = np.random.RandomState(0)
Xs = rng.randn(40, 500)
Rc = np.corrcoef(Xs)
lam = np.linalg.eigvalsh(Rc)
sm = spectrum_measures(lam)
check('spectrum: participation ratio = (tr R)^2 / ||R||_F^2 and top shares from the eigenvalues',
      abs(sm['dim_participation_ratio'] - np.trace(Rc) ** 2 / (Rc ** 2).sum()) < 1e-8
      and abs(sm['dim_top1_share'] - lam.max() / lam.sum()) < 1e-12)
n = 300
A = np.abs(rng.randn(n, n)).astype(np.float32) * 0.1
A = (A + A.T) / 2
np.fill_diagonal(A, 0.0)
lab = rng.randint(0, 6, n)
for c in range(6):
    idx = np.flatnonzero(lab == c)
    A[np.ix_(idx, idx)] += 0.2
np.fill_diagonal(A, 0.0)
seg = segregation(A, lab, 6)
same = (lab[:, None] == lab[None, :]) & ~np.eye(n, dtype=bool)
diff = lab[:, None] != lab[None, :]
k_ = A.sum(1)
Q = sum(A[np.ix_(lab == c, lab == c)].sum() / A.sum() - (k_[lab == c].sum() / A.sum()) ** 2 for c in range(6))
check('segregation: within and between means, ratio and modularity equal a dense computation',
      abs(seg['segregation_within_mean'] - A[same].mean()) < 1e-6
      and abs(seg['segregation_between_mean'] - A[diff].mean()) < 1e-6
      and abs(seg['segregation_modularity'] - Q) < 1e-6 and seg['segregation_ratio'] > 1)
cm = coupling_measures(A, seed=1, n_pairs=200_000)
off = A[~np.eye(n, dtype=bool)]
top = np.sort(A, axis=1)[:, -3:].sum(1) / A.sum(1)
check('coupling: exact mean |r|, sampled median close, concentration and hub gini direct',
      abs(cm['coupling_mean_absr'] - off.mean()) < 1e-6
      and abs(cm['coupling_median_absr'] - np.median(off)) < 5e-3
      and abs(cm['coupling_concentration_top1pct'] - np.median(top)) < 1e-6
      and abs(cm['hub_strength_gini'] - gini(A.sum(1, dtype=np.float64))) < 1e-12)

# ---------------------------------------------------------------- the checkpoint job
model_dir = os.path.join(tmp, 'tiny')
lm = GPTNeoXForCausalLM(GPTNeoXConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                      intermediate_size=64, vocab_size=50304, max_position_embeddings=64)).eval()
lm.save_pretrained(model_dir)
tok.save_pretrained(model_dir)
tree = os.path.join(tmp, 'step7')
conn = dict(model_name=model_dir, revision=None, unit_type='mlp', units_per_layer=None, n_samples=4,
            domains=['random'], seq_len=32, n_tokens=256, split='train', take=100000, wrap=True,
            shuffle=True, batch_size=4, highpass=None, lowpass=None, step=0.2, eps=1e-3,
            null_model='circshift', n_surrogates=0, outputs=['halves'])
PARC = dict(n_networks=4, n_samples=6, clustering='kmeans', binarize_connectivity=False,
            fisher_transform=True, standardize_profiles=True, sparsify_profiles=True,
            connectivity_pca_components=8, pca_whiten=False, store_samples=True,
            parcellate_keys=['halfA', 'halfB'], seed=3, verbose=False)
run_connectivity(output_dir=tree, null_output_dir=tree + '_null', seed=7, verbose=False, **conn)
for t in ('', '_null'):
    run_parcellation(output_dir=tree + t, variant='final', **PARC)
cfg = dict(output_dir=tree, seed=7, connectivity=conn)
out = os.path.join(tmp, 'dyn')
rows, halves = checkpoint_measures(cfg, out, step=7, keep_halves=True, n_sub=50, verbose=False)
ok = True
for h in HALF_NAMES:
    stored = load_h5_data(os.path.join(tree, CONNECTIVITY_NAME, 'connectivity_random_%s.h5' % h),
                          verbose=False)['connectivity']
    ok &= np.allclose(np.nan_to_num(stored), np.nan_to_num(halves['random'][h]), atol=1e-6)
check('recomputed halves equal the connectivity step\'s halves (the stored partitions apply)', ok)
# Direct activation measures on the same samples.
from parcelmate.data import get_dataset
from parcelmate.util import derive_seed
dkw = domain_data_kwargs('random')
dkw['tokenizer'] = tok
ids_all, mask_all = get_dataset(n_tokens=256 * 4, split='train', take=100000, seq_len=32, wrap=True,
                                shuffle=True, seed=derive_seed(7, 'data', 'random'), verbose=False, **dkw)
nps = int(np.ceil(len(ids_all) / 4))
fires, sums, cnts, ntok, loss_sum, loss_cnt = 0, 0, 0, 0, 0.0, 0
for k in range(2):
    ids, mask = ids_all[k * nps:(k + 1) * nps], mask_all[k * nps:(k + 1) * nps]
    X = get_timecourses(lm.base_model, ids, mask, batch_size=4, unit_type='mlp', unit_seed=7,
                        seed=derive_seed(7, 'timecourses', 'random', k + 1), verbose=False)['timecourses']
    cls = token_classes(ids.numpy(), table)[mask.numpy().astype(bool)]
    fires = fires + (X > 0).sum(1)
    sums = sums + np.stack([X[:, cls == c].sum(1) for c in range(len(TOKEN_CLASSES))], 1)
    cnts = cnts + np.array([(cls == c).sum() for c in range(len(TOKEN_CLASSES))])
    ntok += X.shape[1]
    with torch.no_grad():
        for s in range(0, ids.size(0), 4):
            L = lm(input_ids=ids[s:s + 4], labels=ids[s:s + 4]).loss.item()
            loss_sum += L * ids[s:s + 4].numel() - L * ids[s:s + 4].size(0)
            loss_cnt += ids[s:s + 4].numel() - ids[s:s + 4].size(0)
with h5py.File(os.path.join(out, 'units_step7_random.h5'), 'r') as f:
    fr = f['firing_rate_halfA'][()]
    cm_ = f['class_mean_halfA'][()]
    ls, lc = f['loss_sum'][()], f['loss_count'][()]
with np.errstate(invalid='ignore', divide='ignore'):
    direct_cm = sums / cnts[None, :]
check('firing rate equals the direct fraction of positive post-GELU activations', np.allclose(fr, fires / ntok))
check('class means equal direct per-class means of the same timecourses',
      np.allclose(np.nan_to_num(cm_), np.nan_to_num(direct_cm), atol=1e-5))
check('loss of half A equals the model\'s own next-token loss on the same batches',
      abs(ls[:2].sum() / lc[:2].sum() - loss_sum / loss_cnt) < 1e-5)
names = {r['measure'] for r in rows}
check('checkpoint rows: dimensionality, coupling, hubs, activation, loss, segregation of both trees',
      {'dim_participation_ratio', 'dim_top10_share', 'coupling_mean_absr', 'hub_strength_gini',
       'act_firing_rate_median', 'lm_loss', 'segregation_modularity'} <= names
      and {r['partition'] for r in rows if r['measure'] == 'segregation_modularity'} == {'real', 'null'}
      and {(r['key'], r['eval_key']) for r in rows if r['measure'] == 'segregation_modularity'}
      == {('halfA', 'halfA'), ('halfA', 'halfB'), ('halfB', 'halfB'), ('halfB', 'halfA')})
with h5py.File(os.path.join(out, 'subsample_step7_random_halfA.h5'), 'r') as f:
    subA, units = f['absr'][()], f['units'][()]
check('subsample: the |r| of a fixed seeded subset of units, fp16, zero diagonal',
      subA.shape == (50, 50) and subA.dtype == np.float16 and np.all(np.diag(subA) == 0)
      and np.allclose(subA, np.abs(np.nan_to_num(halves['random']['halfA']))[np.ix_(units, units)]
                      * (1 - np.eye(50)), atol=2e-3))

# ---------------------------------------------------------------- partition measures on known cases
N, L = 160, 4   # 40 units per layer, divisible by the 4 networks
coords = np.stack([np.repeat(np.arange(L), N // L), np.tile(np.arange(N // L), L)], 1)
lab_layer = coords[:, 0]                              # every network is one layer
P = np.eye(4)[lab_layer] * 0.9 + 0.025
pm = partition_measures(P, coords)
check('partition measures: single-layer networks, equal sizes, crispness 0.925',
      pm['depth_frac_single_layer'] == 1.0 and abs(pm['depth_median_effective_layers'] - 1) < 1e-12
      and abs(pm['size_gini']) < 1e-12 and pm['size_n_networks'] == 4
      and abs(pm['crisp_median_max_membership'] - 0.925) < 1e-12)
lab_span = np.arange(N) % 4                           # every network spans all four layers
pm2 = partition_measures(np.eye(4)[lab_span], coords)
check('partition measures: networks spanning every layer have 4 effective layers',
      abs(pm2['depth_median_effective_layers'] - 4) < 1e-9 and pm2['depth_frac_single_layer'] == 0)
g = cross_domain_generality({'a': lab_layer, 'b': lab_layer.copy()}, 4)
check('generality: identical partitions in two domains are fully general', g['a']['generality_mean'] == 1.0
      and g['a']['generality_frac_general'] == 1.0)
a = np.repeat([0, 1], 60)
b = np.concatenate([np.repeat([0, 2], 30), np.repeat(1, 60)])  # network 0 splits into 0 and 2
tm = transition_measures(a, b, 3)
check('transitions: one of two networks splits in two, none merge, one continues',
      abs(tm['trans_split_degree'] - 1.5) < 1e-12 and abs(tm['trans_merge_degree'] - 1.0) < 1e-12
      and abs(tm['trans_continued'] - 0.5) < 1e-12)
steps = {0: np.random.RandomState(1).randint(0, 3, 90), 10: np.repeat([0, 1, 2], 30),
         20: np.repeat([0, 1, 2], 30)}
tr, births, st = birth_steps(steps, 3)
check('birth steps: networks present from step 10 on are born at step 10',
      list(st) == [0, 10, 20] and np.all(births == 10) and np.allclose(tr[:, 1:], 1.0))

# ---------------------------------------------------------------- the subcommands
root = os.path.join(tmp, 'model')
for s in (0, 1):
    for t in ('', '_null'):
        shutil.copytree(os.path.join(tree + t, 'final'), os.path.join(root, 'step%d%s' % (s, t), 'final'))
r = subprocess.run([sys.executable, '-m', 'parcelmate.bin.dynamics', 'partitions', '--root', root,
                    '--out', out, '--domains', 'random'], cwd=ROOT, env=ENV, capture_output=True, text=True)
rows_p = list(csv.DictReader(open(os.path.join(out, 'partitions.csv')))) if r.returncode == 0 else []
check('partitions subcommand: measures for both steps and trees, transitions, births',
      r.returncode == 0 and {x['step'] for x in rows_p} == {'0', '1'} and {x['tree'] for x in rows_p} == {'real', 'null'}
      and any(x['measure'] == 'trans_continued' and float(x['value']) == 1.0 for x in rows_p)
      and os.path.exists(os.path.join(out, 'network_tracks.csv')))
shutil.copy(os.path.join(out, 'subsample_step7_random_halfA.h5'), os.path.join(out, 'subsample_step8_random_halfA.h5'))
shutil.copy(os.path.join(out, 'subsample_step7_random_halfB.h5'), os.path.join(out, 'subsample_step8_random_halfB.h5'))
r = subprocess.run([sys.executable, '-m', 'parcelmate.bin.dynamics', 'combine', '--out', out],
                   cwd=ROOT, env=ENV, capture_output=True, text=True)
sim = list(csv.DictReader(open(os.path.join(out, 'similarity.csv')))) if r.returncode == 0 else []
check('combine: identical checkpoints have similarity 1; halves compared; one long connectome table',
      r.returncode == 0
      and any(x['measure'] == 'similarity_to_previous' and abs(float(x['value']) - 1) < 1e-9 for x in sim)
      and any(x['measure'] == 'similarity_between_halves' for x in sim)
      and os.path.exists(os.path.join(out, 'connectome.csv')))
mode = oct(os.stat(os.path.join(out, 'similarity.csv')).st_mode & 0o777)
check('files written by the subcommands are group writable (umask 002)', mode in ('0o664', '0o666'))

shutil.rmtree(tmp)
print('\n%d checks, %d failure(s)' % (n_checks[0], len(failures)))
for f in failures:
    print('  FAIL', f)
sys.exit(1 if failures else 0)
