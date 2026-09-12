"""Verification for the Iteration 14 additions (YOLO runs 1-3).

    PYTHONPATH=. python tests/verify_iter8_yolo.py

Covers: full-k-means and Ward clustering paths, the unwhitened PCA flag, the block-model
refinement (objective identity with fidelity, monotone descent, recovery of a planted block
structure from a corrupted start), the single-sample ceiling handling, the dimension-AMI
triviality metric, MLP-neuron unit extraction on a tiny random GPT-2 (seeded subset, original
indices kept, values equal to the post-GELU activations), and the --variants job filter.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score
from transformers import GPT2Config, GPT2Model

from parcelmate.bin.make_jobs import get_job, DEFAULTS
from parcelmate.constants import CONNECTIVITY_NAME, HALF_NAMES
from parcelmate.metrics import block_sse, blockmodel_refine, fidelity_insample, triviality, block_means
from parcelmate.model import get_timecourses, run_parcellation, sample_parcellations
from parcelmate.util import load_h5_data, read_attrs, save_h5_data

failures = []
n_checks = [0]


def check(name, cond):
    n_checks[0] += 1
    print('%s %s' % ('OK' if cond else 'FAIL', name))
    if not cond:
        failures.append(name)


def planted_abs_r(n=300, k=5, noise=0.05, seed=0):
    """|r|-like symmetric matrix with k planted blocks; returns (R, labels)."""
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, k, n)
    M = rng.uniform(0.05, 0.4, (k, k))
    M = (M + M.T) / 2
    np.fill_diagonal(M, M.diagonal() + 0.3)
    R = M[labels][:, labels] + noise * rng.randn(n, n)
    R = np.abs((R + R.T) / 2)
    np.fill_diagonal(R, 1.0)
    return R, labels


# ---------------------------------------------------------------- block model objective
R, truth = planted_abs_r()
k = truth.max() + 1
onehot = np.eye(k)[truth]
sse = block_sse(R, truth, k)
iu = np.triu_indices(R.shape[0], 1)
sstot = float(np.sum((R[iu] - R[iu].mean()) ** 2))
check('block_sse matches 1 - fidelity_insample on the same partition',
      abs((1 - sse / sstot) - fidelity_insample(R, onehot)) < 1e-9)

rng = np.random.RandomState(1)
start = truth.copy()
flip = rng.rand(len(start)) < 0.3
start[flip] = rng.randint(0, k, flip.sum())
sse_start = block_sse(R, start, k)
refined, sse_ref = blockmodel_refine(R, start, k, max_iter=50)
check('refinement lowers the block SSE from a 30%%-corrupted start (%.4g -> %.4g)'
      % (sse_start, sse_ref), sse_ref < sse_start)
check('refinement returns the SSE of the labels it returns',
      abs(block_sse(R, refined, k) - sse_ref) < 1e-9)
check('refinement recovers the planted blocks (ARI %.3f)' % adjusted_rand_score(truth, refined),
      adjusted_rand_score(truth, refined) > 0.98)
check('refinement is at least as good as the planted truth', sse_ref <= sse + 1e-9)
again, _ = blockmodel_refine(R, refined, k, max_iter=50)
check('refinement is a fixed point of itself', np.array_equal(again, refined))

# ------------------------------------------------------------ clustering paths
n = R.shape[0]
out_km = sample_parcellations(R, n_networks=k, n_samples=3, binarize_connectivity=False,
                              fisher_transform=True, standardize_profiles=True,
                              clustering='kmeans', seed=0, verbose=False)
check('kmeans path: 3 restarts of valid labels',
      out_km['samples'].shape == (3, n) and out_km['samples'].max() < k)
check('kmeans path: inertia recorded per restart', np.all(out_km['scores'] > 0))
check('kmeans path recovers the planted blocks (ARI %.3f)'
      % adjusted_rand_score(truth, out_km['samples'][0]),
      adjusted_rand_score(truth, out_km['samples'][0]) > 0.9)

out_w = sample_parcellations(R, n_networks=k, n_samples=5, binarize_connectivity=False,
                             fisher_transform=True, standardize_profiles=True,
                             clustering='ward', seed=0, verbose=False)
check('ward path forces a single sample', out_w['samples'].shape == (1, n))
out_w2 = sample_parcellations(R, n_networks=k, n_samples=1, binarize_connectivity=False,
                              fisher_transform=True, standardize_profiles=True,
                              clustering='ward', seed=123, verbose=False)
check('ward is deterministic across seeds', np.array_equal(out_w['samples'], out_w2['samples']))

out_bm = sample_parcellations(R, n_networks=k, n_samples=2, binarize_connectivity=False,
                              fisher_transform=True, standardize_profiles=True,
                              clustering='kmeans', blockmodel_refine_labels=True,
                              seed=0, verbose=False)
check('blockmodel path: score is the block SSE of the returned labels',
      all(abs(block_sse(R, out_bm['samples'][i].astype(int), k) - out_bm['scores'][i]) < 1e-6
          for i in range(2)))
check('blockmodel path: SSE no worse than the k-means labels it started from',
      all(out_bm['scores'][i] <= block_sse(R, out_km['samples'][i].astype(int), k) + 1e-9
          for i in range(2)))

out_pca = sample_parcellations(R, n_networks=k, n_samples=2, binarize_connectivity=False,
                               fisher_transform=True, connectivity_pca_components=10,
                               pca_whiten=False, clustering='kmeans', seed=0, verbose=False)
check('unwhitened PCA path runs and labels are valid',
      out_pca['samples'].shape == (2, n) and out_pca['samples'].max() < k)

try:
    sample_parcellations(R, n_networks=k, n_samples=1, clustering='spectral', verbose=False)
    check('unknown clustering rejected', False)
except AssertionError:
    check('unknown clustering rejected', True)

# ---------------------------------------------- run_parcellation: ceiling with one sample
tmp = tempfile.mkdtemp()
conn_dir = os.path.join(tmp, CONNECTIVITY_NAME)
os.makedirs(conn_dir)
coords = np.stack([np.repeat(np.arange(3), n // 3), np.tile(np.arange(n // 3), 3)], 1).astype(np.int32)
R_signed = R.copy()
np.fill_diagonal(R_signed, 1.0)
for key in ('avg',) + HALF_NAMES:
    save_h5_data(dict(connectivity=R_signed, coordinates=coords), os.path.join(
        conn_dir, '%s_dom_%s.h5' % (CONNECTIVITY_NAME, key)), verbose=False)
run_parcellation(output_dir=tmp, n_networks=k, n_samples=4, binarize_connectivity=False,
                 fisher_transform=True, standardize_profiles=True, connectivity_pca_components=None,
                 clustering='ward', seed=0, variant='ward', verbose=False)
path = os.path.join(tmp, 'ward', 'parcellation', 'parcellation_dom_halfA.h5')
d = load_h5_data(path, verbose=False)
a = read_attrs(path)
check('ward run: restart-split halves equal the parcellation (ceiling 1 by construction)',
      np.array_equal(d['parcellation_split1'], d['parcellation']) and
      np.array_equal(d['parcellation_split2'], d['parcellation']))
check('ward run: provenance records clustering=ward, n_samples=1, pca_whiten',
      a.get('clustering') == 'ward' and int(a.get('n_samples')) == 1 and 'pca_whiten' in a)
run_parcellation(output_dir=tmp, n_networks=k, n_samples=2, binarize_connectivity=False,
                 fisher_transform=True, standardize_profiles=True, connectivity_pca_components=None,
                 clustering='kmeans', blockmodel_refine_labels=True, seed=0, variant='bm', verbose=False)
a = read_attrs(os.path.join(tmp, 'bm', 'parcellation', 'parcellation_dom_halfA.h5'))
check('blockmodel run: provenance records the refinement flag',
      a.get('blockmodel_refine_labels') in (True, 'True', 1))

# ------------------------------------------------------------ dimension AMI
chain_labels = coords[:, 1] % k                      # follows the dimension index
layer_labels = coords[:, 0]
tri_chain = triviality(np.eye(k)[chain_labels], coords)
tri_layer = triviality(np.eye(3)[layer_labels], coords)
# AMI is normalized by the mean entropy of the two labelings, so a k-class partition that
# is a deterministic function of a 100-class index tops out well below 1 (here ~0.4); the
# real case, 768 dimensions into 50 clusters, is bounded near 0.74. Compare with ~0.
check('ami_dimension well above 0 (%.2f) and ami_layer ~0 for a chain-following parcellation'
      % tri_chain['ami_dimension'],
      tri_chain['ami_dimension'] > 0.3 and tri_chain['ami_layer'] < 0.05)
check('ami_dimension ~0 and ami_layer high for a layer parcellation',
      tri_layer['ami_dimension'] < 0.05 and tri_layer['ami_layer'] > 0.9)

# ------------------------------------------------------------ MLP units
cfg = GPT2Config(n_layer=2, n_embd=16, n_head=2, vocab_size=100, n_positions=32)
torch.manual_seed(0)
model = GPT2Model(cfg).eval()
ids = torch.randint(0, 100, (3, 12))
mask = torch.ones_like(ids)
mask[2, 8:] = 0
out = get_timecourses(model, ids, mask, batch_size=2, unit_type='mlp', units_per_layer=8,
                      unit_seed=7, verbose=False)
tc, co = out['timecourses'], out['coordinates']
check('mlp units: 2 layers x 8 = 16 units over the unmasked tokens',
      tc.shape == (16, int(mask.sum())) and co.shape == (16, 2))
check('mlp units: coordinates carry layer and ORIGINAL neuron index (< 64), sorted per layer',
      set(co[:, 0]) == {0, 1} and co[:, 1].max() < 64 and
      all(np.all(np.diff(co[co[:, 0] == l, 1]) > 0) for l in (0, 1)))
out_same = get_timecourses(model, ids, mask, batch_size=3, unit_type='mlp', units_per_layer=8,
                           unit_seed=7, verbose=False)
out_other = get_timecourses(model, ids, mask, batch_size=3, unit_type='mlp', units_per_layer=8,
                            unit_seed=8, verbose=False)
check('mlp units: subset fixed by unit_seed, independent of batch size',
      np.array_equal(co, out_same['coordinates']) and np.allclose(tc, out_same['timecourses'], atol=1e-5))
check('mlp units: a different unit_seed draws a different subset',
      not np.array_equal(co, out_other['coordinates']))
# Ground truth: the post-GELU activations from a hook on `mlp.act` itself.
acts = {}
hs = [blk.mlp.act.register_forward_hook(lambda m, i, o, l=l: acts.__setitem__(l, o.detach()))
      for l, blk in enumerate(model.h)]
with torch.no_grad():
    model(input_ids=ids, attention_mask=mask)
for h in hs:
    h.remove()
m = mask.numpy().astype(bool)
ok = True
for l in (0, 1):
    sel = co[:, 0] == l
    ref = acts[l].numpy()[m][:, co[sel, 1]].T
    ok &= np.allclose(tc[sel], ref, atol=1e-5)
check('mlp units: values equal the post-GELU activations', ok)
out_h = get_timecourses(model, ids, mask, batch_size=2, unit_type='hidden', verbose=False)
check('hidden units unchanged: 3 x 16 residual-stream units',
      out_h['timecourses'].shape[0] == 48 and set(out_h['coordinates'][:, 0]) == {0, 1, 2})

# ------------------------------------------------------------ --variants
settings = dict(DEFAULTS)
job = get_job('configs/x.yml', settings, steps=['parcellation'], variants=['a', 'b'])
check('make_jobs: -V reaches main.py and the job name',
      '-V a b' in job and '--job-name=x.parcellation.a_b' in job)
cfg_path = os.path.join(tmp, 'c.yml')
with open(cfg_path, 'w') as f:
    f.write('output_dir: %s\nparcellation_variants:\n  a: {}\n  b: {}\n' % tmp)
r = subprocess.run([sys.executable, '-m', 'parcelmate.bin.main', cfg_path, '-s', 'nothing', '-V', 'zzz'],
                   capture_output=True, text=True, env=dict(os.environ, PYTHONPATH='.'))
check('main.py rejects an unknown -V variant', r.returncode != 0 and 'unknown variant' in r.stderr)
r = subprocess.run([sys.executable, '-m', 'parcelmate.bin.main', cfg_path, '-s', 'nothing', '-V', 'a'],
                   capture_output=True, text=True, env=dict(os.environ, PYTHONPATH='.'))
check('main.py accepts a known -V variant', r.returncode == 0)

# ------------------------------------------------------------ score -V
from parcelmate.bin.score import score_config
cfg = dict(output_dir=tmp, seed=1, connectivity=dict(domains=['dom']),
           parcellation_variants=dict(ward={}, bm={}))
rows, out_path = score_config(cfg, variants=['ward'], cross_domain=False, verbose=False,
                              allow_partial=True)
check('score -V: only the named arm is scored and the file is scores_<arm>.csv',
      {r['variant'] for r in rows if not r['variant'].startswith('(')} == {'ward'}
      and os.path.basename(out_path) == 'scores_ward.csv')
try:
    score_config(cfg, variants=['nope'], cross_domain=False, verbose=False, allow_partial=True)
    check('score -V rejects an unknown arm', False)
except AssertionError:
    check('score -V rejects an unknown arm', True)

print('\n%d check(s), %d failure(s)' % (n_checks[0], len(failures)))
if failures:
    for f in failures:
        print('  - %s' % f)
    sys.exit(1)
