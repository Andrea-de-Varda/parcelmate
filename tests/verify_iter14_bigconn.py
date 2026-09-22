"""Verification for the Iteration 25 additions (out-of-core connectivity, parcellation, scoring).

    PYTHONPATH=. python tests/verify_iter14_bigconn.py

CPU only, a tiny GPT-NeoX with the GPT-2 tokenizer, two baseline domains, 4 samples.
Covers: the tiled halves equal the dense `outputs: [halves]` path in both trees (fp16
tolerance), with equal unit statistics, coordinates and stored row strengths; `roll_rows`
equals numpy.roll per row; `null_offsets` equals the dense path's draw; the streamed
profiles equal `sample_parcellations`' profile transform; the randomized PCA features span
the exact PCA's subspace; `stream_block_means` / `stream_agreements` equal `block_means`,
`fidelity` and `fidelity_ceiling` on dense and tiled input; the tiled parcellation runs
through `run_parcellation` with the dispatch and the confirmed arm's provenance and refuses
another arm; `score_config_big` equals `score_config` on a dense copy of the same tiles and
parcellations; the `-D` domain filter and the `purge_null_connectivity` step through main.py;
the generated configs and the launcher's names.
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
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from transformers import AutoTokenizer, GPTNeoXConfig, GPTNeoXModel

from parcelmate.bigconn import (
    TiledMatrix, is_tiled, null_offsets, profile_block, randomized_pca_features, roll_rows,
    stream_profiles,
)
from parcelmate.bin.score import score_config
from parcelmate.bin.score_big import score_config_big, tree_is_tiled
from parcelmate.constants import CONNECTIVITY_NAME, HALF_NAMES
from parcelmate.data import fisher
from parcelmate.metrics import (
    block_means, fidelity, fidelity_ceiling, fidelity_insample, hard_labels, stream_agreements,
    stream_block_means,
)
from parcelmate.model import (
    run_connectivity, run_parcellation, sparsify_standardized, standardize_rows_inplace,
)
from parcelmate.data import standardize_array
from parcelmate.util import load_h5_data, read_attrs, save_h5_data

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


def raises(fn, exc=AssertionError):
    try:
        fn()
    except exc:
        return True
    return False


torch.manual_seed(0)
os.makedirs(os.path.join(HERE, '.tmp'), exist_ok=True)
tmp = tempfile.mkdtemp(dir=os.path.join(HERE, '.tmp'))
model_dir = os.path.join(tmp, 'tiny-neox')
neox = GPTNeoXModel(GPTNeoXConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                  intermediate_size=64, vocab_size=50304, max_position_embeddings=64))
neox.save_pretrained(model_dir)
AutoTokenizer.from_pretrained('gpt2').save_pretrained(model_dir)

# ---------------------------------------------------------------- roll and offsets
x = torch.randn(5, 17)
off = np.array([0, 1, 5, 16, 9])
rolled = roll_rows(x, off, sub=2).numpy()
check('roll_rows equals numpy.roll per row (sub-blocked gather)',
      all(np.allclose(rolled[i], np.roll(x[i].numpy(), off[i])) for i in range(5)))
rng = np.random.RandomState(123)
check('null_offsets: the dense draw where units < tokens', np.array_equal(
    null_offsets(10, 100, 123), rng.choice(np.arange(1, 100), size=10, replace=False)))
o = null_offsets(300, 100, 5)
check('null_offsets: with replacement where units >= tokens, offsets in [1, T)',
      len(o) == 300 and o.min() >= 1 and o.max() < 100)

# ---------------------------------------------------------------- tiled vs dense halves
COMMON = dict(model_name=model_dir, n_samples=4, domains=('random', 'whitespace'), seq_len=32,
              n_tokens=256, batch_size=4, unit_type='mlp', units_per_layer=None,
              null_model='circshift', seed=7, verbose=False, outputs=['halves'])
dense = os.path.join(tmp, 'dense')
tiled = os.path.join(tmp, 'tiled')
run_connectivity(output_dir=dense, null_output_dir=dense + '_null', **COMMON)
run_connectivity(output_dir=tiled, null_output_dir=tiled + '_null', storage='tiled_fp16', **COMMON)
ok = True
worst = 0.0
# The whitespace domain on a random model has every unit constant to float precision (std
# about 1e-9): the dense path correlates float noise, the tiled path treats the units as
# dead (below its relative threshold) and writes zeros. Equality is checked on `random`.
for tree in ('', '_null'):
    for d in ('random',):
        for h in HALF_NAMES:
            pd_ = os.path.join(dense + tree, CONNECTIVITY_NAME, '%s_%s_%s.h5' % (CONNECTIVITY_NAME, d, h))
            pt = os.path.join(tiled + tree, CONNECTIVITY_NAME, '%s_%s_%s.h5' % (CONNECTIVITY_NAME, d, h))
            a = load_h5_data(pd_, verbose=False)
            ok &= is_tiled(pt) and not is_tiled(pd_)
            T = TiledMatrix(pt)
            Rd = np.abs(np.nan_to_num(a['connectivity']))
            Rt = T[0:T.shape[0]]
            worst = max(worst, float(np.abs(Rd - Rt).max()))
            ok &= np.allclose(Rd, Rt, atol=3e-3)
            ok &= np.allclose(T.strength, Rd.sum(1), rtol=1e-3, atol=1e-2)
            ok &= np.allclose(a['unit_means'], T.attrs and h5py.File(pt)['unit_means'][()], rtol=1e-4, atol=1e-5)
            ok &= np.array_equal(a['coordinates'], T.coordinates)
            ok &= int(a['n_obs']) == int(h5py.File(pt)['n_obs'][()])
            T.close()
check('tiled halves equal the dense halves in both trees (fp16 tolerance; worst |diff| %.2e)' % worst, ok)
Tw = TiledMatrix(os.path.join(tiled, CONNECTIVITY_NAME, 'connectivity_whitespace_halfA.h5'))
check('near-constant units (whitespace on a random model) are dead in the tiled path: zero rows',
      float(Tw[0:Tw.shape[0]].max()) == 0.0 and (h5py.File(Tw.path)['unit_stds'][()] == 0).all())
Tw.close()
attrs = read_attrs(os.path.join(tiled, CONNECTIVITY_NAME, 'connectivity_random_halfA.h5'))
check('tiled provenance: storage, n_units, sources, model, distinct offsets flag',
      attrs.get('storage') == 'tiled_fp16' and int(attrs.get('n_units')) == 128
      and attrs.get('sources') == 'sample1, sample2' and attrs.get('model_name') == model_dir
      and str(attrs.get('null_offsets_distinct')) in ('True', '1'))
check('storage: tiled_fp16 refuses the dense layout',
      raises(lambda: run_connectivity(output_dir=os.path.join(tmp, 'x'), storage='tiled_fp16',
                                      **dict(COMMON, outputs=['samples', 'avg']))))

# ---------------------------------------------------------------- profiles and PCA
pt = os.path.join(tiled, CONNECTIVITY_NAME, 'connectivity_random_halfA.h5')
T = TiledMatrix(pt)
R = T[0:T.shape[0]].astype(np.float64)
X = fisher(np.clip(np.array(R, copy=True), -1.0, 1.0))
np.fill_diagonal(X, 0.0)
Z = sparsify_standardized(standardize_array(X, axis=-1))
P = np.concatenate([p for _, _, p in stream_profiles(T, block=50)])
check('streamed profiles equal the in-memory transform (blocks of 50 rows)', np.allclose(P, Z, atol=1e-4))
feat, sv = randomized_pca_features(T, n_components=6, oversample=100, seed=1, block=50, verbose=False)
exact = PCA(n_components=6, svd_solver='full').fit_transform(Z)
# Same subspace: each randomized feature is (up to sign) an exact one, when k >= N.
corr = np.abs(np.corrcoef(feat.T, exact.T)[:6, 6:])
check('randomized PCA features equal the exact PCA scores up to sign (k >= N)',
      np.allclose(corr.max(1), 1.0, atol=1e-4) and np.allclose(np.sort(np.abs(feat).sum(0)), np.sort(np.abs(exact).sum(0)), rtol=1e-3))
# Approximate regime: fewer random vectors than units, a planted low-rank structure.
rng = np.random.RandomState(0)
n = 400
lab = rng.randint(0, 5, n)
B = (lab[:, None] == lab[None, :]).astype(np.float32) * 0.3 + 0.05 * rng.randn(n, n).astype(np.float32)
B = np.abs((B + B.T) / 2)
np.fill_diagonal(B, 1.0)
np.clip(B, 0, 1, out=B)
pth = os.path.join(tmp, 'planted.h5')
with h5py.File(pth, 'w') as f:
    f.create_dataset('connectivity', data=B.astype(np.float16), chunks=(64, n))
    f.create_dataset('unit_strength', data=B.sum(1))
    f.create_dataset('coordinates', data=np.stack([np.zeros(n), np.arange(n)], 1).astype(np.int32))
    f.attrs['storage'] = 'tiled_fp16'
TB = TiledMatrix(pth)
Zb = sparsify_standardized(standardize_array(profile_block(np.array(B, dtype=np.float32), 0, n) * 0 + fisher(np.clip(np.array(B, dtype=np.float64), -1, 1)) * (1 - 0) if False else None, axis=-1)) if False else None
Pb = np.concatenate([p for _, _, p in stream_profiles(TB, block=64)]).astype(np.float64)
featb, _ = randomized_pca_features(TB, n_components=4, oversample=20, n_iter=1, seed=2, block=64, verbose=False)
exactb = PCA(n_components=4, svd_solver='full').fit(Pb)
proj_exact = exactb.transform(Pb)
# Variance captured by the randomized 4-dim subspace vs the exact one.
Qr, _ = np.linalg.qr(featb.astype(np.float64))
Pc = Pb - Pb.mean(0)
var_r = np.linalg.norm(Pc @ Qr @ Qr.T) ** 2
var_e = np.linalg.norm(Pc @ exactb.components_.T @ exactb.components_) ** 2
featb2, _ = randomized_pca_features(TB, n_components=4, oversample=20, n_iter=2, seed=2, block=64, verbose=False)
Qr2, _ = np.linalg.qr(featb2.astype(np.float64))
var_r2 = np.linalg.norm(Pc @ Qr2 @ Qr2.T) ** 2
check('randomized PCA (k = 24 of 400, q = 1) captures >= 98%% of the exact top-4 variance (%.4f), more with q = 2 (%.4f)'
      % (var_r / var_e, var_r2 / var_e), var_r / var_e >= 0.98 and var_r2 >= var_r)
check('k-means on randomized features recovers the planted blocks',
      adjusted_rand_score(lab, hard_labels(np.eye(5)[__import__('sklearn.cluster', fromlist=['KMeans']).KMeans(5, n_init=5, random_state=0).fit_predict(featb)])) > 0.95)

# ---------------------------------------------------------------- streaming metrics
rng = np.random.RandomState(3)
n = 150
R1 = np.abs(rng.randn(n, n)).astype(np.float32); R1 = (R1 + R1.T) / 2
R2 = np.abs(R1 + 0.3 * rng.randn(n, n)).astype(np.float32); R2 = (R2 + R2.T) / 2
parts = {'a': np.eye(5)[rng.randint(0, 5, n)], 'b': rng.randint(0, 7, n)}
means = stream_block_means(R1, parts, block=40)
ok = True
for name, P in parts.items():
    lab_ = hard_labels(P) if np.ndim(P) == 2 else P
    ok &= np.allclose(means[name][0], block_means(R1, lab_, n_networks=(P.shape[1] if np.ndim(P) == 2 else 7)))
check('stream_block_means equals block_means for soft and hard partitions', ok)
ag = stream_agreements(R2, means, ceiling=R1, block=40)
ok = abs(ag['a']['r2'] - fidelity(R1, R2, parts['a'], 'r2')) < 1e-10
ok &= abs(ag['a']['r'] - fidelity(R1, R2, parts['a'], 'r')) < 1e-10
ok &= abs(ag['(ceiling)']['r2'] - fidelity_ceiling(R1, R2, 'r2')) < 1e-10
ok &= abs(ag['(ceiling)']['r'] - fidelity_ceiling(R1, R2, 'r')) < 1e-10
ins = stream_agreements(R1, means, block=40)
ok &= abs(ins['a']['r2'] - fidelity_insample(R1, parts['a'])) < 1e-10
check('stream_agreements equals fidelity, fidelity_ceiling and fidelity_insample (r2 and r)', ok)
pth2 = os.path.join(tmp, 'r1.h5')
with h5py.File(pth2, 'w') as f:
    f.create_dataset('connectivity', data=R1.astype(np.float16), chunks=(32, n))
    f.create_dataset('unit_strength', data=R1.sum(1)); f.create_dataset('coordinates', data=np.zeros((n, 2), np.int32))
    f.attrs['storage'] = 'tiled_fp16'
T1 = TiledMatrix(pth2)
m16 = stream_block_means(T1, {'a': parts['a']}, block=40)
check('streaming metrics accept a TiledMatrix (fp16 read, tolerance 1e-3)',
      np.allclose(m16['a'][0], means['a'][0], atol=2e-3)
      and abs(fidelity(T1, R2, parts['a'], 'r') - fidelity(R1, R2, parts['a'], 'r')) < 2e-3)

# ---------------------------------------------------------------- parcellation dispatch
PARC = dict(n_networks=4, n_samples=6, clustering='kmeans', binarize_connectivity=False,
            fisher_transform=True, standardize_profiles=True, sparsify_profiles=True,
            connectivity_pca_components=8, pca_whiten=False, store_samples=True,
            parcellate_keys=['halfA', 'halfB'], seed=3, verbose=False)
for tree in ('', '_null'):
    run_parcellation(output_dir=tiled + tree, variant='final', **PARC)
fp = os.path.join(tiled, 'final', 'parcellation', 'parcellation_random_halfA.h5')
d = load_h5_data(fp, verbose=False)
a = read_attrs(fp)
check('tiled parcellation: consensus, splits, samples written with the confirmed arm\'s provenance',
      d['parcellation'].shape == (128, 4) and 'parcellation_split1' in d and d['samples'].shape == (6, 128)
      and str(a.get('out_of_core')) in ('True', '1') and a.get('pca_method', '').startswith('randomized')
      and a.get('sparsify_profiles') in ('True', True, '1') and a.get('consensus') == 'hungarian')
check('tiled parcellation refuses another arm (Ward)',
      raises(lambda: run_parcellation(output_dir=tiled, variant='ward', **dict(PARC, clustering='ward'))))
check('tiled parcellation refuses whitened PCA',
      raises(lambda: run_parcellation(output_dir=tiled, variant='w', **dict(PARC, pca_whiten=True))))
# -D filter: only the named domain's files
run_parcellation(output_dir=tiled, variant='onlyws', domains=['whitespace'], **PARC)
files = sorted(os.listdir(os.path.join(tiled, 'onlyws', 'parcellation')))
check('run_parcellation(domains=...) parcellates only that domain', files == ['parcellation_whitespace_halfA.h5', 'parcellation_whitespace_halfB.h5'])

# ---------------------------------------------------------------- score_big vs dense score
# A dense copy of the SAME tiles with the SAME parcellations, scored the ordinary way.
copy = os.path.join(tmp, 'copy')
for tree in ('', '_null'):
    os.makedirs(os.path.join(copy + tree, CONNECTIVITY_NAME))
    for fn in os.listdir(os.path.join(tiled + tree, CONNECTIVITY_NAME)):
        src = os.path.join(tiled + tree, CONNECTIVITY_NAME, fn)
        with h5py.File(src, 'r') as f:
            data = dict(connectivity=np.asarray(f['connectivity'], dtype=np.float32),
                        coordinates=np.asarray(f['coordinates']), unit_means=np.asarray(f['unit_means']),
                        unit_stds=np.asarray(f['unit_stds']), n_obs=np.asarray(f['n_obs']))
        save_h5_data(data, os.path.join(copy + tree, CONNECTIVITY_NAME, fn), verbose=False)
    shutil.copytree(os.path.join(tiled + tree, 'final'), os.path.join(copy + tree, 'final'))
cfg = dict(seed=7, connectivity=dict(domains=['random', 'whitespace'], null_model='circshift'),
           parcellation_variants={'final': {}}, purge_connectivity=True)
check('tree_is_tiled detects the tiled tree and not the dense copy',
      tree_is_tiled(dict(cfg, output_dir=tiled)) and not tree_is_tiled(dict(cfg, output_dir=copy)))
rows_big, _ = score_config_big(dict(cfg, output_dir=tiled), verbose=False)
rows_small, _ = score_config(dict(cfg, output_dir=copy), verbose=False)
key = lambda r: (r['tree'], r['variant'], r['metric'], r['fit'], r['eval'])
big = {key(r): r['value'] for r in rows_big}
small = {key(r): r['value'] for r in rows_small}
shared = [k for k in big if k in small]
skipped = {k for k in small if k not in big}
ok = len(shared) > 40
worst = 0.0
for k in shared:
    if np.isfinite(big[k]) and np.isfinite(small[k]):
        worst = max(worst, abs(big[k] - small[k]))
        ok &= abs(big[k] - small[k]) < 1e-6
check('score_config_big equals score_config on every shared row (%d rows, worst |diff| %.1e)' % (len(shared), worst), ok)
check('score_config_big omits only the null tree and the co-association rows',
      all(k[0] == 'null' or k[2] in ('reliability_within_coassoc', 'fidelity_across', 'reliability_across') for k in skipped))
check('score_config_big writes ceilings, real, pnull and rand, within and across',
      {k[0] for k in big} == {'real', 'pnull', 'rand'}
      and any(k[2] == 'fidelity_across_halves' and k[1] == '(ceiling)' for k in big)
      and any(k[2] == 'triviality_ami_hubness' and k[0] == 'pnull' for k in big))

# ---------------------------------------------------------------- main.py: -D and purge_null
cfg_path = os.path.join(tmp, 'tiled.yml')
yaml.safe_dump(dict(cfg, output_dir=tiled), open(cfg_path, 'w'))
r = subprocess.run([sys.executable, '-m', 'parcelmate.bin.main', cfg_path, '-s', 'purge_null_connectivity', '-D', 'random'],
                   cwd=ROOT, env=ENV, capture_output=True, text=True)
left = sorted(os.listdir(os.path.join(tiled + '_null', CONNECTIVITY_NAME)))
check('purge_null_connectivity -D random removes only that domain\'s null tiles and leaves a manifest',
      r.returncode == 0 and left == ['connectivity_whitespace_halfA.h5', 'connectivity_whitespace_halfB.h5']
      and 'connectivity_random_halfA.h5' in open(os.path.join(tiled + '_null', CONNECTIVITY_NAME + '_purged.txt')).read())
shutil.rmtree(os.path.join(tiled + '_null', 'final'))
r = subprocess.run([sys.executable, '-m', 'parcelmate.bin.main', cfg_path, '-s', 'purge_null_connectivity', '-D', 'whitespace'],
                   cwd=ROOT, env=ENV, capture_output=True, text=True)
check('purge_null_connectivity refuses when the null parcellations are missing',
      r.returncode != 0 and 'null parcellations missing' in r.stderr
      and len(os.listdir(os.path.join(tiled + '_null', CONNECTIVITY_NAME))) == 2)
r = subprocess.run([sys.executable, '-m', 'parcelmate.bin.main', cfg_path, '-s', 'connectivity', '-D', 'nosuch'],
                   cwd=ROOT, env=ENV, capture_output=True, text=True)
check('-D with an unknown domain is refused', r.returncode != 0 and 'unknown domain' in r.stderr)

# ---------------------------------------------------------------- configs and launcher
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import make_qwen_configs as mqc
from inspect import signature
sig_c, sig_p = signature(run_connectivity), signature(run_parcellation)
ok = True
for name, spec in mqc.MODELS.items():
    c = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'qwen35', name + '.yml')))
    ok &= all(k in sig_c.parameters for k in c['connectivity'])
    ok &= all(k in sig_p.parameters for k in c['parcellation'])
    ok &= c['connectivity']['storage'] == 'tiled_fp16' and c['connectivity']['outputs'] == ['halves']
    ok &= c['connectivity']['units_per_layer'] is None and c['purge_connectivity'] is True
    ok &= c['parcellation']['n_samples'] == 200 and c['parcellation']['n_networks'] == 100
    ok &= c['parcellation']['sparsify_profiles'] and c['parcellation']['connectivity_pca_components'] == 100
    ok &= c['connectivity']['model_name'] == spec['model'] and c['connectivity']['domains'] == spec['domains']
check('configs/qwen35: both configs bind to the pipeline and are the confirmed arm, tiled, on all units', ok)
launcher = open(os.path.join(ROOT, 'scripts', 'launch_qwen.sh')).read()
check('launcher: chains connectivity -> parcellation -> purge_null per domain, then score + purge',
      'purge_null_connectivity' in launcher and 'score purge_connectivity' in launcher and 'umask 002' in launcher)
from parcelmate.bin.make_jobs import get_job
job = get_job('configs/qwen35/qwen3.5-4b.yml', dict(time=1, n_cores=1, memory=4, workdir=None, log_dir='logs',
              python='python', conda_sh=None, conda_env=None, account=None, partition=None, qos=None, gpu=0,
              gpu_type=None, constraint=None, exclude=None, env={}), steps=['connectivity'], domains=['agnews'])
check('make_jobs -D: job name carries the domain, command carries -D, script sets umask 002',
      'qwen3.5-4b.connectivity.agnews' in job and ' -D agnews' in job and 'umask 002' in job)

shutil.rmtree(tmp)
print('\n%d checks, %d failure(s)' % (n_checks[0], len(failures)))
for f in failures:
    print('  FAIL', f)
sys.exit(1 if failures else 0)
