"""Verification for the Iteration 22 additions (Pythia training dynamics, all MLP units).

    PYTHONPATH=. python tests/verify_iter12_pythia.py

Covers: the architecture-aware MLP hook on a GPT-NeoX model (values equal the post-GELU
activations, all neurons when `units_per_layer` is null, loud failure on an unknown
architecture); `revision` reaching the loader and the provenance; `outputs: [halves]` in
`run_connectivity` (halves equal to the ones `run_split_halves` builds from per-sample files,
in both trees, no sample or avg files written, skip-on-cache, the guards); the streaming
fidelity metrics equal to the direct ones; the scorer on a halves-only tree; the
`purge_connectivity` step (refusals, the manifest); `score_checkpoints` on two fake steps;
the generated configs (every one binds to `run_connectivity`/`run_parcellation` and is the
confirmed arm's settings on all units); and the launcher's step list.
"""

import csv
import inspect
import os
import re
import subprocess
import sys
import tempfile

# CPU only: the tiled GPU correlation accumulates in a nondeterministic order, and this
# suite compares two independent connectivity runs entry by entry.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer, GPTNeoXConfig, GPTNeoXModel, GPT2Config, GPT2Model

from parcelmate.bin.score import score_config
from parcelmate.bin.score_checkpoints import concat_scores, score_checkpoints
from parcelmate.constants import CONNECTIVITY_NAME, HALF_NAMES
from parcelmate.metrics import (
    agreement, block_means, fidelity, fidelity_ceiling, fidelity_insample, hard_labels,
    pair_agreement, reliability,
)
from parcelmate.model import (
    get_model_and_tokenizer, get_timecourses, mlp_projections, run_connectivity,
    run_parcellation, run_split_halves,
)
from parcelmate.util import load_h5_data, read_attrs

failures = []
n_checks = [0]
ENV = dict(os.environ, PYTHONPATH='.')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')


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
np.random.seed(0)

# ---------------------------------------------------------------- MLP hook on GPT-NeoX
# The GPT-2 vocabulary, so the saved copy below can be driven by the GPT-2 tokenizer.
cfg = GPTNeoXConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                    intermediate_size=64, vocab_size=50304, max_position_embeddings=64)
neox = GPTNeoXModel(cfg).eval()
ids = torch.randint(0, 100, (4, 16))
mask = torch.ones_like(ids)
check('mlp_projections: one output projection per GPT-NeoX layer',
      [l for l, _ in mlp_projections(neox)] == [0, 1]
      and all(m is neox.layers[l].mlp.dense_4h_to_h for l, m in mlp_projections(neox)))
gpt2 = GPT2Model(GPT2Config(n_layer=2, n_embd=16, n_head=2, vocab_size=100, n_positions=32))
check('mlp_projections: GPT-2 still resolves to c_proj',
      all(m is gpt2.h[l].mlp.c_proj for l, m in mlp_projections(gpt2)))
check('mlp_projections: an unknown architecture fails loudly',
      raises(lambda: mlp_projections(torch.nn.Linear(2, 2))))

out = get_timecourses(neox, ids, mask, batch_size=2, unit_type='mlp', verbose=False)
acts = {}
hooks = [blk.mlp.act.register_forward_hook(lambda m, i, o, l=l: acts.__setitem__(l, o.detach()))
         for l, blk in enumerate(neox.layers)]
with torch.no_grad():
    neox(input_ids=ids, attention_mask=mask)
for h in hooks:
    h.remove()
ref = np.concatenate([acts[l].reshape(-1, 64).numpy().T for l in range(2)])
check('units_per_layer null: ALL 2 x 64 neurons are units', out['timecourses'].shape == (128, 64))
check('GPT-NeoX mlp units equal the post-GELU activations',
      np.allclose(ref, out['timecourses'], atol=1e-5))
check('coordinates: layer and original neuron index',
      out['coordinates'][0].tolist() == [0, 0] and out['coordinates'][-1].tolist() == [1, 63])

# ---------------------------------------------------------------- loader with revision
# Under the repository, not /tmp: the local /tmp has a small quota that this suite's copies
# of the trees exhausted once.
os.makedirs(os.path.join(HERE, '.tmp'), exist_ok=True)
tmp = tempfile.mkdtemp(dir=os.path.join(HERE, '.tmp'))
model_dir = os.path.join(tmp, 'tiny-neox')
neox.save_pretrained(model_dir)
AutoTokenizer.from_pretrained('gpt2').save_pretrained(model_dir)
m, tok = get_model_and_tokenizer(model_dir)
check('get_model_and_tokenizer: loads a GPT-NeoX checkpoint from a path, float32',
      type(m).__name__ == 'GPTNeoXModel' and next(m.parameters()).dtype == torch.float32)
check('get_model_and_tokenizer: accepts revision',
      'revision' in inspect.signature(get_model_and_tokenizer).parameters
      and 'revision' in inspect.signature(run_connectivity).parameters)
import parcelmate.model as pm
seen = {}
_orig = pm.AutoModel.from_pretrained
pm.AutoModel.from_pretrained = classmethod(lambda cls, name, **kw: (seen.update(kw), _orig(name))[1])
get_model_and_tokenizer(model_dir, revision='step999')
pm.AutoModel.from_pretrained = _orig
check('get_model_and_tokenizer: revision is forwarded to from_pretrained',
      seen.get('revision') == 'step999')

# ---------------------------------------------------------------- halves-only connectivity
COMMON = dict(model_name=model_dir, n_samples=4, domains=('random', 'whitespace'), seq_len=32,
              n_tokens=256, batch_size=4, unit_type='mlp', units_per_layer=None,
              null_model='circshift', seed=7, verbose=False)
legacy = os.path.join(tmp, 'legacy')
halves = os.path.join(tmp, 'halves')
run_connectivity(output_dir=legacy, null_output_dir=legacy + '_null', **COMMON)
run_split_halves(output_dir=legacy, verbose=False)
run_split_halves(output_dir=legacy + '_null', verbose=False)
run_connectivity(output_dir=halves, null_output_dir=halves + '_null', outputs=['halves'], **COMMON)

for tree in ('', '_null'):
    files = sorted(os.listdir(os.path.join(halves + tree, CONNECTIVITY_NAME)))
    check('halves%s: exactly the two halves per domain written, no sample or avg files' % tree,
          files == sorted('%s_%s_%s.h5' % (CONNECTIVITY_NAME, d, h)
                          for d in ('random', 'whitespace') for h in HALF_NAMES))
    ok = True
    for d in ('random', 'whitespace'):
        for h in HALF_NAMES:
            a = load_h5_data(os.path.join(legacy + tree, CONNECTIVITY_NAME,
                                          '%s_%s_%s.h5' % (CONNECTIVITY_NAME, d, h)), verbose=False)
            b = load_h5_data(os.path.join(halves + tree, CONNECTIVITY_NAME,
                                          '%s_%s_%s.h5' % (CONNECTIVITY_NAME, d, h)), verbose=False)
            # equal_nan: a constant unit (a dead neuron of the random model) has an undefined
            # correlation, and both paths leave it NaN.
            ok &= np.allclose(a['connectivity'], b['connectivity'], atol=1e-5, equal_nan=True)
            ok &= np.allclose(a['unit_means'], b['unit_means']) and np.allclose(a['unit_stds'], b['unit_stds'])
            ok &= int(a['n_obs']) == int(b['n_obs']) and np.array_equal(a['coordinates'], b['coordinates'])
    check('halves%s: equal to run_split_halves over the per-sample files' % tree, ok)
attrs = read_attrs(os.path.join(halves, CONNECTIVITY_NAME, 'connectivity_random_halfA.h5'))
check('halves: provenance carries model_name, revision, unit_type, n_units and the samples',
      attrs.get('model_name') == model_dir and attrs.get('revision') == ''
      and attrs.get('unit_type') == 'mlp' and int(attrs.get('n_units')) == 128
      and attrs.get('sources') == 'sample1, sample2')
mtime = os.path.getmtime(os.path.join(halves, CONNECTIVITY_NAME, 'connectivity_random_halfA.h5'))
run_connectivity(output_dir=halves, null_output_dir=halves + '_null', outputs=['halves'], **COMMON)
check('halves: a second run skips a domain whose halves exist',
      os.path.getmtime(os.path.join(halves, CONNECTIVITY_NAME, 'connectivity_random_halfA.h5')) == mtime)
check('halves: odd n_samples refused',
      raises(lambda: run_connectivity(output_dir=os.path.join(tmp, 'x'), outputs=['halves'],
                                      **dict(COMMON, n_samples=3, null_model=None))))
# The legacy layout without a null tree still writes its per-sample files (the null-file
# write is guarded by null_model, which a restructuring of the loop once dropped).
nonull = os.path.join(tmp, 'nonull')
run_connectivity(output_dir=nonull, **dict(COMMON, null_model=None, domains=('random',)))
check('legacy outputs without a null: sample and avg files written, nothing else',
      sorted(os.listdir(os.path.join(nonull, CONNECTIVITY_NAME)))
      == ['connectivity_random_avg.h5'] + ['connectivity_random_sample%d.h5' % i for i in (1, 2, 3, 4)])
check('halves: surrogates refused',
      raises(lambda: run_connectivity(output_dir=os.path.join(tmp, 'x'), outputs=['halves'],
                                      n_surrogates=2, **COMMON)))
check('outputs: an unknown name is refused',
      raises(lambda: run_connectivity(output_dir=os.path.join(tmp, 'x'), outputs=['foo'], **COMMON)))
both = os.path.join(tmp, 'both')
run_connectivity(output_dir=both, null_output_dir=both + '_null', outputs=['halves', 'avg'], **COMMON)
a = load_h5_data(os.path.join(legacy, CONNECTIVITY_NAME, 'connectivity_random_avg.h5'), verbose=False)
b = load_h5_data(os.path.join(both, CONNECTIVITY_NAME, 'connectivity_random_avg.h5'), verbose=False)
check('outputs: [halves, avg] writes an avg equal to the legacy Fisher mean',
      np.allclose(a['connectivity'], b['connectivity'], atol=1e-5, equal_nan=True)
      and not os.path.exists(os.path.join(both, CONNECTIVITY_NAME, 'connectivity_random_sample1.h5')))

# ---------------------------------------------------------------- streaming metrics
rng = np.random.RandomState(1)
n = 200
R1 = np.abs(rng.randn(n, n)).astype(np.float32); R1 = (R1 + R1.T) / 2
R2 = (R1 + 0.3 * np.abs(rng.randn(n, n))).astype(np.float32); R2 = (R2 + R2.T) / 2
P = np.eye(6)[rng.randint(0, 6, n)]
lab = hard_labels(P)
M = block_means(R1, lab, 6)
M_direct = None
Rc = np.array(R1, dtype=np.float64); np.fill_diagonal(Rc, 0)
onehot = np.eye(6)[lab]; sizes = onehot.sum(0)
M_direct = (onehot.T @ Rc @ onehot) / np.maximum(np.outer(sizes, sizes) - np.diag(sizes), 1)
check('block_means: blockwise equals the direct computation', np.allclose(M, M_direct, atol=1e-12))
ok = True
for measure in ('r2', 'r'):
    ok &= abs(fidelity(R1, R2, P, measure) - agreement(R2, M[lab][:, lab], measure)) < 1e-10
    ok &= abs(fidelity_ceiling(R1, R2, measure) - agreement(R2, R1, measure)) < 1e-10
    ok &= abs(fidelity_insample(R1, P, measure) - agreement(R1, M[lab][:, lab], measure)) < 1e-10
    ok &= abs(pair_agreement(R2, lambda s, e: R1[s:e], measure, block=7)
              - agreement(R2, R1, measure)) < 1e-10
check('fidelity, ceiling, insample: streaming equals direct (r2 and r, odd block sizes)', ok)

# ---------------------------------------------------------------- parcellation + score on halves only
PARC = dict(n_networks=4, n_samples=6, clustering='kmeans', binarize_connectivity=False,
            fisher_transform=True, standardize_profiles=True, sparsify_profiles=True,
            connectivity_pca_components=8, pca_whiten=False, store_samples=True,
            parcellate_keys=['halfA', 'halfB'], seed=3, verbose=False)
for tree in ('', '_null'):
    run_parcellation(output_dir=halves + tree, variant='final', **PARC)
files = sorted(os.listdir(os.path.join(halves, 'final', 'parcellation')))
check('parcellate_keys halves: only the halves are parcellated', len(files) == 4
      and all('half' in f for f in files))
score_cfg = dict(output_dir=halves, seed=7,
                 connectivity=dict(domains=['random', 'whitespace'], null_model='circshift'),
                 parcellation_variants={'final': {}})
rows, path = score_config(score_cfg, verbose=False)
metrics = {r['metric'] for r in rows}
check('score on a halves-only tree: writes, with across-domain on halves and no avg rows',
      os.path.exists(path) and 'fidelity_across_halves' in metrics
      and 'reliability_across_halves' in metrics and 'fidelity_across' not in metrics
      and 'reliability_across' not in metrics)
check('score on a halves-only tree: pnull and rand references present',
      {r['tree'] for r in rows} == {'real', 'null', 'pnull', 'rand'})

# ---------------------------------------------------------------- purge step
cfg_path = os.path.join(tmp, 'halves.yml')
yaml.safe_dump(dict(score_cfg, purge_connectivity=True), open(cfg_path, 'w'))
cfg_nopurge = os.path.join(tmp, 'nopurge.yml')
yaml.safe_dump(score_cfg, open(cfg_nopurge, 'w'))
r = subprocess.run([sys.executable, '-m', 'parcelmate.bin.main', cfg_nopurge, '-s', 'purge_connectivity'],
                   cwd=ROOT, env=ENV, capture_output=True, text=True)
check('purge: refused without purge_connectivity: true in the config',
      r.returncode != 0 and 'purge_connectivity: true' in r.stderr
      and os.path.isdir(os.path.join(halves, CONNECTIVITY_NAME)))
os.rename(path, path + '.bak')
r = subprocess.run([sys.executable, '-m', 'parcelmate.bin.main', cfg_path, '-s', 'purge_connectivity'],
                   cwd=ROOT, env=ENV, capture_output=True, text=True)
check('purge: refused while scores.csv is absent',
      r.returncode != 0 and 'not scored' in r.stderr and os.path.isdir(os.path.join(halves, CONNECTIVITY_NAME)))
os.rename(path + '.bak', path)
r = subprocess.run([sys.executable, '-m', 'parcelmate.bin.main', cfg_path, '-s', 'purge_connectivity'],
                   cwd=ROOT, env=ENV, capture_output=True, text=True)
manifest = os.path.join(halves, CONNECTIVITY_NAME + '_purged.txt')
check('purge: both trees\' connectivity removed, parcellations kept, manifest lists the files',
      r.returncode == 0
      and not os.path.isdir(os.path.join(halves, CONNECTIVITY_NAME))
      and not os.path.isdir(os.path.join(halves + '_null', CONNECTIVITY_NAME))
      and os.path.isdir(os.path.join(halves, 'final', 'parcellation'))
      and os.path.exists(manifest) and 'connectivity_random_halfA.h5' in open(manifest).read()
      and os.path.exists(os.path.join(halves + '_null', CONNECTIVITY_NAME + '_purged.txt')))
# Static, because running `-s all` on this config would run GPT-2 connectivity at full
# scale (the config has no model_name, and the defaults are the real pipeline's).
main_src = open(os.path.join(ROOT, 'parcelmate', 'bin', 'main.py')).read()
check('purge: never part of -s all',
      "if 'purge_connectivity' in steps:" in main_src
      and "'all' in steps or 'purge_connectivity' in steps" not in main_src)

# ---------------------------------------------------------------- cross-checkpoint scorer
import shutil
model_root = os.path.join(tmp, 'pythia-tiny')
for step, src in ((0, both), (1000, halves)):
    for tree in ('', '_null'):
        dst = os.path.join(model_root, 'step%d%s' % (step, tree))
        shutil.copytree(os.path.join(src + tree), dst) if os.path.isdir(src + tree) else None
# step0 needs parcellations and a score file too. Its source tree carries avg files, so it
# is parcellated with the default keys (avg and halves) for the scorer's avg pass.
for tree in ('', '_null'):
    run_parcellation(output_dir=os.path.join(model_root, 'step0' + tree), variant='final',
                     **dict(PARC, seed=4, parcellate_keys=None))
score_config(dict(score_cfg, output_dir=os.path.join(model_root, 'step0')), verbose=False)
rows, missing = score_checkpoints(model_root, [0, 1000], 'final', ['random', 'whitespace'], verbose=False)
check('score_checkpoints: no missing inputs on a complete pair of steps', not missing)
real = [r for r in rows if r['tree'] == 'real']
check('score_checkpoints: consecutive and to-final rows for every domain and half, real and pnull',
      len(real) == 2 * 2 * 2 and len(rows) == 2 * len(real)
      and {r['metric'] for r in rows} == {'ari_consecutive', 'ari_to_final'})
a = load_h5_data(os.path.join(model_root, 'step0', 'final', 'parcellation',
                              'parcellation_random_halfA.h5'), verbose=False)['parcellation']
b = load_h5_data(os.path.join(model_root, 'step1000', 'final', 'parcellation',
                              'parcellation_random_halfA.h5'), verbose=False)['parcellation']
v = [r['value'] for r in real if r['domain'] == 'random' and r['key'] == 'halfA'
     and r['metric'] == 'ari_consecutive'][0]
check('score_checkpoints: value equals the ARI of the two steps\' partitions',
      abs(v - reliability(a, b)) < 1e-12)
all_rows, miss = concat_scores(model_root, [0, 1000])
check('concat_scores: both steps, with model and step columns',
      not miss and {r['step'] for r in all_rows} == {0, 1000}
      and all(r['model'] == 'pythia-tiny' for r in all_rows))
rows, missing = score_checkpoints(model_root, [0, 1000, 4000], 'final', ['random'], verbose=False)
check('score_checkpoints: a missing step is reported, not skipped silently',
      any('step4000' in m for m in missing))

# ---------------------------------------------------------------- configs and launcher
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import make_pythia_configs as mpc
paths = sorted(os.listdir(os.path.join(ROOT, 'configs', 'pythia')))
check('configs/pythia: 2 sizes x 11 steps = 22 files', len(paths) == 22)
confirm = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'last_mlp.yml')))
confirm_parc = dict(confirm['parcellation'])
confirm_parc.update(confirm['parcellation_variants']['vmf_sparse_pca100_lloyd100_n200'])
sig_c = inspect.signature(run_connectivity)
sig_p = inspect.signature(run_parcellation)
ok = True
for p in paths:
    c = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'pythia', p)))
    ok &= c['purge_connectivity'] is True and c['seed'] == 42
    ok &= all(k in sig_c.parameters for k in c['connectivity'])
    ok &= all(k in sig_p.parameters for k in c['parcellation'])
    ok &= c['connectivity']['units_per_layer'] is None and c['connectivity']['unit_type'] == 'mlp'
    ok &= c['connectivity']['outputs'] == ['halves'] and c['connectivity']['null_model'] == 'circshift'
    ok &= c['parcellation']['parcellate_keys'] == ['halfA', 'halfB']
    ok &= c['parcellation_variants'] == {'final': {}}
    # The confirmed arm's settings, wherever both configs set a key.
    for k, v in confirm_parc.items():
        if k in c['parcellation']:
            ok &= c['parcellation'][k] == v
    # Same data settings as the GPT-2 MLP runs, apart from the model and the units.
    for k in ('n_samples', 'domains', 'seq_len', 'take', 'batch_size', 'step', 'eps', 'n_surrogates'):
        ok &= c['connectivity'][k] == confirm['connectivity'][k]
    m = re.match(r'pythia-(\d+m)_step(\d+)\.yml', p)
    ok &= c['connectivity']['model_name'] == 'EleutherAI/pythia-%s' % m.group(1)
    ok &= c['connectivity']['revision'] == 'step%s' % m.group(2)
    ok &= c['output_dir'] == 'results/pythia/pythia-%s/step%s' % (m.group(1), m.group(2))
check('configs/pythia: every config binds and is the confirmed arm on all units', ok)
check('configs/pythia: steps are step 0 plus ten checkpoints spaced ~4x, all Pythia saves',
      mpc.STEPS == [0, 1, 4, 16, 64, 256, 1000, 4000, 16000, 64000, 143000]
      and all(s in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512) or s % 1000 == 0 for s in mpc.STEPS))
launcher = open(os.path.join(ROOT, 'scripts', 'launch_pythia.sh')).read()
steps_line = re.search(r'^STEPS="([^"]+)"', launcher, re.M).group(1).split()
check('launcher: STEPS equal the generator\'s', [int(s) for s in steps_line] == mpc.STEPS)
check('launcher: SIZES equal the generator\'s',
      re.search(r'^SIZES="([^"]+)"', launcher, re.M).group(1).split() == list(mpc.SIZES))

shutil.rmtree(tmp)
print('\n%d checks, %d failure(s)' % (n_checks[0], len(failures)))
for f in failures:
    print('  FAIL', f)
sys.exit(1 if failures else 0)
