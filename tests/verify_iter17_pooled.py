"""Verification for the Iteration 31 additions (pooled out-of-core connectivity).

    PYTHONPATH=. python tests/verify_iter17_pooled.py

CPU only, the tiny GPT-NeoX of the Iteration 25 suite. Covers: `write_tiled_pooled` on two
"domains" of DIFFERENT lengths equals a numpy reference built from the same z-scored
samples (each sample correlated with its own token count, Fisher mean over all samples of
the half, the null rolled per sample under the single-domain seed keys), in both trees;
the pooled provenance; `run_connectivity(pool_as=...)` end to end writes only the pooled
halves, skips when they exist, and refuses dense storage and bad names; the parcellation
file pattern accepts the pseudo-domain; the generated pooled configs and the launcher.
"""

import os
import shutil
import sys
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import h5py
import numpy as np
import torch
import yaml
from transformers import AutoTokenizer, GPTNeoXConfig, GPTNeoXModel

from parcelmate.bigconn import TiledMatrix, null_offsets, write_tiled_pooled, zscored_timecourses
from parcelmate.constants import CONNECTIVITY_NAME, HALF_NAMES, INPUT_NAME_RE
from parcelmate.model import get_model_and_tokenizer, run_connectivity
from parcelmate.util import derive_seed, read_attrs

failures = []
n_checks = [0]
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
os.makedirs(os.path.join(HERE, '.tmp'), exist_ok=True)
tmp = tempfile.mkdtemp(dir=os.path.join(HERE, '.tmp'))
model_dir = os.path.join(tmp, 'tiny-neox')
GPTNeoXModel(GPTNeoXConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=64, vocab_size=50304, max_position_embeddings=64)
             ).save_pretrained(model_dir)
AutoTokenizer.from_pretrained('gpt2').save_pretrained(model_dir)
model, _ = get_model_and_tokenizer(model_dir)

# ---------------------------------------------------------------- pooled writer vs numpy
rs = np.random.RandomState(0)
seq = 32
lengths = {'alpha': 8, 'beta': 6}          # sequences per domain: 2 samples of 4 and of 3
parts = []
for d, n_seq in lengths.items():
    ids = torch.as_tensor(rs.randint(0, 50000, size=(n_seq, seq)))
    parts.append((d, ids, torch.ones_like(ids)))
seed = 11
cdir, ndir = os.path.join(tmp, 'w', 'conn'), os.path.join(tmp, 'w_null', 'conn')
os.makedirs(cdir)
os.makedirs(ndir)
write_tiled_pooled(model, parts, 2, 'pooled', cdir, ndir, seed, batch_size=2, verbose=False)
ok, worst = True, 0.0
for h, half in enumerate(HALF_NAMES):
    zs, offs = [], []
    for d, ids, mask in parts:
        n = len(ids) // 2
        tc = zscored_timecourses(model, ids[h * n:(h + 1) * n], mask[h * n:(h + 1) * n], batch_size=2, verbose=False)
        z = tc['z'].astype(np.float64)
        zs.append(z)
        offs.append(null_offsets(z.shape[0], z.shape[1], derive_seed(seed, 'null', d, h + 1)))
    ok &= zs[0].shape[1] != zs[1].shape[1]              # the two samples differ in length
    eps = 1 - 1e-3
    for tree, dir_, rolled in (('real', cdir, zs),
                               ('null', ndir, [np.stack([np.roll(z[i], o[i]) for i in range(len(z))]) for z, o in zip(zs, offs)])):
        ref = np.tanh(np.mean([np.arctanh(np.clip(z @ z.T / z.shape[1] * eps, -eps, eps)) for z in rolled], 0))
        T = TiledMatrix(os.path.join(dir_, '%s_pooled_%s.h5' % (CONNECTIVITY_NAME, half)))
        got = T[0:T.shape[0]]
        worst = max(worst, float(np.abs(np.abs(ref) - got).max()))
        ok &= np.allclose(np.abs(ref), got, atol=3e-3)
        T.close()
check('pooled halves equal the numpy reference in both trees, samples of unequal length (worst |diff| %.1e)' % worst, ok)
a = read_attrs(os.path.join(cdir, '%s_pooled_halfB.h5' % CONNECTIVITY_NAME))
check('pooled provenance: pseudo-domain, member domains, per-domain sample names, token counts',
      a.get('domain') == 'pooled' and a.get('pooled_domains') == 'alpha, beta'
      and a.get('sources') == 'alpha:sample2, beta:sample2' and a.get('sample_tokens') == '128, 96'
      and int(a.get('n_obs')) == 224)
check('only the pooled halves are written, no per-domain file',
      sorted(os.listdir(cdir)) == ['%s_pooled_%s.h5' % (CONNECTIVITY_NAME, h) for h in HALF_NAMES])

# ---------------------------------------------------------------- run_connectivity(pool_as=...)
COMMON = dict(model_name=model_dir, n_samples=2, domains=('random', 'whitespace'), seq_len=32,
              n_tokens=128, batch_size=4, unit_type='mlp', units_per_layer=None, null_model='circshift',
              seed=7, verbose=False, outputs=['halves'], storage='tiled_fp16')
out = os.path.join(tmp, 'pool')
run_connectivity(output_dir=out, null_output_dir=out + '_null', pool_as='pooled', **COMMON)
files = sorted(os.listdir(os.path.join(out, CONNECTIVITY_NAME)))
check('run_connectivity(pool_as): the two pooled halves in each tree and nothing else',
      files == ['%s_pooled_%s.h5' % (CONNECTIVITY_NAME, h) for h in HALF_NAMES]
      and sorted(os.listdir(os.path.join(out + '_null', CONNECTIVITY_NAME))) == files)
a = read_attrs(os.path.join(out, CONNECTIVITY_NAME, files[0]))
check('run_connectivity(pool_as): each domain contributes one sample per half',
      a.get('pooled_domains') == 'random, whitespace' and a.get('sources') == 'random:sample1, whitespace:sample1'
      and a.get('storage') == 'tiled_fp16')
mtime = os.path.getmtime(os.path.join(out, CONNECTIVITY_NAME, files[0]))
run_connectivity(output_dir=out, null_output_dir=out + '_null', pool_as='pooled', **COMMON)
check('run_connectivity(pool_as): skips when the pooled halves exist',
      os.path.getmtime(os.path.join(out, CONNECTIVITY_NAME, files[0])) == mtime)
check('pool_as refuses dense storage, underscores, and a name equal to a member domain',
      raises(lambda: run_connectivity(output_dir=os.path.join(tmp, 'x'), null_output_dir=os.path.join(tmp, 'x_null'),
                                      pool_as='pooled', **dict(COMMON, storage='dense')))
      and raises(lambda: run_connectivity(output_dir=os.path.join(tmp, 'y'), null_output_dir=os.path.join(tmp, 'y_null'),
                                          pool_as='all_five', **COMMON))
      and raises(lambda: run_connectivity(output_dir=os.path.join(tmp, 'z'), null_output_dir=os.path.join(tmp, 'z_null'),
                                          pool_as='random', **COMMON)))
m = INPUT_NAME_RE.match('%s_pooled_halfA.h5' % CONNECTIVITY_NAME)
check('the parcellation file pattern reads the pseudo-domain and key', m is not None and m.group(2) == 'pooled' and m.group(3) == 'halfA')

# ---------------------------------------------------------------- configs and launcher
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import make_qwen_configs as mqc
ok = True
for name in ('qwen3.5-2b-pool5', 'qwen3.5-4b-pool5'):
    c = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'qwen35', name + '.yml')))
    cc = c['connectivity']
    ok &= cc['pool_as'] == 'pooled' and c['score']['domains'] == ['pooled']
    ok &= cc['domains'] == ['wikitext', 'bookcorpus', 'agnews', 'tldr17', 'codeparrot']
    ok &= cc['n_samples'] == 2 and cc['n_tokens'] == 40960 and cc['storage'] == 'tiled_fp16'
    ok &= c['output_dir'] == 'results/qwen35/%s' % name
    ok &= c['parcellation']['n_networks'] == 100 and c['parcellation']['sparsify_profiles']
c4 = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'qwen35', 'qwen3.5-4b.yml')))
ok &= c4['connectivity']['domains'] == ['wikitext'] and 'pool_as' not in c4['connectivity']
check('configs: pooled 2B and 4B over five domains in their own trees; 4B single-domain is wikitext only', ok)
launcher = open(os.path.join(ROOT, 'scripts', 'launch_qwen.sh')).read()
from parcelmate.bin.make_jobs import get_job
job = get_job('configs/qwen35/qwen3.5-4b-pool5.yml', dict(time=1, n_cores=1, memory=4, workdir=None, log_dir='logs',
              python='python', conda_sh=None, conda_env=None, account=None, partition=None, qos=None, gpu=0,
              gpu_type=None, constraint=None, exclude=None, env={}), steps=['connectivity'])
check('launcher: pooled configs get one connectivity job without -D, parcellation on the pseudo-domain',
      'pool_of' in launcher and 'job_domains_of' in launcher and 'qwen3.5-4b-pool5.connectivity' in job
      and ' -D ' not in job)

shutil.rmtree(tmp)
print('\n%d checks, %d failure(s)' % (n_checks[0], len(failures)))
for f in failures:
    print('  FAIL', f)
sys.exit(1 if failures else 0)
