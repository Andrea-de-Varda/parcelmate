"""Verification for the Iteration 32 additions (word-shuffled connectomes).

    PYTHONPATH=. python tests/verify_iter18_shuffle.py

CPU only. Covers: the word-start table (whitespace-initial tokens) on the GPT-2 tokenizer,
which Pythia's byte-level BPE shares in this respect; `shuffle_words_within` keeps every
row's token multiset and every word intact, changes the order, is deterministic under its
seed and handles a row that begins mid-word; `get_dataset(shuffle_words=True)` keeps the
windows' token multisets and masks and changes only the order; `run_connectivity` records
the shuffle in the provenance; `connectome_consistency` equals a numpy reference (Pearson r
of the upper triangles of |r|), writes one row per domain pair and purges; the generated
configs differ from the originals only where intended; the launcher.
"""

import csv
import glob
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
from transformers import AutoTokenizer, GPTNeoXConfig, GPTNeoXModel

from parcelmate.data import get_dataset, shuffle_words_within, word_start_table
from parcelmate.model import run_connectivity
from parcelmate.util import read_attrs

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


os.makedirs(os.path.join(HERE, '.tmp'), exist_ok=True)
tmp = tempfile.mkdtemp(dir=os.path.join(HERE, '.tmp'))
tok = AutoTokenizer.from_pretrained('gpt2')

# ---------------------------------------------------------------- word units
starts = word_start_table(tok)
ids = tok(' the cat sat on the unbelievably big mat.')['input_ids']
check('word-start table: whitespace-initial tokens start words, continuations do not',
      len(starts) == len(tok) and starts[tok(' the')['input_ids'][0]] and not starts[tok('.')['input_ids'][0]]
      and word_start_table(tok) is starts)
text = (' dog' + ' The quick brown fox jumps over the lazy dog, unbelievably fast; antidisestablishment rules.' * 6)
row = tok(text)['input_ids']
out = shuffle_words_within([row], starts, 3)[0]
words = lambda r: sorted(tok.decode(r).split())
check('shuffle keeps the token multiset and every word intact, and changes the order',
      sorted(out) == sorted(row) and words(out) == words(row) and out != row)
check('shuffle is deterministic under its seed and differs across seeds',
      shuffle_words_within([row], starts, 3)[0] == out and shuffle_words_within([row], starts, 4)[0] != out)
check('a row that begins mid-word keeps its leading fragment as one unit (tokens and multiset kept)',
      sorted(shuffle_words_within([row[1:]], starts, 5)[0]) == sorted(row[1:]))

# ---------------------------------------------------------------- get_dataset
kw = dict(dataset='random', tokenizer=tok, n_tokens=512, seq_len=64, seed=11, verbose=False)
a_ids, a_mask = get_dataset(**kw)
b_ids, b_mask = get_dataset(shuffle_words=True, **kw)
check('get_dataset(shuffle_words): same windows, masks and per-window token multisets, different order',
      a_ids.shape == b_ids.shape and torch.equal(a_mask, b_mask)
      and all(sorted(x.tolist()) == sorted(y.tolist()) for x, y in zip(a_ids, b_ids))
      and not torch.equal(a_ids, b_ids))

# ---------------------------------------------------------------- connectivity and consistency
model_dir = os.path.join(tmp, 'tiny-neox')
torch.manual_seed(0)
GPTNeoXModel(GPTNeoXConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                           vocab_size=50304, max_position_embeddings=64)).save_pretrained(model_dir)
tok.save_pretrained(model_dir)
out_dir = os.path.join(tmp, 'shuf')
cfg = dict(output_dir=out_dir, seed=7, connectivity=dict(
    model_name=model_dir, n_samples=4, domains=['random', 'whitespace'], seq_len=32, n_tokens=256, batch_size=4,
    unit_type='mlp', units_per_layer=None, null_model=None, outputs=['halves'], data_kwargs=dict(shuffle_words=True)))
cfg_path = os.path.join(tmp, 'shuf.yml')
yaml.safe_dump(cfg, open(cfg_path, 'w'))
r = subprocess.run([sys.executable, '-m', 'parcelmate.bin.main', cfg_path, '-s', 'connectivity'],
                   cwd=ROOT, env=ENV, capture_output=True, text=True)
conn = os.path.join(out_dir, 'connectivity')
check('run_connectivity with shuffle_words writes the halves and records the shuffle',
      r.returncode == 0 and read_attrs(os.path.join(conn, 'connectivity_random_halfA.h5')).get('shuffle_words')
      == 'within each seq_len window')


def absr(path):
    with h5py.File(path, 'r') as f:
        return np.abs(np.nan_to_num(f['connectivity'][()]))


iu = np.triu_indices(absr(os.path.join(conn, 'connectivity_random_halfA.h5')).shape[0], 1)
ref = np.corrcoef(absr(os.path.join(conn, 'connectivity_random_halfA.h5'))[iu],
                  absr(os.path.join(conn, 'connectivity_random_halfB.h5'))[iu])[0, 1]
r = subprocess.run([sys.executable, '-m', 'parcelmate.bin.connectome_consistency', cfg_path, '--purge'],
                   cwd=ROOT, env=ENV, capture_output=True, text=True)
path = os.path.join(out_dir, 'metrics', 'connectome_consistency.csv')
rows = list(csv.DictReader(open(path))) if r.returncode == 0 else []
within = [x for x in rows if x['fit'] == x['eval'] == 'random']
check('connectome_consistency: split-half r equals the numpy reference (upper triangle of |r|)',
      len(within) == 1 and within[0]['metric'] == 'fidelity_within_r' and abs(float(within[0]['value']) - ref) < 1e-6)
check('connectome_consistency: one row per ordered domain pair, shuffle recorded, connectivity purged',
      len(rows) == 4 and sum(x['metric'] == 'fidelity_across_halves' for x in rows) == 2
      and all(x['shuffle_words'] == 'within each seq_len window' for x in rows) and not os.path.exists(conn)
      and oct(os.stat(path).st_mode & 0o777) in ('0o664', '0o666'))

# ---------------------------------------------------------------- configs and launcher
ok = True
paths = sorted(glob.glob(os.path.join(ROOT, 'configs', 'pythia_shuffled', '*.yml')))
for p in paths:
    s = yaml.safe_load(open(p))
    o = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'pythia', os.path.basename(p))))
    sc, oc = s['connectivity'], o['connectivity']
    ok &= s['output_dir'] == o['output_dir'].replace('results/pythia/', 'results/pythia_shuffled/')
    ok &= sc['data_kwargs'] == {'shuffle_words': True} and sc['null_model'] is None
    ok &= all(sc[k] == oc[k] for k in oc if k not in ('null_model', 'data_kwargs'))
    ok &= s['seed'] == o['seed'] and 'parcellation' not in s
check('configs: 22 shuffled twins, identical to the originals except output tree, shuffle and no null',
      ok and len(paths) == 22)
launcher = open(os.path.join(ROOT, 'scripts', 'launch_shuffle.sh')).read()
check('launcher: connectivity then consistency with purge, throttled, umask 002, jagupard32 excluded',
      'connectome_consistency \\$cfg --purge' in launcher and 'afterany' in launcher and 'umask 002' in launcher
      and 'jagupard32' in launcher)

shutil.rmtree(tmp)
print('\n%d checks, %d failure(s)' % (n_checks[0], len(failures)))
for f in failures:
    print('  FAIL', f)
sys.exit(1 if failures else 0)
