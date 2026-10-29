"""Verification for Iteration 48 (document-coherent sampling in `get_dataset`).

    PYTHONPATH=. python tests/verify_iter48_documents.py

CPU only, no downloads: `datasets.load_dataset` is replaced by small fake datasets and the
tokenizer by a word-level fake. Covers: the default `rows` sampling returns exactly what the
loader returned before Iteration 48 (parcelmate/data.py at commit 578059a, the last before Iteration 48, run side by side);
wikitext windows are the first seq_len tokens of distinct articles at least that long; bookcorpus
windows are the start of distinct blocks of consecutive sentences joined by a space; both are
prefix-stable (more tokens returns the same windows first, which the residual fit's reference
windows rely on) and depend on the seed; other datasets carry the end-of-text token after every
document and nothing else changes; a tokenizer without that token is refused; the generated
-docs configs differ from their originals only in the output tree and `data_kwargs`.
"""

import importlib.util
import os
import subprocess
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import yaml

import parcelmate.data as data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
failures = []
n_checks = [0]


def check(name, cond):
    n_checks[0] += 1
    print('%s %s' % ('OK' if cond else 'FAIL', name))
    if not cond:
        failures.append(name)


# ---------------------------------------------------------------- fakes
class FakeTok:
    """One token per whitespace-separated word, ids from a growing vocabulary shared by all
    instances (no collisions); '<|endoftext|>' is id 0."""
    unk_token_id = 1
    vocab = {}

    def __init__(self, has_eos=True):
        self.has_eos = has_eos

    def word_id(self, w):
        return self.vocab.setdefault(w, 2 + len(self.vocab))

    def __call__(self, text):
        ids = [self.word_id(w) for w in text.split()]
        return {'input_ids': ids, 'attention_mask': [1] * len(ids)}

    def convert_tokens_to_ids(self, tok):
        return 0 if (tok == '<|endoftext|>' and self.has_eos) else self.unk_token_id

    def convert_ids_to_tokens(self, i):
        return '<|endoftext|>' if (i == 0 and self.has_eos) else '<unk>'


rs = np.random.RandomState(0)
WIKI = []
for a in range(60):
    WIKI.append(' = Article%d = \n' % a)
    for p in range(rs.randint(1, 6)):
        WIKI.append(' '.join('w%d_%d_%d' % (a, p, i) for i in range(rs.randint(5, 40))) + ' \n')
BOOK = ['s%d end .' % i for i in range(10000)]   # 50 blocks of 200 rows
NEWS = [{'text': ' '.join('n%d_%d' % (d, i) for i in range(rs.randint(3, 15)))} for d in range(3000)]


class Rows:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        if isinstance(k, str):
            return [r[k] for r in self.rows]
        return {'text': [r['text'] for r in self.rows[k]]}

    def take(self, n):
        return Rows(self.rows[:n])

    def __iter__(self):
        return iter(self.rows)


def fake_load_dataset(path, *args, split=None, streaming=False, **kwargs):
    rows = {"Salesforce/wikitext": WIKI, "rojagtap/bookcorpus": BOOK}.get(path)
    rows = [{'text': t} for t in rows] if rows is not None else NEWS
    return Rows(rows)


data.datasets.load_dataset = fake_load_dataset
tok = FakeTok()
SEQ = 16


def get(dataset, n_tokens, sampling='documents', seed=7, mod=data, **kw):
    ids, mask = mod.get_dataset(dataset, tok, n_tokens, SEQ, seed=seed, sampling=sampling, verbose=False, **kw) \
        if mod is data else mod.get_dataset(dataset, tok, n_tokens, SEQ, seed=seed, verbose=False, **kw)
    return ids.numpy(), mask.numpy()


# ---------------------------------------------------------------- rows: unchanged
old_src = subprocess.run(['git', 'show', '578059a:parcelmate/data.py'], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout
old_path = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'parcelmate_data_head.py')
with open(old_path, 'w') as f:
    f.write(old_src)
spec = importlib.util.spec_from_file_location('parcelmate_data_head', old_path)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
old.datasets.load_dataset = fake_load_dataset
same = True
for d in ('Salesforce/wikitext', 'rojagtap/bookcorpus', 'fancyzhx/ag_news'):
    a = get(d, 37 * SEQ + 5, sampling='rows')
    b = get(d, 37 * SEQ + 5, mod=old)
    same &= all(np.array_equal(x, y) for x, y in zip(a, b))
check("rows (default): identical to the loader before Iteration 48, for all three kinds of dataset", same)

# ---------------------------------------------------------------- wikitext
arts = data.wikitext_articles(WIKI)
art_ids = [tok(a)['input_ids'] for a in arts]
w, m = get('Salesforce/wikitext', 20 * SEQ, name='wikitext-103-raw-v1')
match = [[i for i, a in enumerate(art_ids) if len(a) >= SEQ and a[:SEQ] == list(row)] for row in w]
check('wikitext: 20 windows of seq_len, mask all ones', w.shape == (20, SEQ) and m.all())
check('wikitext: every window is the first seq_len tokens of an article at least that long',
      all(len(x) == 1 for x in match))
check('wikitext: no article used twice', len({x[0] for x in match}) == 20)
w2, _ = get('Salesforce/wikitext', 30 * SEQ, name='wikitext-103-raw-v1')
check('wikitext: prefix-stable (30 windows start with the 20)', np.array_equal(w2[:20], w))
w3, _ = get('Salesforce/wikitext', 20 * SEQ, seed=8, name='wikitext-103-raw-v1')
check('wikitext: another seed draws other articles', not np.array_equal(w3, w))
check('wikitext: same seed, same windows', np.array_equal(get('Salesforce/wikitext', 20 * SEQ, name='x')[0], w))

# ---------------------------------------------------------------- bookcorpus
b, _ = get('rojagtap/bookcorpus', 25 * SEQ)
blocks = {s: tok(' '.join(BOOK[s:s + 200]))['input_ids'][:SEQ] for s in range(0, len(BOOK), 200)}
starts = [[s for s, v in blocks.items() if v == list(row)] for row in b]
check('bookcorpus: every window is the start of a block of 200 consecutive sentences joined by spaces',
      all(len(x) == 1 for x in starts))
check('bookcorpus: no block used twice', len({x[0] for x in starts}) == len(b))
check('bookcorpus: prefix-stable', np.array_equal(get('rojagtap/bookcorpus', 40 * SEQ)[0][:25], b))
try:
    get('rojagtap/bookcorpus', 25 * SEQ + 3)
    check('documents: n_tokens not a multiple of seq_len is refused', False)
except AssertionError:
    check('documents: n_tokens not a multiple of seq_len is refused', True)

# ---------------------------------------------------------------- other datasets: separators
n_tok = 50 * SEQ
r_ids, r_mask = get('fancyzhx/ag_news', n_tok, sampling='rows')
d_ids, d_mask = get('fancyzhx/ag_news', n_tok)
flat = d_ids.reshape(-1)[:n_tok]
docs, cur = [], []
for t in flat:
    if t == 0:
        docs.append(cur)
        cur = []
    else:
        cur.append(int(t))
order = np.arange(len(NEWS))
np.random.RandomState(7).shuffle(order)
expect = [tok(NEWS[i]['text'])['input_ids'] for i in order[:len(docs)]]
check('agnews documents: the token stream is the shuffled documents, each followed by end-of-text',
      docs == expect and cur == tok(NEWS[order[len(docs)]]['text'])['input_ids'][:len(cur)])
check('agnews documents: same number of tokens, separators only added',
      d_mask.sum() == r_mask.sum() == n_tok and np.array_equal(
          flat[flat != 0], r_ids.reshape(-1)[:n_tok][:len(flat[flat != 0])]))
check('agnews documents: prefix-stable', np.array_equal(get('fancyzhx/ag_news', 2 * n_tok)[0][:50], d_ids))
try:
    data.get_dataset('fancyzhx/ag_news', FakeTok(has_eos=False), n_tok, SEQ, seed=7, sampling='documents',
                     verbose=False)
    check('documents: a tokenizer without <|endoftext|> is refused', False)
except AssertionError:
    check('documents: a tokenizer without <|endoftext|> is refused', True)
PythiaLike = type('PythiaLike', (FakeTok,), {'unk_token_id': 0})   # Pythia: end-of-text is also unk (id 0)
check('documents: a tokenizer whose end-of-text id is also its unk id is accepted',
      data.document_separator_id(PythiaLike()) == 0)

# ---------------------------------------------------------------- configs
def load(p):
    with open(os.path.join(ROOT, p)) as f:
        return yaml.safe_load(f)


def differ(a, b, path=''):
    out = []
    for k in set(a) | set(b):
        if isinstance(a.get(k), dict) and isinstance(b.get(k), dict):
            out += differ(a[k], b[k], path + k + '.')
        elif a.get(k) != b.get(k):
            out.append(path + k)
    return sorted(out)


ok = True
for m in ('pythia-70m', 'qwen3.5-2b', 'qwen3.5-4b'):
    a = load('configs/richclub/%s-resid-pool4-tiles.yml' % m)
    b = load('configs/richclub/%s-resid-pool4-docs.yml' % m)
    ok &= differ(a, b) == ['connectivity.data_kwargs', 'output_dir'] and b['purge_connectivity'] is False \
        and b['connectivity']['data_kwargs'] == {'sampling': 'documents'}
check('richclub -docs configs: only the output tree and data_kwargs differ from -tiles; tiles kept', ok)
ok = True
for tree in ('pythia_resid',):
    for size in ('70m', '160m'):
        for step in (0, 1000, 143000):
            a = load('configs/%s/%s-%s_step%d.yml' % (tree, tree, size, step))
            b = load('configs/%s_docs/%s_docs-%s_step%d.yml' % (tree, tree, size, step))
            ok &= differ(a, b) == ['connectivity.data_kwargs', 'output_dir'] and \
                b['output_dir'] == a['output_dir'].replace(tree + '/', tree + '_docs/', 1)
check('pythia_resid_docs configs: only the output tree and data_kwargs differ from pythia_resid', ok)

print('\n%d checks, %d failed' % (n_checks[0], len(failures)))
sys.exit(1 if failures else 0)
