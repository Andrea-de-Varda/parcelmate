"""Shared helpers for the syntax group of the task bank (fetch_*/convert_* scripts of CausalGym,
Marks et al., Newman et al., CLAMS). Deterministic; no model needed (tokenizer only)."""
import datetime
import hashlib
import json
import os
import random
import subprocess
from collections import defaultdict

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TB = os.path.join(ROOT, 'external', 'task_bank')
SEED = 42
CAP = 2000
QWEN_TOKENIZER_REPO = 'Qwen/Qwen3.5-2B'


def raw_dir(source_id):
    return os.path.join(TB, 'raw', source_id)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git_clone_pinned(url, dest, sha):
    """Clone `url` into `dest` (if absent) and check out the pinned commit; verify the SHA."""
    if not os.path.isdir(os.path.join(dest, '.git')):
        subprocess.run(['git', 'clone', '-q', url, dest], check=True)
    subprocess.run(['git', '-C', dest, 'checkout', '-q', sha], check=True)
    head = subprocess.run(['git', '-C', dest, 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
    assert head == sha, (head, sha)
    return head


def write_manifest(source_id, info):
    """Written by fetch scripts: download date and pinned revisions. raw/ is gitignored."""
    d = raw_dir(source_id)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, 'MANIFEST.json')
    if os.path.exists(path):  # keep the original download date on reruns
        old = json.load(open(path))
        info['download_date'] = old.get('download_date', info['download_date'])
    json.dump(info, open(path, 'w'), indent=1)


def read_manifest(source_id):
    return json.load(open(os.path.join(raw_dir(source_id), 'MANIFEST.json')))


def today():
    return datetime.date.today().isoformat()


_TOK = None


def tokenizer():
    global _TOK
    if _TOK is None:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        _TOK = Tokenizer.from_file(hf_hub_download(QWEN_TOKENIZER_REPO, 'tokenizer.json'))
    return _TOK


def diagnostics(items, prefix='', concat=''):
    """Share of items whose clean and corrupted prompts (prefix+problem+concatenation, no chat
    template) have equal Qwen3.5 token counts, and the share whose two answers are both single
    tokens (answers encoded as they appear after the prompt, i.e. with their leading space)."""
    tok = tokenizer()
    n = len(items)
    eq = single = 0
    for it in items:
        a = tok.encode(prefix + it['clean_problem'] + concat).ids
        b = tok.encode(prefix + it['corrupted_problem'] + concat).ids
        eq += len(a) == len(b)
        single += all(len(tok.encode(it[k]).ids) == 1 for k in ('clean_correct', 'clean_incorrect'))
    return dict(len_match_share_qwen35=round(eq / n, 4) if n else None,
                single_token_answer_share=round(single / n, 4) if n else None)


def stratified_sample(rows, key, n=CAP, seed=SEED):
    """Deterministic proportional stratified sample of at most n rows (largest-remainder
    allocation, rows drawn with random.Random(seed) per stratum, returned in original order)."""
    if len(rows) <= n:
        return list(rows)
    strata = defaultdict(list)
    for i, r in enumerate(rows):
        strata[key(r)].append(i)
    keys = sorted(strata, key=str)
    total = len(rows)
    quota = {k: n * len(strata[k]) / total for k in keys}
    alloc = {k: int(quota[k]) for k in keys}
    rest = n - sum(alloc.values())
    for k in sorted(keys, key=lambda k: (-(quota[k] - alloc[k]), str(k)))[:rest]:
        alloc[k] += 1
    rng = random.Random(seed)
    chosen = []
    for k in keys:
        chosen += rng.sample(strata[k], min(alloc[k], len(strata[k])))
    return [rows[i] for i in sorted(chosen)]


def check_flip(it):
    return (it['corrupted_correct'] == it['clean_incorrect'] and it['corrupted_incorrect'] == it['clean_correct']
            and it['clean_correct'] != it['clean_incorrect'])


def write_task(source_id, task, items, ids):
    d = os.path.join(TB, 'tasks', source_id)
    os.makedirs(d, exist_ok=True)
    for it in items:
        assert check_flip(it), it
    with open(os.path.join(d, task + '.json'), 'w', encoding='utf8') as f:
        json.dump(items, f, ensure_ascii=False, indent=1)
    with open(os.path.join(d, task + '.ids.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(ids) + '\n')


def write_config(source_id, tasks):
    os.makedirs(os.path.join(TB, 'configs'), exist_ok=True)
    cfg = {t: {'path': t + '.json', 'prefix': None, 'suffix': None, 'concatenation': None,
               'merging_sign': '', 'force_base_evaluation': True} for t in tasks}
    json.dump(cfg, open(os.path.join(TB, 'configs', source_id + '.json'), 'w'), indent=2)


def write_registry(source_id, entry):
    os.makedirs(os.path.join(TB, 'registry'), exist_ok=True)
    with open(os.path.join(TB, 'registry', source_id + '.yaml'), 'w', encoding='utf8') as f:
        yaml.safe_dump(entry, f, sort_keys=False, allow_unicode=True, width=10000)
