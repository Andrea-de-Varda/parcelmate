"""Shared helpers for the group-4 (knowledge, binding, entity tracking) task-bank scripts.

Conventions follow the other task-bank groups (same item fields, same registry layout) but this module is independent
of scripts/task_bank/common.py so that parallel agents never break each other. Contents:
  * fetch helpers: pinned git checkout, pinned URL download, MANIFEST.json with sha256 of every raw file used;
  * Qwen3.5 tokenizer (tokenizer-only; 2B and 4B share one tokenizer.json) and the length / single-token diagnostics;
  * item construction with the symmetric flip, validation, deterministic seed-42 sampling, writing of tasks / ids / configs / registry;
  * a deterministic pair sampler: pairs of facts of one relation (same stratum) with different answers and equal query token length.
No model inference."""
import collections
import datetime
import hashlib
import itertools
import json
import os
import random
import subprocess
import urllib.request

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
TB = os.path.join(REPO, 'external', 'task_bank')
RAW = os.path.join(TB, 'raw')
CAP = 2000
SEED = 42
TODAY = '2026-10-12'
FIELDS = ['clean_problem', 'corrupted_problem', 'clean_correct', 'clean_incorrect', 'corrupted_correct',
          'corrupted_incorrect', 'source_id', 'source_item_id']
QWEN_REPO, QWEN_REV = 'Qwen/Qwen3.5-2B', '15852e8c16360a2fea060d615a32b45270f8a8fc'


# ------------------------------------------------------------------ fetching
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def git_checkout(source_id, name, url, sha):
    """Shallow clone of `url` at the pinned commit `sha` into raw/<source_id>/<name>; reruns are no-ops."""
    d = os.path.join(RAW, source_id, name)
    if not os.path.isdir(os.path.join(d, '.git')):
        os.makedirs(d, exist_ok=True)
        subprocess.run(['git', 'init', '-q', d], check=True)
        subprocess.run(['git', '-C', d, 'remote', 'add', 'origin', url], check=True)
        subprocess.run(['git', '-C', d, 'fetch', '-q', '--depth', '1', 'origin', sha], check=True)
        subprocess.run(['git', '-C', d, 'checkout', '-q', 'FETCH_HEAD'], check=True)
    head = subprocess.run(['git', '-C', d, 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
    assert head == sha, (d, head, sha)
    return d


def download(url, dest):
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        req = urllib.request.Request(url, headers={'User-Agent': 'parcelmate-task-bank'})
        with urllib.request.urlopen(req) as r, open(dest, 'wb') as f:
            f.write(r.read())
    return dest


def write_manifest(source_id, record, files):
    """raw/<source_id>/MANIFEST.json: what was downloaded, from where, at which revision, with sha256 of each raw file used.
    The first download date is kept on reruns."""
    p = os.path.join(RAW, source_id, 'MANIFEST.json')
    date = TODAY
    if os.path.exists(p):
        date = json.load(open(p)).get('download_date', TODAY)
    rec = dict(record)
    rec['download_date'] = date
    rec['raw_files'] = [dict(path=os.path.relpath(f, REPO), sha256=sha256_file(f)) for f in files]
    with open(p, 'w') as f:
        json.dump(rec, f, indent=1)
    return rec


def load_manifest(source_id):
    return json.load(open(os.path.join(RAW, source_id, 'MANIFEST.json')))


# ------------------------------------------------------------------ tokenizer
_tok = None


def qwen_tokenizer():
    global _tok
    if _tok is None:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(hf_hub_download(QWEN_REPO, 'tokenizer.json', revision=QWEN_REV))
    return _tok


_len_cache = {}


def ntok(text):
    """Token count of `text` under the Qwen3.5 tokenizer (no special tokens added)."""
    n = _len_cache.get(text)
    if n is None:
        n = _len_cache[text] = len(qwen_tokenizer().encode(text, add_special_tokens=False).ids)
    return n


# ------------------------------------------------------------------ items
def make_item(clean_problem, corrupted_problem, clean_correct, clean_incorrect, source_id, source_item_id, **meta):
    """Symmetric-flip item: corrupted_correct = clean_incorrect and corrupted_incorrect = clean_correct."""
    d = dict(clean_problem=clean_problem, corrupted_problem=corrupted_problem, clean_correct=clean_correct,
             clean_incorrect=clean_incorrect, corrupted_correct=clean_incorrect, corrupted_incorrect=clean_correct)
    d.update(meta)
    d['source_id'] = source_id
    d['source_item_id'] = source_item_id
    return d


def valid_item(it):
    if any(not isinstance(it.get(k), str) or it.get(k) == '' for k in FIELDS):
        return False
    if it['clean_correct'] == it['clean_incorrect'] or it['clean_problem'] == it['corrupted_problem']:
        return False
    return it['corrupted_correct'] == it['clean_incorrect'] and it['corrupted_incorrect'] == it['clean_correct']


def diagnostics(items):
    """Share of items whose clean / corrupted prompts (prefix + problem + concatenation = the problem here, no chat template)
    have the same Qwen3.5 token count, and share of items whose two answers are both single tokens."""
    same = sum(ntok(i['clean_problem']) == ntok(i['corrupted_problem']) for i in items)
    single = sum(ntok(i['clean_correct']) == 1 and ntok(i['clean_incorrect']) == 1 for i in items)
    n = max(len(items), 1)
    return dict(len_match_share_qwen35=round(same / n, 4), n_len_match=same,
                single_token_answers_share=round(single / n, 4), n_single_token_answers=single)


def stratified_cap(items, strata_key, cap=CAP, seed=SEED):
    """Deterministic proportional stratified sample (largest remainder) of at most `cap` items."""
    items = sorted(items, key=lambda x: x['source_item_id'])
    if len(items) <= cap:
        return items
    groups = collections.OrderedDict()
    for it in items:
        groups.setdefault(strata_key(it), []).append(it)
    keys = sorted(groups, key=str)
    n = len(items)
    quota = {k: cap * len(groups[k]) / n for k in keys}
    alloc = {k: int(quota[k]) for k in keys}
    rest = cap - sum(alloc.values())
    for k in sorted(keys, key=lambda k: (-(quota[k] - alloc[k]), str(k)))[:rest]:
        alloc[k] += 1
    rng = random.Random(seed)
    out = []
    for k in keys:
        g = list(groups[k])
        rng.shuffle(g)
        out += g[:alloc[k]]
    return sorted(out, key=lambda x: x['source_item_id'])


def write_task(source_id, task, family, domain, pool, strata_key, raw_count, adaptations, extra=None, cap=CAP):
    """pool: validated items. Writes tasks/<source_id>/<task>.json and <task>.ids.txt; returns the registry record."""
    bad = [i for i in pool if not valid_item(i)]
    assert not bad, (task, bad[:2])
    seen, uniq = set(), []
    for i in sorted(pool, key=lambda x: x['source_item_id']):
        k = (i['clean_problem'], i['corrupted_problem'], i['clean_correct'], i['clean_incorrect'])
        if k not in seen:
            seen.add(k)
            uniq.append(i)
    sample = stratified_cap(uniq, strata_key, cap)
    d = os.path.join(TB, 'tasks', source_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, task + '.json'), 'w') as f:
        json.dump(sample, f, indent=1, ensure_ascii=False)
    with open(os.path.join(d, task + '.ids.txt'), 'w') as f:
        f.write('\n'.join(i['source_item_id'] for i in sample) + '\n')
    rec = dict(name=task, family=family, domain=domain, n_items_raw=raw_count, n_items_after_validity_and_dedup=len(uniq),
               n_items_converted=len(sample), sampled_ids_file='external/task_bank/tasks/%s/%s.ids.txt' % (source_id, task),
               sampling_rule=('all items kept' if len(uniq) <= cap else 'proportional stratified sample, seed 42, cap %d' % cap),
               adaptations=adaptations)
    rec.update(diagnostics(sample))
    if extra:
        rec.update(extra)
    rec['status'] = 'converted'
    return rec


def clear_tasks(source_id):
    """Remove stale converted tasks of a source before a rerun (task lists can change between versions of a converter)."""
    d = os.path.join(TB, 'tasks', source_id)
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith('.json') or f.endswith('.ids.txt'):
                os.remove(os.path.join(d, f))


def write_config(source_id, tasks):
    os.makedirs(os.path.join(TB, 'configs'), exist_ok=True)
    cfg = {t: {'path': t + '.json', 'prefix': None, 'suffix': None, 'concatenation': None, 'merging_sign': '',
               'force_base_evaluation': True} for t in tasks}
    with open(os.path.join(TB, 'configs', source_id + '.json'), 'w') as f:
        json.dump(cfg, f, indent=2)


def write_registry(source_id, meta, tasks):
    import yaml
    meta = dict(meta)
    meta['tasks'] = tasks
    os.makedirs(os.path.join(TB, 'registry'), exist_ok=True)
    with open(os.path.join(TB, 'registry', source_id + '.yaml'), 'w') as f:
        yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True, width=200)


def tokenizer_meta():
    return dict(tokenizer_repo=QWEN_REPO, tokenizer_revision=QWEN_REV,
                note='Qwen3.5-2B tokenizer.json; the 4B tokenizer is the same (checked by the circuits group)')


def base_registry(source_id, manifest, **kw):
    """Registry fields that every source shares; `kw` carries title, bibkey, licence etc."""
    reg = dict(source_id=source_id)
    reg.update(kw)
    reg['download_date'] = manifest['download_date']
    reg['raw_files'] = manifest['raw_files']
    return reg


# ------------------------------------------------------------------ pair sampler
def _npairs(m, answer_counts):
    return m * (m - 1) // 2 - sum(c * (c - 1) // 2 for c in answer_counts)


def sample_pairs(facts, n_target, seed_str, stratum_of, answer_of, len_of, compatible=None):
    """Deterministically sample up to `n_target` distinct unordered pairs of facts.

    A pair is valid when the two facts share a stratum (e.g. the same prompt template) and an equal query token length
    (`len_of`), their answers differ and `compatible(fa, fb)` (if given) holds. Allocation over strata is water-filling
    (equal shares, leftovers redistributed to strata that have capacity), so rare templates are not drowned out.
    Orientation (which member is the clean prompt) is a fair coin. Returns a list of (fact_a, fact_b, stratum) and a stats dict."""
    rng = random.Random(seed_str)
    buckets = collections.defaultdict(list)
    for f in facts:
        buckets[(stratum_of(f), len_of(f))].append(f)
    strata = collections.defaultdict(list)  # stratum -> list of bucket keys
    cap = {}
    for key in sorted(buckets, key=str):
        fs = buckets[key]
        if len(fs) < 2:
            continue
        cnt = collections.Counter(answer_of(f) for f in fs)
        c = _npairs(len(fs), cnt.values())
        if c > 0:
            strata[key[0]].append(key)
            cap[key] = c
    scap = {s: sum(cap[k] for k in ks) for s, ks in strata.items()}
    order = sorted(scap, key=str)
    alloc = {s: 0 for s in order}
    remaining = n_target
    active = list(order)
    while remaining > 0 and active:
        share = max(remaining // len(active), 1)
        nxt = []
        for s in active:
            take = min(share, scap[s] - alloc[s], remaining)
            alloc[s] += take
            remaining -= take
            if alloc[s] < scap[s]:
                nxt.append(s)
        active = nxt
    out = []
    for s in order:
        quota = alloc[s]
        if quota == 0:
            continue
        keys = strata[s]
        if quota * 3 >= scap[s]:
            allp = []
            for k in keys:
                fs = buckets[k]
                for a, b in itertools.combinations(range(len(fs)), 2):
                    if answer_of(fs[a]) != answer_of(fs[b]) and (compatible is None or compatible(fs[a], fs[b])):
                        allp.append((fs[a], fs[b]))
            rng.shuffle(allp)
            chosen = allp[:quota]
        else:
            weights = [cap[k] for k in keys]
            chosen, seen, tries = [], set(), 0
            while len(chosen) < quota and tries < quota * 200:
                tries += 1
                k = rng.choices(keys, weights)[0]
                fs = buckets[k]
                a, b = rng.sample(range(len(fs)), 2)
                if a > b:
                    a, b = b, a
                if (k, a, b) in seen:
                    continue
                seen.add((k, a, b))
                if answer_of(fs[a]) == answer_of(fs[b]) or (compatible is not None and not compatible(fs[a], fs[b])):
                    continue
                chosen.append((fs[a], fs[b]))
        for a, b in chosen:
            if rng.random() < 0.5:
                a, b = b, a
            out.append((a, b, s))
    stats = dict(n_strata_with_pairs=len(order), n_requested=n_target, n_pairs=len(out), capacity=sum(scap.values()))
    return out, stats
