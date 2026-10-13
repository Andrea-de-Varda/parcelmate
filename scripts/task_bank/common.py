"""Shared helpers for the task-bank converters of group 1 (circuits): deterministic stratified sampling,
validation of the symmetric flip, Qwen3.5 tokenizer diagnostics, and writing of tasks / ids / configs / diagnostics.
No model inference; tokenizer-only."""
import hashlib, json, os, random, collections

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
TB = os.path.normpath(os.path.join(REPO, 'external', 'task_bank'))
CAP = 2000
SEED = 42
FIELDS = ['clean_problem', 'corrupted_problem', 'clean_correct', 'clean_incorrect', 'corrupted_correct',
          'corrupted_incorrect', 'source_id', 'source_item_id']
QWEN_REPO, QWEN_REV = 'Qwen/Qwen3.5-2B', '15852e8c16360a2fea060d615a32b45270f8a8fc'  # 4B tokenizer.json is identical (checked)

_tok = None
def qwen_tokenizer():
    global _tok
    if _tok is None:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(hf_hub_download(QWEN_REPO, 'tokenizer.json', revision=QWEN_REV))
    return _tok

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def make_item(clean_problem, corrupted_problem, clean_correct, clean_incorrect, source_id, source_item_id, **meta):
    """Build a symmetric-flip item: corrupted_correct = clean_incorrect and corrupted_incorrect = clean_correct."""
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

def stratified_sample(items, strata_key, cap=CAP, seed=SEED):
    """Deterministic proportional stratified sample (largest remainder), seed 42. Items sorted by source_item_id first."""
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

def diagnostics(items):
    tok = qwen_tokenizer()
    same = sum(len(tok.encode(i['clean_problem']).ids) == len(tok.encode(i['corrupted_problem']).ids) for i in items)
    single = sum(len(tok.encode(i['clean_correct']).ids) == 1 and len(tok.encode(i['clean_incorrect']).ids) == 1 for i in items)
    eq_ans = sum(len(tok.encode(i['clean_correct']).ids) == len(tok.encode(i['clean_incorrect']).ids) for i in items)
    n = max(len(items), 1)
    return dict(len_match_share_qwen35=round(same / n, 4), n_len_match=same,
                single_token_answers_share=round(single / n, 4),
                equal_answer_token_count_share=round(eq_ans / n, 4))

def write_task(source_id, task, family, domain, pool, strata_key, raw_count, adaptations, extra=None, cap=CAP):
    """pool: list of valid items (all that the source yields); writes sample + ids; returns registry record."""
    bad = [i for i in pool if not valid_item(i)]
    assert not bad, (task, bad[:2])
    seen, uniq = set(), []
    for i in sorted(pool, key=lambda x: x['source_item_id']):
        k = (i['clean_problem'], i['corrupted_problem'], i['clean_correct'], i['clean_incorrect'])
        if k not in seen:
            seen.add(k)
            uniq.append(i)
    sample = stratified_sample(uniq, strata_key, cap)
    d = os.path.join(TB, 'tasks', source_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, task + '.json'), 'w') as f:
        json.dump(sample, f, indent=1, ensure_ascii=False)
    with open(os.path.join(d, task + '.ids.txt'), 'w') as f:
        f.write('\n'.join(i['source_item_id'] for i in sample) + '\n')
    rec = dict(name=task, family=family, domain=domain, n_items_raw=raw_count, n_items_after_validity_and_dedup=len(uniq),
               n_items_converted=len(sample),
               sampling_rule=('all items kept' if len(uniq) <= cap else 'proportional stratified sample, seed 42, cap %d' % cap),
               adaptations=adaptations)
    rec.update(diagnostics(sample))
    if extra:
        rec.update(extra)
    return rec

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
    return dict(tokenizer_repo=QWEN_REPO, tokenizer_revision=QWEN_REV, note='Qwen3.5-4B tokenizer.json is byte-identical in content to 2B')
