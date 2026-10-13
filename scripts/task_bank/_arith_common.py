"""Shared helpers for the group-3 (arithmetic) task-bank converters."""
import hashlib, json, os, datetime
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TB = os.path.join(ROOT, 'external', 'task_bank')
_TOK = None


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def tokenizer():
    global _TOK
    if _TOK is None:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        _TOK = Tokenizer.from_file(hf_hub_download('Qwen/Qwen3.5-2B', 'tokenizer.json'))
    return _TOK


def ntok(s):
    return len(tokenizer().encode(s).ids)


def length_stats(items, cfg):
    """Share of items whose clean/corrupted prompts (prefix+problem+concatenation, no chat template) have equal Qwen3.5 token count; share of single-token answers (clean_correct, measured as the token-count increase when appended to the clean prompt)."""
    pre, con = cfg.get('prefix') or '', cfg.get('concatenation') or ''
    eq = single = 0
    for it in items:
        c = pre + it['clean_problem'] + con
        k = pre + it['corrupted_problem'] + con
        eq += ntok(c) == ntok(k)
        single += (ntok(c + it['clean_correct']) - ntok(c)) == 1
    n = len(items)
    return round(eq / n, 4), round(single / n, 4)


def check_flip(items):
    bad = [it['source_item_id'] for it in items
           if it['corrupted_correct'] != it['clean_incorrect'] or it['corrupted_incorrect'] != it['clean_correct']
           or it['clean_correct'] == it['clean_incorrect']]
    assert not bad, bad[:5]


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write('\n')


def write_yaml(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True, width=10**6)


def today():
    return datetime.date.today().isoformat()
