"""Checks for the Qwen3.5 length-matched re-pairings (`<task>__qwen35`) of mib, eap_ig and devarda_arithmetic (scripts/task_bank/repair_qwen35_other.py):
symmetric flip, equal Qwen3.5 prompt token counts (prefix + problem + concatenation) for every item, count <= 2,000 and equal to the registry, ids sidecar, config
and registry entries, originals untouched in the registry. Run: python tests/verify_task_bank_qwen35_other.py"""
import json, os, sys
import yaml
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

TB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'external', 'task_bank')
FIELDS = ['clean_problem', 'corrupted_problem', 'clean_correct', 'clean_incorrect', 'corrupted_correct', 'corrupted_incorrect', 'source_id', 'source_item_id']
TASKS = {'mib': ['ravel_attribute_cf', 'ravel_template_cf', 'ravel_entity_swap'], 'eap_ig': ['capital_country', 'hypernymy_comma', 'hypernymy_type_of'],
         'devarda_arithmetic': ['embedded', 'embedded_verbal']}
tok = Tokenizer.from_file(hf_hub_download('Qwen/Qwen3.5-2B', 'tokenizer.json'))
n = lambda s: len(tok.encode(s, add_special_tokens=False).ids)
errors, n_items = [], 0

def err(m):
    errors.append(m); print('FAIL', m)

for sid, bases in TASKS.items():
    reg = {t['name']: t for t in yaml.safe_load(open(os.path.join(TB, 'registry', sid + '.yaml')))['tasks']}
    cfg = json.load(open(os.path.join(TB, 'configs', sid + '.json')))
    for b in bases:
        name = b + '__qwen35'
        t = reg.get(name)
        if t is None or name not in cfg:
            err('%s/%s: missing registry or config entry' % (sid, name)); continue
        if t.get('variant_of') != b or b not in reg or b not in cfg:
            err('%s/%s: variant_of / original entry' % (sid, name))
        c, c0 = cfg[name], cfg[b]
        if c != dict(c0, path=name + '.json'):
            err('%s/%s: prompt settings differ from the original' % (sid, name))
        items = json.load(open(os.path.join(TB, 'tasks', sid, name + '.json')))
        if not (0 < len(items) <= 2000) or len(items) != t['n_items_converted']:
            err('%s/%s: count %d vs registry %s' % (sid, name, len(items), t['n_items_converted']))
        ids = open(os.path.join(TB, 'tasks', sid, name + '.ids.txt')).read().split('\n')
        if ids[-1] == '':
            ids = ids[:-1]
        if ids != [i['source_item_id'] for i in items] or len(set(ids)) != len(ids):
            err('%s/%s: ids sidecar / uniqueness' % (sid, name))
        if t.get('len_match_share_qwen35') != 1.0 or 'single_token_answer_share_qwen35' not in t or not t.get('adaptations'):
            err('%s/%s: registry diagnostics' % (sid, name))
        pre, con = c.get('prefix') or '', c.get('concatenation') or ''
        seen = set()
        for it in items:
            n_items += 1
            if any(not isinstance(it.get(k), str) or it[k] == '' for k in FIELDS):
                err('%s/%s/%s: empty field' % (sid, name, it.get('source_item_id')))
            if it['corrupted_correct'] != it['clean_incorrect'] or it['corrupted_incorrect'] != it['clean_correct'] or it['clean_correct'] == it['clean_incorrect']:
                err('%s/%s/%s: flip' % (sid, name, it['source_item_id']))
            if it['clean_problem'] == it['corrupted_problem']:
                err('%s/%s/%s: identical prompts' % (sid, name, it['source_item_id']))
            if n(pre + it['clean_problem'] + con) != n(pre + it['corrupted_problem'] + con):
                err('%s/%s/%s: unequal Qwen3.5 length' % (sid, name, it['source_item_id']))
            k = (it['clean_problem'], it['corrupted_problem'])
            if k in seen:
                err('%s/%s: duplicate pair' % (sid, name))
            seen.add(k)
        if sid == 'eap_ig' and b == 'capital_country' and len(items) < 100:
            err('capital_country__qwen35 too small')
print('pragworld_flip: no variant by design:', 'pragworld_flip__qwen35' not in {t['name'] for t in yaml.safe_load(open(os.path.join(TB, 'registry', 'pragworld.yaml')))['tasks']})
print('%d variants, %d items checked, %d failures' % (sum(map(len, TASKS.values())), n_items, len(errors)))
sys.exit(1 if errors else 0)
