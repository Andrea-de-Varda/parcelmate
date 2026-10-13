"""Verify the Qwen3.5 length-matched variants (`<task>__qwen35`) of the syntax-group task bank.

For every registry task with `variant_of`: the original task exists and is untouched in the registry; the config entry equals the
original's except for the path; item count in (0, 2000] and equal to the registry and the ids sidecar; flip symmetry and non-empty
fields for every item; clean and corrupted prompts (prefix + problem + concatenation of the config, no chat template) have EQUAL
Qwen3.5 token counts for EVERY item (registry len_match_share_qwen35 == 1.0); the registry shares match a recount; repaired items (not CausalGym, whose
nouns can be multi-word) keep the word count of the original prompts they replace; items marked unchanged exist verbatim among the original task's items (when the original is not a 2000-item sample); no
duplicate items; and the 8 CausalGym tasks that cannot be re-paired (the contrast inserts or deletes a token) have no variant file.
Run: PYTHONPATH=. python tests/verify_task_bank_qwen35_syntax.py
"""
import json
import os
import sys

import yaml
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

TB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'external', 'task_bank')
SOURCES = ['causalgym', 'marks_sfc_sva', 'newman_sva', 'clams']
SUFFIX = '__qwen35'
NOT_REPAIRABLE = ['garden_mvrr', 'garden_mvrr_mod', 'garden_npz_obj', 'garden_npz_obj_mod', 'gss_subord', 'gss_subord_obj-relc',
                  'gss_subord_pp', 'gss_subord_subj-relc']
REQ = ['clean_problem', 'corrupted_problem', 'clean_correct', 'clean_incorrect', 'corrupted_correct', 'corrupted_incorrect',
       'source_id', 'source_item_id', 'family']
fails = []
tok = Tokenizer.from_file(hf_hub_download('Qwen/Qwen3.5-2B', 'tokenizer.json'))


def check(c, m):
    if not c:
        fails.append(m)


def n(s):
    return len(tok.encode(s).ids)


def main():
    total = 0
    for sid in SOURCES:
        reg = yaml.safe_load(open(os.path.join(TB, 'registry', sid + '.yaml'), encoding='utf8'))
        cfg = json.load(open(os.path.join(TB, 'configs', sid + '.json')))
        by = {t['name']: t for t in reg['tasks']}
        variants = [t for t in reg['tasks'] if t['name'].endswith(SUFFIX)]
        check(variants, f'{sid}: no variants in the registry')
        for t in variants:
            name, base = t['name'], t.get('variant_of')
            total += 1
            check(base and name == base + SUFFIX and base in by, f'{sid}/{name}: bad variant_of')
            check(name in cfg and base in cfg and cfg[name] == dict(cfg[base], path=name + '.json'), f'{sid}/{name}: config differs from the original')
            c = cfg[name]
            prefix, concat = c.get('prefix') or '', c.get('concatenation') or ''
            check(not c.get('suffix'), f'{sid}/{name}: unexpected suffix')
            p = os.path.join(TB, 'tasks', sid, name + '.json')
            items = json.load(open(p, encoding='utf8'))
            ids = [l for l in open(p.replace('.json', '.ids.txt'), encoding='utf8').read().split('\n') if l]
            orig = json.load(open(os.path.join(TB, 'tasks', sid, base + '.json'), encoding='utf8'))
            orig_keys = {(i['clean_problem'], i['corrupted_problem'], i['clean_correct'], i['clean_incorrect']) for i in orig}
            check(0 < len(items) <= 2000, f'{sid}/{name}: count {len(items)} outside (0, 2000]')
            check(len(items) == t['n_items_converted'], f'{sid}/{name}: count differs from registry')
            check(len(ids) == len(items), f'{sid}/{name}: ids length differs')
            check(t.get('n_items_original_task') == len(orig), f'{sid}/{name}: n_items_original_task differs from the original file')
            check(t['len_match_share_qwen35'] == 1.0, f'{sid}/{name}: registry len_match_share_qwen35 != 1.0')
            single = sum(n(i['clean_correct']) == 1 and n(i['clean_incorrect']) == 1 for i in items) / len(items)
            check(abs(single - t['single_token_answer_share_qwen35']) < 1e-3, f'{sid}/{name}: single-token share differs from registry')
            seen = set()
            for it in items:
                for f in REQ:
                    check(isinstance(it.get(f), str) and it[f] != '', f'{sid}/{name}: empty field {f} in {it.get("source_item_id")}')
                check(it['corrupted_correct'] == it['clean_incorrect'] and it['corrupted_incorrect'] == it['clean_correct']
                      and it['clean_correct'] != it['clean_incorrect'], f'{sid}/{name}: flip broken {it["source_item_id"]}')
                check(it['clean_problem'] != it['corrupted_problem'], f'{sid}/{name}: identical prompts {it["source_item_id"]}')
                check(n(prefix + it['clean_problem'] + concat) == n(prefix + it['corrupted_problem'] + concat),
                      f'{sid}/{name}: unequal Qwen3.5 lengths {it["source_item_id"]}')
                check(it['source_id'] == sid and it['family'] == t['family'], f'{sid}/{name}: source/family mismatch')
                k = (it['clean_problem'], it['corrupted_problem'], it['clean_correct'], it['clean_incorrect'])
                check(k not in seen, f'{sid}/{name}: duplicate item {it["source_item_id"]}')
                seen.add(k)
                how = it.get('repaired_qwen35')
                check(how in ('unchanged', 'src', 'both', 'corrupted', 'clean'), f'{sid}/{name}: bad repaired_qwen35 {how}')
                if how == 'unchanged':
                    check(len(orig) >= 2000 or k in orig_keys,  # originals of 2000 items are samples of a larger pool
                           f'{sid}/{name}: unchanged item not in the original {it["source_item_id"]}')
                elif sid != 'causalgym':  # CausalGym nouns can be multi-word (taxi driver)
                    check(len(it['clean_problem'].split()) == len(it['original_clean_problem'].split())
                          and len(it['corrupted_problem'].split()) == len(it['original_corrupted_problem'].split()),
                          f'{sid}/{name}: repaired item changed the word count {it["source_item_id"]}')
        if sid == 'causalgym':
            for b in NOT_REPAIRABLE:
                check(b + SUFFIX not in by and b + SUFFIX not in cfg and not os.path.exists(os.path.join(TB, 'tasks', sid, b + SUFFIX + '.json')),
                      f'{sid}/{b}: variant should not exist')
    print(f'{total} variants checked, {len(fails)} failures')
    for f in fails[:40]:
        print('FAIL', f)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
