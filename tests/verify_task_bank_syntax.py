"""Verify the converted syntax-group task bank (causalgym, marks_sfc_sva, newman_sva, clams).

For each converted task: valid JSON; required fields present and non-empty; every item flips
symmetrically; item count <= 2000, equals the registry and the ids sidecar; configs list exactly the
tasks; spot checks (first, middle, last item plus 20 seeded random ones) against the raw source.
Run: PYTHONPATH=. python tests/verify_task_bank_syntax.py
"""
import json
import os
import random
import re
import sys

import yaml

TB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'external', 'task_bank')
SOURCES = ['causalgym', 'marks_sfc_sva', 'newman_sva', 'clams']
REQ = ['clean_problem', 'corrupted_problem', 'clean_correct', 'clean_incorrect', 'corrupted_correct',
       'corrupted_incorrect', 'source_id', 'source_item_id', 'family']
fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)


def norm(s):
    s = re.sub(r'\s+', ' ', s.strip().lower())
    return re.sub(r' ([,.;:!?])', r'\1', s)


def raw_sentence(sid, item_id, which='good'):
    """Return the raw grammatical sentence for a source_item_id of newman_sva / clams."""
    fn, ln = item_id.rsplit(':', 1)
    ln = int(ln)
    if sid == 'clams':
        lines = open(os.path.join(TB, 'raw/clams/repo', fn), encoding='utf8').read().split('\n')
        flag, s = lines[ln - 1].split('\t')
        assert flag == 'True'
        return s
    line = open(os.path.join(TB, 'raw/newman_sva/repo/data', fn), encoding='utf8').read().split('\n')[ln - 1]
    return json.loads(line)['sentence_good']


def spot(sid, task, items, rng):
    pick = [items[0], items[len(items) // 2], items[-1]] + rng.sample(items, min(20, len(items)))
    for it in pick:
        if sid == 'causalgym':
            split, idx = it['source_item_id'].split(':')
            r = json.load(open(os.path.join(TB, 'raw/causalgym/hf', split + '.json')))[int(idx)]
            base = ''.join(r['base']).replace('<|endoftext|>', '', 1)
            check(r['task'] == it['task'] and base == it['clean_problem'] and r['base_label'] == it['clean_correct']
                  and ''.join(r['src']).replace('<|endoftext|>', '', 1) == it['corrupted_problem'], f'{sid}/{task}: raw mismatch {it["source_item_id"]}')
        elif sid == 'marks_sfc_sva':
            fn, ln = it['source_item_id'].split(':')
            r = json.loads(open(os.path.join(TB, 'raw/marks_sfc_sva/repo/data', fn), encoding='utf8').read().split('\n')[int(ln) - 1])
            check((r['clean_prefix'], r['patch_prefix'], r['clean_answer'], r['patch_answer']) ==
                  (it['clean_problem'], it['corrupted_problem'], it['clean_correct'], it['clean_incorrect']), f'{sid}/{task}: raw mismatch {it["source_item_id"]}')
        else:
            for pid, prob, ans in ((it['source_item_id'], it['clean_problem'], it['clean_correct']),
                                   (it['corrupted_source_item_id'], it['corrupted_problem'], it['corrupted_correct'])):
                s = norm(raw_sentence(sid, pid))
                check(s.startswith(norm(prob)), f'{sid}/{task}: prompt not a prefix of raw sentence {pid}')
                check(norm(prob + ' ' + ans.strip()) in s, f'{sid}/{task}: prompt+answer not in raw sentence {pid}')


def main():
    rng = random.Random(0)
    total = 0
    for sid in SOURCES:
        reg = yaml.safe_load(open(os.path.join(TB, 'registry', sid + '.yaml')))
        cfg = json.load(open(os.path.join(TB, 'configs', sid + '.json')))
        names = [t['name'] for t in reg['tasks']]
        check(sorted(cfg) == sorted(names), f'{sid}: config tasks differ from registry')
        for k, v in cfg.items():
            check(v['force_base_evaluation'] is True and v['path'] == k + '.json', f'{sid}/{k}: bad config entry')
        for t in reg['tasks']:
            p = os.path.join(TB, 'tasks', sid, t['name'] + '.json')
            items = json.load(open(p, encoding='utf8'))
            ids = [l for l in open(p.replace('.json', '.ids.txt'), encoding='utf8').read().split('\n') if l]
            total += 1
            check(0 < len(items) <= 2000, f'{sid}/{t["name"]}: count {len(items)} outside (0, 2000]')
            check(len(items) == t['n_items_converted'], f'{sid}/{t["name"]}: count differs from registry')
            check(len(ids) == len(items), f'{sid}/{t["name"]}: ids file length differs')
            for it in items:
                for f in REQ:
                    check(isinstance(it.get(f), str) and it[f] != '', f'{sid}/{t["name"]}: empty field {f} in {it.get("source_item_id")}')
                check(it['corrupted_correct'] == it['clean_incorrect'] and it['corrupted_incorrect'] == it['clean_correct']
                      and it['clean_correct'] != it['clean_incorrect'], f'{sid}/{t["name"]}: flip broken {it["source_item_id"]}')
                check(it['clean_problem'] != it['corrupted_problem'], f'{sid}/{t["name"]}: identical prompts {it["source_item_id"]}')
                check(it['source_id'] == sid and it['family'] == t['family'], f'{sid}/{t["name"]}: source/family mismatch')
            spot(sid, t['name'], items, rng)
    print(f'{total} tasks checked, {len(fails)} failures')
    for f in fails[:40]:
        print('FAIL', f)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
