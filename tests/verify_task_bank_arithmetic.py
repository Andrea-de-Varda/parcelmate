"""Checks for the group-3 (arithmetic) task-bank sources: devarda_arithmetic, nikankin_arithmetic, stolfo_arithmetic (format_invariant_arithmetic is registry-only).

Run: PYTHONPATH=. python tests/verify_task_bank_arithmetic.py
"""
import json, os, re
import yaml

TB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'external', 'task_bank')
REQ = ['clean_problem', 'corrupted_problem', 'clean_correct', 'clean_incorrect', 'corrupted_correct', 'corrupted_incorrect', 'source_id', 'source_item_id']
SOURCES = ['devarda_arithmetic', 'nikankin_arithmetic', 'stolfo_arithmetic']
n_checks = 0


def check(cond, msg):
    global n_checks
    n_checks += 1
    assert cond, msg


def num(s): return int(s.strip())


for sid in SOURCES:
    reg = yaml.safe_load(open(os.path.join(TB, 'registry', f'{sid}.yaml')))
    cfg = json.load(open(os.path.join(TB, 'configs', f'{sid}.json')))
    check(reg['source_id'] == sid and reg['status'] == 'converted', sid)
    check(len(reg['revision']) == 40, 'revision is a commit sha')
    for t in reg['tasks']:
        name = t['name']
        items = json.load(open(os.path.join(TB, 'tasks', sid, f'{name}.json')))
        check(name in cfg and cfg[name]['path'] == f'{name}.json', f'{sid}/{name} config')
        check(len(items) == t['n_items_converted'] <= 2000, f'{sid}/{name} count')
        ids = open(os.path.join(TB, 'tasks', sid, f'{name}.ids.txt')).read().split()
        check(ids == [i['source_item_id'] for i in items], f'{sid}/{name} ids sidecar')
        check(len(set(ids)) == len(ids), 'unique ids')
        for it in items:
            for k in REQ:
                check(isinstance(it.get(k), str) and it[k] != '', f'{sid}/{name} field {k}')
            check(it['clean_correct'] != it['clean_incorrect'], 'correct != incorrect')
            check(it['corrupted_correct'] == it['clean_incorrect'] and it['corrupted_incorrect'] == it['clean_correct'], 'symmetric flip')
            check(it['clean_problem'] != it['corrupted_problem'], 'prompts differ')
        # spot checks against the arithmetic itself
        for it in items[::97]:
            if sid == 'nikankin_arithmetic':
                for p, a in ((it['clean_problem'], it['clean_correct']), (it['corrupted_problem'], it['corrupted_correct'])):
                    check(int(eval(p[:-1])) == num(a) and eval(p[:-1]) == num(a), f'{p}{a}')
            elif sid == 'stolfo_arithmetic':
                if name != 'operator_change':
                    for p, a in ((it['clean_problem'], it['clean_correct']), (it['corrupted_problem'], it['corrupted_correct'])):
                        x, y = map(int, re.findall(r'\d+', p)[-2:] if p.endswith('=') else re.findall(r'\d+', p))
                        check(eval(f'{x}{it["operator"]}{y}') == num(a), p)
            elif sid == 'devarda_arithmetic' and name == 'numeric':
                p = it['clean_problem']
                toks = p[:-2].split(' ')
                check(eval(' '.join(toks)) == num(it['clean_correct']) == it['ground_truth'], p)
                check(eval(' '.join(toks).translate(str.maketrans('+-', '-+'))) == num(it['clean_incorrect']), p)

# de Varda: same ids across formats, same selection as raw generated file, formats match the regenerated raw data
dv = {t['name']: json.load(open(os.path.join(TB, 'tasks', 'devarda_arithmetic', f"{t['name']}.json"))) for t in yaml.safe_load(open(os.path.join(TB, 'registry', 'devarda_arithmetic.yaml')))['tasks'] if 'variant_of' not in t}  # __qwen35 variants: tests/verify_task_bank_qwen35_other.py
check(len(dv) == 6, 'six formats')
base = [i['source_item_id'] for i in dv['numeric']]
for name, items in dv.items():
    check([i['source_item_id'] for i in items] == base, f'{name} ids match numeric')
    check([(i['n_terms'], i['n_digits'], i['has_carry'], i['operators']) for i in items] == [(i['n_terms'], i['n_digits'], i['has_carry'], i['operators']) for i in dv['numeric']], f'{name} metadata matches')
    raw = {r['id']: r for r in json.load(open(os.path.join(TB, 'raw', 'devarda_arithmetic', 'generated', 'data', 'json', f'arith_dataset_{name}.json')))}
    for it in items[::50]:
        r = raw[it['source_item_id']]
        check(r['prompt'] == it['clean_problem'] and r['answer'] == it['clean_correct'], f'{name} {it["source_item_id"]} vs raw')
        check(it['ground_truth_prime'] >= 0, 'no negative flipped result')
print(f'OK: {n_checks} checks passed')
