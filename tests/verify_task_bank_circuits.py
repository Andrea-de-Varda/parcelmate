"""Verify the converted task bank of group 1 (circuits): valid JSON, required fields, symmetric flip, count cap and registry match, spot checks against raw."""
import glob, json, os, sys
import yaml

TB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'external', 'task_bank')
FIELDS = ['clean_problem', 'corrupted_problem', 'clean_correct', 'clean_incorrect', 'corrupted_correct', 'corrupted_incorrect', 'source_id', 'source_item_id']
SOURCES = ['mib', 'eap_ig', 'wang_ioi', 'docstring', 'gpt2_syntax_circuits']
errors, n_tasks = [], 0

def err(m):
    errors.append(m); print('FAIL', m)

for sid in SOURCES:
    reg = yaml.safe_load(open(os.path.join(TB, 'registry', sid + '.yaml')))
    cfg = json.load(open(os.path.join(TB, 'configs', sid + '.json')))
    conv = [t for t in reg['tasks'] if t.get('n_items_converted')]
    if set(cfg) != {t['name'] for t in conv}:
        err('%s: config tasks differ from registry' % sid)
    for t in conv:
        n_tasks += 1
        name = t['name']
        c = cfg[name]
        if c.get('path') != name + '.json' or c.get('force_base_evaluation') is not True or c.get('prefix') is not None:
            err('%s/%s: bad config' % (sid, name))
        p = os.path.join(TB, 'tasks', sid, name + '.json')
        items = json.load(open(p))
        if len(items) > 2000 or len(items) != t['n_items_converted']:
            err('%s/%s: count %d vs registry %d' % (sid, name, len(items), t['n_items_converted']))
        ids = open(os.path.join(TB, 'tasks', sid, name + '.ids.txt')).read().split()
        if ids != [i['source_item_id'] for i in items]:
            err('%s/%s: ids file mismatch' % (sid, name))
        if len(set(ids)) != len(ids):
            err('%s/%s: duplicate source ids' % (sid, name))
        for i in items:
            for f in FIELDS:
                if not isinstance(i.get(f), str) or not i[f]:
                    err('%s/%s: missing %s in %s' % (sid, name, f, i.get('source_item_id')))
            if i['corrupted_correct'] != i['clean_incorrect'] or i['corrupted_incorrect'] != i['clean_correct'] or i['clean_correct'] == i['clean_incorrect']:
                err('%s/%s: flip violated in %s' % (sid, name, i['source_item_id']))
            if i['source_id'] != sid:
                err('%s/%s: wrong source_id' % (sid, name))

# ---- spot checks against the raw files
def chk(cond, msg):
    if not cond:
        err(msg)
mib = os.path.join(TB, 'raw', 'mib')
ioi = {x['prompt']: x for x in json.load(open(os.path.join(mib, 'ioi', 'train.json')))}
for i in json.load(open(os.path.join(TB, 'tasks', 'mib', 'ioi_s2_flip.json')))[:50]:
    x = ioi[i['clean_problem']]
    chk(x['s2_io_flip_counterfactual']['prompt'] == i['corrupted_problem'] and i['clean_correct'] == ' ' + x['choices'][x['answerKey']], 'mib ioi spot check ' + i['source_item_id'])
rows = [json.loads(l) for l in open(os.path.join(mib, 'arithmetic_addition', 'train.jsonl'))]
for i in json.load(open(os.path.join(TB, 'tasks', 'mib', 'addition_random.json')))[:50]:
    x = rows[int(i['source_item_id'].split('/')[-2])]
    chk(x['prompt'] == i['clean_problem'] and str(x['label']) == i['clean_correct'] and x['random_counterfactual']['prompt'] == i['corrupted_problem'] and str(x['random_counterfactual']['label']) == i['clean_incorrect'], 'mib addition spot check ' + i['source_item_id'])
import pandas as pd
D = os.path.join(TB, 'raw', 'eap_ig', 'eap-ig-faithfulness', 'data')
g = pd.read_csv(os.path.join(D, 'gender-bias', 'gpt2.csv'))
for i in json.load(open(os.path.join(TB, 'tasks', 'eap_ig', 'gender_bias.json')))[:50]:
    r = g.iloc[int(i['source_item_id'].split('/')[-1])]
    chk(r['clean'] == i['clean_problem'] and r['corrupted'] == i['corrupted_problem'], 'eap gender spot check ' + i['source_item_id'])
s = pd.read_csv(os.path.join(D, 'sva', 'gpt2.csv'))
for i in json.load(open(os.path.join(TB, 'tasks', 'eap_ig', 'sva.json')))[:50]:
    r = s.iloc[int(i['source_item_id'].split('/')[-1])]
    chk(r['clean'] == i['clean_problem'] and ((' are' if r['plural'] else ' is') == i['clean_correct']), 'eap sva spot check ' + i['source_item_id'])
# frozen generator outputs match the registry sha256
import hashlib
reg = yaml.safe_load(open(os.path.join(TB, 'registry', 'wang_ioi.yaml')))
h = hashlib.sha256(open(os.path.join(TB, 'tasks', 'wang_ioi', 'ioi_wang_s2_flip.json'), 'rb').read()).hexdigest()
chk(h == reg['generator']['frozen_output_sha256'], 'wang_ioi frozen output sha256 differs from registry')
# docstring: the clean answer is in the def line but undocumented; the clean incorrect answer is documented in the clean prompt only
for i in json.load(open(os.path.join(TB, 'tasks', 'docstring', 'docstring_rest.json')))[:100]:
    a = i['clean_correct'].strip(); b = i['clean_incorrect'].strip()
    chk(a in i['clean_problem'].split('\n')[1] and (':param %s:' % a) not in i['clean_problem'] and (':param %s:' % b) in i['clean_problem'] and (':param %s:' % b) not in i['corrupted_problem'] and (':param %s:' % a) not in i['corrupted_problem'], 'docstring spot check ' + i['source_item_id'])
# no wang prompt ends with the answer name (generator text includes it; must be stripped)
for i in json.load(open(os.path.join(TB, 'tasks', 'wang_ioi', 'ioi_wang_s2_flip.json')))[:200]:
    chk(i['clean_problem'].endswith(' to'), 'wang prompt not truncated: ' + i['clean_problem'][-20:])
print('%d tasks checked, %d failures' % (n_tasks, len(errors)))
sys.exit(1 if errors else 0)
