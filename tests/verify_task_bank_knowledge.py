"""Verification of the group-4 (knowledge / binding / entity tracking) task bank.

For every converted task of the ten sources: valid JSON list; required fields present and non-empty; the symmetric flip holds
(corrupted_correct == clean_incorrect, corrupted_incorrect == clean_correct, correct != incorrect, clean != corrupted prompt); at most 2000 items and the count matches the registry;
the ids sidecar matches the items; the config entry exists with force_base_evaluation true; a handful of items match the raw source (source-specific checks below).
Run:  PYTHONPATH=. python tests/verify_task_bank_knowledge.py"""
import collections, csv, glob, json, os, random, re, sys
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TB = os.path.join(REPO, 'external', 'task_bank')
RAW = os.path.join(TB, 'raw')
SOURCES = ['ravel', 'counterfact', 'lre_relations', 'linear_relational_concepts', 'function_vectors', 'binding_iclr', 'oi_subspace', 'entity_tracking_prakash', 'entity_tracking_mi', 'pragworld']
FIELDS = ['clean_problem', 'corrupted_problem', 'clean_correct', 'clean_incorrect', 'corrupted_correct', 'corrupted_incorrect', 'source_id', 'source_item_id']
fails = []

def check(cond, msg):
    if not cond:
        fails.append(msg)

def words(t):
    return sorted(re.sub(r'[^\w ]', ' ', t).split())

def jl(p):
    return [json.loads(l) for l in open(p)]

# ---- source-specific spot checks against the raw files -------------------------------------------------------
def spot_counterfact(items, task):
    raw = {str(r['case_id']): r for r in json.load(open(os.path.join(RAW, 'counterfact', 'official', 'counterfact.json')))}
    for it in random.Random(0).sample(items, min(8, len(items))):
        a, b = it['source_item_id'].split('|')
        for cid, prob, ans in ((a, it['clean_problem'], it['clean_correct']), (b, it['corrupted_problem'], it['corrupted_correct'])):
            rw = raw[cid]['requested_rewrite']
            check(prob == rw['prompt'].format(rw['subject']) and ans == ' ' + rw['target_true']['str'], '%s: counterfact mismatch %s' % (task, cid))
        check(raw[a]['requested_rewrite']['relation_id'] == raw[b]['requested_rewrite']['relation_id'] == it['relation_id'], task + ': relation mismatch')
        check(raw[a]['requested_rewrite']['prompt'] == raw[b]['requested_rewrite']['prompt'], task + ': template mismatch')

def spot_lre(items, task):
    for it in random.Random(0).sample(items, min(8, len(items))):
        a, b, t = it['source_item_id'].split('|')
        rel = json.load(open(os.path.join(RAW, 'lre_relations', 'repo', 'data', a.split('#')[0] + '.json')))
        un = lambda s: re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), s)
        for ident, prob, ans in ((a, it['clean_problem'], it['clean_correct']), (b, it['corrupted_problem'], it['corrupted_correct'])):
            s = rel['samples'][int(ident.split('#')[1])]
            check(prob.endswith(rel['prompt_templates'][int(t[1:])].format(un(s['subject']))) and ans == ' ' + un(s['object']), '%s: lre mismatch %s' % (task, ident))
        check(it['clean_problem'].split('\n')[:-1] == it['corrupted_problem'].split('\n')[:-1] and len(it['clean_problem'].split('\n')) == 6, task + ': demos not shared / not 5-shot')

def spot_fv(items, task):
    for it in random.Random(0).sample(items, min(8, len(items))):
        a, b = it['source_item_id'].split('|')
        for ident, q, ans in ((a, it['clean_query'], it['clean_correct']), (b, it['corrupted_query'], it['corrupted_correct'])):
            rel, idx = ident.split('#')
            row = json.load(open(os.path.join(RAW, 'function_vectors', 'repo', 'dataset_files', rel + '.json')))[int(idx)]
            check(row['input'] == q and ' ' + row['output'] == ans, '%s: fv mismatch %s' % (task, ident))
        check(it['clean_problem'].endswith('Q: %s\nA:' % it['clean_query']) and it['corrupted_problem'].endswith('Q: %s\nA:' % it['corrupted_query']), task + ': query line')
        check(it['clean_problem'].split('Q: ')[:-1] == it['corrupted_problem'].split('Q: ')[:-1] and it['clean_problem'].count('Q: ') == 6, task + ': demos not shared / not 5-shot')

def spot_ravel(items, task):
    d = os.path.join(RAW, 'ravel', 'data_unpacked', 'data')
    for it in random.Random(0).sample(items, min(8, len(items))):
        ea = json.load(open(os.path.join(d, 'ravel_%s_entity_attributes.json' % it['entity_type'])))
        tmpl = it['template']
        for ent, prob, ans in ((it['clean_entity'], it['clean_problem'], it['clean_correct']), (it['corrupted_entity'], it['corrupted_problem'], it['corrupted_correct'])):
            v = ea[ent][it['attribute']]
            if v == 'non-living things':
                v = 'non-living thing'
            check(prob == tmpl % ent and ans.strip() == v and (ans == v) == tmpl.endswith('"'), '%s: ravel mismatch %s' % (task, ent))

def spot_binding(items, task):
    raw = os.path.join(RAW, 'binding_iclr', 'repo', 'coref', 'datasets', 'raw')
    cap = {r['country']: r['capital'] for r in csv.DictReader(open(os.path.join(raw, 'country-list.csv'), encoding='utf-8'))}
    for it in random.Random(0).sample(items, min(20, len(items))):
        check(words(it['clean_problem']) == words(it['corrupted_problem']) or task == 'binding_onehop', task + ': not a permutation')
        if task in ('binding_capitals', 'binding_parallel_capitals', 'binding_onehop'):
            q = it['query_index']
            check(it['clean_attributes'][q] in cap and it['corrupted_attributes'][q] == it['clean_attributes'][it['swapped_with_index']], task + ': swap bookkeeping')
            if task != 'binding_onehop':
                check(' ' + cap[it['clean_attributes'][q]] == it['clean_correct'] and ' ' + cap[it['corrupted_attributes'][q]] == it['clean_incorrect'], task + ': capital answer')
        check(it['clean_problem'].rsplit('\n\n', 1)[-1].count(it['entities'][it['query_index']]) >= 1, task + ': query entity missing')

def spot_prakash(items, task):
    rows = jl(os.path.join(RAW, 'entity_tracking_prakash', 'repo', 'data', 'dataset.jsonl'))
    for it in random.Random(0).sample(items, min(15, len(items))):
        i = int(it['source_item_id'].split(':')[1])
        check(it['clean_problem'] + it['clean_correct'] + '.' == rows[i]['sentence'], '%s: row %d differs from raw' % (task, i))
        check(words(it['clean_problem']) == words(it['corrupted_problem']), task + ': not a permutation')

def spot_oi(items, task):
    for it in random.Random(0).sample(items, min(15, len(items))):
        f, i = it['source_item_id'].split(':')
        rows = jl(os.path.join(RAW, 'oi_subspace', 'acl_software', 'model', 'bi_data', os.path.basename(f)))
        s = rows[int(i)]['sentence']
        check(s.startswith(it['clean_problem']) and s.endswith(it['clean_correct'] + '.'), '%s: row %s differs from raw' % (task, i))
        check(words(it['clean_problem']) == words(it['corrupted_problem']), task + ': not a permutation')

def spot_mi(items, task):
    cache = {}
    for it in random.Random(0).sample(items, min(15, len(items))):
        run, lns = it['source_item_id'].split(':')
        l1, l2 = map(int, lns.split('|'))
        if run not in cache:
            cache[run] = open(os.path.join(RAW, 'entity_tracking_mi', 'generated', run, 'train-t5.jsonl')).read().split('\n')
        r1, r2 = json.loads(cache[run][l1]), json.loads(cache[run][l2])
        check(r1['sentence_masked'] + ' the' == it['clean_problem'] and r1['masked_content'] == 'the' + it['clean_correct'], '%s: clean row %d' % (task, l1))
        check(r2['sentence_masked'] + ' the' == it['corrupted_problem'] and r2['masked_content'] == 'the' + it['clean_incorrect'], '%s: corrupted row %d' % (task, l2))

def spot_pragworld(items, task):
    rows = {'manual': list(csv.DictReader(open(os.path.join(RAW, 'pragworld', 'repo', 'Manual_Dataset_500_Final.csv'), encoding='utf-8'))),
            'synthetic': list(csv.DictReader(open(os.path.join(RAW, 'pragworld', 'repo', 'synthetic_final.csv'), encoding='utf-8')))}
    for it in random.Random(0).sample(items, min(15, len(items))):
        v, ln = it['source_item_id'].split(':')
        row = rows[v][int(ln) - 2]
        check(row['Perturbation?'].strip() == 'TRUE', task + ': not a perturbed row')
        pert_is_clean = not it['clean_is_original']
        pa = ' ' + row['Answer'].strip().lower().capitalize()
        check((it['clean_correct'] if pert_is_clean else it['clean_incorrect']) == pa, '%s: perturbed answer %s' % (task, it['source_item_id']))
        ctx = row['Local Context'].strip()
        check(ctx in (it['clean_problem'] if pert_is_clean else it['corrupted_problem']), task + ': perturbed context')

SPOT = {'counterfact': spot_counterfact, 'lre_relations': spot_lre, 'function_vectors': spot_fv, 'ravel': spot_ravel, 'binding_iclr': spot_binding,
        'entity_tracking_prakash': spot_prakash, 'oi_subspace': spot_oi, 'entity_tracking_mi': spot_mi, 'pragworld': spot_pragworld}

def main():
    n_tasks = n_items = 0
    no_redistribute = []
    for sid in SOURCES:
        reg = yaml.safe_load(open(os.path.join(TB, 'registry', sid + '.yaml')))
        check(reg['source_id'] == sid, sid + ': registry source_id')
        for k in ['title', 'bibkey', 'url', 'revision', 'download_date', 'raw_files', 'licence', 'licence_source', 'redistribute', 'tier', 'tasks']:
            check(k in reg, '%s: registry lacks %s' % (sid, k))
        check(os.path.exists(os.path.join(TB, 'references', sid + '.bib')), sid + ': no .bib')
        for rf in reg['raw_files']:
            check(os.path.exists(os.path.join(REPO, rf['path'])) and len(rf['sha256']) == 64, '%s: raw file %s' % (sid, rf['path']))
        if str(reg['redistribute']).startswith('no'):
            no_redistribute.append(sid)
        conv = [t for t in reg['tasks'] if t.get('status') == 'converted']
        if sid == 'linear_relational_concepts':
            check(not conv and not glob.glob(os.path.join(TB, 'tasks', sid, '*.json')), sid + ': must not contain converted tasks (duplicate of lre_relations)')
            continue
        cfg = json.load(open(os.path.join(TB, 'configs', sid + '.json')))
        check(sorted(cfg) == sorted(t['name'] for t in conv), sid + ': config tasks differ from registry')
        on_disk = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(TB, 'tasks', sid, '*.json')))
        check(on_disk == sorted(t['name'] for t in conv), '%s: task files %s != registry' % (sid, set(on_disk) ^ {t['name'] for t in conv}))
        for t in conv:
            name = t['name']
            n_tasks += 1
            items = json.load(open(os.path.join(TB, 'tasks', sid, name + '.json')))
            check(isinstance(items, list) and len(items) > 0, name + ': empty')
            check(len(items) <= 2000, name + ': > 2000 items')
            check(len(items) == t['n_items_converted'], '%s: registry says %s, file has %d' % (name, t['n_items_converted'], len(items)))
            ids = open(os.path.join(TB, 'tasks', sid, name + '.ids.txt')).read().split('\n')[:-1]
            check(ids == [i['source_item_id'] for i in items], name + ': ids sidecar')
            check(len(set(ids)) == len(ids), name + ': duplicate source ids')
            c = cfg[name]
            check(c['path'] == name + '.json' and c['force_base_evaluation'] is True and c['prefix'] is None and c['suffix'] is None and c['concatenation'] is None, name + ': config entry')
            for i, it in enumerate(items):
                ok = all(isinstance(it.get(f), str) and it[f] != '' for f in FIELDS)
                ok = ok and it['source_id'] == sid and it['corrupted_correct'] == it['clean_incorrect'] and it['corrupted_incorrect'] == it['clean_correct']
                ok = ok and it['clean_correct'] != it['clean_incorrect'] and it['clean_problem'] != it['corrupted_problem']
                if not ok:
                    check(False, '%s: item %d invalid' % (name, i))
                    break
            if sid in SPOT and items:
                SPOT[sid](items, name)
            n_items += len(items)
            for k in ['family', 'domain', 'n_items_raw', 'len_match_share_qwen35', 'single_token_answers_share', 'adaptations', 'sampled_ids_file']:
                check(k in t, '%s: registry task lacks %s' % (name, k))
    print('sources checked: %d, converted tasks: %d, items: %d' % (len(SOURCES), n_tasks, n_items))
    print('redistribute: no ->', no_redistribute)
    if fails:
        print('FAILURES (%d):' % len(fails))
        for f in fails[:60]:
            print('  ', f)
        sys.exit(1)
    print('ALL CHECKS PASSED')

if __name__ == '__main__':
    main()
