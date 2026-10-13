"""Convert the MIB datasets (raw/mib, fetched by fetch_mib.py) to tier-1 tasks in the LLM_Modularity format.
Deterministic (seed 42, cap 2000/task). Usage: python convert_mib.py"""
import json, os, glob, re, random, datetime
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

SID = 'mib'
RAW = os.path.join(TB, 'raw', 'mib')
man = json.load(open(os.path.join(RAW, 'MANIFEST.json')))
tasks = []

def jl(p):
    out = []
    for n, l in enumerate(open(os.path.join(RAW, p))):
        x = json.loads(l); x['_line'] = n   # the source `idx` field is NOT unique (repeats across templates): ids use the 0-based line number
        out.append(x)
    return out

# ---------------------------------------------------------------- IOI
ioi = json.load(open(os.path.join(RAW, 'ioi', 'train.json')))
def ioi_task(name, cf_key, adapt):
    pool = []
    for idx, x in enumerate(ioi):
        cf = x.get(cf_key)
        if cf is None:
            continue
        ca, cb = x['choices'][x['answerKey']], x['choices'][1 - x['answerKey']]
        fa = cf['choices'][cf['answerKey']]
        if fa != cb or sorted(cf['choices']) != sorted(x['choices']):
            continue
        pool.append(make_item(x['prompt'], cf['prompt'], ' ' + ca, ' ' + cb, SID, 'ioi/train/%d' % idx,
                              template=x['template'], io_name=ca, s_name=cb, counterfactual=cf_key))
    return write_task(SID, name, 'ioi', 'syntax/semantics (name movement)', pool, lambda i: i['source_item_id'] and ioi[int(i['source_item_id'].split('/')[-1])]['template'],
                      len(ioi), adapt)
tasks.append(ioi_task('ioi_s2_flip', 's2_io_flip_counterfactual',
    ['MIB `s2_io_flip_counterfactual`: the second mention of the subject (S2) is replaced by the IO name, so IO and S exchange roles and the answer flips (Maria <-> Carl); prompts differ in one token. Answers are the two names with a leading space.',
     'Other MIB ioi counterfactuals not used: abc (answerKey -1, no flip), random_names (different names, answers not the clean pair), s1_io_flip (order of the first clause only, answer unchanged), random_names_* variants.']))
tasks.append(ioi_task('ioi_s1_s2_flip', 's1_ioi_flip_s2_ioi_flip_counterfactual',
    ['MIB `s1_ioi_flip_s2_ioi_flip_counterfactual`: both the order of the names in the first clause and the repeated name are swapped (ABBA <-> BAAB structure), answer flips; three tokens differ.']))

# ---------------------------------------------------------------- arithmetic
CF_GROUPS = {'random': ['random_counterfactual'],
             'single_digit': ['ones_op1_counterfactual', 'ones_op2_counterfactual', 'tens_op1_counterfactual', 'tens_op2_counterfactual'],
             'carry': ['ones_carry_counterfactual', 'tens_carry_counterfactual']}
for ds, opname in (('arithmetic_addition', 'addition'), ('arithmetic_subtraction', 'subtraction')):
    rows = jl(ds + '/train.jsonl')
    for gname, cfs in CF_GROUPS.items():
        pool = []
        for x in rows:
            for cf in cfs:
                c = x[cf]
                if c is None or str(c['label']) == str(x['label']):
                    continue  # counterfactual does not change the answer: cannot flip
                pool.append(make_item(x['prompt'], c['prompt'], str(x['label']), str(c['label']), SID,
                                      '%s/train/%d/%s' % (ds, x['_line'], cf.replace('_counterfactual', '')),
                                      source_idx=x['idx'], template=x['template'], num_digit=x['num_digit'], operator=x['operator'],
                                      counterfactual=cf.replace('_counterfactual', ''), operand1=x['operand1'], operand2=x['operand2'],
                                      cf_operand1=c['operand1'], cf_operand2=c['operand2']))
        tasks.append(write_task(SID, '%s_%s' % (opname, gname), 'arithmetic_' + opname, 'arithmetic', pool,
                                lambda i: (i['template'], i['counterfactual']), len(rows) * len(cfs),
                                ['Source split: train. One item per (problem, counterfactual); items whose counterfactual has the same result as the clean problem are dropped (no flip possible). Answers are the numeric strings without a leading space (the prompts end in "A: " or "=", digits are tokenised one by one by Qwen).',
                                 'Counterfactual group `%s` = %s.' % (gname, ', '.join(c.replace('_counterfactual', '') for c in cfs)),
                                 'Strata: template x counterfactual type.']))

# ---------------------------------------------------------------- MCQA / ARC
def mcq(prompt_obj):
    ch = prompt_obj['choices']
    return ' ' + str(ch['label'][prompt_obj['answerKey']])

def mcq_task(name, family, rows, cf_key, id_fn, meta_fn, domain, adapt, strata):
    pool = []
    for x in rows:
        c = x.get(cf_key)
        if c is None:
            continue
        ca, cb = mcq(x), mcq(c)
        if ca == cb:
            continue
        pool.append(make_item(x['prompt'], c['prompt'], ca, cb, SID, id_fn(x), counterfactual=cf_key.replace('_counterfactual', ''), **meta_fn(x)))
    return write_task(SID, name, family, domain, pool, strata, len(rows), adapt)

rows = []
for p in sorted(glob.glob(os.path.join(RAW, 'copycolors_mcqa', '*_answer_choices', '*.jsonl'))):
    n = int(re.search(r'(\d+)_answer_choices', p).group(1)); sp = os.path.basename(p)[:-6]
    for ln, x in enumerate(json.loads(l) for l in open(p)):
        x['_n'] = n; x['_split'] = sp; x['_line'] = ln
        rows.append(x)
for cf, nm in (('answerPosition_counterfactual', 'copycolors_answer_position'), ('answerPosition_color_counterfactual', 'copycolors_answer_position_color')):
    tasks.append(mcq_task(nm, 'mcqa_copycolors', rows, cf, lambda x: 'copycolors_mcqa/%d_choices/%s/%d' % (x['_n'], x['_split'], x['_line']),
        lambda x: dict(n_choices=x['_n'], split=x['_split'], dataset=x['dataset']), 'reasoning (multiple choice, answer-letter binding)',
        ['All 9 choice-count configs (2..10) and all three public splits pooled (1,890 rows in total, below the cap, so no sampling). The same underlying question can appear under several choice counts.',
         'Clean and corrupted differ only in the position of the correct option (`%s`)%s; the answer is the option letter with a leading space; flip holds because the correct letter changes.' % (cf, ' and, additionally, the colour of the correct option' if 'color' in cf else ''),
         'Not used: noun, color, noun_color counterfactuals (correct letter unchanged, no flip), symbol and randomLetter variants (answer in another label vocabulary than the clean prompt offers).'],
        lambda i: i['n_choices']))
for ds in ('arc_easy', 'arc_challenge'):
    rows = jl(ds + '/train.jsonl')
    tasks.append(mcq_task(ds + '_answer_position', 'mcqa_arc', rows, 'answerPosition_counterfactual', lambda x, ds=ds: '%s/train/%d' % (ds, x['_line']),
        lambda x: dict(arc_id=x['arc_id'], n_choices=len(x['choices']['text'])), 'science QA (multiple choice)',
        ['Source split: train. MIB `answerPosition_counterfactual`: the options are rotated so that the correct letter changes; the question and option texts are identical. Answer = option letter with a leading space.',
         'Other counterfactuals (symbol, randomLetter, answerPosition_randomLetter) use a different label vocabulary and are not used.',
         'Strata: clean answer letter x counterfactual answer letter.'],
        lambda i: (i['clean_correct'], i['clean_incorrect'])))

# ---------------------------------------------------------------- RAVEL
rv = pd.read_parquet(os.path.join(RAW, 'ravel', 'data', 'train-00000-of-00001.parquet'))
rv = rv[rv.attribute.isin(['Country', 'Continent', 'Language'])].reset_index()  # drops 'wikipedia' rows (no attribute answer)
def lead(prompt):  # leading space of the answer given how the prompt ends
    return '' if prompt.endswith('"') or prompt.endswith(' ') else ' '
rows = rv.to_dict('records')
pool = []
for r in rows:
    for cfk in ('prompt_template_counterfactual', 'attribute_counterfactual'):
        c = r[cfk]
        if c is None or c['attribute'] not in ('Country', 'Continent', 'Language'):
            continue
        if lead(c['prompt']) != lead(r['prompt']):
            continue
        a, b = r[r['attribute']], c[c['attribute']]
        if a == b:
            continue
        key = 'template' if cfk.startswith('prompt_template') else 'attribute'
        pool.append((key, make_item(r['prompt'], c['prompt'], lead(r['prompt']) + a, lead(r['prompt']) + b, SID, 'ravel/train/%d/%s' % (r['index'], key),
                                    attribute=r['attribute'], cf_attribute=c['attribute'], entity=r['entity'], cf_entity=c['entity'], template=r['template'])))
for key, nm in (('template', 'ravel_template_cf'), ('attribute', 'ravel_attribute_cf')):
    tasks.append(write_task(SID, nm, 'factual_ravel', 'factual knowledge (city attributes)', [p for k, p in pool if k == key],
        lambda i: (i['attribute'], i['cf_attribute']), len(rows),
        ['Source split: train; rows with attribute `wikipedia` dropped (no attribute answer). MIB `%s_counterfactual` used as the corrupted prompt: %s.' % ('prompt_template' if key == 'template' else 'attribute',
          'a different entity in a different template, same attribute' if key == 'template' else 'a different entity asked for a different attribute'),
         'Clean answer = attribute value of the clean entity; clean_incorrect = the counterfactual\'s attribute value, both teacher-forced after the clean prompt (leading space omitted when the prompt ends in a quote, i.e. JSON-style prompts). Only pairs whose prompts end in the same kind of character (quote vs word) are kept, so the answer spacing is valid under both prompts.',
         'Clean and corrupted prompts use different templates, so token lengths rarely match; entity-swap variant below is the better length-matched version.']))
# own construction: swap the entity inside the same template
rng = random.Random(SEED)
by_key = {}
for r in rows:
    by_key.setdefault((r['template'], r['attribute']), []).append(r)
pool = []
for (tpl, att), g in sorted(by_key.items()):
    g = sorted(g, key=lambda r: r['index'])
    for r in g:
        cands = [o for o in g if o[att] != r[att] and len(o['entity'].split()) == len(r['entity'].split())]
        if not cands:
            continue
        m = min(abs(len(o['entity']) - len(r['entity'])) for o in cands)
        close = [o for o in cands if abs(len(o['entity']) - len(r['entity'])) == m]
        o = close[rng.randrange(len(close))]
        l = lead(r['prompt'])
        pool.append(make_item(r['prompt'], o['prompt'], l + r[att], l + o[att], SID, 'ravel/train/%d~%d' % (r['index'], o['index']),
                              attribute=att, entity=r['entity'], cf_entity=o['entity'], template=tpl))
tasks.append(write_task(SID, 'ravel_entity_swap', 'factual_ravel', 'factual knowledge (city attributes)', pool, lambda i: (i['attribute'], i['template']), len(rows),
    ['OUR CONSTRUCTION (not in MIB): for each train row, the entity is replaced by another entity from the same template and attribute with a different attribute value (same number of words, closest character length, ties broken with random.Random(42) in template/index order). Prompts then differ only in the entity name; token length equality depends on the entity names (diagnostic).',
     'Answers as in the other RAVEL tasks (no leading space after a quote). Rows with attribute `wikipedia` excluded.']))

write_config(SID, [t['name'] for t in tasks])
meta = dict(source_id=SID, title='MIB: A Mechanistic Interpretability Benchmark (Hugging Face datasets mib-bench/*)', bibkey='mueller2025mibmechanisticinterpretabilitybenchmark',
            bibkey_data=['wang2022interpretabilitywildcircuitindirect (IOI)', 'clark2018thinksolvedquestionanswering (ARC)', 'huang-etal-2024-ravel (RAVEL)'],
            url='https://huggingface.co/mib-bench', download_date=man['_download_date'], tier=1,
            revisions={k: v['revision'] for k, v in man.items() if not k.startswith('_')},
            raw_files=[dict(path='external/task_bank/raw/mib/' + p, sha256=s) for k, v in man.items() if not k.startswith('_') for p, s in v['files'].items()],
            licence={'ioi': 'MIT', 'copycolors_mcqa': 'Apache-2.0', 'arc_easy': 'Apache-2.0', 'arc_challenge': 'Apache-2.0', 'arithmetic_addition': 'none stated', 'arithmetic_subtraction': 'none stated', 'ravel': 'none stated'},
            licence_source='YAML front matter of README.md of each dataset at the pinned revision (raw/mib/<dataset>/README.md); arithmetic and ravel cards state no licence',
            redistribute={'ioi': 'yes', 'copycolors_mcqa': 'yes', 'arc_easy': 'yes (card says Apache-2.0; ARC itself is CC BY-SA 4.0 per Clark et al., not verified here)', 'arc_challenge': 'yes (same caveat)',
                          'arithmetic_addition': 'unclear', 'arithmetic_subtraction': 'unclear', 'ravel': 'unclear (cards silent; RAVEL repo licence not checked)'},
            tokenizer=tokenizer_meta(), converter='scripts/task_bank/convert_mib.py', seed=SEED, status='converted',
            notes='Only public splits are downloadable; all tasks sample from the train split except copycolors (all splits pooled).')
write_registry(SID, meta, tasks)
for t in tasks:
    print(t['name'], t['n_items_raw'], t['n_items_converted'], t['len_match_share_qwen35'], t['single_token_answers_share'])
