"""Convert the function-vector task datasets (Todd et al., ICLR 2024; ericwtodd/function_vectors) to tier-1 items by a query swap.

Each dataset is a list of {input, output} pairs defining an input-output function (antonym, capital, translation, 'choose the last of 5 words' ...).
The paper's prompt format is reproduced (src/utils/prompt_utils.py: prefixes Q:/A:, separators "\\n" and "\\n\\n", a space before every input and output):
    "Q: <in_1>\\nA: <out_1>\\n\\n ... Q: <query>\\nA:"   and the answer is " <output>".
An item = N_SHOTS=5 demonstrations (the paper uses 10; shortened to keep prompts cheap for all-position attribution) shared by the clean and corrupted prompt, plus two
queries x1, x2 of the same task whose outputs differ (neither a prefix of the other) and whose query tokens have equal Qwen3.5 length: the clean prompt queries x1
(answer y1), the corrupted prompt queries x2 (answer y2). The task identity is therefore constant (carried by the demonstrations) and only the queried entity changes.
This is OUR construction; the original paper builds corruptions by shuffling the demonstration labels, which does not flip the answer.
Inputs with several different outputs in the source file (e.g. an antonym list that maps one word to two antonyms) are dropped; duplicates kept once.
Not converted: ag_news, commonsense_qa, squad_val, conll2003_* (long or multi-line inputs; tier-1 feasible but expensive), and the *_3 variants of the extractive
tasks (near duplicates of the *_5 variants; the script option INCLUDE_3 enables them). Seed 42, cap 1000 items per task."""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knowledge_common as kc

SID = 'function_vectors'
N_SHOTS, CAP, MIN_CAPACITY = 5, 1000, 200
INCLUDE_3 = False
# (file, task name, family, domain)
ABSTRACTIVE = [
    ('antonym', 'antonym', 'lexical_antonym', 'lexical semantics'), ('synonym', 'synonym', 'lexical_synonym', 'lexical semantics'),
    ('capitalize', 'capitalize', 'string_letter_ops', 'string manipulation'), ('capitalize_first_letter', 'capitalize_first_letter', 'string_letter_ops', 'string manipulation'),
    ('capitalize_last_letter', 'capitalize_last_letter', 'string_letter_ops', 'string manipulation'), ('capitalize_second_letter', 'capitalize_second_letter', 'string_letter_ops', 'string manipulation'),
    ('lowercase_first_letter', 'lowercase_first_letter', 'string_letter_ops', 'string manipulation'), ('lowercase_last_letter', 'lowercase_last_letter', 'string_letter_ops', 'string manipulation'),
    ('next_capital_letter', 'next_capital_letter', 'string_letter_ops', 'string manipulation'), ('word_length', 'word_length', 'string_length', 'string manipulation'),
    ('next_item', 'next_item', 'sequence_next_prev', 'sequence'), ('prev_item', 'prev_item', 'sequence_next_prev', 'sequence'),
    ('country-capital', 'country_capital', 'factual_capital', 'factual knowledge'), ('country-currency', 'country_currency', 'factual_currency', 'factual knowledge'),
    ('landmark-country', 'landmark_country', 'factual_country', 'factual knowledge'), ('park-country', 'park_country', 'factual_country', 'factual knowledge'),
    ('national_parks', 'national_parks_state', 'factual_us_state', 'factual knowledge'), ('person-instrument', 'person_instrument', 'factual_instrument', 'factual knowledge'),
    ('person-occupation', 'person_occupation', 'factual_occupation', 'factual knowledge'), ('person-sport', 'person_sport', 'factual_sport', 'factual knowledge'),
    ('product-company', 'product_company', 'factual_product_company', 'factual knowledge'), ('english-french', 'english_french', 'translation_french', 'translation'),
    ('english-german', 'english_german', 'translation_german', 'translation'), ('english-spanish', 'english_spanish', 'translation_spanish', 'translation'),
    ('present-past', 'present_past', 'morphology_past_tense', 'morphology'), ('singular-plural', 'singular_plural', 'morphology_plural', 'morphology'),
    ('sentiment', 'sentiment', 'sentiment', 'sentiment'),
]
EXTRACTIVE = ['adjective_v_verb', 'alphabetically_first', 'alphabetically_last', 'animal_v_object', 'choose_first_of', 'choose_last_of', 'choose_middle_of',
              'color_v_animal', 'concept_v_object', 'fruit_v_animal', 'object_v_concept', 'verb_v_adjective']
NOT_CONVERTED = {
    'abstractive/ag_news': 'long news articles as input; tier-1 feasible but costly for all-position attribution', 'abstractive/commonsense_qa': 'multi-line multiple-choice prompts (MCQA covered by the MIB group)',
    'extractive/squad_val': 'reading-comprehension contexts of ~150 tokens per demonstration', 'extractive/conll2003_location': 'NER spans from running text, long inputs',
    'extractive/conll2003_organization': 'NER spans from running text, long inputs', 'extractive/conll2003_person': 'NER spans from running text, long inputs',
}

def main():
    man = kc.load_manifest(SID)
    kc.clear_tasks(SID)
    base = os.path.join(kc.RAW, SID, 'repo', 'dataset_files')
    jobs = [('abstractive/' + f, n, fam, dom) for f, n, fam, dom in ABSTRACTIVE]
    for e in EXTRACTIVE:
        for k in ((5, 3) if INCLUDE_3 else (5,)):
            jobs.append(('extractive/%s_%d' % (e, k), 'extractive_%s_%d' % (e, k), 'fv_extractive_' + e, 'in-context selection'))
    tasks_rec, kept = [], []
    for rel, name, fam, dom in jobs:
        rows = json.load(open(os.path.join(base, rel + '.json')))
        outs = {}
        for r in rows:
            outs.setdefault(r['input'], set()).add(r['output'])
        seen, facts, n_amb = set(), [], 0
        for i, r in enumerate(rows):
            if len(outs[r['input']]) > 1:
                n_amb += 1
            elif r['input'] not in seen:
                seen.add(r['input'])
                facts.append(dict(id='%s#%d' % (rel, i), inp=r['input'], answer=' ' + r['output'], qline='Q: ' + r['input'] + '\nA:'))
        name_t = 'fv_' + name
        pairs, st = kc.sample_pairs(facts, CAP, '42:' + name_t, lambda f: 0, lambda f: f['answer'], lambda f: kc.ntok(f['qline']),
                                    compatible=lambda a, b: not (a['answer'].startswith(b['answer']) or b['answer'].startswith(a['answer'])))
        common = dict(name=name_t, family=fam, domain=dom, n_items_raw=len(rows), n_ambiguous_inputs_dropped=n_amb, n_facts_used=len(facts), valid_pair_capacity=st['capacity'], source_file='dataset_files/%s.json' % rel)
        if st['capacity'] < MIN_CAPACITY:
            common.update(status='not_converted', n_items_converted=0, notes='fewer than %d valid pairs' % MIN_CAPACITY)
            tasks_rec.append(common)
            continue
        rng = random.Random('42:demos:' + name_t)
        items = []
        for a, b, _ in pairs:
            others = [f for f in facts if f['id'] not in (a['id'], b['id'])]
            demos = rng.sample(others, N_SHOTS)
            ctx = ''.join('Q: %s\nA:%s\n\n' % (d['inp'], d['answer']) for d in demos)
            items.append(kc.make_item(ctx + a['qline'], ctx + b['qline'], a['answer'], b['answer'], SID, a['id'] + '|' + b['id'], task=name, clean_query=a['inp'], corrupted_query=b['inp'],
                                      demo_ids=[d['id'] for d in demos]))
        ad = ['query swap (our construction): same %d demonstrations in both prompts, clean queries x1 (answer y1), corrupted queries x2 (answer y2 != y1; neither a prefix of the other), equal Qwen3.5 token length of the query line' % N_SHOTS,
              'prompt format of the paper (Q:/A:, "\\n", "\\n\\n", space before inputs/outputs) with %d instead of 10 demonstrations; answer " " + output' % N_SHOTS,
              'inputs with several outputs in the source dropped (%d rows here); duplicates kept once' % n_amb]
        rec = kc.write_task(SID, name_t, fam, dom, items, lambda i: 0, len(rows), ad, extra={k: v for k, v in common.items() if k not in ('name', 'family', 'domain', 'n_items_raw')} | dict(n_shots=N_SHOTS), cap=CAP)
        if rec['n_items_converted'] < 300:
            rec['notes'] = 'below the 300-item target (valid-pair capacity %d)' % st['capacity']
        tasks_rec.append(rec)
        kept.append(name_t)
    for k, why in NOT_CONVERTED.items():
        tasks_rec.append(dict(name='fv_' + k.split('/')[1], status='not_converted', source_file='dataset_files/%s.json' % k, notes=why))
    kc.write_config(SID, kept)
    reg = kc.base_registry(SID, man, title='Function-vector task datasets (Todd et al.): abstractive and extractive in-context tasks',
                           bibkey='todd2024functionvectorslargelanguage', url=man['url'], revision=man['revision'],
                           licence='MIT (Copyright 2023 Eric Todd)', licence_source='external/task_bank/raw/function_vectors/repo/LICENSE', redistribute='yes (MIT; underlying word lists come from third-party sources named in dataset_files/generate, see report)',
                           tier=1, tokenizer=kc.tokenizer_meta(), status='converted',
                           notes='Cap lowered to 1000 items per task (the plan allows 2000) because this source contributes ~40 tasks.')
    kc.write_registry(SID, reg, tasks_rec)
    for t in tasks_rec:
        print(t['name'], t.get('n_items_raw'), t.get('valid_pair_capacity'), t.get('n_items_converted'), t.get('len_match_share_qwen35'), t.get('single_token_answers_share'))

if __name__ == '__main__':
    main()
