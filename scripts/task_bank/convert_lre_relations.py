"""Convert the LRE relation data (Hernandez et al., ICLR 2024; evandez/relations) to tier-1 items: in-context relation completion with a subject swap.

Per relation: facts = (subject, object) pairs; templates = the relation's `prompt_templates` (1-3 strings with one {}; `prompt_templates_zs` are not used).
An item = a pair of facts (s1,o1),(s2,o2) with o1 != o2 (neither a string prefix of the other), the same template, and equal Qwen3.5 token length of the query line
template.format(subject). Both prompts share the same 5 in-context demonstrations (other facts of the same relation, distinct subjects, none of the two query
facts), written one per line as template.format(subject) + " " + object, followed by the query line; the answer is " " + object. This mirrors the few-shot
format of the LRE paper's ICL prompts (their own prompt builder was not run) and gives the model the answer format (a zero-shot version is not produced).
Relations with fewer than MIN_FACTS facts or fewer than MIN_CAPACITY valid pairs are not converted. Seed 42; equal allocation over templates; cap 2000 items.
Data cleaning: double-escaped unicode ('Bras\\\\u00edlia' in country_capital_city.json) is decoded. The identical copy of this data in
chanind/linear-relational-concepts (checked file by file) is NOT converted again."""
import glob, json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knowledge_common as kc

SID = 'lre_relations'
N_SHOTS = 5
MIN_FACTS, MIN_CAPACITY = 75, 200  # 75: below this the same few subjects recur in too many items
# relation file stem -> family (shared across sources when the same relation occurs elsewhere)
FAMILY = {
    'country_capital_city': 'factual_capital', 'city_in_country': 'factual_country', 'landmark_in_country': 'factual_country',
    'landmark_on_continent': 'factual_continent', 'person_native_language': 'factual_language', 'country_language': 'factual_language',
    'person_occupation': 'factual_occupation', 'person_plays_pro_sport': 'factual_sport', 'person_plays_instrument': 'factual_instrument',
    'product_by_company': 'factual_product_company', 'person_plays_position_in_sport': 'factual_sport_position', 'adj_antonym': 'lexical_antonym', 'verb_past_tense': 'morphology_past_tense',
}

def unesc(s):
    return re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), s)

def main():
    man = kc.load_manifest(SID)
    kc.clear_tasks(SID)
    root = os.path.join(kc.RAW, SID, 'repo', 'data')
    lrc = os.path.join(kc.RAW, 'linear_relational_concepts', 'repo', 'data', 'lre')
    tasks_rec, kept = [], []
    for path in sorted(glob.glob(os.path.join(root, '*', '*.json'))):
        cat, stem = path.split(os.sep)[-2], os.path.basename(path)[:-5]
        rel = json.load(open(path))
        facts = [dict(id='%s/%s#%d' % (cat, stem, i), subject=unesc(s['subject']), object=unesc(s['object'])) for i, s in enumerate(rel['samples'])]
        name = 'lre_%s_%s' % (cat, stem)
        fam = FAMILY.get(stem, 'factual_relation_' + stem if cat == 'factual' else '%s_relation_%s' % (cat, stem))
        # subjects with several different objects in the relation (e.g. 'carrot' -> vegetable and plant) are ambiguous: dropped; exact duplicates kept once
        objs = {}
        for f in facts:
            objs.setdefault(f['subject'], set()).add(f['object'])
        n_amb = sum(len(objs[f['subject']]) > 1 for f in facts)
        seen_s, uniq_facts = set(), []
        for f in facts:
            if len(objs[f['subject']]) == 1 and f['subject'] not in seen_s:
                seen_s.add(f['subject'])
                uniq_facts.append(f)
        n_raw = len(facts)
        facts = uniq_facts
        common = dict(name=name, family=fam, domain='%s relation (LRE category)' % cat, n_items_raw=n_raw, n_ambiguous_subject_facts_dropped=n_amb, n_facts_used=len(facts), relation=rel['name'], category=cat,
                      templates=rel['prompt_templates'])
        # relation of this copy to the LRC copy
        twin = [p for p in glob.glob(os.path.join(lrc, '*', stem + '.json'))]
        common['identical_copy_in_linear_relational_concepts'] = bool(twin) and json.load(open(twin[0])) == rel
        if len(facts) < MIN_FACTS:
            common.update(status='not_converted', n_items_converted=0, notes='fewer than %d facts' % MIN_FACTS)
            tasks_rec.append(common)
            continue
        templ = rel['prompt_templates']
        for f in facts:
            f['qlines'] = [t.format(f['subject']) for t in templ]
        pool = [dict(f, tmpl=ti, answer=' ' + f['object'], qline=f['qlines'][ti]) for f in facts for ti in range(len(templ))]
        pairs, st = kc.sample_pairs(pool, kc.CAP, '42:' + name, lambda f: f['tmpl'], lambda f: f['answer'], lambda f: kc.ntok(f['qline']),
                                    compatible=lambda a, b: a['subject'] != b['subject'] and not (a['answer'].startswith(b['answer']) or b['answer'].startswith(a['answer'])))
        if st['capacity'] < MIN_CAPACITY:
            common.update(status='not_converted', n_items_converted=0, valid_pair_capacity=st['capacity'], notes='fewer than %d valid pairs' % MIN_CAPACITY)
            tasks_rec.append(common)
            continue
        rng = random.Random('42:demos:' + name)
        items = []
        for a, b, ti in pairs:
            others = [f for f in facts if f['id'] not in (a['id'], b['id'])]
            demos = rng.sample(others, N_SHOTS)
            lines = [templ[ti].format(d['subject']) + ' ' + d['object'] for d in demos]
            ctx = '\n'.join(lines) + '\n'
            items.append(kc.make_item(ctx + a['qline'], ctx + b['qline'], a['answer'], b['answer'], SID, a['id'] + '|' + b['id'] + '|t%d' % ti,
                                      relation=rel['name'], category=cat, template=templ[ti], clean_subject=a['subject'], corrupted_subject=b['subject'],
                                      demo_subjects=[d['subject'] for d in demos]))
        ad = ['subject swap within one relation and one template: (s1,o1) vs (s2,o2) with o1 != o2, equal Qwen3.5 token length of the query line, answers not prefixes of each other',
              '5-shot in-context prompt (demonstrations are other facts of the relation, shared by the clean and corrupted prompt, one per line "<template with subject> <object>"); answer " " + object',
              'only prompt_templates are used (not prompt_templates_zs); double-escaped unicode decoded']
        rec = kc.write_task(SID, name, fam, common['domain'], items, lambda i: i['template'], n_raw, ad + ['facts whose subject has several different objects in the relation (%d here) are dropped; duplicates kept once' % n_amb],
                            extra={k: v for k, v in common.items() if k not in ('name', 'family', 'domain', 'n_items_raw')} | dict(valid_pair_capacity=st['capacity'], n_shots=N_SHOTS,
                                   n_distinct_subjects_in_items=len({s for i in items for s in (i['clean_subject'], i['corrupted_subject'])})))
        if rec['n_items_converted'] < 300:
            rec['notes'] = 'below the 300-item target (valid-pair capacity %d)' % st['capacity']
        tasks_rec.append(rec)
        kept.append(name)
    kc.write_config(SID, kept)
    reg = kc.base_registry(SID, man, title='LRE relations (Hernandez et al.): 48 relations (factual, commonsense, linguistic, bias) as subject-object fact lists',
                           bibkey='hernandez2024linearityrelationdecodingtransformer', url=man['url'], revision=man['revision'],
                           licence='MIT (Copyright 2022 Evan Hernandez)', licence_source='external/task_bank/raw/lre_relations/repo/LICENSE', redistribute='yes (MIT; the repository states no separate data licence; facts were scraped/curated by the authors)',
                           tier=1, tokenizer=kc.tokenizer_meta(), status='converted',
                           notes='Chanin et al. (linear_relational_concepts) ship a copy of this data (data/lre/*, 47 shared files byte-for-byte identical as parsed JSON; person_native_language sits under bias/ there and factual/ here). Converted once, here.')
    kc.write_registry(SID, reg, tasks_rec)
    for t in tasks_rec:
        print(t['name'], t['n_items_raw'], t.get('valid_pair_capacity'), t['n_items_converted'], t.get('len_match_share_qwen35'), t.get('single_token_answers_share'), t.get('n_distinct_subjects_in_items'))

if __name__ == '__main__':
    main()
