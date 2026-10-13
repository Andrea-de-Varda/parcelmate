"""Convert CounterFact (Meng et al., NeurIPS 2022) to tier-1 items by subject swap within one relation and one prompt template.

Each record gives a template ("The mother tongue of {} is"), a subject and the TRUE object (target_true). The counterfactual target_new of the source is NOT
used (a false object has no ground-truth status for a model). An item pairs two records of the same relation_id AND the same template whose true objects differ:
clean "The mother tongue of Danielle Darrieux is" -> " French", corrupted "The mother tongue of Raymond Barre is" -> " English"... Pairs are restricted to equal
Qwen3.5 token length of the full prompt (so the clean/corrupted lengths match by construction) and to answers where neither is a string prefix of the other.
Seed 42; water-filling allocation over the templates of a relation (equal shares) up to 2000 items; relations whose valid-pair capacity is below MIN_CAPACITY
are not converted (recorded in the registry).  Task = one relation_id."""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knowledge_common as kc

SID = 'counterfact'
MIN_CAPACITY = 200  # the plan aims at >= 300 items per task; relations with 200-299 valid pairs are kept and flagged below target
# Wikidata property labels, read from the Wikidata API (wbgetentities, English labels) on 2026-10-12
WD = {'P17': 'country', 'P19': 'place of birth', 'P20': 'place of death', 'P27': 'country of citizenship', 'P30': 'continent', 'P36': 'capital',
      'P37': 'official language', 'P39': 'position held', 'P101': 'field of work', 'P103': 'native language', 'P106': 'occupation', 'P108': 'employer',
      'P127': 'owned by', 'P131': 'located in the administrative territorial entity', 'P136': 'genre', 'P138': 'named after', 'P140': 'religion or worldview',
      'P159': 'headquarters location', 'P176': 'manufacturer', 'P178': 'developer', 'P190': 'twinned administrative body', 'P264': 'record label',
      'P276': 'location', 'P364': 'original language of film or TV show', 'P407': 'language of work or name', 'P413': 'position played on team / speciality',
      'P449': 'original broadcaster', 'P463': 'member of', 'P495': 'country of origin', 'P641': 'sport', 'P740': 'location of formation',
      'P937': 'work location', 'P1303': 'instrument', 'P1412': 'languages spoken, written or signed'}
# families shared with the other sources that cover the same relation (FV, LRE, RAVEL); every other relation gets 'factual_relation_<label>'
FAMILY = {'P36': 'factual_capital', 'P17': 'factual_country', 'P30': 'factual_continent', 'P37': 'factual_language', 'P103': 'factual_language',
          'P407': 'factual_language', 'P364': 'factual_language', 'P1412': 'factual_language', 'P106': 'factual_occupation', 'P1303': 'factual_instrument',
          'P641': 'factual_sport', 'P176': 'factual_product_company', 'P413': 'factual_sport_position'}

def slug(s):
    return ''.join(c if c.isalnum() else '_' for c in s.lower()).strip('_')[:40]

def main():
    man = kc.load_manifest(SID)
    kc.clear_tasks(SID)
    recs = json.load(open(os.path.join(kc.RAW, SID, 'official', 'counterfact.json')))
    by_rel = collections.defaultdict(list)
    for r in recs:
        rw = r['requested_rewrite']
        tmpl, subj, ans = rw['prompt'], rw['subject'], rw['target_true']['str']
        assert tmpl.count('{}') == 1
        by_rel[rw['relation_id']].append(dict(id=str(r['case_id']), template=tmpl, subject=subj, text=tmpl.format(subj), answer=' ' + ans, relation=rw['relation_id']))
    tasks_rec, kept = [], []
    for rel in sorted(by_rel, key=lambda s: int(s[1:])):
        # prompts that occur with several different true objects (e.g. 'In European Union, an official language is') are ambiguous and dropped;
        # exact duplicates (same text and answer) are kept once
        ans_by_text = collections.defaultdict(set)
        for f in by_rel[rel]:
            ans_by_text[f['text']].add(f['answer'])
        seen_text, facts, n_ambiguous = set(), [], 0
        for f in by_rel[rel]:
            if len(ans_by_text[f['text']]) > 1:
                n_ambiguous += 1
            elif f['text'] not in seen_text:
                seen_text.add(f['text'])
                facts.append(f)
        name = 'cf_%s_%s' % (rel, slug(WD[rel]))
        pairs, st = kc.sample_pairs(facts, kc.CAP, '42:' + name, lambda f: f['template'], lambda f: f['answer'], lambda f: kc.ntok(f['text']),
                                    compatible=lambda a, b: not (a['answer'].startswith(b['answer']) or b['answer'].startswith(a['answer'])))
        rec_common = dict(name=name, family=FAMILY.get(rel, 'factual_relation_' + slug(WD[rel])), domain='factual knowledge', n_items_raw=len(by_rel[rel]), n_ambiguous_prompts_dropped=n_ambiguous, n_facts_used=len(facts),
                          relation='%s (%s)' % (rel, WD[rel]), n_templates=len({f['template'] for f in facts}), valid_pair_capacity=st['capacity'])
        if st['capacity'] < MIN_CAPACITY:
            rec_common.update(status='not_converted', n_items_converted=0, notes='fewer than %d valid pairs (same template, equal token length, different answers)' % MIN_CAPACITY)
            tasks_rec.append(rec_common)
            continue
        items = []
        for a, b, tmpl in pairs:
            items.append(kc.make_item(a['text'], b['text'], a['answer'], b['answer'], SID, a['id'] + '|' + b['id'], relation_id=rel, relation=WD[rel], template=tmpl,
                                      clean_subject=a['subject'], corrupted_subject=b['subject']))
        ad = ['pairs two records of the same relation_id and the same prompt template whose TRUE objects differ (subject swap); target_new (the counterfactual object) is unused',
              'pairs restricted to equal Qwen3.5 token length of the full prompt and to answers that are not prefixes of each other',
              'answers are " " + target_true; zero-shot, no demonstrations']
        rec = kc.write_task(SID, name, rec_common['family'], rec_common['domain'], items, lambda i: i['template'], len(by_rel[rel]), ad + ['records whose prompt text has several distinct true objects in the source (%d here) are dropped; duplicate records kept once' % n_ambiguous],
                            extra={k: v for k, v in rec_common.items() if k not in ('name', 'family', 'domain', 'n_items_raw')})
        if rec['n_items_converted'] < 300:
            rec['notes'] = 'below the 300-item target (valid-pair capacity %d)' % st['capacity']
        tasks_rec.append(rec)
        kept.append(name)
    kc.write_config(SID, kept)
    reg = kc.base_registry(SID, man, title='CounterFact (ROME): factual statements with counterfactual targets, used here as true-fact subject-swap pairs',
                           bibkey='meng2023locatingeditingfactualassociations', bibkey_data='meng2023locatingeditingfactualassociations (CounterFact is introduced in the same paper; NeurIPS 2022)',
                           url=man['url'], revision=man['revision'], code_repo=man['code_repo'], code_repo_revision=man['code_repo_revision'],
                           licence='unclear for the data: the code repository kmeng01/rome is MIT (LICENSE at the pinned commit) but neither the repository nor the host page states a licence for counterfact.json; the data derives from ParaRel (Elazar et al., MIT) and Wikidata (CC0)',
                           licence_source='external/task_bank/raw/counterfact/rome_repo/LICENSE (code only)', redistribute='no (data licence unclear; keep converted items out of git until decided)',
                           tier=1, tokenizer=kc.tokenizer_meta(), status='converted',
                           notes='Mirrors on Hugging Face (azhx/counterfact sha c01c413f..., 19,728+2,191 = 21,919 records; NeelNanda/counterfact-tracing sha c945b082..., 21,919 rows) carry the same records; they were not used. 21,919 records = the official file.')
    kc.write_registry(SID, reg, tasks_rec)
    for t in tasks_rec:
        print(t['name'], t['n_items_raw'], t.get('valid_pair_capacity'), t['n_items_converted'], t.get('len_match_share_qwen35'), t.get('single_token_answers_share'))

if __name__ == '__main__':
    main()
