"""Convert RAVEL (Huang et al., ACL 2024; explanare/ravel, data.tgz) to tier-1 items by an entity swap within one (entity type, attribute) and one prompt template.

RAVEL stores, per entity type, entity -> {attribute: value} and attribute -> list of prompt templates with one '%s' (entity slot). Many templates carry in-context
facts or are JSON-like ('{"city": "%s", "country": "'). An item pairs two entities e1, e2 of the same entity type whose gold values of the attribute differ, rendered in the
same template: the clean prompt template % e1 -> value1, the corrupted prompt template % e2 -> value2. Pairs are restricted to equal Qwen3.5 token length of the full prompt.
The gold attribute values of the dataset are used (the benchmark's own pipeline instead keeps only prompts a given model completes correctly and labels with the model output; that is model-dependent and not reproduced here).
Answer spacing: templates ending in a double quote take the value as is (the JSON continuation); all others take " " + value. Templates ending in a space, '(' or '/' or in 'UTC'
(numeric / IPA / UTC-offset continuations that do not match the gold strings) are dropped. Seed 42; equal allocation over the surviving templates; cap 2000.
Attributes not converted: Latitude, Longitude, Award Year, Birth Year (numeric), Size (gold 'cm/m/mm/km' vs templates that show spelled-out units), Definition, Pronunciation, Duty (free text / IPA), Gender Bias (no gold values in the entity file).
Cleaning: city Language values that are comma-separated lists (467 of 3552 cities) are dropped (ambiguous answer); empty values dropped; Category 'non-living things' is mapped to 'non-living thing'.
Relation to the MIB RAVEL split (mib-bench/ravel, converted by another agent): that split covers only the city type and the attributes Continent/Country/Language and was built by the
benchmark's pipeline with counterfactual entities per attribute; this conversion is independent and covers all five entity types."""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knowledge_common as kc

SID = 'ravel'
MIN_CAPACITY = 200
# (entity_type_file, attribute, task name, family)
TASKS = [
    ('city', 'Country', 'city_country', 'factual_country'), ('city', 'Continent', 'city_continent', 'factual_continent'), ('city', 'Language', 'city_language', 'factual_language'),
    ('city', 'Timezone', 'city_timezone', 'factual_timezone'),
    ('nobel_prize_winner', 'Field', 'nobel_field', 'factual_nobel_field'), ('nobel_prize_winner', 'Country of Birth', 'nobel_country_of_birth', 'factual_country'),
    ('nobel_prize_winner', 'Gender', 'nobel_gender', 'factual_gender'),
    ('verb', 'Past Tense', 'verb_past_tense', 'morphology_past_tense'), ('verb', 'Singular', 'verb_third_person', 'morphology_third_person'),
    ('physical_object', 'Category', 'object_category', 'commonsense_object_category'), ('physical_object', 'Color', 'object_color', 'commonsense_object_color'),
    ('physical_object', 'Texture', 'object_texture', 'commonsense_object_texture'),
    ('occupation', 'Industry', 'occupation_industry', 'commonsense_occupation_industry'), ('occupation', 'Work Location', 'occupation_work_location', 'commonsense_occupation_location'),
]
NOT_CONVERTED = {('city', 'Latitude'): 'numeric', ('city', 'Longitude'): 'numeric', ('nobel_prize_winner', 'Award Year'): 'numeric year', ('nobel_prize_winner', 'Birth Year'): 'numeric year',
                 ('verb', 'Definition'): 'free-text definitions', ('verb', 'Pronunciation'): 'IPA strings', ('physical_object', 'Size'): 'gold abbreviations (cm/m/mm/km) do not match the spelled-out units shown in the templates',
                 ('occupation', 'Duty'): 'free-text duties', ('occupation', 'Gender Bias'): 'no gold values in the entity file'}

# per-attribute substring a template must contain: the Nobel Gender templates of the form '<Name>: for his contributions in chemistry. %s: for' elicit a pronoun (his/her), not the gold 'male'/'female'
REQUIRE = {('nobel_prize_winner', 'Gender'): 'gender'}

def template_ok(t, et, attr):
    return t[-1] not in ' (/' and not t.endswith('UTC') and t.count('%s') == 1 and REQUIRE.get((et, attr), '') in t

def main():
    man = kc.load_manifest(SID)
    kc.clear_tasks(SID)
    d = os.path.join(kc.RAW, SID, 'data_unpacked', 'data')
    tasks_rec, kept = [], []
    for et, attr, name, fam in TASKS:
        ea = json.load(open(os.path.join(d, 'ravel_%s_entity_attributes.json' % et)))
        ap = json.load(open(os.path.join(d, 'ravel_%s_attribute_to_prompts.json' % et)))
        es = json.load(open(os.path.join(d, 'ravel_%s_entity_to_split.json' % et)))
        templ = [t for t in ap[attr] if template_ok(t, et, attr)]
        n_drop_t = len(ap[attr]) - len(templ)
        ents, n_bad = [], 0
        for e in sorted(ea):
            v = ea[e].get(attr)
            if not v or (et == 'city' and attr == 'Language' and ',' in v) or (et == 'nobel_prize_winner' and ',' in v):
                n_bad += 1
                continue
            if attr == 'Category' and v == 'non-living things':
                v = 'non-living thing'
            ents.append((e, v))
        pool = []
        for ti, t in enumerate(templ):
            sp = '' if t.endswith('"') else ' '
            for e, v in ents:
                pool.append(dict(id='%s/%s/%s' % (et, attr, e), entity=e, value=v, tmpl=ti, text=t % e, answer=sp + v))
        tname = 'ravel_' + name
        pairs, st = kc.sample_pairs(pool, kc.CAP, '42:' + tname, lambda f: f['tmpl'], lambda f: f['answer'], lambda f: kc.ntok(f['text']),
                                    compatible=lambda a, b: a['entity'] != b['entity'] and not (a['answer'].strip().startswith(b['answer'].strip()) or b['answer'].strip().startswith(a['answer'].strip())))
        common = dict(name=tname, family=fam, domain='entity-attribute knowledge (%s)' % et, n_items_raw=len(ea), entity_type=et, attribute=attr, n_entities_used=len(ents), n_entities_dropped=n_bad,
                      n_templates_total=len(ap[attr]), n_templates_used=len(templ), n_templates_dropped_by_ending=n_drop_t, n_distinct_values=len({v for _, v in ents}), valid_pair_capacity=st['capacity'])
        if st['capacity'] < MIN_CAPACITY:
            common.update(status='not_converted', n_items_converted=0, notes='fewer than %d valid pairs' % MIN_CAPACITY)
            tasks_rec.append(common)
            continue
        items = [kc.make_item(a['text'], b['text'], a['answer'], b['answer'], SID, a['id'] + '|' + b['id'].split('/')[-1] + '|t%d' % a['tmpl'], entity_type=et, attribute=attr, template=templ[a['tmpl']],
                              clean_entity=a['entity'], corrupted_entity=b['entity'], clean_entity_split=es.get(a['entity']), corrupted_entity_split=es.get(b['entity'])) for a, b, _ in pairs]
        ad = ['entity swap within one entity type, attribute and prompt template: gold values must differ and neither answer may be a prefix of the other; equal Qwen3.5 token length of the full prompt',
              'gold attribute values from the dataset (not model-labelled completions); answer = value (templates ending in a double quote) or " " + value; templates ending in space, "(", "/" or "UTC" dropped, and for Nobel Gender only templates mentioning gender kept (%d of %d dropped)' % (n_drop_t, len(ap[attr])),
              '%d entities dropped (missing value, multi-valued comma list, or empty)' % n_bad]
        rec = kc.write_task(SID, tname, fam, common['domain'], items, lambda i: i['template'], len(ea), ad, extra={k: v for k, v in common.items() if k not in ('name', 'family', 'domain', 'n_items_raw')})
        if rec['n_items_converted'] < 300:
            rec['notes'] = 'below the 300-item target (valid-pair capacity %d)' % st['capacity']
        tasks_rec.append(rec)
        kept.append(tname)
    for (et, attr), why in NOT_CONVERTED.items():
        tasks_rec.append(dict(name='ravel_%s_%s' % (et, attr.lower().replace(' ', '_')), status='not_converted', entity_type=et, attribute=attr, notes=why))
    kc.write_config(SID, kept)
    reg = kc.base_registry(SID, man, title='RAVEL: entity-attribute dataset (cities, Nobel laureates, verbs, physical objects, occupations), full version',
                           bibkey='huang-etal-2024-ravel', url=man['url'], revision=man['revision'],
                           licence='MIT (Copyright 2024 Jing Huang); the Hugging Face dataset card hij/ravel (revision 2c45eb232b9944c235509db5320345eaba563e07, city type only) also states license: mit', licence_source='external/task_bank/raw/ravel/repo/LICENSE',
                           redistribute='yes (MIT)', tier=1, tokenizer=kc.tokenizer_meta(), status='converted',
                           notes='MIB RAVEL (mib-bench/ravel, revision 53dc7a4dbe84276567b895d8fb608b3f169b9276, 100,347+15,950+1,000 rows, city only, attributes Continent/Country/Language, no licence field on its card) derives from the same data; it is converted by another agent and not duplicated here. data.tgz in the GitHub repo is the version used (not the Hugging Face copy).')
    kc.write_registry(SID, reg, tasks_rec)
    for t in tasks_rec:
        print(t['name'], t.get('n_entities_used'), t.get('n_templates_used'), t.get('valid_pair_capacity'), t.get('n_items_converted'), t.get('len_match_share_qwen35'), t.get('single_token_answers_share'))

if __name__ == '__main__':
    main()
