"""Convert the ordering-ID binding data of Dai, Heinzerling and Inui (EMNLP 2024; software attachment of the paper / cl-tohoku/OI-Subspace) to tier-1 items.

model/bi_data/ has six JSONL files of 15,400 rows (2,200 contexts x 7 queries):
  dataset.jsonl                      the boxes data of Prakash et al. (byte-identical to entity_tracking_prakash data/dataset.jsonl; NOT converted here)
  dataset_{sell,apply,move,bring,push}_a.jsonl   person-verb-object binding with random common words: "The cool is pushed by person Julian, the hair is pushed by person Juli, ...
                                     (7 clauses). Person Julian pushes the cool."
We convert the five verb files, one task each ("oi_sell", ...). Prompt = context + " Person <Name> <verb phrase> the", answer " <object>". Flip (our construction): the person names of the queried
clause and of one other clause (random.Random('42:<task>:<row>')) are exchanged in the context; the query is unchanged; the answer flips from the object of the queried person to that of the partner.
Clean and corrupted prompts are permutations of the same tokens (equal length by construction). Cap 2000 items per task, stratified on the position of the queried clause.
The 'Ordering ID' of the paper (the entity's position in the list) is what the swap manipulates: the entity-attribute binding is exchanged while positions stay fixed.
Licence: the GitHub repo cl-tohoku/OI-Subspace carries an MIT LICENSE (Tohoku NLP Group) and its README points to the Anthology attachment for the data; the attachment itself has no licence statement. Treated as MIT, flagged."""
import json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knowledge_common as kc

SID = 'oi_subspace'
VERBS = ['sell', 'apply', 'move', 'bring', 'push']
CL = re.compile(r'(?:The|the) (\w+) is (\w+) by person (\w+)')

def main():
    man = kc.load_manifest(SID)
    kc.clear_tasks(SID)
    d = os.path.join(kc.RAW, SID, 'acl_software', 'model', 'bi_data')
    tasks_rec, kept = [], []
    for v in VERBS:
        task = 'oi_' + v
        rows = [json.loads(l) for l in open(os.path.join(d, 'dataset_%s_a.jsonl' % v))]
        pool, n_dup = [], 0
        for i, r in enumerate(rows):
            s = r['sentence']
            ctx, query = s.rsplit('. ', 1)
            ctx += '.'
            m = re.fullmatch(r'(Person (\w+) .*? the) (\w+)\.', query)
            assert m, query
            qprefix, qname, qobj = m.groups()
            clauses = CL.findall(ctx)
            assert len(clauses) == 7, s
            names = [c[2] for c in clauses]
            if len(set(names)) < 7 or len({c[0] for c in clauses}) < 7:  # a name or object occurs twice in the context: the binding is ambiguous
                n_dup += 1
                continue
            assert dict((n, o) for o, _, n in clauses)[qname] == qobj, s
            qidx = names.index(qname)
            rng = random.Random('42:%s:%d' % (task, i))
            pidx = rng.choice([j for j in range(7) if j != qidx])
            parts = re.split(r'(?<=\w)(?=, the )', ctx)
            def swap_name(part, new):
                return re.sub(r'by person (\w+)(\.?)$', lambda mm: 'by person %s%s' % (new, mm.group(2)), part)
            parts[qidx], parts[pidx] = swap_name(parts[qidx], names[pidx]), swap_name(parts[pidx], names[qidx])
            pool.append(kc.make_item(ctx + ' ' + qprefix, ''.join(parts) + ' ' + qprefix, ' ' + qobj, ' ' + clauses[pidx][0], SID, '%s:%d' % (os.path.basename(d) + '/dataset_%s_a.jsonl' % v, i),
                                     verb=v, query_position=qidx, swapped_with_position=pidx, query_person=qname, partner_person=names[pidx]))
        ad = ['query cut before the object: prompt = context + " Person <Name> <verb phrase> the", answer " <object>"',
              'flip (our construction): exchange the person names of the queried clause and one other clause (random.Random("42:<task>:<row>")); query unchanged; token permutation, equal length by construction',
              'random common words as objects (the paper uses them to avoid semantic cues), so answers are often not intuitive nouns',
              'rows whose context repeats a person name or an object (%d of %d) are dropped because the binding is ambiguous' % (n_dup, len(rows))]
        rec = kc.write_task(SID, task, 'binding_ordering_id', 'variable binding', pool, lambda i: i['query_position'], len(rows), ad, extra=dict(verb=v, n_clauses=7, n_shots=0))
        tasks_rec.append(rec)
        kept.append(task)
    tasks_rec.append(dict(name='oi_boxes', status='not_converted', notes='bi_data/dataset.jsonl is byte-identical to the Prakash et al. boxes data; converted once as entity_tracking_prakash/boxes_binding'))
    kc.write_config(SID, kept)
    reg = kc.base_registry(SID, man, title='Representational Analysis of Binding in Language Models (ordering-ID binding data)',
                           bibkey='dai-etal-2024-representational', bibkey_data='feng2024languagemodelsbindentities (Binding ID mechanism); prakash2024finetuningenhancesexistingmechanisms (boxes data)', url=man['url'], revision=man['revision'], data_url=man['data_url'], data_revision=man['data_revision'],
                           licence='MIT for the GitHub repository (Copyright 2024 Tohoku NLP Group); the data attachment on the ACL Anthology page has no licence statement of its own',
                           licence_source='external/task_bank/raw/oi_subspace/repo/LICENSE', redistribute='yes (assumed: MIT repo whose README designates the attachment as its data; flagged as an assumption)', tier=1,
                           tokenizer=kc.tokenizer_meta(), status='converted',
                           notes='The repository itself contains only README and LICENSE (README: "GitHub page for more comprehensive analysis will be completed soon").')
    kc.write_registry(SID, reg, tasks_rec)
    for t in tasks_rec:
        print(t['name'], t.get('n_items_converted'), t.get('len_match_share_qwen35'), t.get('single_token_answers_share'))

if __name__ == '__main__':
    main()
