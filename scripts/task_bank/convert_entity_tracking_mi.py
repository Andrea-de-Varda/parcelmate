"""Convert the box data of "Do Language Models Track Entities Across State Changes?" (Tang et al., ICML 2026; PootieT/entity-tracking-mi generator, seed 42) to tier-1 items.

fetch_entity_tracking_mi.py generated three datasets with the paper's own parameter sets (1 remove, 1 move, 1 put with one object per phrase). Each JSONL row is one
(world state, box) pair: `sentence_masked` = initial-state description + operation(s) + "Box <q> contains", `masked_content` = the gold content ("the sugar", "the photo and the bell", "nothing").
Rows of the same sample and the same context form a group (7 boxes). We keep rows whose gold content is a single object ("the X") and cut the prompt at "Box <q> contains the" (answer " X").
Flip (our construction): within a group, the clean prompt queries box q1 (answer X1) and the corrupted prompt queries another box q2 with a different single object X2; the context is shared and only
the box digit changes (equal length). This is the query-box counterfactual of the authors' generator ("rand_query_id") restricted to boxes with a unique gold answer; their rand_obj / rand_box_id
counterfactuals were not used (they do not define a flipped answer).
Tasks: et_state_noop (initial state only; the 0-operation rows of the 1remove run: any two boxes with single objects), et_state_1remove / et_state_1move (the clean box is a box changed by the single
operation and still holds exactly one object, so the model must track the operation; the partner is any other single-object box, one random partner per clean box).
Not fitting tier 1: 1put (the changed box always ends with two objects and the generator writes them in random order, so the gold answer is a set, no single teacher-forced continuation; tier-2 candidate if the
first-mentioned object is accepted), and the three pilot files shipped in data/ (noop_remove, reintroduce_removed_object, shared_label_object_remove: set-valued gold contents).
Difficulty: the paper evaluates 2B-70B models and reports that zero-shot state tracking degrades with the number of operations; with one operation on 7 boxes, Qwen3.5-2B/4B may sit near the
both-correct threshold, expected difficulty unknown (no inference run). Seed 42; all valid pairs kept up to the 2000 cap, stratified on the sample's operation type."""
import collections, json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knowledge_common as kc

SID = 'entity_tracking_mi'
RUNS = {'et_state_noop': ('boxes_AltForm_1remove', 0), 'et_state_1remove': ('boxes_AltForm_1remove', 1), 'et_state_1move': ('boxes_AltForm_1move', 1)}

def groups_of(path):
    g = collections.OrderedDict()
    with open(path) as f:
        for ln, line in enumerate(f):
            r = json.loads(line)
            ctx, q = r['sentence_masked'].rsplit(' Box ', 1)
            g.setdefault((r['sample_id'], ctx), []).append((int(q.split()[0]), r, ln))
    return g

def main():
    man = kc.load_manifest(SID)
    kc.clear_tasks(SID)
    tasks_rec, kept = [], []
    for task, (run, nops) in RUNS.items():
        path = os.path.join(kc.RAW, SID, 'generated', run, 'train-t5.jsonl')
        g = groups_of(path)
        pool = []
        n_rows = 0
        for (sid, ctx), members in g.items():
            n_rows += len(members)
            has_op = bool(re.search(r' (Remove|Move|Put) ', ' ' + ctx))
            if has_op != bool(nops):
                continue
            single = [(q, r, ln) for q, r, ln in members if re.fullmatch(r'the \w+', r['masked_content'])]
            clean_c = [m for m in single if (m[1]['numops'] >= 1)] if nops else single
            rng = random.Random('42:%s:%d' % (task, sid))
            if not nops:
                rng.shuffle(clean_c)
                clean_c = clean_c[:1]  # one pair per context for the 0-operation task
            for q1, r1, ln1 in clean_c:
                others = [m for m in single if m[0] != q1 and m[1]['masked_content'] != r1['masked_content']]
                if not others:
                    continue
                q2, r2, ln2 = rng.choice(others)
                a1, a2 = r1['masked_content'][4:], r2['masked_content'][4:]
                op = re.findall(r'(Remove|Move|Put) ([^.]*)\.', ctx)
                pool.append(kc.make_item(ctx + ' Box %d contains the' % q1, ctx + ' Box %d contains the' % q2, ' ' + a1, ' ' + a2, SID, '%s:%d|%d' % (run, ln1, ln2),
                                         sample_id=sid, operation=(op[0][0].lower() if op else 'none'), operation_text=(op[0][0] + ' ' + op[0][1] + '.') if op else '',
                                         clean_box=q1, corrupted_box=q2, clean_box_numops=r1['numops']))
        ad = ['generated with the paper\'s own parameter set (%s) and seed 42 instead of their default 2255 (patched copy of their generator for Python >= 3.11, PYTHONHASHSEED=0)' % run,
              'rows with a single-object gold content only; prompt cut at "Box <q> contains the", answer " <object>"',
              'flip (our construction): same context, query box q1 vs a different query box q2 holding a different single object (the generator\'s rand_query_id counterfactual restricted to unambiguous boxes); equal length (one digit)',
              ('clean box = a box changed by the operation that still holds exactly one object; partner = random other single-object box with a different object' if nops else 'one random pair of single-object boxes per context')]
        rec = kc.write_task(SID, task, 'entity_tracking_boxes_state' if nops else 'entity_tracking_boxes', 'entity tracking', pool, lambda i: i['operation'], n_rows, ad,
                            extra=dict(generator_run=run, n_operations=nops, n_boxes=7, n_shots=0))
        if rec['n_items_converted'] < 300:
            rec['notes'] = 'below the 300-item target'
        tasks_rec.append(rec)
        kept.append(task)
    tasks_rec.append(dict(name='et_state_1put', status='not_fitting_tier1', notes='the changed box ends with two objects listed in random order (set-valued gold); candidate for tier 2 only if the first-mentioned object is accepted'))
    for f in ['noop_remove', 'reintroduce_removed_object', 'shared_label_object_remove']:
        tasks_rec.append(dict(name='pilot_' + f, status='not_fitting_tier1', notes='pilot file of the paper for the Remove degeneracy analysis; gold contents are sets (e.g. ["drink","sheet"]) and prompts end in "Box k contains the"'))
    kc.write_config(SID, kept)
    gen_runs = man.get('generator_runs')
    reg = kc.base_registry(SID, man, title='Do Language Models Track Entities Across State Changes? (box-data generator and pilot datasets)',
                           bibkey='tang2026languagemodelstrackentities', bibkey_data='kim-schuster-2023-entity (the generator derives from their code)', url=man['url'], revision=man['revision'],
                           licence='MIT (Copyright 2026 Zilu (Peter) Tang)', licence_source='external/task_bank/raw/entity_tracking_mi/repo/LICENSE', redistribute='yes (MIT; the generator derives from Kim & Schuster 2023 code, whose licence was not checked; their password-protected data is not used)',
                           tier=1, tokenizer=kc.tokenizer_meta(), status='converted',
                           generator=dict(commit=man['revision'], seed=42, runs=gen_runs, patch='random.sample(x, k) -> random.sample(sorted(x), k) at two call sites (Python >= 3.11 compatibility); PYTHONHASHSEED=0; run with python -s -P'),
                           notes='The paper additionally offers data on Google Drive (not fetched; the generator reproduces it).')
    kc.write_registry(SID, reg, tasks_rec)
    for t in tasks_rec:
        print(t['name'], t.get('n_items_raw'), t.get('n_items_converted'), t.get('len_match_share_qwen35'), t.get('single_token_answers_share'))

if __name__ == '__main__':
    main()
