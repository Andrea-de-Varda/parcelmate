"""Convert the entity-tracking (boxes) data of Prakash et al. (ICLR 2024; Nix07/finetuning, data/dataset.jsonl) to tier-1 items by a binding swap.

dataset.jsonl: 15,400 rows = 2,200 contexts x 7 queries. A context is "The document is in Box X, the pot is in Box T, ... the map is in Box D." (7 object-box clauses, all
numops = 0, i.e. no state-changing operations) followed by one query "Box X contains the document." We cut the query before the object: prompt = context + " Box X contains the",
answer = " document". Flip (our construction): the labels of the queried box and of one other box (chosen with random.Random('42:<row>')) are exchanged in the context clauses, the query box
is unchanged, so the object found in the queried box flips from the document to the pot; the two prompts are permutations of the same tokens (equal length by construction).
Cap 2000 items, stratified on the position of the queried box (7 strata). The identical file also ships in the Dai et al. OI-Subspace bundle (sha256 checked in the registry) and is converted only here.
Difficulty note: the paper's models are 7B (LLaMA, Vicuna, Goat, Float; base LLaMA-7B ~ 0.6 on its harder tasks); this file has no state changes, so it is a pure binding lookup and should be much easier than the later operations data."""
import json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knowledge_common as kc

SID = 'entity_tracking_prakash'
CLAUSE = re.compile(r'(?:The|the) (\w+) is in Box (\w)')

def main():
    man = kc.load_manifest(SID)
    kc.clear_tasks(SID)
    path = os.path.join(kc.RAW, SID, 'repo', 'data', 'dataset.jsonl')
    rows = [json.loads(l) for l in open(path)]
    pool = []
    for i, r in enumerate(rows):
        s = r['sentence']
        ctx, query = s.rsplit('. ', 1)
        ctx += '.'
        m = re.fullmatch(r'Box (\w) contains the (\w+)\.', query)
        assert m, query
        qbox, qobj = m.groups()
        clauses = CLAUSE.findall(ctx)
        assert len(clauses) == 7 and dict((b, o) for o, b in clauses)[qbox] == qobj, s
        boxes = [b for o, b in clauses]
        qidx = boxes.index(qbox)
        rng = random.Random('42:%d' % i)
        pidx = rng.choice([j for j in range(7) if j != qidx])
        pbox, pobj = boxes[pidx], clauses[pidx][0]
        # exchange the box labels of clauses qidx and pidx
        parts = re.split(r'(?<=\w)(?=, the )', ctx)
        def swap_label(part, a, b):
            return re.sub(r'Box (\w)(\.?)$', lambda mm: 'Box %s%s' % (b, mm.group(2)), part)
        parts[qidx] = swap_label(parts[qidx], qbox, pbox)
        parts[pidx] = swap_label(parts[pidx], pbox, qbox)
        cor_ctx = ''.join(parts)
        clean = ctx + ' Box %s contains the' % qbox
        corr = cor_ctx + ' Box %s contains the' % qbox
        pool.append(kc.make_item(clean, corr, ' ' + qobj, ' ' + pobj, SID, 'dataset.jsonl:%d' % i, sample_id=r['sample_id'], query_position=qidx, swapped_with_position=pidx, query_box=qbox,
                                 partner_box=pbox))
    ad = ['query cut before the object: prompt = context + " Box X contains the", answer " <object>" (the source row is the full sentence "Box X contains the <object>.")',
          'flip (our construction): exchange the box labels of the queried box and one other box in the context (partner drawn with random.Random("42:<row>")); the query is unchanged; token permutation, equal length by construction',
          'only the numops = 0 data of the file exists (no put/move/remove operations)']
    rec = kc.write_task(SID, 'boxes_binding', 'entity_tracking_boxes', 'entity tracking', pool, lambda i: i['query_position'], len(rows), ad,
                        extra=dict(n_contexts=len({r['sample_id'] for r in rows}), n_boxes=7, n_shots=0))
    kc.write_config(SID, ['boxes_binding'])
    reg = kc.base_registry(SID, man, title='Entity tracking (boxes) data of "Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking"',
                           bibkey='prakash2024finetuningenhancesexistingmechanisms', bibkey_data='kim-schuster-2023-entity (original boxes task; its password-protected data is not used)', url=man['url'], revision=man['revision'],
                           licence='none: the repository has no LICENSE file at the pinned commit and the README states none (the README only asks to cite the paper)',
                           licence_source='external/task_bank/raw/entity_tracking_prakash/repo (no LICENSE file)', redistribute='no (no licence: keep converted items out of git)', tier=1,
                           tokenizer=kc.tokenizer_meta(), status='converted',
                           notes='data/dataset.jsonl is byte-identical to model/bi_data/dataset.jsonl of the Dai et al. OI-Subspace bundle (sha256 c2631807e1900a3f66c323bc6df2b3ae88da1a31f4ce9ee6c2424dbe6d3702b9); converted once here. The repo also has data/data_utils.py (the authors desiderata-based counterfactuals for path patching: not used, they do not flip the answer symmetrically).')
    kc.write_registry(SID, reg, [rec])
    print(rec['name'], rec['n_items_converted'], rec['len_match_share_qwen35'], rec['single_token_answers_share'])

if __name__ == '__main__':
    main()
