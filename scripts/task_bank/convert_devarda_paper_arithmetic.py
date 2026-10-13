"""Convert the arithmetic data of de Varda et al. (arXiv 2609.04463) into four tier-1 tasks.

Raw: raw/devarda_paper_arithmetic (see fetch_devarda_paper_arithmetic.py): 5,000 problems x numeric /
English / Spanish / Italian, ids aligned, each with its operator-flipped counterfactual (x, x', y, y').

Decisions (Andrea, 2026-10-12): the paper's data replace the public-generator conversion for these four
formats; one-shot prompts with the paper's exemplars; problems whose original or flipped result is negative
are skipped. Sample: the paper's 2,000-id attribution fold restricted to non-negative problems, filled to
2,000 with a seed-42 draw from the remaining non-negative problems, so the bank overlaps the paper's items
as much as possible. The same ids are used in every format.
"""
import ast, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _arith_common as C

SID = 'devarda_paper_arithmetic'
RAW = os.path.join(C.TB, 'raw', SID)
CAP, SEED = 2000, 42
FORMATS = {'numeric': 'arithmetic_addsub_numeric', 'english': 'arithmetic_addsub_verbal_en',
           'spanish': 'arithmetic_addsub_verbal_es', 'italian': 'arithmetic_addsub_verbal_it'}


def exemplars():
    """ONE_SHOT_EXEMPLARS read from the paper's own data.py, so the prompts match it exactly."""
    tree = ast.parse(open(os.path.join(RAW, 'attribution', 'src', 'data.py')).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'ONE_SHOT_EXEMPLARS':
            return ast.literal_eval(node.value)
    raise KeyError('ONE_SHOT_EXEMPLARS')


def main():
    ex = exemplars()
    data = {f: json.load(open(os.path.join(RAW, 'arithmetic-dataset', 'arith_dataset_%s_counterfactual.json' % f)))
            for f in FORMATS}
    ids = [x['id'] for x in data['numeric']]
    assert all([x['id'] for x in data[f]] == ids for f in FORMATS)
    num = {x['id']: x for x in data['numeric']}
    ok = sorted(i for i in ids if num[i]['ground_truth'] >= 0 and num[i]['ground_truth_prime'] >= 0
                and num[i]['ground_truth'] != num[i]['ground_truth_prime'])
    fold = json.load(open(os.path.join(RAW, 'attribution', 'data', 'attribution_fold_ids.json')))
    in_fold = sorted(set(fold) & set(ok))
    rest = sorted(set(ok) - set(fold))
    random.Random(SEED).shuffle(rest)
    kept = sorted(in_fold + rest[:CAP - len(in_fold)])
    assert len(kept) == CAP

    outdir = os.path.join(C.TB, 'tasks', SID)
    os.makedirs(outdir, exist_ok=True)
    cfgs, tasks_reg = {}, []
    for f, family in FORMATS.items():
        by_id = {x['id']: x for x in data[f]}
        items = []
        for pid in kept:
            x = by_id[pid]
            items.append(dict(clean_problem=x['x'], corrupted_problem=x['x_prime'], clean_correct=x['y'],
                              clean_incorrect=x['y_prime'], corrupted_correct=x['y_prime'], corrupted_incorrect=x['y'],
                              source_id=SID, source_item_id=pid, format=f, in_paper_fold=pid in fold,
                              n_terms=x['n_terms'], n_digits=x['n_digits'], has_carry=x['has_carry'],
                              ground_truth=x['ground_truth'], ground_truth_prime=x['ground_truth_prime'], split=x['split']))
        C.check_flip(items)
        C.write_json(os.path.join(outdir, '%s.json' % f), items)
        open(os.path.join(outdir, '%s.ids.txt' % f), 'w').write('\n'.join(kept) + '\n')
        cfgs[f] = dict(path='%s.json' % f, prefix=ex[f], suffix=None, concatenation=None, merging_sign='',
                       force_base_evaluation=True)
        eq, single = C.length_stats(items, cfgs[f])
        tasks_reg.append(dict(name=f, family=family, domain='arithmetic', n_items_raw=len(ids),
                              n_items_converted=len(items), sampled_ids_file='tasks/%s/%s.ids.txt' % (SID, f),
                              len_match_share_qwen35=eq, single_token_answer_share_qwen35=single,
                              adaptations='none to the items (the paper\'s own x, x\', y, y\'); one-shot exemplar of the paper as prefix; negative-result problems skipped'))
        print(f, len(items), 'len-match', eq, 'single-token answers', single)
    C.write_json(os.path.join(C.TB, 'configs', '%s.json' % SID), cfgs)

    man = json.load(open(os.path.join(RAW, 'MANIFEST.json')))
    C.write_yaml(os.path.join(C.TB, 'registry', '%s.yaml' % SID), dict(
        source_id=SID, title='Arithmetic counterfactual data of "Shared circuits predict whether LLMs generalize across formats in arithmetic reasoning"',
        bibkey='devarda2026sharedcircuits', bibkey_data='devarda2026arithmeticdataset',
        url=man['url'], revision=man['commit'], download_date=man['download_date'],
        raw_files=[dict(path='external/task_bank/raw/%s/%s' % (SID, p), sha256=s) for p, s in man['files'].items()],
        licence='project owner\'s own data (generated with github.com/Andrea-de-Varda/arithmetic-dataset, MIT per its README)',
        licence_source='Andrea de Varda (author)', redistribute='yes (author\'s own data)',
        tier=1, tasks=tasks_reg, status='converted',
        notes=dict(
            replaces='the numeric/english/spanish/italian tasks of source devarda_arithmetic (public-generator conversion), which stay on disk but are excluded from evaluation (eval_tasks.txt).',
            sample='%d non-negative problems of %d; %d of the paper\'s 2,000-id fold are non-negative and all kept; %d filled with a seed-42 draw from the other non-negative problems.' % (len(ok), len(ids), len(in_fold), CAP - len(in_fold)),
            prompt_framing='prefix = the paper\'s ONE_SHOT_EXEMPLARS (attribution/src/data.py), no suffix, no chat template, as in the paper.',
            word_problems='the paper\'s data have no word-problem (embedded) formats; those remain in source devarda_arithmetic, where the corrupted story is not a minimal edit.')))


if __name__ == '__main__':
    main()
