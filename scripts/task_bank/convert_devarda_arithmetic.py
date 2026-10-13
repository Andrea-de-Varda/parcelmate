"""Convert the de Varda et al. arithmetic dataset (raw/devarda_arithmetic, regenerated at seed 42) into tier-1 operator-flip tasks, one per format.

Corruption: every operator flipped (+ <-> -), as in the repo's generate_arithmetic_counterfactual.py (which only covers the numeric format and was run for a different 5,000-problem sample; so we rebuild x, x', y, y' for the 10,000 seed-42 problems using the repo's own DatasetFormatter for all six formats).
Embedded formats: the repo's formatter seeds characters/object from sha256(id|operands|operators), so a flipped problem would get different names; we replay the CLEAN problem's seed so names and object are shared (the story template itself necessarily changes between addition and subtraction).
"""
import hashlib, importlib.util, json, os, random, re, sys, types, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _arith_common as C

RAW = os.path.join(C.TB, 'raw', 'devarda_arithmetic')
REPO, GEN = os.path.join(RAW, 'repo'), os.path.join(RAW, 'generated')
SID = 'devarda_arithmetic'
SEED, CAP = 42, 2000
FORMATS = {  # task -> (format key, family, domain)
    'numeric': ('numeric', 'arithmetic_addsub_numeric'),
    'english': ('en', 'arithmetic_addsub_verbal_en'),
    'spanish': ('es', 'arithmetic_addsub_verbal_es'),
    'italian': ('it', 'arithmetic_addsub_verbal_it'),
    'embedded': ('embedded', 'arithmetic_addsub_wordproblem'),
    'embedded_verbal': ('embedded_verbal', 'arithmetic_addsub_wordproblem_verbal'),
}
EX = {  # one-shot exemplars from eval_models.py ONE_SHOT_EXAMPLES (text before "{prompt}"); suffixes from PROMPT_SUFFIXES
    'numeric': "7 + 5 = 12\n", 'english': "seven plus five equals twelve\n",
    'spanish': "siete más cinco es igual a doce\n", 'italian': "sette più cinque fa dodici\n",
    'embedded': "Question: Alice has 3 marbles and finds 2 more. How many does she have now?\nAnswer: 5\n",
    'embedded_verbal': "Question: Alice has three marbles and finds two more. How many does she have now?\nAnswer: five\n",
}
SUFFIX = {'embedded': ' The answer is', 'embedded_verbal': ' The answer is'}


def load_module():
    spec = importlib.util.spec_from_file_location('gad', os.path.join(REPO, 'generate_arithmetic_data.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _Hash:  # shim so the formatter's story seed can be replayed from the clean problem
    override = None

    @classmethod
    def sha256(cls, data):
        return hashlib.sha256(cls.override.encode() if cls.override is not None else data)


def parse(prompt):
    toks = prompt[:-2].split(' ')  # strip ' ='
    return [int(t) for t in toks[0::2]], toks[1::2]


def evaluate(operands, ops):
    r = operands[0]
    for o, n in zip(ops, operands[1:]):
        r = r + n if o == '+' else r - n
    return r


def main():
    gad = load_module()
    fmt = gad.DatasetFormatter()
    gad.hashlib = _Hash
    data = {t: json.load(open(os.path.join(GEN, 'data', 'json', f'arith_dataset_{t}.json'))) for t in FORMATS}
    n = len(data['numeric'])
    per_item = {}
    for i, base in enumerate(data['numeric']):
        operands, ops = parse(base['prompt'])
        assert evaluate(operands, ops) == base['ground_truth']
        flipped = ['-' if o == '+' else '+' for o in ops]
        clean = dict(id=base['id'], operands=operands, operators=ops, result=base['ground_truth'])
        corr = dict(id=base['id'], operands=operands, operators=flipped, result=evaluate(operands, flipped))
        rec = dict(meta=base, ops=''.join(ops), gt=clean['result'], gt_prime=corr['result'], fm={})
        for task, (k, _) in FORMATS.items():
            def f(p, replay=None):
                _Hash.override = replay
                try:
                    if k == 'numeric': return fmt.format_numeric(p)
                    if k in ('en', 'es', 'it'): return fmt.format_verbal(p, k)
                    return fmt.format_embedded_context(p, verbal=(k == 'embedded_verbal'))
                finally:
                    _Hash.override = None
            x, y = f(clean)
            seed_clean = f"{clean['id']}|{clean['operands']}|{clean['operators']}"
            xp, yp = f(corr, replay=seed_clean)
            xv, yv = f(corr)  # repo-verbatim corrupted story (different names), kept for the diagnostic only
            ref = data[task][i]
            assert ref['id'] == base['id'] and ref['prompt'] == x and ref['answer'] == y, (task, i)
            rec['fm'][task] = (x, xp, y, yp, xv)
        per_item[base['id']] = rec

    # eligibility: identical for all formats so ids stay matched
    drop = collections.Counter()
    eligible = []
    for pid, r in per_item.items():
        if r['gt_prime'] < 0:
            drop['negative_flipped_result'] += 1
        elif any(v[2] == v[3] for v in r['fm'].values()):
            drop['y_equals_y_prime_in_some_format'] += 1
        else:
            eligible.append(pid)
    # stratified deterministic sample on (n_terms, n_digits, has_carry, operator pattern); proportional allocation, largest remainder
    strata = collections.defaultdict(list)
    for pid in eligible:
        m = per_item[pid]['meta']
        strata[(m['n_terms'], m['n_digits'], m['has_carry'], per_item[pid]['ops'])].append(pid)
    rng = random.Random(SEED)
    keys = sorted(strata, key=str)
    quota = {k: CAP * len(strata[k]) / len(eligible) for k in keys}
    alloc = {k: int(quota[k]) for k in keys}
    for k in sorted(keys, key=lambda k: (-(quota[k] - alloc[k]), str(k)))[:CAP - sum(alloc.values())]:
        alloc[k] += 1
    kept = []
    for k in keys:
        pool = sorted(strata[k])
        rng.shuffle(pool)
        kept += pool[:alloc[k]]
    kept = sorted(kept)
    assert len(kept) == CAP
    outdir = os.path.join(C.TB, 'tasks', SID)
    os.makedirs(outdir, exist_ok=True)
    open(os.path.join(outdir, 'ids.txt'), 'w').write('\n'.join(kept) + '\n')

    cfgs, tasks_reg = {}, []
    for task, (k, family) in FORMATS.items():
        items = []
        for pid in kept:
            r = per_item[pid]
            x, xp, y, yp, _ = r['fm'][task]
            m = r['meta']
            items.append(dict(clean_problem=x, corrupted_problem=xp, clean_correct=y, clean_incorrect=yp,
                              corrupted_correct=yp, corrupted_incorrect=y, source_id=SID, source_item_id=pid,
                              format=task, n_terms=m['n_terms'], n_digits=m['n_digits'], has_carry=m['has_carry'],
                              operators=r['ops'], ground_truth=r['gt'], ground_truth_prime=r['gt_prime'], split=m['split']))
        C.check_flip(items)
        C.write_json(os.path.join(outdir, f'{task}.json'), items)
        open(os.path.join(outdir, f'{task}.ids.txt'), 'w').write('\n'.join(kept) + '\n')
        cfgs[task] = dict(path=f'{task}.json', prefix=EX[task], suffix=None, concatenation=SUFFIX.get(task),
                          merging_sign='', force_base_evaluation=True)
        eq, single = C.length_stats(items, cfgs[task])
        # diagnostic: verbatim repo corruption (names not matched) for embedded formats
        verb = sum(C.ntok(per_item[p]['fm'][task][0]) == C.ntok(per_item[p]['fm'][task][4]) for p in kept) / len(kept)
        tasks_reg.append(dict(name=task, family=family, domain='arithmetic', n_items_raw=n, n_items_converted=len(items),
                              sampled_ids_file=f'tasks/{SID}/{task}.ids.txt', len_match_share_qwen35=eq,
                              single_token_answer_share_qwen35=single,
                              len_match_share_qwen35_repo_verbatim_corruption=round(verb, 4),
                              adaptations='operator flip applied with the repo formatter to the 10,000 seed-42 problems; '
                                          + ('story names/object replayed from the clean problem; ' if k.startswith('embedded') else '')
                                          + f'items with negative flipped result and items with y==y\' dropped before sampling; 2,000 stratified sample'))
        print(task, len(items), 'len-match', eq, 'single-token answers', single, 'verbatim-len-match', round(verb, 4))
    C.write_json(os.path.join(C.TB, 'configs', f'{SID}.json'), cfgs)

    man = json.load(open(os.path.join(RAW, 'MANIFEST.json')))
    neg_in_pool = sum(r['gt_prime'] < 0 for r in per_item.values())
    reg = dict(
        source_id=SID, title='Arithmetic problem dataset generator (numeric / English / Spanish / Italian / embedded word problems)',
        bibkey='devarda2026sharedcircuits', bibkey_data='devarda2026arithmeticdataset',
        url=man['url'], revision=man['commit'], download_date=man['download_date'],
        raw_files=[dict(path=f'external/task_bank/raw/devarda_arithmetic/{p}', sha256=s) for p, s in man['files'].items()],
        licence='MIT (README.md section "License" says "MIT License"; the repo has no LICENSE file)',
        licence_source='README.md at the pinned commit (last line); no LICENSE file present', redistribute='yes (MIT per README; author of this dataset is the project owner)',
        generator=dict(commit=man['commit'], seed=42, params='python generate_arithmetic_data.py -n 10000 --seed 42 (defaults: terms 0.5/0.5, digits 0.5/0.5, carry 0.5); output reproduces the committed data/json files exactly (all six formats compared); num2words 0.5.14',
                       frozen_output_sha256={p: s for p, s in man['files'].items() if p.startswith('generated/')}),
        tier=1, tasks=tasks_reg, status='converted',
        notes=dict(
            prompt_framing='config prefix = the one-shot exemplar of eval_models.py ONE_SHOT_EXAMPLES, concatenation = PROMPT_SUFFIXES (" The answer is" for the embedded formats, none otherwise). The repo evaluates zero-shot by default (--one-shot is opt-in); the paper attributes with one exemplar per format (per the arXiv HTML, read via a summariser). For a zero-shot run set prefix to null.',
            negative_results=f'{neg_in_pool} of {n} problems have a negative result after the flip; dropped (not representable in the story formats, ambiguous with the word "minus" in verbal formats). Dropped counts: {dict(drop)}; eligible pool {len(eligible)}.',
            answers='answers keep the repo convention of a leading space (" 73", " forty-two"); verbal formats answer in words, numeric and embedded in digits, embedded_verbal in words.',
            sixth_format='the repo also ships embedded_verbal (word problems with number words); converted as a sixth task.',
            repo_counterfactual_file='data/json/arith_dataset_numeric_counterfactual.json in the repo is a different 5,000-problem sample (first item 37 - 63 + 73), not the 10,000 seed-42 set; not used.',
            embedded_corruption='addition and subtraction stories use different template lists, so the embedded corruption is not a minimal edit; only names and object are matched.'))
    C.write_yaml(os.path.join(C.TB, 'registry', f'{SID}.yaml'), reg)
    print('dropped', dict(drop), 'eligible', len(eligible), 'negative flips', neg_in_pool)


if __name__ == '__main__':
    main()
