"""Build tier-1 flipping variants of the Nikankin et al. (ICLR 2025) arithmetic prompts, one task per operator.

The repo ships no prompt files: prompts are generated at run time by prompt_generation.py ("x{op}y=" without spaces; operands 0..299 as in the notebook's max_op=300 and op_ranges; legal second-operand ranges from _get_operand_range; results limited to 0..999, Llama-3's single-token range) and then filtered by Llama-3's own correctness (model-dependent, NOT reproduced here). Their patching experiments pair each prompt with a random other prompt (clean/corrupt answers differ). We re-implement the model-independent part with seed 42 and pair each clean prompt with a random partner of the same operator, same digit counts for both operands, and a different answer.
Division: only exact integer quotients are kept (the original keeps non-integer quotients and truncates in its correctness check).
"""
import collections, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _arith_common as C

SID = 'nikankin_arithmetic'
RAW = os.path.join(C.TB, 'raw', SID)
SEED, CAP = 42, 2000
MAX_OP, MAX_RES = 300, 999
OPS = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}


def second_range(op, x):  # copied logic of prompt_generation._get_operand_range
    lo = 1 if op == '/' else 0
    if op == '+': return range(lo, min(MAX_RES - x, MAX_OP))
    if op == '-': return range(lo, x + 1)
    if op == '*': return range(lo, MAX_OP) if x == 0 else range(lo, min(MAX_RES // x + 1, MAX_OP))
    return range(1, MAX_OP)


def pool(op):
    out = []
    for x in range(0, MAX_OP):
        for y in second_range(op, x):
            r = {'+': x + y, '-': x - y, '*': x * y}.get(op)
            if op == '/':
                if x % y: continue
                r = x // y
            if 0 <= r <= MAX_RES:
                out.append((x, y, r))
    return out


def main():
    man = __import__('json').load(open(os.path.join(RAW, 'MANIFEST.json')))
    cfgs, tasks_reg = {}, []
    for name, op in OPS.items():
        P = pool(op)
        rng = random.Random(SEED)
        strata = collections.defaultdict(list)
        for p in P: strata[(len(str(p[0])), len(str(p[1])))].append(p)
        keys = sorted(strata)
        quota = {k: CAP * len(strata[k]) / len(P) for k in keys}
        alloc = {k: int(quota[k]) for k in keys}
        for k in sorted(keys, key=lambda k: (-(quota[k] - alloc[k]), k))[:CAP - sum(alloc.values())]: alloc[k] += 1
        clean = []
        for k in keys:
            s = sorted(strata[k]); rng.shuffle(s); clean += s[:alloc[k]]
        clean.sort()
        items, dropped = [], 0
        for i, (x, y, r) in enumerate(clean):
            cands = [q for q in strata[(len(str(x)), len(str(y)))] if q[2] != r]
            if not cands: dropped += 1; continue
            x2, y2, r2 = rng.choice(cands)
            items.append(dict(clean_problem=f'{x}{op}{y}=', corrupted_problem=f'{x2}{op}{y2}=',
                              clean_correct=str(r), clean_incorrect=str(r2), corrupted_correct=str(r2), corrupted_incorrect=str(r),
                              source_id=SID, source_item_id=f'{name}_{x}_{y}', operator=op, operand1=x, operand2=y,
                              operand1_corrupted=x2, operand2_corrupted=y2))
        C.check_flip(items)
        C.write_json(os.path.join(C.TB, 'tasks', SID, f'{name}.json'), items)
        open(os.path.join(C.TB, 'tasks', SID, f'{name}.ids.txt'), 'w').write('\n'.join(i['source_item_id'] for i in items) + '\n')
        cfgs[name] = dict(path=f'{name}.json', prefix=None, suffix=None, concatenation=None, merging_sign='', force_base_evaluation=True)
        eq, single = C.length_stats(items, cfgs[name])
        tasks_reg.append(dict(name=name, family=f'arithmetic_{name}_numeric_operand_swap', domain='arithmetic', n_items_raw=len(P),
                              n_items_converted=len(items), sampled_ids_file=f'tasks/{SID}/{name}.ids.txt',
                              len_match_share_qwen35=eq, single_token_answer_share_qwen35=single,
                              adaptations='model-independent re-implementation of prompt generation (no Llama-3 correctness filter); partner prompt = random same-operator prompt with equal operand digit counts and different answer; ' + ('exact integer quotients only; ' if op == '/' else '') + f'{dropped} clean prompts without partner dropped; stratified on operand digit counts'))
        print(name, len(P), len(items), eq, single)
    C.write_json(os.path.join(C.TB, 'configs', f'{SID}.json'), cfgs)
    key = {p: s for p, s in man['files'].items() if p.split('/')[-1] in ('LICENSE', 'prompt_generation.py', 'README.md', 'activation_patching.py', 'model_analysis_consts.py', 'llm-arithmetic-analysis-main.ipynb', '.gitignore')}
    C.write_yaml(os.path.join(C.TB, 'registry', f'{SID}.yaml'), dict(
        source_id=SID, title='Arithmetic Without Algorithms: Language Models Solve Math With a Bag of Heuristics (prompt generator)',
        bibkey='nikankin2025arithmetic', bibkey_data=None, url=man['url'], revision=man['commit'], download_date=man['download_date'],
        raw_files=[dict(path=f'external/task_bank/raw/{SID}/{p}', sha256=s) for p, s in key.items()] + [dict(path=f'external/task_bank/raw/{SID}/MANIFEST.json', sha256=C.sha256(os.path.join(RAW, 'MANIFEST.json')), note='sha256 of all 40 tracked files')],
        licence='MIT (Copyright (c) 2024 technion-cs-nlp)', licence_source='LICENSE at the pinned commit', redistribute='yes',
        generator=dict(commit=man['commit'], seed=SEED, params=f'own re-implementation of prompt_generation.generate_all_prompts_for_operator: operands 0..{MAX_OP - 1}, results 0..{MAX_RES}, cap {CAP}/operator'),
        tier=1, tasks=tasks_reg, status='converted',
        notes=dict(neuron_lists='No neuron index files are committed (.gitignore excludes *.pt, per_neuron_*, mlp_input_*, data/mean_cache*; the repo has no data/ directory). Neurons must be recomputed with script_per_neuron_analysis.py / script_topk_neuron_eval.py on Llama-3-8B/70B or Pythia-6.9B; per-neuron results are produced, not released. Not usable as a neuron list for Qwen anyway.',
                   model_dependence='the original prompts are those Llama-3-8B answers correctly; here all legal prompts are kept (model-independent), the run-time both-correct filter plays that role for Qwen.',
                   operators='the original also pairs prompts of different operators in some experiments ("Corrupt prompts ... differ in operator and/or operands"); only same-operator operand swaps are built here.',
                   answers='no leading space ("x+y=" then digits), as in the original')))


if __name__ == '__main__':
    main()
