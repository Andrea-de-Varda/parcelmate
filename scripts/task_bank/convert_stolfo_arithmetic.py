"""Build tier-1 tasks following Stolfo, Belinkov & Sachan (EMNLP 2023) two-operand arithmetic interventions.

The repo ships no data files: interventions/intervention.py samples prompts at run time (Hydra config, seed 0, GPT-J tokenizer). We re-implement the sampling with seed 42, using the repo's templates verbatim, operands 1..300 and results 1..299 (conf/config.yaml: max_n=300), intervention types:
  type 1 (operand swap, one task per operator): clean and alternative operands drawn independently; we require equal digit counts per operand position (as the repo does in its Llama setting, max_n=9) and different results (the repo only enforces the latter in that setting).
  type 3 (operator change, one task): same operands, different operator (random other operator), same template index; both results must be integers in 1..299 and differ.
Templates are cycled round-robin so every task is template-balanced. Division keeps integer quotients only (the repo's is_int check). Zero-shot, representation 'arabic' only; the 'words' representation and the three-operand equations are not converted (see report).
"""
import json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _arith_common as C

SID = 'stolfo_arithmetic'
RAW = os.path.join(C.TB, 'raw', SID)
SEED, N = 42, 2000
MAX_N = 300
TEMPL = {
    '+': ['Q: How much is {x} plus {y}? A:', 'Q: What is {x} plus {y}? A:', 'Q: What is the result of {x} plus {y}? A:',
          'Q: What is the sum of {x} and {y}? A:', 'The sum of {x} and {y} is', '{x} + {y} ='],
    '-': ['Q: How much is {x} minus {y}? A:', 'Q: What is {x} minus {y}? A:', 'Q: What is the result of {x} minus {y}? A:',
          'Q: What is the difference between {x} and {y}? A:', 'The difference between {x} and {y} is', '{x} - {y} ='],
    '*': ['Q: How much is {x} times {y}? A:', 'Q: What is {x} times {y}? A:', 'Q: What is the result of {x} times {y}? A:',
          'Q: What is the product of {x} and {y}? A:', 'The product of {x} and {y} is', '{x} * {y} ='],
    '/': ['Q: How much is {x} over {y}? A:', 'Q: What is {x} over {y}? A:', 'Q: What is the result of {x} over {y}? A:',
          'Q: What is the ratio between {x} and {y}? A:', 'The ratio of {x} and {y} is', '{x} / {y} ='],
}
NAMES = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div'}


def ev(x, op, y):
    r = {'+': x + y, '-': x - y, '*': x * y}.get(op) if op != '/' else (x // y if x % y == 0 else None)
    return r if r is not None and 1 <= r < MAX_N else None


def fill(t, x, y): return t.replace('{x}', str(x)).replace('{y}', str(y))


def main():
    rng = random.Random(SEED)
    man = json.load(open(os.path.join(RAW, 'MANIFEST.json')))
    cfgs, treg = {}, []
    spec = [(NAMES[o], o) for o in '+-*/'] + [('operator_change', None)]
    for name, op in spec:
        items, seen = [], set()
        while len(items) < N:
            t_id = len(items) % 6
            if op:
                while True:
                    x, y = rng.randint(1, MAX_N), rng.randint(1, MAX_N)
                    r = ev(x, op, y)
                    if r: break
                while True:
                    x2, y2 = rng.randint(1, MAX_N), rng.randint(1, MAX_N)
                    r2 = ev(x2, op, y2)
                    if r2 and r2 != r and len(str(x)) == len(str(x2)) and len(str(y)) == len(str(y2)): break
                t = TEMPL[op][t_id]; t2 = t; ops = op
                a, b = fill(t, x, y), fill(t2, x2, y2)
                meta = dict(operator=op, operands=[x, y], operands_corrupted=[x2, y2], template_id=t_id)
            else:
                while True:
                    o1 = rng.choice('+-*/'); o2 = rng.choice([o for o in '+-*/' if o != o1])
                    x, y = rng.randint(1, MAX_N), rng.randint(1, MAX_N)
                    r, r2 = ev(x, o1, y), ev(x, o2, y)
                    if r and r2 and r != r2: break
                a, b = fill(TEMPL[o1][t_id], x, y), fill(TEMPL[o2][t_id], x, y)
                meta = dict(operator=o1, operator_corrupted=o2, operands=[x, y], template_id=t_id)
            key = (a, b)
            if key in seen: continue
            seen.add(key)
            items.append(dict(clean_problem=a, corrupted_problem=b, clean_correct=f' {r}', clean_incorrect=f' {r2}',
                              corrupted_correct=f' {r2}', corrupted_incorrect=f' {r}', source_id=SID,
                              source_item_id=f'{name}_{len(items):04d}', **meta))
        C.check_flip(items)
        C.write_json(os.path.join(C.TB, 'tasks', SID, f'{name}.json'), items)
        open(os.path.join(C.TB, 'tasks', SID, f'{name}.ids.txt'), 'w').write('\n'.join(i['source_item_id'] for i in items) + '\n')
        cfgs[name] = dict(path=f'{name}.json', prefix=None, suffix=None, concatenation=None, merging_sign='', force_base_evaluation=True)
        eq, single = C.length_stats(items, cfgs[name])
        fam = 'arithmetic_operator_change_numeric' if not op else f'arithmetic_{NAMES[op]}_numeric_operand_swap'
        treg.append(dict(name=name, family=fam, domain='arithmetic', n_items_raw='generated (no raw data files)', n_items_converted=len(items),
                         sampled_ids_file=f'tasks/{SID}/{name}.ids.txt', len_match_share_qwen35=eq, single_token_answer_share_qwen35=single,
                         adaptations='own seed-42 re-implementation of intervention.py sampling (repo has no data files); templates verbatim; template round-robin; ' + ('equal digit counts per operand position and different results; ' if op else 'same operands, random different operator, results differ; ') + 'integer results 1..299 only'))
        print(name, len(items), eq, single)
    C.write_json(os.path.join(C.TB, 'configs', f'{SID}.json'), cfgs)
    key = {p: s for p, s in man['files'].items() if p.split('/')[-1] in ('LICENSE', 'README.md', 'intervention.py', 'config.yaml', 'math_cma.py', 'three_operand_questions.json')}
    C.write_yaml(os.path.join(C.TB, 'registry', f'{SID}.yaml'), dict(
        source_id=SID, title='A Mechanistic Interpretation of Arithmetic Reasoning in Language Models using Causal Mediation Analysis (prompt generator)',
        bibkey='stolfo2023mechanistic', bibkey_data=None, url=man['url'], revision=man['commit'], download_date=man['download_date'],
        raw_files=[dict(path=f'external/task_bank/raw/{SID}/{p}', sha256=s) for p, s in key.items()] + [dict(path=f'external/task_bank/raw/{SID}/MANIFEST.json', sha256=C.sha256(os.path.join(RAW, 'MANIFEST.json')), note='sha256 of all 19 tracked files')],
        licence='MIT (Copyright (c) 2023 Alessandro Stolfo)', licence_source='LICENSE at the pinned commit', redistribute='yes',
        generator=dict(commit=man['commit'], seed=SEED, params=f'own re-implementation (not the repo code, which needs Hydra, torch and a model tokenizer): max_n={MAX_N}, n_shots=0, representation=arabic, {N} items per task'),
        tier=1, tasks=treg, status='converted',
        notes=dict(not_converted='intervention type 2 (same result, no flip), words representation, three-operand equations (three_operand_questions.json), number-retrieval and LAMA factual tasks',
                   answers='leading space before the digits (the repo appends " {res}" in few-shot exemplars)',
                   operator_change='the repo draws operator pairs among + - * /; so some pairs are not add/sub.')))


if __name__ == '__main__':
    main()
