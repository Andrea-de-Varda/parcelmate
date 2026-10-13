"""Docstring task (Heimersheim & Janiak 2023) from the ACDC copy of the prompt generator (acdc/docstring/prompts.py).
Third-party code is only parsed with `ast` (word lists) and re-implemented here: docstring_prompt_templ and the clean-prompt part of
docstring_induction_prompt_generator, with the ACDC parameters n_matching_args=3, n_def_prefix_args=2, n_def_suffix_args=1, n_doc_prefix_args=0,
met_desc_len=3, arg_desc_len=2. Item seeds are drawn from random.Random(42). N=2000 per style (rest and goog)."""
import ast, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

SID = 'docstring'
RAW = os.path.join(TB, 'raw', 'docstring')
man = json.load(open(os.path.join(RAW, 'MANIFEST.json')))
SRC = os.path.join(RAW, 'Automatic-Circuit-Discovery', 'acdc', 'docstring', 'prompts.py')
tree = ast.parse(open(SRC).read())
VN = CN = None
for node in tree.body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        if node.targets[0].id == 'variable_names':
            VN = ast.literal_eval(node.value.func.value).split(' ')   # "...".split(" ")
        if node.targets[0].id == 'common_single_token_nouns':
            CN = ast.literal_eval(node.value)
assert VN and CN

def templ(style, met_name, met_desc_words, def_args, doc_args, doc_args_desc_words):
    def_args_str = ', '.join(def_args)
    met_desc_str = ' '.join(met_desc_words)
    ind4 = 4 * ' '
    args_line = '\n%sArgs:' % ind4 if style == 'goog' else ''
    def_and_desc = 'def %s(self, %s):\n%s"""%s\n%s' % (met_name, def_args_str, ind4, met_desc_str, args_line)
    param_prefix = '%s:param' % ind4 if style == 'rest' else '%s   ' % ind4
    doc_lines = ['%s %s: %s' % (param_prefix, a, ' '.join(d)) for a, d in zip(doc_args, doc_args_desc_words)]
    return '\n%s\n%s\n%s' % (def_and_desc, '\n'.join(doc_lines), param_prefix)

def gen(style, seed, n_matching=3, n_def_prefix=2, n_def_suffix=1, n_doc_prefix=0, met_desc_len=3, arg_desc_len=2):
    rng = random.Random(seed)
    n_not = n_matching - 1
    met_name, *all_args = rng.sample(VN, 2 + n_matching + n_not + n_not + n_matching + n_def_prefix + n_def_suffix + n_doc_prefix)
    a_ = list(all_args)
    random_answer, *a_ = a_
    _, a_ = a_[:n_matching], a_[n_matching:]       # random_random_mid_def_args (unused here)
    _, a_ = a_[:n_not], a_[n_not:]                 # random_random_mid_doc_args (unused here)
    not_matching, a_ = a_[:n_not], a_[n_not:]
    matching, a_ = a_[:n_matching], a_[n_matching:]
    def_prefix, a_ = a_[:n_def_prefix], a_[n_def_prefix:]
    def_suffix, a_ = a_[:n_def_suffix], a_[n_def_suffix:]
    doc_prefix, a_ = a_[:n_doc_prefix], a_[n_doc_prefix:]
    assert not a_
    clean_def = def_prefix + matching + def_suffix
    clean_doc = doc_prefix + matching[:-1]
    met_desc = rng.sample(CN, met_desc_len)
    doc_desc = [rng.sample(CN, arg_desc_len) for _ in clean_doc]
    clean = templ(style, met_name, met_desc, clean_def, clean_doc, doc_desc)
    # OUR flipping corruption: document the two def-prefix args instead of the two matching args; the next undocumented
    # def arg is then matching[0] instead of matching[-1]
    flip_doc = doc_prefix + def_prefix[:len(clean_doc)]
    assert len(flip_doc) == len(clean_doc)
    flipped = templ(style, met_name, met_desc, clean_def, flip_doc, doc_desc)
    return clean, flipped, ' ' + matching[-1], ' ' + matching[0], dict(met_name=met_name, def_args=clean_def, clean_doc_args=clean_doc, corrupted_doc_args=flip_doc)

seeds = [random.Random(SEED).randrange(10 ** 9) for _ in range(1)]
srng = random.Random(SEED)
item_seeds = [srng.randrange(10 ** 9) for _ in range(2000)]
tasks = []
for style in ('rest', 'goog'):
    pool = []
    for n, sd in enumerate(item_seeds):
        c, f, a, b, md = gen(style, sd)
        pool.append(make_item(c, f, a, b, SID, '%s/%05d/seed%d' % (style, n, sd), style=style, **{k: json.dumps(v) if isinstance(v, list) else v for k, v in md.items()}))
    tasks.append(write_task(SID, 'docstring_' + style, 'docstring', 'code (induction over argument names)', pool, lambda i: 0, 2000,
        ['Clean prompts reproduce the ACDC `docstring_induction_prompt_generator("%s", ...)` clean prompt (same RNG call order, per-item seeds from random.Random(42)). The prompt documents the first two of three consecutive def args (`:param a: ...`), and the answer is the third (single-word variable name, leading space).' % style,
         'OUR flipping corruption (the generator\'s own corruptions, e.g. random_doc, give no defined answer): the docstring documents the two def args that precede the matching block instead; the correct next argument is then the FIRST matching arg, which is the clean prompt\'s clean_incorrect. Same descriptions, so equal length when variable names are single tokens.',
         'Source designed for a 4-layer attention-only model; the `goog` style adds an "Args:" line and indentation-only param lines.',
         'Variable names / description nouns are single GPT-2 tokens by construction; Qwen tokenisation checked in the diagnostic.']))
write_config(SID, [t['name'] for t in tasks])
r = man['repos']['Automatic-Circuit-Discovery']
meta = dict(source_id=SID, title='Docstring task (A circuit for Python docstrings in a 4-layer attention-only transformer); generator from the ACDC repo', bibkey='heimersheim2023docstring',
            bibkey_data=['conmy2023automatedcircuitdiscoverymechanistic'], url='https://github.com/ArthurConmy/Automatic-Circuit-Discovery (acdc/docstring/prompts.py); https://www.alignmentforum.org/posts/u6KXXmKFbXfWzoAXn',
            download_date=man['download_date'], tier=1, revision=r['commit'],
            raw_files=[dict(path=p, sha256=s) for p, s in r['files'].items() if p.endswith(('acdc/docstring/prompts.py', 'acdc/docstring/utils.py', 'LICENSE'))],
            licence='MIT-style grant text, Copyright 2023 Arthur Conmy, Adria Garriga-Alonso', licence_source='raw/docstring/Automatic-Circuit-Discovery/LICENSE', redistribute='yes (generator output; word lists come from the MIT-licensed repo)',
            generator=dict(commit=r['commit'], seed=SEED, N=2000, per_item_seeds='random.Random(42).randrange(1e9)', params=dict(n_matching_args=3, n_def_prefix_args=2, n_def_suffix_args=1, n_doc_prefix_args=0, met_desc_len=3, arg_desc_len=2),
                           frozen_output_sha256={t['name']: sha256_file(os.path.join(TB, 'tasks', SID, t['name'] + '.json')) for t in tasks}),
            tokenizer=tokenizer_meta(), converter='scripts/task_bank/convert_docstring.py', status='converted',
            notes='The docstring task has no paper or arXiv id; the reference is the Alignment Forum post (see the bib entry, composed from page metadata). Third-party code only parsed, not executed.')
write_registry(SID, meta, tasks)
for t in tasks:
    print(t['name'], t['n_items_converted'], t['len_match_share_qwen35'], t['single_token_answers_share'])
