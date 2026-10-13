"""Frozen IOI dataset from the generator of Wang et al. 2023 (Easy-Transformer ioi_dataset.py).
The third-party module is NOT imported or run: constants (NAMES, templates, places, objects) are read from its source with `ast`, and the
generation logic (gen_prompt_uniform with symmetric=False, and the ("S2","IO") flip of gen_flipped_prompts) is re-implemented here.
Parameters: seed 42, N=2000, templates = all 15 BABA + all 15 ABBA templates (the original `mixed` option would use only the first 7 of each), uniform over templates."""
import ast, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

SID = 'wang_ioi'
RAW = os.path.join(TB, 'raw', 'wang_ioi')
man = json.load(open(os.path.join(RAW, 'MANIFEST.json')))
SRC = os.path.join(RAW, 'Easy-Transformer', 'easy_transformer', 'ioi_dataset.py')
tree = ast.parse(open(SRC).read())
const = {}
for node in tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('NAMES', 'BABA_TEMPLATES', 'PLACES', 'OBJECTS'):
        const[node.targets[0].id] = ast.literal_eval(node.value)
NAMES, BABA, PLACES, OBJECTS = const['NAMES'], const['BABA_TEMPLATES'], const['PLACES'], const['OBJECTS']
assert len(NAMES) == 99 and len(BABA) == 15

def to_abba(t):  # same transformation as the module's loop: in the first clause, swap which of [B]/[A] comes first
    t = list(t); first = True
    s = ''.join(t)
    out = s
    for j in range(1, len(out) - 1):
        if out[j - 1:j + 2] == '[B]' and first:
            out = out[:j] + 'A' + out[j + 1:]
        elif out[j - 1:j + 2] == '[A]' and first:
            first = False
            out = out[:j] + 'B' + out[j + 1:]
    return out
ABBA = [to_abba(t) for t in BABA]
TEMPL = [('BABA', i, t) for i, t in enumerate(BABA)] + [('ABBA', i, t) for i, t in enumerate(ABBA)]

N = 2000
rng = random.Random(SEED)
pool = []
for n in range(N):
    kind, ti, temp = TEMPL[rng.randrange(len(TEMPL))]
    while True:
        a, b = rng.choice(NAMES), rng.choice(NAMES)
        if a != b:
            break
    place, obj = rng.choice(PLACES), rng.choice(OBJECTS)
    base = temp.replace('[PLACE]', place).replace('[OBJECT]', obj)
    clean = base.replace('[A]', a).replace('[B]', b)         # IO = A, S = B
    toks = clean.split(' ')
    assert toks[-1] == a   # generator text ends with the IO name (the answer); the prompt excludes it
    toks = toks[:-1]
    clean = ' '.join(toks)
    last = len(toks) - toks[::-1].index(b) - 1                 # S2 = last mention of S
    toks[last] = a                                              # flip ("S2", "IO"): S2 replaced by IO, roles exchange
    corrupted = ' '.join(toks)
    pool.append(make_item(clean, corrupted, ' ' + a, ' ' + b, SID, 'gen/%05d' % n, template_kind=kind, template_idx=ti, io_name=a, s_name=b, place=place, object=obj))
rec = write_task(SID, 'ioi_wang_s2_flip', 'ioi', 'syntax/semantics (name movement)', pool, lambda i: (i['template_kind'], i['template_idx']), N,
    ['Generated with our re-implementation of Wang et al.\'s gen_prompt_uniform (seed 42, N=2000, all 30 BABA/ABBA templates, names drawn from the 99 NAMES, 8 places, 8 objects, symmetric=False); the names are single GPT-2 tokens by construction.',
     'Corrupted prompt = the original generator\'s ("S2","IO") flip: the last mention of S is replaced by IO. IO and S exchange roles, so the answer flips; one token differs, equal length when the names are single tokens. The ABC corruption (random third name) used in the paper does not flip the answer and is not used.',
     'Differences from the original: random.Random(42) instead of the global `random` state, no prefixes/endoftext, and all templates (not the first 7 of the `mixed` option).'])
tasks = [rec]
write_config(SID, [rec['name']])
r = man['repos']
meta = dict(source_id=SID, title='IOI dataset generator of Wang et al. (Easy-Transformer ioi_dataset.py; copy in the ACDC repo)', bibkey='wang2022interpretabilitywildcircuitindirect',
            bibkey_data=[], url='https://github.com/redwoodresearch/Easy-Transformer', download_date=man['download_date'], tier=1,
            revision={k: v['commit'] for k, v in r.items()},
            raw_files=[dict(path=p, sha256=s) for k, v in r.items() for p, s in v['files'].items() if p.endswith(('ioi_dataset.py', 'LICENSE')) and 'subnetwork' not in p],
            licence={'Easy-Transformer': 'MIT', 'Automatic-Circuit-Discovery': 'MIT-style grant text (Copyright 2023 Arthur Conmy, Adria Garriga-Alonso)'},
            licence_source='raw/wang_ioi/Easy-Transformer/LICENSE; raw/wang_ioi/Automatic-Circuit-Discovery/LICENSE', redistribute='yes (generator output from MIT-licensed code and word lists)',
            generator=dict(commit=r['Easy-Transformer']['commit'], seed=SEED, N=2000, script='scripts/task_bank/convert_wang_ioi.py',
                           frozen_output='external/task_bank/tasks/wang_ioi/ioi_wang_s2_flip.json', frozen_output_sha256=sha256_file(os.path.join(TB, 'tasks', SID, 'ioi_wang_s2_flip.json'))),
            tokenizer=tokenizer_meta(), status='converted',
            notes='Third-party code was only parsed (ast), never executed. ACDC copy of ioi_dataset.py is a slightly edited version (see its header) and was not used.')
write_registry(SID, meta, tasks)
print(rec['n_items_converted'], rec['len_match_share_qwen35'], rec['single_token_answers_share'])
