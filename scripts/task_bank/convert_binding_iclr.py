"""Generate tier-1 binding tasks in the style of Feng & Steinhardt (ICLR 2024, jiahai-feng/binding-iclr).

The repository ships generators (coref/datasets/*.py) and word lists (coref/datasets/raw/*.csv), not frozen datasets. We re-implement the templates of the repo
(read from the pinned commit; the repo's own code needs LLaMA weights and torch and was not run) and generate items with seed 42:

  binding_capitals          "<Name> lives in the capital city of <Country>." x n  / Which city does <Name> live in? / Answer: <Name> lives in the city of  -> " <Capital>"
  binding_fruits            "<Name> likes eating the <fruit>." x n               / Which fruit does <Name> like?  / Answer: <Name> likes the            -> " <fruit>"
  binding_shapes            "The <shape> is <color>." x n                         / Which shape is colored <color>? / Answer: The <color> shape is the   -> " <shape>"
  binding_parallel_capitals "<N1>, <N2> and <N3> live in the capital cities of <C1>, <C2> and <C3> respectively." (one sentence) / same question as capitals
  binding_onehop            capitals context + "Question: The person living in <Capital_i> likes <color_i>. ... What color does <Name> like?" / Answer: <Name> likes -> " <color>"  (two hops)

Flip (our construction): n in {2,3,4} entities with distinct attributes; the clean context binds entity i to attribute a_i; the corrupted context swaps the attributes of the QUERIED
entity and one other entity, the query is unchanged. The answer flips from a_q to a_p, the clean and corrupted prompts are permutations of the same tokens, so their Qwen3.5 length is equal by construction.
(The paper's own interventions swap representations, they do not define a flipped text pair.) Word lists: the repo's CSVs, restricted to entries that are a single Qwen3.5 token with a leading space
(names, countries AND capitals, colors, fruits, shapes), as the original restricted to single LLaMA tokens. Items are drawn with random.Random('42:<task>'), 2000 per task, n_entities stratified equally.
Not converted: BIOSBIAS (Bias in Bios; 56 MB of personal biographies, licence unclear, free-text attributes), the MCQ/ICL-SST tasks (classification formats), compound variants not in the paper's main task list.
The repo has no top-level LICENSE file: redistribute = no."""
import csv, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knowledge_common as kc

SID = 'binding_iclr'
N_ITEMS = 2000
NS = (2, 3, 4)
HEAD = 'Answer the question based on the context below. Keep the answer short.\n\nContext:'

def rows(path):
    return list(csv.DictReader(open(path, encoding='utf-8')))

def single(w):
    return kc.ntok(' ' + w) == 1

def fmt_parallel(things):
    return ', '.join(things[:-1]) + ' and ' + things[-1]

def main():
    man = kc.load_manifest(SID)
    kc.clear_tasks(SID)
    raw = os.path.join(kc.RAW, SID, 'repo', 'coref', 'datasets', 'raw')
    names_all = [r['name'] for r in rows(os.path.join(raw, 'new-top-firstNames.csv'))]
    cc_all = [(r['country'], r['capital']) for r in rows(os.path.join(raw, 'country-list.csv'))]
    colors_all = [r['color'] for r in rows(os.path.join(raw, 'colors.csv'))]
    fruits_all = [r['fruits'] for r in rows(os.path.join(raw, 'fruits.csv'))]
    shapes_all = [r['shape'] for r in rows(os.path.join(raw, 'shapes.csv'))]
    names = [n for n in names_all if single(n)]
    cc = [(c, k) for c, k in cc_all if single(c) and single(k)]
    cap_count = {}
    for c, k in cc:
        cap_count[k] = cap_count.get(k, 0) + 1
    cc = [(c, k) for c, k in cc if cap_count[k] == 1]  # capitals must identify the country
    colors = [c for c in colors_all if single(c)]
    fruits = [f for f in fruits_all if single(f)]
    shapes = [s for s in shapes_all if single(s)]
    vocab = dict(names=(len(names_all), len(names)), country_capital=(len(cc_all), len(cc)), colors=(len(colors_all), len(colors)), fruits=(len(fruits_all), len(fruits)), shapes=(len(shapes_all), len(shapes)))

    specs = {}
    # each spec: (entity pool, attribute pool, answer fn, context fn, query fn, domain/family)
    specs['binding_capitals'] = dict(
        ents=names, attrs=cc, ans=lambda a: a[1],
        ctx=lambda E, A: ''.join(' %s lives in the capital city of %s.' % (e, a[0]) for e, a in zip(E, A)),
        query=lambda E, A, q: '\n\nQuestion: Which city does %s live in?\n\nAnswer: %s lives in the city of' % (E[q], E[q]))
    specs['binding_parallel_capitals'] = dict(
        ents=names, attrs=cc, ans=lambda a: a[1],
        ctx=lambda E, A: ' %s live in the capital cities of %s respectively.' % (fmt_parallel(E), fmt_parallel([a[0] for a in A])),
        query=lambda E, A, q: '\n\nQuestion: Which city does %s live in?\n\nAnswer: %s lives in the city of' % (E[q], E[q]))
    specs['binding_fruits'] = dict(
        ents=names, attrs=fruits, ans=lambda a: a,
        ctx=lambda E, A: ''.join(' %s likes eating the %s.' % (e, a) for e, a in zip(E, A)),
        query=lambda E, A, q: '\n\nQuestion: Which fruit does %s like?\n\nAnswer: %s likes the' % (E[q], E[q]))
    specs['binding_shapes'] = dict(
        ents=colors, attrs=shapes, ans=lambda a: a,
        ctx=lambda E, A: ''.join(' The %s is %s.' % (a, e) for e, a in zip(E, A)),
        query=lambda E, A, q: '\n\nQuestion: Which shape is colored %s?\n\nAnswer: The %s shape is the' % (E[q], E[q]))
    # onehop: each attribute is a capital with a colour; the colour is what the question asks for. The pool entry is (country, capital); the colour of attribute i is fixed per item.
    specs['binding_onehop'] = dict(ents=names, attrs=cc, ans=None, ctx=None, query=None)
    fam = {'binding_capitals': ('binding_capital', 'variable binding'), 'binding_parallel_capitals': ('binding_capital_parallel', 'variable binding'),
           'binding_fruits': ('binding_fruit', 'variable binding'), 'binding_shapes': ('binding_shape_color', 'variable binding'), 'binding_onehop': ('binding_onehop', 'variable binding + one hop')}
    tasks_rec, kept = [], []
    for task, sp in specs.items():
        rng = random.Random('42:' + task)
        items, seen = [], set()
        tries = 0
        while len(items) < N_ITEMS and tries < N_ITEMS * 20:
            tries += 1
            n = NS[len(items) % len(NS)]
            E = rng.sample(sp['ents'], n)
            A = rng.sample(sp['attrs'], n)
            q = rng.randrange(n)
            p = rng.choice([i for i in range(n) if i != q])
            A2 = list(A)
            A2[q], A2[p] = A[p], A[q]
            if task == 'binding_onehop':
                cols = rng.sample(colors, n)  # colour of the city of attribute i (stays with the attribute, i.e. with the city, in both prompts)
                order = rng.sample(range(n), n)
                def mk(Aa):
                    ctx = ''.join(' %s lives in the capital city of %s.' % (e, a[0]) for e, a in zip(E, Aa))
                    qn = ''.join(' The person living in %s likes %s.' % (A[i][1], cols[i]) for i in order)  # city<->colour pairs fixed by the ORIGINAL attributes A
                    return HEAD + ctx + '\n\nQuestion:' + qn + ' What color does %s like?\n\nAnswer: %s likes' % (E[q], E[q])
                cp, op = mk(A), mk(A2)
                ca, ia = ' ' + cols[q], ' ' + cols[p]
            else:
                cp = HEAD + sp['ctx'](E, A) + sp['query'](E, A, q)
                op = HEAD + sp['ctx'](E, A2) + sp['query'](E, A2, q)
                ca, ia = ' ' + sp['ans'](A[q]), ' ' + sp['ans'](A[p])
            if cp in seen:
                continue
            seen.add(cp)
            items.append(kc.make_item(cp, op, ca, ia, SID, '%s#%d' % (task, len(items)), n_entities=n, query_index=q, swapped_with_index=p, entities=E,
                                      clean_attributes=[a if isinstance(a, str) else a[0] for a in A], corrupted_attributes=[a if isinstance(a, str) else a[0] for a in A2]))
        ad = ['re-implemented generator (templates and word lists of the pinned repo commit), seed 42, %d items, n_entities in %s equally' % (len(items), list(NS)),
              'flip = swap of the attributes of the queried entity and one other entity in the context; the query is unchanged; clean and corrupted prompts are permutations of the same tokens',
              'word lists restricted to single-token (with leading space) entries under the Qwen3.5 tokenizer' + ('; one-hop: the city-colour sentences in the question keep the original city-colour pairs in both prompts' if task == 'binding_onehop' else '')]
        f, dom = fam[task]
        rec = kc.write_task(SID, task, f, dom, items, lambda i: i['n_entities'], len(items), ad, extra=dict(n_shots=0, n_entities_values=list(NS), vocab_sizes_raw_vs_single_token=vocab))
        rec['n_items_raw'] = 'generated (%d)' % len(items)
        tasks_rec.append(rec)
        kept.append(task)
    tasks_rec.append(dict(name='binding_biosbias', status='not_converted', notes='Bias in Bios biographies (56 MB, personal data, free-text attributes); not a clean entity-attribute lookup'))
    kc.write_config(SID, kept)
    reg = kc.base_registry(SID, man, title='How do language models bind entities in context? (binding-ID tasks: generators and word lists)',
                           bibkey='feng2024languagemodelsbindentities', url=man['url'], revision=man['revision'],
                           licence='none: no top-level LICENSE file at the pinned commit (only transformer_lens/LICENSE, which covers the vendored TransformerLens code); the README has no licence statement',
                           licence_source='external/task_bank/raw/binding_iclr/repo (file listing); external/task_bank/raw/binding_iclr/repo/transformer_lens/LICENSE (vendored code only)',
                           redistribute='no (no licence: keep converted items out of git)', tier=1, tokenizer=kc.tokenizer_meta(), status='converted',
                           generator=dict(commit=man['revision'], seed=42, params=dict(n_items_per_task=N_ITEMS, n_entities=list(NS), implementation='scripts/task_bank/convert_binding_iclr.py (re-implementation of coref/datasets templates)')),
                           notes='Frozen output = tasks/binding_iclr/*.json (sha256 in the report).')
    kc.write_registry(SID, reg, tasks_rec)
    for t in tasks_rec:
        print(t['name'], t.get('n_items_converted'), t.get('len_match_share_qwen35'), t.get('single_token_answers_share'))
    print(vocab)

if __name__ == '__main__':
    main()
