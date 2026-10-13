"""Qwen3.5 length-matched re-pairings (`<task>__qwen35`) of the circuits (mib, eap_ig) and arithmetic (devarda_arithmetic) tasks that lose more
than 10% of their items to the evaluation's token-length alignment filter (parcelmate/patching.py: items whose clean and corrupted prompts, built as
prefix + problem + concatenation, tokenize to different lengths are dropped).

Each variant keeps the source's own contrast and re-pairs it so that every item has EQUAL clean/corrupted prompt token counts under the Qwen3.5 tokenizer:
  mib ravel_template_cf / ravel_attribute_cf : the corrupted prompt is the source's own counterfactual (of that type) of ANOTHER train row with the same
        template and attribute as the clean row (so the counterfactual-template distribution given the clean template is the source's), chosen among those
        whose prompt has the clean prompt's token length (and the same answer-spacing kind); the original partner is preferred when it already matches.
  mib ravel_entity_swap : same construction as the original (entity swapped within template and attribute, same word count, different attribute value,
        closest character length) but the partner must give a prompt of equal token length instead of only being closest in characters.
  eap_ig capital_country, hypernymy_comma, hypernymy_type_of : the subject swap, drawn over ALL pairs of the raw file's prompt pool (clean and corrupted
        columns pooled, deduplicated) with equal token length and different answers; unordered pairs (the both-correct criterion is symmetric in the
        orientation), orientation by a seeded coin.
  devarda_arithmetic embedded / embedded_verbal : operator flip with names/object replayed from the clean problem (as the original); the corrupted story
        template is chosen among all templates of the generator for the flipped operators (2-term problems have 5; 3-term ones have one) so that the prompt
        length (exemplar prefix + story + " The answer is") equals the clean one. The default (hash-chosen) template is preferred when it matches. Problems
        without a matching template are dropped before the same stratified sampling as the original.
  pragworld_flip : NOT re-paired (pairs are fixed by the source), see the printed note; no file is written.

Deterministic (seed 42), cap 2,000 items. Run AFTER the converters (they rewrite configs and registries; this script only adds/replaces `__qwen35` entries):
    python scripts/task_bank/repair_qwen35_other.py [ravel|eap|arith|all]
Needs raw/ data (not committed) and the Qwen3.5 tokenizer.json. No model inference."""
import ast, collections, itertools, json, os, random, sys, types
import yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm
from common import TB, SEED, CAP, make_item, valid_item, stratified_sample, qwen_tokenizer

SUFFIX = '__qwen35'
_len = {}


def ntoks(texts):
    """Qwen3.5 token counts (no special tokens), cached."""
    todo = [t for t in set(texts) if t not in _len]
    if todo:
        for t, e in zip(todo, qwen_tokenizer().encode_batch(todo, add_special_tokens=False)):
            _len[t] = len(e.ids)
    return [_len[t] for t in texts]


def ntok(t):
    return ntoks([t])[0]


def load_cfg(src):
    return json.load(open(os.path.join(TB, 'configs', src + '.json')))


def aligned(items, cfg):
    pre, con = cfg.get('prefix') or '', cfg.get('concatenation') or ''
    return sum(ntok(pre + i['clean_problem'] + con) == ntok(pre + i['corrupted_problem'] + con) for i in items)


def stats(items, cfg):
    pre, con = cfg.get('prefix') or '', cfg.get('concatenation') or ''
    n = max(len(items), 1)
    eq = sum(ntok(pre + i['clean_problem'] + con) == ntok(pre + i['corrupted_problem'] + con) for i in items)
    single = 0
    for i in items:
        c = pre + i['clean_problem'] + con
        single += (ntok(c + i['clean_correct']) - ntok(c) == 1) and (ntok(c + i['clean_incorrect']) - ntok(c) == 1)
    return round(eq / n, 4), round(single / n, 4)


def emit(src, base, pool, strata_key, adaptations, extra=None, yaml_width=200):
    """Write tasks/<src>/<base>__qwen35.{json,ids.txt}, a config entry and a registry entry; returns the registry record."""
    task = base + SUFFIX
    bad = [i for i in pool if not valid_item(i)]
    assert not bad, bad[:2]
    seen, uniq = set(), []
    for i in sorted(pool, key=lambda x: x['source_item_id']):
        k = (i['clean_problem'], i['corrupted_problem'], i['clean_correct'], i['clean_incorrect'])
        if k not in seen:
            seen.add(k)
            uniq.append(i)
    sample = stratified_sample(uniq, strata_key, CAP)
    ids = [i['source_item_id'] for i in sample]
    assert len(set(ids)) == len(ids)
    d = os.path.join(TB, 'tasks', src)
    with open(os.path.join(d, task + '.json'), 'w') as f:
        json.dump(sample, f, indent=1, ensure_ascii=False)
    with open(os.path.join(d, task + '.ids.txt'), 'w') as f:
        f.write('\n'.join(ids) + '\n')
    cfgs = load_cfg(src)
    cfg = dict(cfgs[base])
    cfg['path'] = task + '.json'
    cfgs[task] = cfg
    with open(os.path.join(TB, 'configs', src + '.json'), 'w', encoding='utf-8') as f:     # same serialisation as each source's converter
        if yaml_width == 200:
            json.dump(cfgs, f, indent=2)
        else:
            json.dump(cfgs, f, ensure_ascii=False, indent=1)
            f.write('\n')
    reg_path = os.path.join(TB, 'registry', src + '.yaml')
    reg = yaml.safe_load(open(reg_path))
    orig = next(t for t in reg['tasks'] if t['name'] == base)
    orig_items = json.load(open(os.path.join(d, base + '.json')))
    eq, single = stats(sample, cfg)
    assert eq == 1.0, (task, eq)
    rec = dict(name=task, variant_of=base, family=orig.get('family'), domain=orig.get('domain'), n_items_raw=orig.get('n_items_raw'),
               n_items_original_task=len(orig_items), n_items_original_task_aligned_qwen35=aligned(orig_items, cfgs[base]),
               n_items_after_validity_and_dedup=len(uniq), n_items_converted=len(sample),
               sampling_rule=('all items kept' if len(uniq) <= CAP else 'proportional stratified sample, seed 42, cap %d' % CAP),
               sampled_ids_file='external/task_bank/tasks/%s/%s.ids.txt' % (src, task),
               len_match_share_qwen35=eq, single_token_answer_share_qwen35=single, adaptations=adaptations, status='converted')
    if extra:
        rec.update(extra)
    reg['tasks'] = [t for t in reg['tasks'] if t['name'] != task] + [rec]
    with open(reg_path, 'w') as f:
        yaml.safe_dump(reg, f, sort_keys=False, allow_unicode=True, width=yaml_width)
    print('%-34s original %4d, aligned originally %4d, variant %4d (pool %d)' % (task, len(orig_items), rec['n_items_original_task_aligned_qwen35'], len(sample), len(uniq)))
    return rec


# ------------------------------------------------------------------------------------------------ MIB RAVEL
def lead(prompt):
    return '' if prompt.endswith('"') or prompt.endswith(' ') else ' '


def ravel():
    import pandas as pd
    RAW = os.path.join(TB, 'raw', 'mib')
    rv = pd.read_parquet(os.path.join(RAW, 'ravel', 'data', 'train-00000-of-00001.parquet'))
    rv = rv[rv.attribute.isin(['Country', 'Continent', 'Language'])].reset_index()
    rows = rv.to_dict('records')
    ATT = ('Country', 'Continent', 'Language')
    L = ntoks([r['prompt'] for r in rows])
    for r, n in zip(rows, L):
        r['_n'] = n
    for key, cfk, base in (('template', 'prompt_template_counterfactual', 'ravel_template_cf'), ('attribute', 'attribute_counterfactual', 'ravel_attribute_cf')):
        cand = [r for r in rows if r[cfk] is not None and r[cfk]['attribute'] in ATT and lead(r[cfk]['prompt']) == lead(r[cfk]['prompt'])]
        cn = ntoks([r[cfk]['prompt'] for r in cand])
        buckets = collections.defaultdict(list)       # (template, attribute, lead, cf prompt length) -> source rows whose counterfactual qualifies
        for r, n in zip(cand, cn):
            buckets[(r['template'], r['attribute'], lead(r[cfk]['prompt']), n)].append(r)
        rng = random.Random(SEED)
        pool, n_orig, n_none = [], 0, 0
        for r in sorted(rows, key=lambda r: r['index']):
            a = r[r['attribute']]
            ok = lambda s: s[cfk] is not None and s[cfk]['attribute'] in ATT and s[cfk][s[cfk]['attribute']] != a and s[cfk]['entity'] != r['entity']
            b = buckets.get((r['template'], r['attribute'], lead(r['prompt']), r['_n']), [])
            c0 = [s for s in b if s is r and ok(s)]
            if c0:
                s, n_orig = c0[0], n_orig + 1
            else:
                cs = [s for s in b if ok(s)]
                if not cs:
                    n_none += 1
                    continue
                s = cs[rng.randrange(len(cs))]
            c = s[cfk]
            l = lead(r['prompt'])
            sid = 'ravel/train/%d/%s' % (r['index'], key) + ('' if s is r else '~%d' % s['index'])
            pool.append(make_item(r['prompt'], c['prompt'], l + a, l + c[c['attribute']], 'mib', sid, attribute=r['attribute'], cf_attribute=c['attribute'],
                                  entity=r['entity'], cf_entity=c['entity'], template=r['template'], cf_template=c['template'],
                                  partner_row_index=int(s['index']), original_partner=s is r))
        print(base, 'pool', len(pool), 'original partner kept', n_orig, 'no length-matched partner', n_none)
        emit('mib', base, pool, lambda i: (i['attribute'], i['cf_attribute']),
             ['RE-PAIRED for equal Qwen3.5 token length (variant of `%s`). Clean row = every Country/Continent/Language train row; the corrupted prompt is MIB\'s own `%s` of a train row with the SAME template and attribute as the clean row (the clean row\'s own counterfactual when its length already matches, otherwise a seeded random draw among the matching ones), so that the counterfactual type and the clean template are preserved while the counterfactual entity/template/attribute come from the source\'s distribution.' % (base, 'prompt_template_counterfactual' if key == 'template' else 'attribute_counterfactual'),
              'Constraints on the partner: prompt token length equal to the clean prompt (Qwen3.5), same answer-spacing kind (quote vs word ending), counterfactual attribute value different from the clean value, counterfactual entity different from the clean entity. Answers as in the original task. Strata for the 2,000 sample: attribute x counterfactual attribute. Note that the clean-entity / counterfactual-entity pairing is no longer the source\'s row-wise pairing (a counterfactual of another row is used).'],
             extra=dict(n_original_partner_kept=n_orig, n_clean_rows_without_match=n_none))
    # entity swap
    by_key = collections.defaultdict(list)
    for r in rows:
        by_key[(r['template'], r['attribute'])].append(r)
    rng = random.Random(SEED)
    pool = []
    for (tpl, att), g in sorted(by_key.items()):
        g = sorted(g, key=lambda r: r['index'])
        bk = collections.defaultdict(list)
        for o in g:
            bk[(len(o['entity'].split()), o['_n'])].append(o)
        for r in g:
            cands = [o for o in bk[(len(r['entity'].split()), r['_n'])] if o[att] != r[att]]
            if not cands:
                continue
            m = min(abs(len(o['entity']) - len(r['entity'])) for o in cands)
            close = [o for o in cands if abs(len(o['entity']) - len(r['entity'])) == m]
            o = close[rng.randrange(len(close))]
            l = lead(r['prompt'])
            pool.append(make_item(r['prompt'], o['prompt'], l + r[att], l + o[att], 'mib', 'ravel/train/%d~%d' % (r['index'], o['index']),
                                  attribute=att, entity=r['entity'], cf_entity=o['entity'], template=tpl))
    print('ravel_entity_swap pool', len(pool))
    emit('mib', 'ravel_entity_swap', pool, lambda i: (i['attribute'], i['template']),
         ['RE-PAIRED for equal Qwen3.5 token length (variant of `ravel_entity_swap`). Same construction as the original (OUR CONSTRUCTION, not in MIB): the entity is replaced by another entity of the same template and attribute with a different attribute value and the same number of words; the partner is additionally required to give a prompt of EQUAL Qwen3.5 token length; among those, the closest character length wins, ties broken with random.Random(42) in template/index order. Answers as in the original.'])


# ------------------------------------------------------------------------------------------------ EAP-IG
def eap():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    D = os.path.join(TB, 'raw', 'eap_ig', 'eap-ig-faithfulness', 'data')
    g2 = Tokenizer.from_file(hf_hub_download('openai-community/gpt2', 'tokenizer.json', revision='607a30d783dfa663caf39e06633721c8d4cfcd7e'))
    dec = lambda i: g2.decode([int(i)])

    MAX_USE = 2

    def pairs(nodes, base, label, meta_fn, adapt):
        """nodes: list of (prompt, answer-set tuple, meta dict); all unordered pairs of distinct prompts with equal token length and a distinguishing answer."""
        rng = random.Random(SEED)
        lens = ntoks([n[0] for n in nodes])
        buckets = collections.defaultdict(list)
        for n, l in zip(nodes, lens):
            buckets[l].append(n)
        pool, use, cand = [], collections.Counter(), []
        for l in sorted(buckets):
            for a, b in itertools.combinations(sorted(buckets[l], key=lambda n: n[0]), 2):
                if rng.random() < 0.5:
                    a, b = b, a
                if set(a[1]) & set(b[1]):      # overlapping answer sets: the contrast would be blurred (the original only required one non-shared answer each)
                    continue
                cand.append((a, b))
        # Each prompt enters at most MAX_USE items (main session, 2026-10-13): all-pairs would
        # reuse one prompt in up to 180 items and inflate the item count the inclusion rule
        # relies on. Greedy over a seeded shuffle of the candidate pairs.
        rng.shuffle(cand)
        for a, b in cand:
            if use[a[0]] < MAX_USE and use[b[0]] < MAX_USE:
                pool.append(meta_fn(a, b, a[1][0], b[1][0]))
                use[a[0]] += 1
                use[b[0]] += 1
        print(base, 'prompts', len(nodes), 'candidate pairs', len(cand), 'items', len(pool), 'max uses of one prompt', max(use.values()) if use else 0)
        return pool

    # capital -> country
    df = pd.read_csv(os.path.join(D, 'fact-retrieval-comma', 'gpt2.csv'))
    nodes = {}
    for k, r in df.iterrows():
        for col, p, cap, ctry, idx in (('c', r['clean'], r['capital'], r['country'], r['country_idx']), ('k', r['corrupted'], r['corrupted_capital'], r['corrupted_country'], r['corrupted_country_idx'])):
            nodes.setdefault(p, (p, (' ' + ctry,), dict(capital=cap, first=dec(idx), tag='%s%d' % (col, k))))   # tag: column (c = clean, k = corrupted) and row of the first occurrence
    nodes = list(nodes.values())
    mk = lambda a, b, a0, b0: make_item(a[0], b[0], a0, b0, 'eap_ig', 'fact-retrieval-comma/gpt2/pair:%s~%s' % (a[2]['tag'], b[2]['tag']), capital=a[2]['capital'],
                                        corrupted_capital=b[2]['capital'], country_first_token_gpt2=a[2]['first'], corrupted_country_first_token_gpt2=b[2]['first'])
    pool = pairs(nodes, 'capital_country', '', mk, None)
    emit('eap_ig', 'capital_country', pool, lambda i: 0,
         ['RE-PAIRED for equal Qwen3.5 token length (variant of `capital_country`). The prompt pool is the raw file\'s clean and corrupted capital prompts pooled and deduplicated (%d distinct capitals); an item is any unordered pair of capitals with the same Qwen3.5 prompt length and different countries (orientation by a seeded coin; the both-correct criterion is symmetric in the orientation). The subject swap (another capital) is the source\'s contrast; only the pairing differs from the source\'s row-wise (clean, corrupted) columns. Answers as in the original (full country name with a leading space).' % len(nodes),
          'Each capital prompt occurs in at most 2 items (greedy selection over a seeded shuffle of all equal-length pairs), so the item count stays close to the number of distinct prompts.'])
    # hypernymy
    df = pd.read_csv(os.path.join(D, 'hypernymy-comma', 'gpt2.csv'))
    for base, ck, ok, txt in (('hypernymy_comma', 'clean', 'corrupted', 'the `clean`/`corrupted` columns (", gemstones and other")'), ('hypernymy_type_of', 'clean2', 'corrupted2', 'the `clean2`/`corrupted2` columns ("gemstones are a type of")')):
        nodes = {}
        for k, r in df.iterrows():
            a = tuple(dec(i) for i in ast.literal_eval(r['answers_idx']))
            b = tuple(dec(i) for i in ast.literal_eval(r['corrupted_answers_idx']))
            nodes.setdefault(r[ck], (r[ck], a, dict(h=r['hypernym'], tag='c%d' % k)))
            nodes.setdefault(r[ok], (r[ok], b, dict(h=r['corrupted_hypernym'], tag='k%d' % k)))
        nodes = list(nodes.values())
        mk = lambda a, b, a0, b0: make_item(a[0], b[0], a0, b0, 'eap_ig', 'hypernymy-comma/gpt2/%s/pair:%s~%s' % (ck, a[2]['tag'], b[2]['tag']),
                                            clean_answers=json.dumps(list(a[1])), corrupted_answers=json.dumps(list(b[1])), hypernym=a[2]['h'], corrupted_hypernym=b[2]['h'])
        pool = pairs(nodes, base, '', mk, None)
        emit('eap_ig', base, pool, lambda i: 0,
             ['RE-PAIRED for equal Qwen3.5 token length (variant of `%s`). Prompt pool: %s of the raw file, each with its answer set (clean and corrupted columns pooled and deduplicated by prompt; %d prompts). An item is any unordered pair of prompts with the same Qwen3.5 length whose answer sets are DISJOINT (stricter than the original, which only needed one non-shared answer each; the correct answer is the first listed one, as in the original; orientation by a seeded coin). The subject swap (another hypernymy category) is the source\'s contrast.' % (base, txt, len(nodes)),
              'Each prompt occurs in at most 2 items (greedy selection over a seeded shuffle of all equal-length pairs); pairs with partly overlapping answer sets are excluded.'])


# ------------------------------------------------------------------------------------------------ devarda arithmetic word problems
def arith():
    import convert_devarda_arithmetic as cd
    import _arith_common as C
    import hashlib
    gad = cd.load_module()
    fmt = gad.DatasetFormatter()
    gad.hashlib = cd._Hash

    class _R(random.Random):
        force = None

        def choice(self, seq):
            v = super().choice(seq)
            if _R.force is not None and isinstance(seq, list) and len(seq) == 5:   # the addition / subtraction template lists
                return seq[_R.force]
            return v
    gad.random = types.SimpleNamespace(Random=_R)
    cfgs = load_cfg('devarda_arithmetic')
    GEN = os.path.join(TB, 'raw', 'devarda_arithmetic', 'generated', 'data', 'json')
    base_data = json.load(open(os.path.join(GEN, 'arith_dataset_numeric.json')))
    for task in ('embedded', 'embedded_verbal'):
        cfg = cfgs[task]
        pre, con = cfg['prefix'], cfg['concatenation']
        verbal = task == 'embedded_verbal'
        ref = json.load(open(os.path.join(GEN, 'arith_dataset_%s.json' % task)))
        pool, n_default, n_alt, n_nomatch, n_neg = [], 0, 0, 0, 0
        for base, rf in zip(base_data, ref):
            operands, ops = cd.parse(base['prompt'])
            flipped = ['-' if o == '+' else '+' for o in ops]
            clean = dict(id=base['id'], operands=operands, operators=ops, result=base['ground_truth'])
            corr = dict(id=base['id'], operands=operands, operators=flipped, result=cd.evaluate(operands, flipped))
            if corr['result'] < 0:
                n_neg += 1
                continue
            seed_clean = f"{clean['id']}|{clean['operands']}|{clean['operators']}"
            def render(problem, force):
                cd._Hash.override = seed_clean
                _R.force = force
                try:
                    return fmt.format_embedded_context(problem, verbal=verbal)
                finally:
                    _R.force = None
                    cd._Hash.override = None
            x, y = render(clean, None)
            assert x == rf['prompt'] and y == rf['answer']
            xd, yd = render(corr, None)                                  # the original task's corrupted story
            two = len(operands) == 2
            c_opts = [(x, y, 'default')] + ([render(clean, t) + (t,) for t in range(5)] if two else [])
            k_opts = [(xd, yd, 'default')] + ([render(corr, t) + (t,) for t in range(5)] if two else [])
            good = []                                                     # preference: (default, default), (default, other), (other, any)
            for ci, (x_, y_, ct) in enumerate(c_opts):
                for xp_, yp_, kt in k_opts:
                    if ntok(pre + x_ + con) == ntok(pre + xp_ + con) and y_ != yp_ and x_ != xp_:
                        good.append((0 if ct == 'default' and kt == 'default' else 1 if ct == 'default' else 2, x_, xp_, y_, yp_, ct, kt))
            if not good:
                n_nomatch += 1
                continue
            best = min(g[0] for g in good)
            cand = sorted(set(g for g in good if g[0] == best), key=str)
            _, x, xp, y, yp, ct, kt = cand[random.Random('42:%s' % base['id']).randrange(len(cand))]
            if best == 0:
                n_default += 1
            else:
                n_alt += 1
            gt = base['ground_truth']
            pool.append(make_item(x, xp, y, yp, 'devarda_arithmetic', base['id'], format=task, n_terms=base['n_terms'], n_digits=base['n_digits'],
                                  has_carry=base['has_carry'], operators=''.join(ops), ground_truth=gt, ground_truth_prime=corr['result'], split=base['split'],
                                  clean_template=str(ct), corrupted_template=str(kt)))
        print(task, 'eligible matched', len(pool), 'default template', n_default, 'other template', n_alt, 'no template matches', n_nomatch, 'negative flip', n_neg)
        emit('devarda_arithmetic', task, pool, lambda i: (i['n_terms'], i['n_digits'], i['has_carry'], i['operators']),
             ['RE-PAIRED for equal Qwen3.5 token length (variant of `%s`; prompt = exemplar prefix + story + " The answer is"). Same operator flip with names and object replayed from the clean problem. For two-term problems the generator has 5 story templates per operator class (three-term problems have one); the clean and corrupted templates are chosen among ALL of them so that the prompt token lengths are equal, preferring in this order (1) the original task\'s pair (hash-chosen templates; %d problems), (2) the original clean story with another corrupted template, (3) another clean template, a seeded random choice among the matching combinations (together %d problems; metadata `clean_template` / `corrupted_template`: `default` or the forced template index). Answers are recomputed by the generator for the chosen templates (some subtraction templates ask for the amount given, or the initial amount, so the answer is not always the arithmetic result; the symmetric flip is checked).' % (task, n_default, n_alt),
              'Problems with no matching template (%d), with a negative flipped result (%d) or with y == y\' are dropped before the original stratified sampling (n_terms x n_digits x has_carry x operator pattern, seed 42, cap 2,000). The clean stories are those of the original task; matching changes the stratum mix (the original task\'s strata shares are not exactly reproduced). Item ids are the original problem ids (not matched across formats any more: eligibility is per format).' % (n_nomatch, n_neg),
              'Template change between addition and subtraction stories remains: the corrupted story is not a minimal edit (unchanged from the original task).'],
             extra=dict(n_matched_original_templates=n_default, n_matched_other_templates=n_alt, n_dropped_no_length_match=n_nomatch), yaml_width=10 ** 6)


def pragworld():
    print('pragworld_flip: not re-paired. The pairs are fixed by the source (original vs perturbed conversation of the same item, answers yes/no differ); '
          'the only alternative partner would be a different perturbation of the same dialogue, which changes the contrast (perturbation vs perturbation, not original vs perturbation), '
          'and perturbations edit words, so lengths are not controllable. No file written.')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('ravel', 'all'):
        ravel()
    if what in ('eap', 'all'):
        eap()
    if what in ('arith', 'all'):
        arith()
    if what in ('all', 'pragworld'):
        pragworld()
