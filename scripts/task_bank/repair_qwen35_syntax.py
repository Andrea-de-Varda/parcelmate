"""Qwen3.5 length-matched re-pairings (`<task>__qwen35`) of the syntax-group tasks. Deterministic (seed 42), tokenizer only.

Why: the evaluation drops every item whose clean and corrupted prompts (prefix + problem + concatenation, no chat template, no
BOS) tokenize to different lengths under the Qwen3.5 tokenizer. For the tasks listed below more than 10% of the items were lost.
Each variant keeps the source's own contrast and template and only chooses different partners, from the source's own pools,
so that every item has equal clean/corrupted token counts. Originals are never touched; the variants are written next to
them (tasks/<source>/<task>__qwen35.json + .ids.txt) and appended to configs/<source>.json and registry/<source>.yaml
(idempotent: earlier __qwen35 entries are replaced). Rerun after convert_*.py (which rewrite the registry and config).

Usage: python scripts/task_bank/repair_qwen35_syntax.py [source ...]   (default: all four sources)

CausalGym. Source = SyntaxGym templates (raw/causalgym/repo/data/templates/syntaxgym.json): base and src differ only in the
span(s) of the label variable, each drawn from the variable's list for its type (plural/singular nouns, did/other verbs,
transitive/intransitive verbs, names by gender); all other variables are shared. For every converted item (same dedupe and
ids as convert_causalgym.py) the shared spans and the two types are kept, and the label-variable spans are re-chosen from the
type's lists: the original pair is kept if its token counts are equal; otherwise a src span giving equal length (clean span
kept), otherwise any (clean, src) span combination giving equal length; items with no such combination are dropped.
Labels (base_label, src_label) are kept (agr_sv_num_*_matched: the number-flipped form of the clean label). Tasks whose
contrast inserts or deletes a token (who was / by, Since the / The, comma / no comma) have no equal-length combination at
all and get no variant (the script reports them).

Marks, Newman, CLAMS. Every prompt of the pool is a (words, good answer, bad answer) entry; the source pairs two entries of
the same file whose answers are swapped. Pairs with equal token counts are kept unchanged. For the others, one side is kept
and the other is replaced by another pool entry that (i) has the swapped answers, (ii) has the same words everywhere except
the positions where the original pair differed plus the noun slot, (iii) has a close morphological variant (same stem, or
the article/verb flip rules of _syntax_pairing.py) at every differing position other than the noun slot, and (iv) gives equal
token counts. The noun slot is the first differing position (the position after the article for French and German, whose
nouns may be invariant). So the subject noun of the corrupted prompt is another noun of the same lexicon with the other
number; the structure, the object/attractor words, and the agreeing words (same lexeme) are unchanged. If the corrupted side
cannot be replaced, the clean side is replaced; if neither, the pair is dropped.
"""
import functools
import json
import os
import random
import re
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _syntax_common import *  # noqa
from _syntax_agreement import parse_pair, detok, sample as pair_sample
from _syntax_pairing import find_counterfactuals, _make_close, EXTRA_FLIPS
import convert_causalgym as CG
import convert_clams as CL
import convert_marks_sfc_sva as MK
import convert_newman_sva as NW

SUFFIX = '__qwen35'
ARTICLES = {'fr': {'le', 'la', 'les', "l'"}, 'de': {'der', 'die', 'den', 'dem', 'das'}}

CAUSALGYM_TASKS = ['agr_gender', 'agr_refl_num_obj-relc', 'agr_refl_num_pp', 'agr_refl_num_subj-relc',
                   'agr_sv_num_obj-relc_matched', 'agr_sv_num_pp_matched', 'agr_sv_num_subj-relc_matched', 'cleft',
                   'garden_mvrr', 'garden_mvrr_mod', 'garden_npz_obj', 'garden_npz_obj_mod', 'garden_npz_v-trans',
                   'garden_npz_v-trans_mod', 'gss_subord', 'gss_subord_obj-relc', 'gss_subord_pp', 'gss_subord_subj-relc']
MARKS_TASKS = ['nounpp', 'rc', 'simple']
NEWMAN_TASKS = ['subj_rel']


@functools.lru_cache(maxsize=None)
def ntok(s):
    return len(tokenizer().encode(s).ids)


def single_token_share(items):
    tok = tokenizer()
    return round(sum(all(len(tok.encode(it[k]).ids) == 1 for k in ('clean_correct', 'clean_incorrect')) for it in items) / len(items), 4)


def len_match_share(items):
    return round(sum(ntok(it['clean_problem']) == ntok(it['corrupted_problem']) for it in items) / len(items), 4)


# ------------------------------------------------------------------------------------------------ agreement sources

class Pool:
    """Entries (words tuple, good, bad); candidate search for number-flip partners with a free noun slot."""

    def __init__(self, entries, lang=None):
        self.e = entries
        self.lang = lang
        self.close = _make_close(entries, EXTRA_FLIPS.get(lang, set()))
        self.index = {}

    def prompt(self, k):
        return detok(' '.join(self.e[k][0]))

    def noun_slot(self, k, D):
        p0 = D[0]
        if self.e[k][0][p0].lower() in ARTICLES.get(self.lang, ()):
            return p0 + 1 if p0 + 1 < len(self.e[k][0]) else None
        return p0

    def _index(self, allowed):
        if allowed not in self.index:
            idx = {}
            for j, (q, g, b) in enumerate(self.e):
                if all(a < len(q) for a in allowed):
                    idx.setdefault((len(q), g, b, tuple(w for p, w in enumerate(q) if p not in allowed)), []).append(j)
            self.index[allowed] = idx
        return self.index[allowed]

    def candidates(self, i, D):
        """Equal-length partners for entry i (swapped answers, differing at D plus the noun slot only, carriers close)."""
        p, g, b = self.e[i]
        free = self.noun_slot(i, D)
        allowed = tuple(sorted(set(D) | ({free} if free is not None else set())))
        rest = tuple(w for k, w in enumerate(p) if k not in allowed)
        out = []
        for j in self._index(allowed).get((len(p), b, g, rest), []):
            q = self.e[j][0]
            if j == i or not all(q[k] != p[k] and self.close(p[k], q[k]) for k in D if k != free):
                continue
            if ntok(self.prompt(j)) == ntok(self.prompt(i)):
                out.append(j)
        return out


def diff_positions(a, b):
    return tuple(k for k in range(len(a)) if a[k] != b[k])


def repair_pairs(pool, oriented, rng):
    """oriented: list of (clean index, corrupted index, meta). Returns list of (clean, corrupted, meta, how) and stats."""
    out, seen = [], set()
    st = dict(n_pairs_original=len(oriented), n_equal_original=0, n_replaced_corrupted=0, n_replaced_clean=0, n_dropped_no_partner=0, n_dropped_duplicate=0)
    for i, j, meta in oriented:
        D = diff_positions(pool.e[i][0], pool.e[j][0])
        if ntok(pool.prompt(i)) == ntok(pool.prompt(j)):
            new, how = (i, j), 'unchanged'
            st['n_equal_original'] += 1
        else:
            c = pool.candidates(i, D)
            if c:
                new, how = (i, rng.choice(sorted(c))), 'corrupted'
                st['n_replaced_corrupted'] += 1
            else:
                c = pool.candidates(j, D)
                if c:
                    new, how = (rng.choice(sorted(c)), j), 'clean'
                    st['n_replaced_clean'] += 1
                else:
                    st['n_dropped_no_partner'] += 1
                    continue
        key = frozenset(new) if meta.get('unordered', True) else new
        if key in seen:
            st['n_dropped_duplicate'] += 1
            continue
        seen.add(key)
        out.append((new[0], new[1], meta, how, (i, j)))
    return out, st


def newman_clams_variant(pairs, base_fields, stem, lang):
    """Same pool/partner/orientation steps as _syntax_agreement.build_items, then repair_pairs."""
    pool_e, first_line, seen = [], [], {}
    for ln, g, b in pairs:
        p = parse_pair(g, b)
        if p is None or p in seen:
            continue
        seen[p] = len(pool_e)
        pool_e.append(p)
        first_line.append(ln)
    partner, _ = find_counterfactuals(pool_e, lang=lang)
    unordered = sorted({tuple(sorted((i, j))) for i, j in partner.items()})
    coin = random.Random(SEED)
    oriented = []
    for i, j in unordered:
        if coin.random() < 0.5:
            i, j = j, i
        oriented.append((i, j, {}))
    pool = Pool(pool_e, lang)
    res, st = repair_pairs(pool, oriented, random.Random(SEED))
    items = []
    for i, j, _, how, (oi, oj) in res:
        (pa, ga, ba), (pb, gb, bb) = pool_e[i], pool_e[j]
        assert gb == ba and bb == ga
        it = dict(clean_problem=detok(' '.join(pa)), corrupted_problem=detok(' '.join(pb)),
                  clean_correct=' ' + detok(ga), clean_incorrect=' ' + detok(ba),
                  corrupted_correct=' ' + detok(gb), corrupted_incorrect=' ' + detok(bb),
                  n_words_differing_in_prompt=sum(1 for x, y in zip(pa, pb) if x != y),
                  source_item_id=f'{stem}:{first_line[i]}', corrupted_source_item_id=f'{stem}:{first_line[j]}',
                  repaired_qwen35=how, **base_fields)
        if how != 'unchanged':
            it['original_clean_problem'] = detok(' '.join(pool_e[oi][0]))
            it['original_corrupted_problem'] = detok(' '.join(pool_e[oj][0]))
        items.append(it)
    return items, st


def marks_variant(task, stem, family):
    d = os.path.join(raw_dir(MK.SID), 'repo', 'data')
    entries, eid, first_id, pairs, seen = [], {}, [], [], set()
    n_raw = 0

    def entry(prefix, g, b, ident):
        assert prefix == ' '.join(prefix.split())
        k = (tuple(prefix.split()), g, b)
        if k not in eid:
            eid[k] = len(entries)
            entries.append(k)
            first_id.append(ident)
        return eid[k]
    for split in ('train', 'test'):
        for ln, line in enumerate(open(os.path.join(d, f'{stem}_{split}.json'), encoding='utf8'), 1):
            r = json.loads(line)
            n_raw += 1
            key = (r['clean_prefix'], r['patch_prefix'], r['clean_answer'], r['patch_answer'])
            if key in seen:
                continue
            seen.add(key)
            ident = f'{stem}_{split}.json:{ln}'
            a = entry(r['clean_prefix'], r['clean_answer'], r['patch_answer'], ident)
            b = entry(r['patch_prefix'], r['patch_answer'], r['clean_answer'], ident)
            pairs.append((a, b, dict(case=r['case'], split=split, source_item_id=ident, unordered=False)))
    pool = Pool(entries)
    res, st = repair_pairs(pool, pairs, random.Random(SEED))
    items = []
    for i, j, meta, how, (oi, oj) in res:
        (pa, ga, ba), (pb, gb, bb) = entries[i], entries[j]
        assert gb == ba and bb == ga
        it = dict(clean_problem=' '.join(pa), corrupted_problem=' '.join(pb), clean_correct=ga, clean_incorrect=ba,
                  corrupted_correct=gb, corrupted_incorrect=bb, task=task, family=family, case=meta['case'],
                  split=meta['split'], source_id=MK.SID,
                  source_item_id=meta['source_item_id'] if how != 'clean' else first_id[i],
                  repaired_qwen35=how)
        if how != 'unchanged':
            it['original_clean_problem'] = ' '.join(entries[oi][0])
            it['original_corrupted_problem'] = ' '.join(entries[oj][0])
            it['corrupted_source_item_id'] = first_id[j]
        items.append(it)
    return items, st, n_raw, len(pairs)


# ------------------------------------------------------------------------------------------------------ CausalGym

def cg_template(task):
    base = task[:-len('_matched')] if task.endswith('_matched') else task
    t = json.load(open(os.path.join(raw_dir(CG.SID), 'repo', 'data', 'templates', 'syntaxgym.json')))[base]
    assert len(t['templates']) == 1
    spans = [x for x in re.split(r"(?<=\})|(?= \{)|(?<! )(?=\{)", CG.BOS + t['templates'][0]) if x != '']
    label_vars = t['label'] if isinstance(t['label'], list) else [t['label']]
    pos = [k for k, s in enumerate(spans) if any('{' + v + '}' == m for v in label_vars for m in re.findall(r"\{.+?\}", s))]
    return t, spans, label_vars, pos


def cg_variants(t, spans, label_vars, pos, typ):
    """All label-variable span tuples (at positions pos) for a type, from the template lists."""
    import itertools
    choices = []
    for p in pos:
        var = re.findall(r"\{(.+?)\}", spans[p])[0]
        assert var in label_vars
        choices.append(sorted({spans[p].replace('{' + var + '}', c) for c in t['variables'][var][typ]}))
    return [tuple(c) for c in itertools.product(*choices)]


def causalgym_variant(task, rows_by_task):
    t, spans, label_vars, pos = cg_template(task)
    base_task = task[:-len('_matched')] if task.endswith('_matched') else task
    matched = task.endswith('_matched')
    cache = {}

    def variants(typ):
        if typ not in cache:
            cache[typ] = cg_variants(t, spans, label_vars, pos, typ)
        return cache[typ]
    seen, seen_new, items, st = set(), set(), [], dict(n_original=0, n_equal_original=0, n_src_changed=0, n_both_changed=0, n_dropped=0, n_dropped_duplicate=0)
    for r in rows_by_task[base_task]:
        key = (CG.text(r['base']), CG.text(r['src']), r['base_label'], r['src_label'])
        if key in seen:
            continue
        seen.add(key)
        st['n_original'] += 1
        sid = f"{r['split']}:{r['idx']}"
        bs, ss = list(r['base']), list(r['src'])
        assert len(bs) == len(spans) and all(bs[k] == ss[k] for k in range(len(bs)) if k not in pos), (task, sid)
        b0, s0 = tuple(bs[p] for p in pos), tuple(ss[p] for p in pos)
        assert b0 in variants(r['base_type']) and s0 in variants(r['src_type']), (task, sid, b0, s0)

        def text(v):
            sp = list(bs)
            for p, x in zip(pos, v):
                sp[p] = x
            return CG.text(sp)
        if ntok(text(b0)) == ntok(text(s0)):
            b1, s1, how = b0, s0, 'unchanged'
            st['n_equal_original'] += 1
        else:
            rng = random.Random(f'{SEED}:{task}:{sid}')
            c = [s for s in variants(r['src_type']) if ntok(text(s)) == ntok(text(b0))]
            if c:
                b1, s1, how = b0, rng.choice(c), 'src'
                st['n_src_changed'] += 1
            else:
                c = [(b, s) for b in variants(r['base_type']) for s in variants(r['src_type']) if ntok(text(b)) == ntok(text(s))]
                if not c:
                    st['n_dropped'] += 1
                    continue
                b1, s1 = rng.choice(c)
                how = 'both'
                st['n_both_changed'] += 1
        clean, corr = text(b1), text(s1)
        cc, ci = r['base_label'], r['src_label']
        if matched:
            ci = CG.FLIP[cc]
        k2 = (clean, corr, cc, ci)
        if k2 in seen_new:
            st['n_dropped_duplicate'] += 1
            continue
        seen_new.add(k2)
        it = dict(clean_problem=clean, corrupted_problem=corr, clean_correct=cc, clean_incorrect=ci,
                  corrupted_correct=ci, corrupted_incorrect=cc, task=task, family=CG.FAMILY[base_task],
                  base_type=r['base_type'], src_type=r['src_type'], split=r['split'], source_id=CG.SID,
                  source_item_id=sid, repaired_qwen35=how)
        if matched:
            it['original_src_label'] = r['src_label']
        if how != 'unchanged':
            it['original_clean_problem'], it['original_corrupted_problem'] = key[0], key[1]
        items.append(it)
    return items, st


# -------------------------------------------------------------------------------------------------------- outputs

def read_json(p):
    return json.load(open(p, encoding='utf8'))


def finish(source, task, items, ids, orig_entry, extra_adapt, extra):
    """Write the variant files and return its registry entry."""
    name = task + SUFFIX
    for it in items:
        assert ntok(it['clean_problem']) == ntok(it['corrupted_problem']), (name, it['source_item_id'])
    write_task(source, name, items, ids)
    orig_items = read_json(os.path.join(TB, 'tasks', source, task + '.json'))
    entry = dict(orig_entry)
    entry.update(name=name, variant_of=task, n_items_original_task=len(orig_items),
                 n_items_aligned_original_task=sum(ntok(i['clean_problem']) == ntok(i['corrupted_problem']) for i in orig_items),
                 n_items_converted=len(items), sampled_ids_file=f'tasks/{source}/{name}.ids.txt',
                 len_match_share_qwen35=len_match_share(items), single_token_answer_share_qwen35=single_token_share(items))
    entry.pop('single_token_answer_share', None)
    entry['adaptations'] = list(orig_entry['adaptations']) + extra_adapt
    entry.update(extra)
    assert entry['len_match_share_qwen35'] == 1.0
    return entry


def update_registry_config(source, entries):
    rp = os.path.join(TB, 'registry', source + '.yaml')
    reg = yaml.safe_load(open(rp, encoding='utf8'))
    reg['tasks'] = [t for t in reg['tasks'] if not t['name'].endswith(SUFFIX)] + entries
    write_registry(source, reg)
    cp = os.path.join(TB, 'configs', source + '.json')
    cfg = {k: v for k, v in json.load(open(cp)).items() if not k.endswith(SUFFIX)}
    for e in entries:
        cfg[e['name']] = dict(cfg[e['variant_of']], path=e['name'] + '.json')
    json.dump(cfg, open(cp, 'w'), indent=2)


def reg_entry(source, task):
    return next(t for t in yaml.safe_load(open(os.path.join(TB, 'registry', source + '.yaml'), encoding='utf8'))['tasks'] if t['name'] == task)


def run_causalgym():
    rows = []
    for split in ('train', 'dev', 'test'):
        for i, r in enumerate(read_json(os.path.join(raw_dir(CG.SID), 'hf', split + '.json'))):
            r['split'], r['idx'] = split, i
            rows.append(r)
    by_task = {}
    for r in rows:
        by_task.setdefault(r['task'], []).append(r)
    entries, skipped = [], []
    for task in CAUSALGYM_TASKS:
        items, st = causalgym_variant(task, by_task)
        if not items:
            skipped.append(task)
            print('CAUSALGYM', task, 'NOT re-paired:', st, flush=True)
            continue
        entry = finish(CG.SID, task, items, [i['source_item_id'] for i in items], reg_entry(CG.SID, task),
                       ['QWEN35 RE-PAIRING: label-variable spans (the only text that differs between base and src) re-chosen from the SyntaxGym template lists of the same types so that clean and corrupted prompts have equal Qwen3.5 token counts; shared spans, types and labels kept; equal-length original pairs kept unchanged'],
                       dict(repair_stats=st))
        entries.append(entry)
        print('CAUSALGYM', task, st, '->', len(items), flush=True)
    update_registry_config(CG.SID, entries)
    return skipped


def run_marks():
    entries = []
    for task in MARKS_TASKS:
        stem, family = MK.FILES[task]
        items, st, n_raw, n_pairs = marks_variant(task, stem, family)
        kept = stratified_sample(items, lambda it: (it['split'], it['case']))
        entry = finish(MK.SID, task, kept, [i['source_item_id'] for i in kept], reg_entry(MK.SID, task),
                       ['QWEN35 RE-PAIRING: pairs with equal Qwen3.5 prompt lengths kept; for the others the corrupted (else the clean) prompt is replaced by another pool prompt with the swapped answers, the same words except a different subject noun of the other number, equal token count'],
                       dict(repair_stats=st, n_items_after_repair=len(items),
                            sampling_rule='exact duplicates dropped; if > 2000, seed-42 proportional sample stratified on (split, case)'))
        entries.append(entry)
        print('MARKS', task, st, len(items), '->', len(kept), flush=True)
    update_registry_config(MK.SID, entries)


def run_newman():
    entries = []
    d = os.path.join(raw_dir(NW.SID), 'repo', 'data', 'ML')
    for task in NEWMAN_TASKS:
        pairs = []
        for ln, line in enumerate(open(os.path.join(d, task + '_all.jsonl'), encoding='utf8'), 1):
            r = json.loads(line)
            pairs.append((ln, r['sentence_good'].strip(), r['sentence_bad'].strip()))
        items, st = newman_clams_variant(pairs, dict(task=task, family=NW.FAMILY[task], structure=task, source_id=NW.SID), f'ML/{task}_all.jsonl', None)
        kept = pair_sample(items)
        ids = [i['source_item_id'] + '|' + i['corrupted_source_item_id'] for i in kept]
        entry = finish(NW.SID, task, kept, ids, reg_entry(NW.SID, task), [ADAPT_PAIRS], dict(repair_stats=st, n_items_after_repair=len(items)))
        entries.append(entry)
        print('NEWMAN', task, st, len(items), '->', len(kept), flush=True)
    update_registry_config(NW.SID, entries)


ADAPT_PAIRS = ('QWEN35 RE-PAIRING: pairs with equal Qwen3.5 prompt lengths kept; for the others the corrupted (else the clean) prompt is replaced by another prompt of the same file with the swapped '
               'answers, the same words except a different subject noun of the other number and close variants (same lexeme) of the agreeing words, equal token count')


def run_clams():
    entries = []
    for lang, stem in CL.JOBS:
        if lang == 'en':
            continue
        task = f'{lang}_{stem}'
        path = os.path.join(raw_dir(CL.SID), 'repo', f'{lang}_evalset', stem + '.txt')
        family = CL.STRUCT_FAMILY[stem] + '_' + lang
        items, st = newman_clams_variant(CL.read_pairs(path), dict(task=task, family=family, language=lang, structure=stem, source_id=CL.SID),
                                         f'{lang}_evalset/{stem}.txt', lang)
        kept = pair_sample(items)
        if not kept:
            print('CLAMS', task, 'NOT re-paired', st, flush=True)
            continue
        ids = [i['source_item_id'] + '|' + i['corrupted_source_item_id'] for i in kept]
        entry = finish(CL.SID, task, kept, ids, reg_entry(CL.SID, task), [ADAPT_PAIRS], dict(repair_stats=st, n_items_after_repair=len(items)))
        entries.append(entry)
        print('CLAMS', task, st, len(items), '->', len(kept), flush=True)
    update_registry_config(CL.SID, entries)


def main():
    which = sys.argv[1:] or ['causalgym', 'marks_sfc_sva', 'newman_sva', 'clams']
    if 'causalgym' in which:
        run_causalgym()
    if 'marks_sfc_sva' in which:
        run_marks()
    if 'newman_sva' in which:
        run_newman()
    if 'clams' in which:
        run_clams()


if __name__ == '__main__':
    main()
