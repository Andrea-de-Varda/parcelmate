"""Build number-flip counterfactual items from files of (grammatical, ungrammatical) sentence pairs
(CLAMS evalsets and the Marvin & Linzen templates shipped by Newman et al.). See _syntax_pairing.py
for the pairing rule. Used by convert_clams.py and convert_newman_sva.py.

Per raw pair: words of both sentences are compared; the first and last differing word delimit the
answer span (the words between them, inclusive, joined by spaces, with a leading space); everything
before is the prompt (prompt ends right before the answer), everything after is dropped. Pairs whose
sentences differ in word count are skipped. Items are deduplicated on (prefix, good, bad) (the dropped
tail makes many raw pairs identical). Counterfactual partners come from the same file. Each unordered
pair {A,B} yields one item; which of the two is 'clean' is decided by a seed-42 coin (so number of the
clean subject is balanced). Items with no unique partner are dropped and counted.
"""
import random
import re

from _syntax_common import SEED, stratified_sample
from _syntax_pairing import find_counterfactuals


def detok(s):
    return re.sub(r' ([,;:.!?])', r'\1', s)


def parse_pair(good, bad):
    g, b = good.split(), bad.split()
    # split final sentence punctuation off the last word (Newman files end with '.')
    for w in (g, b):
        if w and re.search(r'[.!?]$', w[-1]) and len(w[-1]) > 1:
            w[-1], p = w[-1][:-1], w[-1][-1]
            w.append(p)
    if len(g) != len(b):
        return None
    d = [i for i in range(len(g)) if g[i] != b[i]]
    if not d:
        return None
    f, l = d[0], d[-1]
    return tuple(g[:f]), ' '.join(g[f:l + 1]), ' '.join(b[f:l + 1])


def build_items(pairs, base_fields, stem, lang=None):
    """pairs: list of (line_number, good_sentence, bad_sentence) in file order. Returns (items, stats)."""
    stats = dict(n_pairs_raw=len(pairs), skipped_unparsable=0)
    pool, first_line, seen = [], [], {}
    for ln, g, b in pairs:
        p = parse_pair(g, b)
        if p is None:
            stats['skipped_unparsable'] += 1
            continue
        if p in seen:
            continue
        seen[p] = len(pool)
        pool.append(p)
        first_line.append(ln)
    stats['n_unique_prefix_answer'] = len(pool)
    partner, pst = find_counterfactuals(pool, lang=lang)
    stats.update({'dropped_' + k: v for k, v in pst.items()})
    unordered = sorted({tuple(sorted((i, j))) for i, j in partner.items()})
    rng = random.Random(SEED)
    items = []
    for i, j in unordered:
        if rng.random() < 0.5:
            i, j = j, i
        (pa, ga, ba), (pb, gb, bb) = pool[i], pool[j]
        assert gb == ba and bb == ga
        items.append(dict(
            clean_problem=detok(' '.join(pa)), corrupted_problem=detok(' '.join(pb)),
            clean_correct=' ' + detok(ga), clean_incorrect=' ' + detok(ba),
            corrupted_correct=' ' + detok(gb), corrupted_incorrect=' ' + detok(bb),
            n_words_differing_in_prompt=sum(1 for x, y in zip(pa, pb) if x != y),
            source_item_id=f'{stem}:{first_line[i]}', corrupted_source_item_id=f'{stem}:{first_line[j]}',
            **base_fields))
    stats['n_items_all_pairs'] = len(items)
    return items, stats


def sample(items):
    return stratified_sample(items, lambda it: (it['clean_correct'], it['clean_incorrect']))
