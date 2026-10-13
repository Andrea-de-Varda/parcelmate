"""Shared helper for the syntax group: build number-flip counterfactual pairs from a pool of
(prefix_words, good, bad) agreement items (CLAMS, Newman et al. / Marvin & Linzen templates).

Item A = (words of the prompt, good answer g, bad answer b). Its counterfactual B is an item of the
SAME pool whose good answer equals A's bad answer and whose bad answer equals A's good answer (so the
answer flips symmetrically by construction), whose prompt has the same number of words and differs
from A in d positions, where every differing word pair is a morphological variant pair (see
_make_close). The smallest d (1..max_diff) with a candidate is used; among candidates at that d the
one with the smallest character difference. If the best candidate is not unique, or none exists, A is
dropped and counted. Deterministic: ties are never broken arbitrarily.
"""
from collections import defaultdict
from itertools import combinations

EXTRA_FLIPS = {  # closed-class number flips that are not stem variants (article agreement); per language
    'de': {('der', 'die'), ('den', 'die'), ('dem', 'den'), ('der', 'den')},
}


def _common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x.lower() != y.lower():
            break
        n += 1
    return n


def _make_close(items, extra):
    """Two prompt words are variants of one another if they share a stem (common prefix of at least
    max(2, min length - 3) characters), or are answer-position flips seen in this very pool (e.g.
    French est/sont, German mag/moegen), or are listed in `extra` (article flips)."""
    learned = set()
    for _, g, b in items:
        for x, y in zip(g.split(), b.split()):
            if x != y:
                learned.add((x, y))
                learned.add((y, x))
    for x, y in extra:
        learned.add((x, y))
        learned.add((y, x))

    def close(a, b):
        if a == b or (a, b) in learned:
            return True
        return _common_prefix(a, b) >= max(2, min(len(a), len(b)) - 3)
    return close


def find_counterfactuals(items, max_diff=3, lang=None):
    """items: list of (prefix_words tuple, good, bad). Returns (pairs, stats): pairs is a dict
    index_A -> index_B, stats counts the dropped items by reason."""
    close = _make_close(items, EXTRA_FLIPS.get(lang, set()))
    out, stats = {}, defaultdict(int)
    pending = set(range(len(items)))
    for d in range(1, max_diff + 1):
        if not pending:
            break
        index = defaultdict(list)  # (len, good, bad, positions, remaining words) -> indices
        for j, (q, g, b) in enumerate(items):
            for pos in combinations(range(len(q)), d):
                rest = tuple(w for k, w in enumerate(q) if k not in pos)
                index[(len(q), g, b, pos, rest)].append(j)
        for i in sorted(pending):
            p, g, b = items[i]
            found = []
            for pos in combinations(range(len(p)), d):
                rest = tuple(w for k, w in enumerate(p) if k not in pos)
                for j in index.get((len(p), b, g, pos, rest), []):
                    q = items[j][0]
                    if all(p[k] != q[k] and close(p[k], q[k]) for k in pos):
                        cost = sum(abs(len(p[k]) - len(q[k])) + sum(1 for x, y in zip(p[k], q[k]) if x != y) for k in pos)
                        found.append((cost, j))
            if not found:
                continue
            found.sort()
            pending.discard(i)
            if len(found) > 1 and found[0][0] == found[1][0]:
                stats['ambiguous'] += 1
            else:
                out[i] = found[0][1]
    stats['no_partner'] = len(pending)
    return out, dict(stats)
