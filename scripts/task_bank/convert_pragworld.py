"""Convert PragWorld (Vashistha et al., AAAI 2026; arXiv 2511.13021) to tier-1 items from its original/perturbed conversation pairs whose yes/no answer differs.

Two CSVs (SachinVashisth/PRAGWORLD): Manual_Dataset_500_Final.csv (88 original dialogues x their perturbed variants; rows with Perturbation? = FALSE are originals, the TRUE rows that follow
are their perturbations; 'Perturbation Effect' = Flip / Invarianat) and synthetic_final.csv (Seed Index + Question identify the original; FALSE rows = originals, TRUE rows = perturbations).
An item = (original prompt, perturbed prompt) whose gold answers are 'yes' and 'no' in either order (we use the answers, not the 'Perturbation Effect' label: they disagree on 4 manual rows).
Orientation (which member is the clean prompt) is a seeded coin (random.Random('42:pragworld')). Prompt = "<dialogue>\\n\\nQuestion: <question>\\nAnswer (Yes or No):", answer " Yes" / " No";
whitespace in questions is collapsed and a leading list number ("1. ") removed. Rows with answers 'notsure' / 'cannot answer' / typos are unusable for a yes/no flip and are dropped.
Manual and synthetic items are kept in ONE task (219 items in total, below the 300 target; each half alone would be about 135 and 84). Clean and corrupted prompts differ by a word substitution / swap / quantifier / connective change, so
token lengths often differ (reported). Caveats: dialogues derive from GRICE / CICERO (their licences are not stated in the repo), answers are human or LLM-assisted labels without model verification, and the
question is sometimes the perturbed part (102 of 412 manual perturbations rewrite the question rather than the dialogue)."""
import csv, collections, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knowledge_common as kc

SID = 'pragworld'

def norm_q(q):
    q = ' '.join(q.split())
    return re.sub(r'^\d+\.\s*', '', q)

def prompt(ctx, q):
    return ctx.strip() + '\n\nQuestion: ' + norm_q(q) + '\nAnswer (Yes or No):'

def yn(a):
    a = a.strip().lower()
    return a if a in ('yes', 'no') else None

def main():
    man = kc.load_manifest(SID)
    kc.clear_tasks(SID)
    base = os.path.join(kc.RAW, SID, 'repo')
    pairs = []  # (variant, orig_row, pert_row, id)
    man_rows = list(csv.DictReader(open(os.path.join(base, 'Manual_Dataset_500_Final.csv'), encoding='utf-8')))
    cur = None
    for ln, r in enumerate(man_rows):
        p = r['Perturbation?'].strip()
        if p == 'FALSE':
            cur = r
        elif p == 'TRUE' and cur is not None:
            pairs.append(('manual', cur, r, 'manual:%d' % (ln + 2)))
    syn_rows = list(csv.DictReader(open(os.path.join(base, 'synthetic_final.csv'), encoding='utf-8')))
    orig = {}
    for ln, r in enumerate(syn_rows):
        if r['Perturbation?'] == 'FALSE':
            orig[(r['Seed Index'], r['Question'])] = r
    for ln, r in enumerate(syn_rows):
        if r['Perturbation?'] == 'TRUE' and (r['Seed Index'], r['Question']) in orig:
            pairs.append(('synthetic', orig[(r['Seed Index'], r['Question'])], r, 'synthetic:%d' % (ln + 2)))
    rng = random.Random('42:pragworld')
    pool, n_same, n_bad = [], 0, 0
    n_label_mismatch = 0
    for variant, o, p, rid in pairs:
        ao, ap = yn(o['Answer']), yn(p['Answer'])
        if ao is None or ap is None:
            n_bad += 1
            continue
        if ao == ap:
            n_same += 1
            continue
        if variant == 'manual' and (p['Perturbation Effect'].strip() != 'Flip'):
            n_label_mismatch += 1
        co, cp = prompt(o['Local Context'], o['Question']), prompt(p['Local Context'], p['Question'])
        orig_clean = rng.random() < 0.5
        (cc, ca, cq), (xc, xa, xq) = ((co, ao, o), (cp, ap, p)) if orig_clean else ((cp, ap, p), (co, ao, o))
        pool.append(kc.make_item(cc, xc, ' ' + ca.capitalize(), ' ' + xa.capitalize(), SID, rid, variant=variant, clean_is_original=orig_clean, perturbation_category=' '.join(p['Perturbation Category'].split()),
                                 perturbation_effect_label=p['Perturbation Effect'].strip()))
    ad = ['pairs original and perturbed conversation of the same item whose yes/no gold answers differ (from the answers, not from the Perturbation Effect label; %d manual rows disagree with the label)' % n_label_mismatch,
          'prompt "<dialogue>\\n\\nQuestion: <q>\\nAnswer (Yes or No):", answers " Yes"/" No"; zero-shot; whitespace normalised', 'orientation by seeded coin',
          'dropped: %d pairs with a non yes/no answer (notsure, cannot answer, typos), %d pairs with equal answers (answer-invariant perturbations)' % (n_bad, n_same)]
    rec = kc.write_task(SID, 'pragworld_flip', 'pragmatics_dialogue_yesno', 'pragmatics / local world model', pool, lambda i: i['variant'], len(pairs), ad, extra=dict(n_manual=sum(i['variant'] == 'manual' for i in pool), n_synthetic=sum(i['variant'] == 'synthetic' for i in pool)))
    rec['notes'] = 'below the 300-item target'
    kc.write_config(SID, ['pragworld_flip'])
    reg = kc.base_registry(SID, man, title='PragWorld: a benchmark evaluating LLMs\' local world model under minimal linguistic alterations and conversational dynamics',
                           bibkey='Vashistha_2026', bibkey_data='vashistha2025pragworldbenchmarkevaluatingllms (arXiv version)', url=man['url'], revision=man['revision'],
                           licence='MIT (LICENSE at the pinned commit); seed dialogues come from GRICE and CICERO, whose licences are not stated in the repository', licence_source='external/task_bank/raw/pragworld/repo/LICENSE',
                           redistribute='yes (MIT repo; upstream seed-data licences unchecked, flagged)', tier=1, tokenizer=kc.tokenizer_meta(), status='converted')
    kc.write_registry(SID, reg, [rec])
    print(rec['name'], rec['n_items_raw'], rec['n_items_converted'], rec['len_match_share_qwen35'], rec['single_token_answers_share'], rec['n_manual'], rec['n_synthetic'])

if __name__ == '__main__':
    main()
