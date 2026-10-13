"""Convert the EAP-IG task set (Hanna, Pezzelle & Belinkov 2024) to tier-1 tasks. Reads raw/eap_ig (fetch_eap_ig.py).
Token ids in the source CSVs are GPT-2 ids; answers are recovered by decoding them with the GPT-2 tokenizer (tokenizer-only)."""
import ast, json, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

SID = 'eap_ig'
RAW = os.path.join(TB, 'raw', 'eap_ig')
man = json.load(open(os.path.join(RAW, 'MANIFEST.json')))
D = os.path.join(RAW, 'eap-ig-faithfulness', 'data')
GPT2_REV = '607a30d783dfa663caf39e06633721c8d4cfcd7e'
g2 = Tokenizer.from_file(hf_hub_download('openai-community/gpt2', 'tokenizer.json', revision=GPT2_REV))
dec = lambda i: g2.decode([int(i)])
tasks = []

# ------------------------------------------------------------------ IOI
def ioi_pool(df, tag, ids=True):
    pool, nmis = [], 0
    for k, r in df.iterrows():
        c, h = r['clean'].split(' '), r['corrupted_hard'].split(' ')
        diff = [i for i in range(len(c)) if c[i] != h[i]]
        assert len(c) == len(h) and len(diff) == 1
        s, io = c[diff[0]], h[diff[0]]     # S2 in clean; IO name written in its place in corrupted_hard
        if ids:
            assert dec(r['correct_idx']) == ' ' + io and dec(r['incorrect_idx']) == ' ' + s, (r.to_dict(), s, io)
        pool.append(make_item(r['clean'], r['corrupted_hard'], ' ' + io, ' ' + s, SID, 'ioi_%s/%s' % (tag, k), io_name=io, s_name=s))
    return pool
df = pd.read_csv(os.path.join(D, 'ioi', 'gpt2.csv'))
tasks.append(write_task(SID, 'ioi_eapig_gpt2', 'ioi', 'syntax/semantics (name movement)', ioi_pool(df, 'gpt2'), lambda i: 0, len(df),
    ['data/ioi/gpt2.csv of eap-ig-faithfulness (1,000 items from the Wang et al. generator, filtered so names are single GPT-2 tokens). The CSV has `corrupted` (ABC: S2 replaced by a third name; the answer does not flip) and `corrupted_hard` (S2 replaced by the IO name, so IO and S exchange roles). We use `corrupted_hard` as the corrupted prompt: this is the flipping variant. Answers decoded from correct_idx/incorrect_idx (asserted equal to the names in the text).',
     'The ABC `corrupted` column is not used (no flip).']))
df2 = pd.read_csv(os.path.join(RAW, 'EAP-IG', 'ioi_llama.csv'))
tasks.append(write_task(SID, 'ioi_eapig_llama', 'ioi', 'syntax/semantics (name movement)', ioi_pool(df2, 'llama', ids=False), lambda i: 0, len(df2),
    ['ioi_llama.csv of the hannamw/EAP-IG repo (499 rows; names single tokens for the Llama tokenizer, ids are Llama ids and are not used); answers recovered from the text (name written in place of S2 in `corrupted_hard`). Same structure and template family as ioi_eapig_gpt2, different item sample.']))

# ------------------------------------------------------------------ gender bias
df = pd.read_csv(os.path.join(D, 'gender-bias', 'gpt2.csv'))
pool = []
for k, r in df.iterrows():
    a, b = dec(r['clean_answer_idx']), dec(r['corrupted_answer_idx'])
    pool.append(make_item(r['clean'], r['corrupted'], a, b, SID, 'gender-bias/gpt2/%d' % k, label=int(r['label']),
                          clean_subject=r['clean'].split(' ')[1], corrupted_subject=r['corrupted'].split(' ')[1]))
tasks.append(write_task(SID, 'gender_bias', 'gender_pronoun', 'social/bias (stereotyped pronoun)', pool, lambda i: i['clean_correct'], len(df),
    ['data/gender-bias/gpt2.csv (986 items). Clean = stereotyped profession ("The nurse said that"); corrupted = a gendered noun ("The man ..."); answers are the pronoun tokens decoded from clean_answer_idx / corrupted_answer_idx with GPT-2 (e.g. " she"/" he"); they differ in every kept item, so the flip is natural. "Correct" is stereotype-defined, not grammatical.']))

# ------------------------------------------------------------------ capital -> country
df = pd.read_csv(os.path.join(D, 'fact-retrieval-comma', 'gpt2.csv'))
pool = []
for k, r in df.iterrows():
    if r['country'] == r['corrupted_country']:
        continue
    pool.append(make_item(r['clean'], r['corrupted'], ' ' + r['country'], ' ' + r['corrupted_country'], SID, 'fact-retrieval-comma/gpt2/%d' % k,
                          capital=r['capital'], corrupted_capital=r['corrupted_capital'], country_first_token_gpt2=dec(r['country_idx']),
                          corrupted_country_first_token_gpt2=dec(r['corrupted_country_idx'])))
tasks.append(write_task(SID, 'capital_country', 'factual_capital', 'factual knowledge (capital -> country)', pool, lambda i: 0, len(df),
    ['data/fact-retrieval-comma/gpt2.csv (190 items): " Kabul, the capital of" -> " Afghanistan"; the corrupted prompt uses another capital. The source scores the FIRST GPT-2 token of the country (country_idx, e.g. " United" for "United Kingdom"); we use the full country name with a leading space as the answer (multi-token answers are allowed), and keep the first-token string in metadata. Answer token counts can differ between clean and corrupted answer, which biases summed log-probabilities slightly.',
     'Country -> capital direction: not in the repo (the repo only contains capital -> country, "fact-retrieval-comma"); not built.']))

# ------------------------------------------------------------------ hypernymy
df = pd.read_csv(os.path.join(D, 'hypernymy-comma', 'gpt2.csv'))
for nm, ck, ok, txt in (('hypernymy_comma', 'clean', 'corrupted', 'prompt ", gemstones and other" (the source `clean`/`corrupted` columns; note the leading comma)'),
                        ('hypernymy_type_of', 'clean2', 'corrupted2', 'prompt "gemstones are a type of" (the source `clean2`/`corrupted2` columns)')):
    pool = []
    for k, r in df.iterrows():
        a = [dec(i) for i in ast.literal_eval(r['answers_idx'])]
        b = [dec(i) for i in ast.literal_eval(r['corrupted_answers_idx'])]
        a0 = next((x for x in a if x not in b), None)
        b0 = next((x for x in b if x not in a), None)
        if a0 is None or b0 is None:
            continue
        pool.append(make_item(r[ck], r[ok], a0, b0, SID, 'hypernymy-comma/gpt2/%d' % k, clean_answers=json.dumps(a), corrupted_answers=json.dumps(b),
                              hypernym=r['hypernym'], corrupted_hypernym=r['corrupted_hypernym'], prob_diff_gpt2=float(r['prob_diff'])))
    tasks.append(write_task(SID, nm, 'hypernymy', 'lexical semantics (hypernym completion)', pool, lambda i: 0, len(df),
        ['data/hypernymy-comma/gpt2.csv (198 items), ' + txt + '. Each item has a SET of acceptable answers; the source metric sums probability over the set. Tier 1 needs one answer: we take the first listed answer (token decoded from answers_idx, leading space) that is not in the other item\'s answer set, for the clean and for the corrupted item; the order of the lists is as in the source and its principle is not documented. Full sets kept in metadata.',
         'Source reports a low GPT-2 baseline for this task (prob_diff mean about 0.2).']))

# ------------------------------------------------------------------ SVA
df = pd.read_csv(os.path.join(D, 'sva', 'gpt2.csv'))
pool = []
# Structures whose next word is not naturally an auxiliary (the `within` groups continue
# with the embedded verb, vp_coord with a verb after "and"). Dropped because the remaining
# pool still exceeds the 2,000-item cap (decided by Andrea 2026-10-12).
NOT_AUX = ('within', 'vp_coord')
for k, r in df.iterrows():
    if any(s in r['group'] for s in NOT_AUX):
        continue
    pl = int(r['plural'])
    a, b = (' are', ' is') if pl else (' is', ' are')
    pool.append(make_item(r['clean'], r['corrupted'], a, b, SID, 'sva/gpt2/%d' % k, group=r['group'], clean_plural=pl))
assert len(pool) > 2000, len(pool)
tasks.append(write_task(SID, 'sva', 'sva', 'syntax (subject-verb agreement)', pool, lambda i: (i['group'], i['clean_plural']), len(df),
    ['data/sva/gpt2.csv (12,400 items, 15 Marvin & Linzen-style structures `group`, half plural-clean, half singular-clean; clean/corrupted differ by the number of one noun, filtered to equal GPT-2 length). The source metric sums logit/prob over a whole list of singular vs plural verbs (data/sva/combined_verb_list.csv); the CSV has no single answer word.',
     'OUR ADAPTATION: answer pair " is"/" are" for all structures (plural clean -> " are" correct). Not a perfect fit for the `within` groups (next word is the embedded verb) and vp_coord groups (next word is a verb after "and"); the `group` field allows restricting to structures where an auxiliary is natural (simple_agrmt, prep_*, subj_rel, obj_rel_across_*, obj_rel_no_comp_across_*, sent_comp).',
     'RESTRICTED (2026-10-12): the `within` and vp_coord groups are excluded, since the remaining pool exceeds the 2,000-item cap.',
     'Pooled across groups, stratified by group x clean number. Rows come from the gpt2-filtered CSV; pythia.csv (different filtered subset) not converted. The raw template data (sva_raw_data) also contain BLiMP files, not converted (BLiMP is covered by group 2).',
     'Matching of the repo-level `gpt2 copy.csv` ignored (stray duplicate).']))

# ------------------------------------------------------------------ greater-than (not fitting)
gt_reason = ('The corrupted prompt (start year ending in "01") makes every two-digit end year valid, so no answer pair flips; and no flipping variant exists for the "to the year XX" frame: '
             'with a monotone constraint (end > start), if token a is valid after start s_c and token b is invalid, a corrupted start s_k can make b valid only if s_k < b, but then a (> s_c) cannot be invalid unless a <= s_k < b, contradicting b <= s_c < a. '
             'Fits tier 2 (prompt + one valid and one invalid year token, no corrupted prompt).')
for nm, p, n in (('greater_than', 'greater-than/gpt2.csv', 1000), ('greater_than_price', 'greater-than-price/gpt2.csv', 1000), ('greater_than_sequence', 'greater-than-sequence/gpt2.csv', 825)):
    tasks.append(dict(name=nm, family='greater_than', domain='arithmetic/comparison', n_items_raw=n, n_items_converted=0, status='not_fitting_tier1', source_file='data/' + p,
                      adaptations=[gt_reason], suits_tier2='yes'))
tasks.append(dict(name='greater_than_data_csv_EAP-IG', family='greater_than', domain='arithmetic/comparison', n_items_raw=1000, n_items_converted=0, status='not_fitting_tier1',
                  source_file='EAP-IG/greater_than_data.csv', adaptations=['Same as greater_than (clean/corrupted "01"/label), a duplicate-style copy in the EAP-IG repo; see greater_than.'], suits_tier2='yes'))

conv = [t['name'] for t in tasks if t.get('n_items_converted')]
write_config(SID, conv)
repos = man['repos']
meta = dict(source_id=SID, title='EAP-IG task set (ioi, gender-bias, fact-retrieval-comma, hypernymy-comma, sva, greater-than*) of "Have Faith in Faithfulness"',
            bibkey='hanna2024faithfaithfulnessgoingcircuit', bibkey_data=['wang2022interpretabilitywildcircuitindirect', 'hanna2023doesgpt2computegreaterthan', 'marvin-linzen-2018-targeted'],
            url='https://github.com/hannamw/eap-ig-faithfulness', download_date=man['download_date'], tier=1,
            revision={k: v['commit'] for k, v in repos.items()},
            raw_files=[dict(path=p, sha256=s) for v in repos.values() for p, s in v['files'].items() if p.endswith(('.csv', 'LICENSE', '.json', '.py')) and '/data/' in p or p.endswith(('EAP-IG/LICENSE', 'greater_than_data.csv', 'ioi_llama.csv'))],
            licence={'eap-ig-faithfulness': 'none (no LICENSE file in the repo)', 'EAP-IG': 'MIT', 'gpt2-greater-than': 'no LICENSE file'},
            licence_source='raw/eap_ig/EAP-IG/LICENSE (MIT); eap-ig-faithfulness and gpt2-greater-than contain no licence file; Tigges et al. 2024 appendix also states greater-than and gendered-pronoun data are released without a licence',
            redistribute='unclear (unlicensed repo; data is derived from public templates and word lists, but no permission is recorded)',
            tokenizer=tokenizer_meta(), gpt2_tokenizer_revision=GPT2_REV, converter='scripts/task_bank/convert_eap_ig.py', seed=SEED, status='converted',
            notes='gpt2.csv files are filtered for GPT-2 tokenisation; pythia.csv variants (different filtered subsets) not converted.')
write_registry(SID, meta, tasks)
for t in tasks:
    print(t['name'], t['n_items_raw'], t['n_items_converted'], t.get('len_match_share_qwen35'), t.get('single_token_answers_share'))
