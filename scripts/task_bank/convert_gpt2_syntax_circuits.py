"""Nomber1-1/gpt2-syntax-circuits: SVN and IOI test cases. The two small generator modules (Code/core/svn_dataset.py, Code/core/ioi_dataset.py; read in full before use,
they import only dataclasses/itertools/random/tqdm) are loaded by path and run with their own seed (42)."""
import importlib.util, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

SID = 'gpt2_syntax_circuits'
RAW = os.path.join(TB, 'raw', 'gpt2_syntax_circuits')
man = json.load(open(os.path.join(RAW, 'MANIFEST.json')))
CORE = os.path.join(RAW, 'gpt2-syntax-circuits', 'Code', 'core')
def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(CORE, name + '.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
svn = load('svn_dataset').get_svn_test_cases()
ioi = load('ioi_dataset').get_ioi_test_cases()
tasks = []

pool, n_neg = [], 0
for k, c in enumerate(svn):
    if '_neg_' in c.case_type:
        n_neg += 1
        continue  # negated cases: the correct verb form is the same (plural stem) under clean and corrupted prompt, no flip
    assert c.correct_token != c.incorrect_token
    # corrupted answer must be the clean incorrect one: true for affirmative cases (singular <-> plural subject)
    pool.append(make_item(c.clean_prompt, c.corrupted_prompt, c.correct_token, c.incorrect_token, SID, 'svn/%d' % k, case_type=c.case_type, subject=c.subject, subject_number=c.subject_number))
tasks.append(write_task(SID, 'svn', 'sva', 'syntax (subject-verb agreement)', pool, lambda i: i['case_type'], len(svn),
    ['Generator `_generate_balanced_svn_cases(samples_per_perm=5, seed=42)`: pronoun or name subjects (singular vs "X and Y" plural), optional time prefix, regular verbs (walks/walk ...), irregular (goes/go, eats/eat, has/have, is/are, was/were). The prompt ends right before the verb; answer = verb form with leading space.',
     'Corrupted prompt = the same frame with a subject of the opposite number. NEGATED cases (%d of %d, "does not"/"do not") are dropped: the source marks the plural stem as correct for both numbers, so the answer does not flip.' % (n_neg, len(svn)),
     'The README states "162 unique test cases"; the generator as committed yields %d cases at samples_per_perm=5 (discrepancy not investigated); exact duplicates are removed by our dedup step.' % len(svn),
     'Plural names ("Alice and Bob") are two extra tokens, so many clean/corrupted lengths differ (see diagnostic).']))
pool = [make_item(c.clean_prompt, c.corrupted_prompt, c.correct_token, c.incorrect_token, SID, 'ioi/%d' % k, case_type=c.case_type, io_name=c.subject_io, s_name=c.subject_s) for k, c in enumerate(ioi)]
tasks.append(write_task(SID, 'ioi', 'ioi', 'syntax/semantics (name movement)', pool, lambda i: 0, len(ioi),
    ['4 templates x 6 name pairs = 24 hand-written cases (ABA and BBA). Corrupted prompt swaps the two names in the whole sentence (IO <-> S), so the answer flips (equivalent to MIB `s1_ioi_flip_s2_ioi_flip`). Tiny and fully contained in the IOI family of larger sources; fewer than 300 items, so it cannot meet the >=300 both-correct requirement alone.']))
write_config(SID, [t['name'] for t in tasks])
r = man['repos']['gpt2-syntax-circuits']
meta = dict(source_id=SID, title='gpt2-syntax-circuits (COMP 579 McGill project): SVN and IOI test cases', bibkey=None, bibkey_data=[],
            url='https://github.com/Nomber1-1/gpt2-syntax-circuits', download_date=man['download_date'], tier=1, revision=r['commit'],
            raw_files=[dict(path=p, sha256=s) for p, s in r['files'].items() if p.endswith(('svn_dataset.py', 'ioi_dataset.py', 'LICENSE'))],
            licence='MIT (Copyright 2026 Patrick Zakaria)', licence_source='raw/gpt2_syntax_circuits/gpt2-syntax-circuits/LICENSE', redistribute='yes',
            generator=dict(commit=r['commit'], seed=42, params='samples_per_perm=5', note='svn generator seeds itself with 42; ioi cases are hand-written'),
            tokenizer=tokenizer_meta(), converter='scripts/task_bank/convert_gpt2_syntax_circuits.py', status='converted',
            notes='Student project without a paper; unreviewed. The SVN generator is "inspired by" DavidDemitriAfrica/verb-conjugation-mechinterp (not fetched). Low value beyond the other SVA/IOI sources; kept for completeness.')
write_registry(SID, meta, tasks)
for t in tasks:
    print(t['name'], t['n_items_raw'], t['n_items_converted'], t['len_match_share_qwen35'], t['single_token_answers_share'])
