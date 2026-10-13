"""Marks et al. 2025 (Sparse Feature Circuits) subject-verb agreement -> tier-1 tasks. Deterministic.

Raw files are JSON lines: clean_prefix, patch_prefix, clean_answer, patch_answer, case. The two
prefixes differ in the number of one noun and the two answers are the singular/plural form of one
verb, so the flip is symmetric by construction (clean_correct = clean_answer = patch's incorrect
answer, corrupted_correct = patch_answer). Mapping: clean_problem = clean_prefix, corrupted_problem =
patch_prefix, clean_correct = clean_answer, clean_incorrect = patch_answer. The *_train and *_test
files (different splits, original use: discovery vs evaluation) are pooled; exact duplicates are
dropped; if the pool exceeds 2,000 items a seed-42 sample stratified on (split, case) is drawn.
Tasks: simple, rc (across relative clause), within_rc, nounpp (across prepositional phrase).
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _syntax_common import *  # noqa

SID = 'marks_sfc_sva'
FILES = {'simple': ('simple', 'sva_simple'), 'rc': ('rc', 'sva_across_obj_rc'), 'within_rc': ('within_rc', 'sva_within_obj_rc'),
         'nounpp': ('nounpp', 'sva_across_pp')}


def main():
    man = read_manifest(SID)
    d = os.path.join(raw_dir(SID), 'repo', 'data')
    reg_tasks, names, raw_files = [], [], []
    for task, (stem, family) in FILES.items():
        pool, seen, n_raw = [], set(), 0
        for split in ('train', 'test'):
            fn = os.path.join(d, f'{stem}_{split}.json')
            raw_files.append(fn)
            for ln, line in enumerate(open(fn, encoding='utf8'), 1):
                r = json.loads(line)
                n_raw += 1
                key = (r['clean_prefix'], r['patch_prefix'], r['clean_answer'], r['patch_answer'])
                if key in seen:
                    continue
                seen.add(key)
                pool.append(dict(
                    clean_problem=r['clean_prefix'], corrupted_problem=r['patch_prefix'],
                    clean_correct=r['clean_answer'], clean_incorrect=r['patch_answer'],
                    corrupted_correct=r['patch_answer'], corrupted_incorrect=r['clean_answer'],
                    task=task, family=family, case=r['case'], split=split, source_id=SID,
                    source_item_id=f'{stem}_{split}.json:{ln}'))
        items = stratified_sample(pool, lambda it: (it['split'], it['case']))
        names.append(task)
        write_task(SID, task, items, [it['source_item_id'] for it in items])
        reg_tasks.append(dict(
            name=task, family=family, domain='syntax', n_items_raw=n_raw, n_items_converted=len(items),
            n_items_after_dedup=len(pool), sampled_ids_file=f'tasks/{SID}/{task}.ids.txt',
            sampling_rule='exact duplicates dropped; if > 2000, seed-42 proportional sample stratified on (split, case)',
            adaptations=['train and test files pooled (split kept as metadata)', 'no change to prompts or answers'],
            **diagnostics(items)))
    write_config(SID, names)
    write_registry(SID, dict(
        source_id=SID, title='Sparse Feature Circuits subject-verb agreement data (Marks et al.)',
        bibkey='ICLR2025_3ba4d47a', bibkey_data='finlayson-etal-2021-causal (stimuli the data are modified from)',
        url='https://github.com/saprmarks/feature-circuits (data/)', revision=dict(github_commit=man['github_commit']),
        download_date=man['download_date'], raw_files=[dict(path=os.path.relpath(f, TB), sha256=sha256(f)) for f in raw_files],
        licence='MIT', licence_source='external/task_bank/raw/marks_sfc_sva/repo/LICENSE', redistribute='yes',
        tier=1, tasks=reg_tasks, status='converted',
        notes='Verb sets restricted to those whose singular and plural forms are single tokens in the Pythia tokenizer; stimuli modified from Finlayson et al. (2021) (same nouns and structures). README states this; Finlayson et al. licence not checked here.'))
    print(names)


if __name__ == '__main__':
    main()
