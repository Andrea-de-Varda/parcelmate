"""Newman et al. 2021 (Refining TSE) subject-verb agreement templates -> tier-1 tasks. Deterministic (seed 42).

The repo ships (data/ML) the Marvin & Linzen (2018) number-agreement templates as JSON lines
(sentence_good, sentence_bad, label) and (data/blimp) four BLiMP agreement paradigms. The ML sentences
are the same sentence sets as the CLAMS English evalsets (verified: identical after lower-casing and
removing the final period). They carry no counterfactual prompts; counterfactuals are BUILT by subject
number swap (see _syntax_pairing.py / _syntax_agreement.py): partner = another item of the same file
whose answers are the swapped pair and whose prompt differs only by morphological variants (subject
noun, plus agreeing words inside the prompt). The label field is not documented and is ignored. The
BLiMP files are not converted (no number-swap partner exists: items are independent sentences); they
are tier-2 candidates (prompt + correct/incorrect word given in one_prefix_*).
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _syntax_common import *  # noqa
from _syntax_agreement import build_items, sample

SID = 'newman_sva'
FAMILY = {
    'simple_agrmt': 'sva_simple', 'vp_coord': 'sva_vp_coord', 'long_vp_coord': 'sva_long_vp_coord', 'sent_comp': 'sva_sent_comp',
    'prep_anim': 'sva_across_pp', 'prep_inanim': 'sva_across_pp', 'subj_rel': 'sva_across_subj_rc',
    'obj_rel_across_anim': 'sva_across_obj_rc', 'obj_rel_across_inanim': 'sva_across_obj_rc',
    'obj_rel_within_anim': 'sva_within_obj_rc', 'obj_rel_within_inanim': 'sva_within_obj_rc',
    'obj_rel_no_comp_across_anim': 'sva_across_obj_rc_nocomp', 'obj_rel_no_comp_across_inanim': 'sva_across_obj_rc_nocomp',
    'obj_rel_no_comp_within_anim': 'sva_within_obj_rc_nocomp', 'obj_rel_no_comp_within_inanim': 'sva_within_obj_rc_nocomp'}


def main():
    man = read_manifest(SID)
    d = os.path.join(raw_dir(SID), 'repo', 'data', 'ML')
    reg_tasks, names, raw_files = [], [], []
    for stem in sorted(FAMILY):
        path = os.path.join(d, stem + '_all.jsonl')
        raw_files.append(path)
        pairs = []
        for ln, line in enumerate(open(path, encoding='utf8'), 1):
            r = json.loads(line)
            pairs.append((ln, r['sentence_good'].strip(), r['sentence_bad'].strip()))
        task = stem
        items, st = build_items(pairs, dict(task=task, family=FAMILY[stem], structure=stem, source_id=SID), f'ML/{stem}_all.jsonl')
        kept = sample(items)
        names.append(task)
        write_task(SID, task, kept, [it['source_item_id'] + '|' + it['corrupted_source_item_id'] for it in kept])
        reg_tasks.append(dict(
            name=task, family=FAMILY[stem], domain='syntax', n_items_raw=st['n_pairs_raw'], n_items_converted=len(kept),
            pairing_stats=st, sampled_ids_file=f'tasks/{SID}/{task}.ids.txt',
            sampling_rule='all unordered counterfactual pairs; if > 2000, seed-42 sample stratified on (clean_correct, clean_incorrect)',
            adaptations=['counterfactual built by subject-number swap (partner = other item of the same file with swapped answers and morphologically close prompt words)',
                         'answer = words from first to last differing word, leading space; shared tail (and final period) dropped',
                         'clean/corrupted orientation of each pair chosen by seed-42 coin'],
            **diagnostics(kept)))
        print(task, st, len(kept), flush=True)
    write_config(SID, names)
    write_registry(SID, dict(
        source_id=SID, title='Refining Targeted Syntactic Evaluation of Language Models (code and data of Newman et al.)',
        bibkey='newman-etal-2021-refining', bibkey_data='marvin-linzen-2018-targeted (templates); warstadt-etal-2020-blimp-benchmark (BLiMP files, not converted)',
        url='https://github.com/bnewm0609/refining-tse', revision=dict(github_commit=man['github_commit']), download_date=man['download_date'],
        raw_files=[dict(path=os.path.relpath(f, TB), sha256=sha256(f)) for f in raw_files],
        licence='MIT', licence_source='external/task_bank/raw/newman_sva/repo/LICENSE (Copyright 2020 kaiang)', redistribute='yes (MIT for the repo; the templates originate from Marvin and Linzen 2018, whose licence was not checked)',
        tier=1, tasks=reg_tasks, status='converted',
        notes='ML sentence sets are identical to the CLAMS English evalsets (same sentences, case and final period aside). Not converted: data/blimp (4 files) and data/verbs (verb list for the MW/EW metrics).'))


if __name__ == '__main__':
    main()
