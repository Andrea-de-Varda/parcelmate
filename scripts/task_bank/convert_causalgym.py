"""CausalGym (Arora, Jurafsky & Potts 2024) -> tier-1 tasks. Deterministic.

Each row has base/src spans (lists of strings) and base_label/src_label (the next token expected after
each prompt). Mapping: clean = base, corrupted = src, clean_correct = base_label, clean_incorrect =
src_label; the flip is symmetric by construction (corrupted_correct = src_label = clean_incorrect).
The leading '<|endoftext|>' (Pythia BOS written into the first span) is stripped; the remaining spans
are concatenated with ''. All three HF splits (train/dev/test, 400/100/100 per task) are pooled; exact
duplicates (same task, base, src and labels) are dropped; both orientations of a pair (base<->src) are
kept because the source contains both. Pool per task (<= 600) is below the 2,000 cap, so all kept.
For the three agr_sv_num_* tasks the source labels are drawn independently from the class verb sets
(e.g. ' have' vs ' is'), so clean_incorrect is not the number-flipped form of clean_correct; extra
tasks agr_sv_num_*_matched replace the incorrect answer by the flipped form of the correct one
(is/are, has/have, was/were), the original src_label is kept as metadata.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _syntax_common import *  # noqa

SID = 'causalgym'
BOS = '<|endoftext|>'
FLIP = {' is': ' are', ' are': ' is', ' was': ' were', ' were': ' was', ' has': ' have', ' have': ' has'}
FAMILY = {
    'agr_gender': 'gender_pronoun_agreement',
    'agr_sv_num_pp': 'sva_across_pp', 'agr_sv_num_subj-relc': 'sva_across_subj_rc', 'agr_sv_num_obj-relc': 'sva_across_obj_rc',
    'agr_refl_num_pp': 'reflexive_number_across_pp', 'agr_refl_num_subj-relc': 'reflexive_number_across_subj_rc',
    'agr_refl_num_obj-relc': 'reflexive_number_across_obj_rc',
    'npi_any_subj-relc': 'npi_any', 'npi_any_obj-relc': 'npi_any', 'npi_ever_subj-relc': 'npi_ever', 'npi_ever_obj-relc': 'npi_ever',
    'garden_mvrr': 'garden_path_mvrr', 'garden_mvrr_mod': 'garden_path_mvrr', 'garden_npz_obj': 'garden_path_npz',
    'garden_npz_obj_mod': 'garden_path_npz', 'garden_npz_v-trans': 'garden_path_npz', 'garden_npz_v-trans_mod': 'garden_path_npz',
    'gss_subord': 'gss_subordination', 'gss_subord_subj-relc': 'gss_subordination', 'gss_subord_obj-relc': 'gss_subordination',
    'gss_subord_pp': 'gss_subordination', 'cleft': 'cleft', 'cleft_mod': 'cleft',
    'filler_gap_embed_3': 'filler_gap', 'filler_gap_embed_4': 'filler_gap', 'filler_gap_hierarchy': 'filler_gap',
    'filler_gap_obj': 'filler_gap', 'filler_gap_pp': 'filler_gap', 'filler_gap_subj': 'filler_gap',
}


def text(spans):
    s = ''.join(spans)
    assert s.startswith(BOS), s
    return s[len(BOS):]


def main():
    man = read_manifest(SID)
    d = os.path.join(raw_dir(SID), 'hf')
    rows = []
    for split in ('train', 'dev', 'test'):
        for i, r in enumerate(json.load(open(os.path.join(d, split + '.json')))):
            r['split'], r['idx'] = split, i
            rows.append(r)
    tasks_raw = {}
    for r in rows:
        tasks_raw.setdefault(r['task'], []).append(r)
    reg_tasks, names = [], []
    for task in sorted(tasks_raw):
        seen, items, ids = set(), [], []
        for r in tasks_raw[task]:
            key = (text(r['base']), text(r['src']), r['base_label'], r['src_label'])
            if key in seen:
                continue
            seen.add(key)
            items.append(dict(
                clean_problem=key[0], corrupted_problem=key[1],
                clean_correct=r['base_label'], clean_incorrect=r['src_label'],
                corrupted_correct=r['src_label'], corrupted_incorrect=r['base_label'],
                task=task, family=FAMILY[task], base_type=r['base_type'], src_type=r['src_type'], split=r['split'],
                source_id=SID, source_item_id=f"{r['split']}:{r['idx']}"))
        variants = [(task, items, [])]
        if task.startswith('agr_sv_num'):
            m = []
            for it in items:
                it2 = dict(it)
                it2['clean_incorrect'] = it2['corrupted_correct'] = FLIP[it['clean_correct']]
                it2['corrupted_incorrect'] = it['clean_correct']
                it2['original_src_label'] = it['clean_incorrect']
                m.append(it2)
            variants.append((task + '_matched', m, ['clean_incorrect replaced by the number-flipped form of clean_correct (is/are, has/have, was/were); original src_label kept as original_src_label']))
        for name, its, adapt in variants:
            ids = [it['source_item_id'] for it in its]
            write_task(SID, name, its, ids)
            n_raw = len(tasks_raw[task])
            reg_tasks.append(dict(name=name, family=FAMILY[task], domain='syntax', n_items_raw=n_raw, n_items_converted=len(its),
                                  sampled_ids_file=f'tasks/{SID}/{name}.ids.txt',
                                  sampling_rule='all items kept (pool <= 2000 after dropping exact duplicates)',
                                  adaptations=['leading <|endoftext|> span stripped', 'train/dev/test pooled', 'exact duplicates dropped'] + adapt,
                                  **diagnostics(its)))
            names.append(name)
    write_config(SID, names)
    files = [os.path.join(d, f'{s}.json') for s in ('train', 'dev', 'test')]
    write_registry(SID, dict(
        source_id=SID, title='CausalGym: Benchmarking causal interpretability methods on linguistic tasks',
        bibkey='arora-etal-2024-causalgym', bibkey_data='gauthier-etal-2020-syntaxgym; hu-etal-2020-systematic',
        url='https://huggingface.co/datasets/aryaman/causalgym (templates: https://github.com/aryamanarora/causalgym)',
        revision=dict(hf_dataset=man['hf_revision'], github_commit=man['github_commit']), download_date=man['download_date'],
        raw_files=[dict(path=os.path.relpath(f, TB), sha256=sha256(f)) for f in files],
        licence='MIT', licence_source='Hugging Face dataset card (hf/README.md front matter, license: mit) at the pinned revision; the GitHub repo at the pinned commit has no LICENSE file',
        redistribute='yes (MIT per dataset card; the GitHub repo has no licence file, derived SyntaxGym stimuli are covered by the card)',
        tier=1, tasks=reg_tasks, status='converted',
        notes='Labels are the next token after the prompt. Spans are aligned by index and can differ in token count; no alignment is needed here because prompts are used whole. Several task contrasts are class contrasts, not minimal lexical pairs (see REPORT_syntax.md).'))
    print(len(names), 'tasks')


if __name__ == '__main__':
    main()
