"""Write figures/bank_task_labels.csv: display name, figure domain and data-source citation of
every task that has a neuron attribution (task bank + LLM_Modularity).

    python figures/make_bank_task_labels.py figures/bank_concentration_2b_resid_pool4.csv [more csvs]

The CSV is the single place to edit names, domains and citations for the per-task figures.
Domains (9) are the figure's grouping, chosen 2026-10-14; the registries' own `domain` strings
were written by different agents and are not uniform. Citations name the source the items
come from (for derived releases, the release: e.g. MIB for its IOI and arithmetic sets),
with the venue year of the paper in external/task_bank/references/<source>.bib.
"""

import csv
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

DOMAINS = ['Syntax', 'Morphology & lexical semantics', 'Factual knowledge', 'Arithmetic',
           'Logic, code & multiple choice', 'Entity tracking & binding', 'In-context string tasks',
           'Physical reasoning', 'Social reasoning']
# figure order (index into DOMAINS); factual knowledge after arithmetic and logic so that the
# per-task figure's pages split at domain boundaries
ORDER = [0, 1, 3, 4, 2, 5, 6, 7, 8]

CITE = {
    'binding_iclr': 'Feng & Steinhardt (2024)', 'causalgym': 'Arora et al. (2024)',
    'clams': 'Mueller et al. (2020)', 'counterfact': 'Meng et al. (2022)',
    'devarda_arithmetic': 'de Varda et al. (2026)', 'devarda_paper_arithmetic': 'de Varda et al. (2026)',
    'docstring': 'Heimersheim & Janiak (2023)', 'eap_ig': 'Hanna et al. (2024)',
    'entity_tracking_mi': 'Tang et al. (2026)', 'entity_tracking_prakash': 'Prakash et al. (2024)',
    'function_vectors': 'Todd et al. (2024)', 'lre_relations': 'Hernandez et al. (2024)',
    'marks_sfc_sva': 'Marks et al. (2025)', 'mib': 'Mueller et al. (2025)',
    'newman_sva': 'Newman et al. (2021)', 'nikankin_arithmetic': 'Nikankin et al. (2025)',
    'oi_subspace': 'Dai et al. (2024)', 'pragworld': 'Vashistha et al. (2026)',
    'ravel': 'Huang et al. (2024)', 'stolfo_arithmetic': 'Stolfo et al. (2023)',
    'wang_ioi': 'Wang et al. (2023)',
    'Lan': 'Han et al. (2026)', 'MD': 'Han et al. (2026)', 'ToM': 'Han et al. (2026)', 'phys': 'Han et al. (2026)',
}

LANG = {'de': 'German', 'fr': 'French', 'he': 'Hebrew', 'ru': 'Russian'}
STRUCT = {'long_vp_coord': 'long VP coordination', 'vp_coord': 'VP coordination',
          'obj_rel_across_anim': 'across object RC', 'obj_rel_across_inanim': 'across object RC, inanimate',
          'obj_rel_within_anim': 'within object RC', 'obj_rel_within_inanim': 'within object RC, inanimate',
          'obj_rel_no_comp_within_inanim': 'within reduced RC, inanimate',
          'subj_rel': 'across subject RC', 'simple_agrmt': 'simple', 'prep_anim': 'across PP'}
STRUCT_CG = {'obj-relc': 'across object RC', 'subj-relc': 'across subject RC', 'pp': 'across PP'}

NAMES = {
    # binding and entity tracking
    'binding_capitals': 'Binding: person, capital', 'binding_fruits': 'Binding: person, fruit',
    'binding_parallel_capitals': 'Binding: parallel capitals', 'binding_shapes': 'Binding: shape, color',
    'binding_onehop': 'Binding: one-hop',
    'et_state_1move': 'Boxes after one move', 'et_state_1remove': 'Boxes after one removal',
    'et_state_noop': 'Boxes, no operation', 'boxes_binding': 'Boxes',
    'ioi_s2_flip': 'IOI (S2 flip)', 'ioi_s1_s2_flip': 'IOI (S1 and S2 flip)',
    'ioi_eapig_gpt2': 'IOI (GPT-2 set)', 'ioi_eapig_llama': 'IOI (Llama set)', 'ioi_wang_s2_flip': 'IOI (S2 flip)',
    # CausalGym
    'agr_gender': 'Pronoun gender agreement', 'cleft': 'Cleft', 'filler_gap_pp': 'Filler-gap, PP gap',
    'filler_gap_subj': 'Filler-gap, subject gap', 'garden_npz_v-trans': 'Garden path, NP/Z (transitive)',
    # EAP-IG
    'capital_country': 'Capital to country', 'gender_bias': 'Gendered pronoun (stereotype)',
    'hypernymy_comma': 'Hypernym (", X and other")', 'hypernymy_type_of': 'Hypernym ("is a type of")',
    'sva': 'Subject-verb agreement',
    # docstring
    'docstring_goog': 'Docstring argument (Google style)', 'docstring_rest': 'Docstring argument (reST)',
    # arithmetic
    'embedded': 'Word problems (+, -)', 'numeric': 'Addition/subtraction, digits',
    'add': 'Addition', 'sub': 'Subtraction', 'mul': 'Multiplication', 'div': 'Division',
    'operator_change': 'Operator change',
    'addition_carry': 'Addition, carry', 'addition_random': 'Addition, random operands',
    'addition_single_digit': 'Addition, one digit changed', 'subtraction_carry': 'Subtraction, borrow',
    'subtraction_random': 'Subtraction, random operands', 'subtraction_single_digit': 'Subtraction, one digit changed',
    'arc_challenge_answer_position': 'ARC-Challenge, answer position', 'arc_easy_answer_position': 'ARC-Easy, answer position',
    'copycolors_answer_position': 'Copy colors, answer position', 'copycolors_answer_position_color': 'Copy colors, position and color',
    'ravel_attribute_cf': 'City attributes (attribute CF)', 'ravel_entity_swap': 'City attributes (entity swap)',
    'ravel_template_cf': 'City attributes (template CF)',
    # Marks et al.
    'nounpp': 'Subject-verb, across PP', 'rc': 'Subject-verb, across object RC', 'simple': 'Subject-verb, simple',
    'within_rc': 'Subject-verb, within object RC',
    # ordering ID
    'oi_apply': 'Ordering ID: apply', 'oi_bring': 'Ordering ID: bring', 'oi_move': 'Ordering ID: move',
    'oi_push': 'Ordering ID: push', 'oi_sell': 'Ordering ID: sell',
    # RAVEL
    'ravel_city_continent': 'City to continent', 'ravel_city_country': 'City to country',
    'ravel_city_language': 'City to language', 'ravel_city_timezone': 'City to time zone',
    'ravel_nobel_country_of_birth': 'Nobel laureate to birth country', 'ravel_nobel_field': 'Nobel laureate to field',
    'ravel_nobel_gender': 'Nobel laureate to gender', 'ravel_object_category': 'Object to category',
    'ravel_object_color': 'Object to color', 'ravel_object_texture': 'Object to texture',
    'ravel_occupation_industry': 'Occupation to industry', 'ravel_occupation_work_location': 'Occupation to workplace',
    'ravel_verb_past_tense': 'Verb to past tense', 'ravel_verb_third_person': 'Verb to third person',
    # LLM_Modularity
    'anaphor_gender_agreement': 'Anaphor gender agreement', 'det_noun_agreement_irregular': 'Det-noun agreement, irregular',
    'det_noun_agreement_regular': 'Det-noun agreement, regular', 'det_noun_agreement_with_adjective': 'Det-noun agreement, with adjective',
    'hypernymy': 'Hypernymy', 'npi': 'NPI licensing', 'subject_verb_agreement': 'Subject-verb agreement',
    'wug': 'Wug test (novel plurals)',
    'add_sub_2op_symbolic': 'Add/subtract, 2 operands, digits', 'add_sub_2op_verbal': 'Add/subtract, 2 operands, words',
    'add_sub_3op_symbolic': 'Add/subtract, 3 operands, digits', 'mul_div_2op_symbolic': 'Multiply/divide, 2 operands, digits',
    'number_sorting': 'Number sorting', 'code_A': 'Code output: modulo branch', 'code_B': 'Code output: string split',
    'code_conditional': 'Code output: threshold branch', 'code_list': 'Code output: list index',
    'logic_syllogism_1': 'Syllogism validity', 'logic_syllogism_symbolic': 'Syllogism, symbolic',
    'desires_goals': 'Desires and goals', 'emotion_fewshot': 'Emotion inference (few-shot)',
    'norm_appropriate': 'Norm appropriateness', 'primary_emotions': 'Primary emotions',
    'secondary_emotions': 'Secondary emotions', 'social_interactions': 'Social interactions',
    'phys_newton': 'Packing order (Newton)', 'phys_prost': 'Object affordances (PROST)',
    'physics_brightness': 'Brightness', 'physics_solubility': 'Solubility', 'physics_temperature': 'Temperature',
    'pragworld_flip': 'Pragmatic world model',
}

FV = {'antonym': 'Antonym', 'synonym': 'Synonym', 'capitalize': 'Capitalize word',
      'capitalize_first_letter': 'Capitalized first letter', 'capitalize_last_letter': 'Capitalized last letter',
      'capitalize_second_letter': 'Capitalized second letter', 'lowercase_first_letter': 'Lowercase first letter',
      'lowercase_last_letter': 'Lowercase last letter', 'next_capital_letter': 'Next capital letter',
      'word_length': 'Word length', 'next_item': 'Next item in sequence', 'prev_item': 'Previous item in sequence',
      'country_capital': 'Country to capital', 'country_currency': 'Country to currency',
      'landmark_country': 'Landmark to country', 'park_country': 'Park to country',
      'national_parks_state': 'National park to state', 'person_instrument': 'Person to instrument',
      'person_occupation': 'Person to occupation', 'person_sport': 'Person to sport',
      'product_company': 'Product to company', 'english_french': 'English to French',
      'english_german': 'English to German', 'english_spanish': 'English to Spanish',
      'present_past': 'Present to past tense', 'singular_plural': 'Singular to plural', 'sentiment': 'Sentiment',
      'extractive_adjective_v_verb_5': 'Pick the adjective', 'extractive_verb_v_adjective_5': 'Pick the verb',
      'extractive_alphabetically_first_5': 'Alphabetically first word', 'extractive_alphabetically_last_5': 'Alphabetically last word',
      'extractive_animal_v_object_5': 'Pick the animal', 'extractive_object_v_concept_5': 'Pick the object',
      'extractive_concept_v_object_5': 'Pick the concept', 'extractive_color_v_animal_5': 'Pick the color',
      'extractive_fruit_v_animal_5': 'Pick the fruit', 'extractive_choose_first_of_5': 'First item of list',
      'extractive_choose_last_of_5': 'Last item of list', 'extractive_choose_middle_of_5': 'Middle item of list'}
FV_DOMAIN = {'antonym': 1, 'synonym': 1, 'present_past': 1, 'singular_plural': 1, 'english_french': 1,
             'english_german': 1, 'english_spanish': 1, 'sentiment': 1,
             'country_capital': 2, 'country_currency': 2, 'landmark_country': 2, 'park_country': 2,
             'national_parks_state': 2, 'person_instrument': 2, 'person_occupation': 2, 'person_sport': 2,
             'product_company': 2}

LRE = {'landmark_in_country': 'Landmark to country', 'landmark_on_continent': 'Landmark to continent',
       'person_father': 'Person to father', 'person_mother': 'Person to mother',
       'person_native_language': 'Person to native language', 'person_occupation': 'Person to occupation',
       'person_plays_instrument': 'Person to instrument', 'person_plays_position_in_sport': 'Person to sport position',
       'person_plays_pro_sport': 'Person to sport', 'person_university': 'Person to university',
       'product_by_company': 'Product to company', 'star_constellation': 'Star to constellation',
       'superhero_person': 'Superhero to alter ego', 'superhero_archnemesis': 'Superhero to archnemesis',
       'company_ceo': 'Company to CEO', 'company_hq': 'Company to headquarters',
       'adj_superlative': 'Adjective to superlative', 'verb_past_tense': 'Verb to past tense',
       'word_first_letter': 'Word to first letter', 'word_last_letter': 'Word to last letter'}

CF_FIX = {'located_in_the_administrative_territoria': 'located in (admin. territory)',
          'languages_spoken__written_or_signed': 'languages spoken', 'religion_or_worldview': 'religion',
          'original_language_of_film_or_tv_show': 'original language of film',
          'position_played_on_team___speciality': 'position played', 'language_of_work_or_name': 'language of work'}


def label(group, task):
    """(display name, figure domain index)."""
    t = task[:-len('__qwen35')] if task.endswith('__qwen35') else task
    if group == 'causalgym':
        m = re.match(r'agr_(sv|refl)_num_(.+?)(_matched)?$', t)
        if m:
            what = 'Subject-verb number' if m.group(1) == 'sv' else 'Reflexive number'
            return '%s, %s' % (what, STRUCT_CG[m.group(2)]), 0
        m = re.match(r'npi_(any|ever)_(.+)$', t)
        if m:
            return 'NPI "%s", %s' % (m.group(1), STRUCT_CG[m.group(2)]), 0
        return NAMES[t], 0
    if group == 'clams':
        lang, rest = t.split('_', 1)
        return '%s subject-verb, %s' % (LANG[lang], STRUCT[rest]), 0
    if group == 'newman_sva':
        return 'Subject-verb, %s' % STRUCT[t], 0
    if group == 'counterfact':
        m = re.match(r'cf_(P\d+)_(.+)$', t)
        rel = CF_FIX.get(m.group(2), m.group(2).replace('_', ' '))
        return '%s (%s)' % (rel[0].upper() + rel[1:], m.group(1)), 2
    if group == 'function_vectors':
        k = t[len('fv_'):]
        return FV[k], FV_DOMAIN.get(k, 6)
    if group == 'lre_relations':
        k = re.sub(r'^lre_(factual|linguistic|commonsense|bias)_', '', t)
        return LRE[k], (1 if t.startswith('lre_linguistic') and 'letter' not in k else 6 if 'letter' in k else 2)
    if group == 'ravel':
        return NAMES[t], (1 if 'verb_' in t else 2)
    name = NAMES[t]
    dom = {'Lan': 0, 'MD': 3, 'ToM': 8, 'phys': 7, 'binding_iclr': 5, 'oi_subspace': 5, 'entity_tracking_mi': 5,
           'entity_tracking_prakash': 5, 'wang_ioi': 5, 'marks_sfc_sva': 0, 'nikankin_arithmetic': 3,
           'stolfo_arithmetic': 3, 'devarda_arithmetic': 3, 'devarda_paper_arithmetic': 3, 'docstring': 4,
           'pragworld': 8}.get(group)
    if group == 'Lan' and t in ('hypernymy', 'wug'):
        dom = 1
    if group == 'MD' and (t.startswith('code') or t.startswith('logic') or t == 'number_sorting'):
        dom = 4
    if group == 'eap_ig':
        dom = {'capital_country': 2, 'gender_bias': 8, 'hypernymy_comma': 1, 'hypernymy_type_of': 1, 'sva': 0}.get(t, 5)
    if group == 'mib':
        dom = 5 if t.startswith('ioi') else 2 if t.startswith('ravel') else 3 if t.startswith(('add', 'sub')) else 4
    return name, dom


def main():
    tasks = pd.concat([pd.read_csv(p)[['root', 'group', 'task']] for p in sys.argv[1:]]).drop_duplicates()
    rows = []
    for _, r in tasks.iterrows():
        name, dom = label(r.group, r.task)
        rows.append(dict(root=r.root, group=r.group, task=r.task, name=name, domain=DOMAINS[dom],
                         domain_order=ORDER.index(dom), citation=CITE[r.group],
                         variant='length-matched for Qwen3.5' if r.task.endswith('__qwen35') else ''))
    out = pd.DataFrame(rows).sort_values(['domain_order', 'citation', 'name'])
    path = os.path.join(HERE, 'bank_task_labels.csv')
    out.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    print('wrote %s (%d tasks)' % (path, len(out)))
    print(out.groupby('domain', sort=False).size().to_string())


if __name__ == '__main__':
    main()
