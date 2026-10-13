# Task bank, group 1 (circuits): report

Date: 2026-10-12. Scripts: `scripts/task_bank/{fetch,convert}_<source_id>.py` (+ `common.py`, `fetchlib.py`). Test: `PYTHONPATH=. python tests/verify_task_bank_circuits.py` (27 converted tasks, 0 failures). Everything is tier 1 in the LLM_Modularity format with `force_base_evaluation: true`, no prefix/suffix/concatenation. Every item satisfies `corrupted_correct == clean_incorrect` and `corrupted_incorrect == clean_correct` (asserted at conversion and re-tested). Cap 2,000 items per task, proportional stratified sample with seed 42 (strata named per task in the registry); the sampled ids are in `tasks/<source_id>/<task>.ids.txt`. Diagnostics: `len_match` = share of items whose clean and corrupted prompt (no chat template) have the same token count under the Qwen3.5 tokenizer (tokenizer.json of Qwen/Qwen3.5-2B at revision 15852e8c; the 4B file has identical content); `single` = share of items whose clean_correct and clean_incorrect are both single tokens. Answers are tokenised separately from the prompt in `patching.py` (`encode`), so raw-text answers carry a leading space unless the prompt already ends in a space/quote (arithmetic, RAVEL JSON prompts).

Model capacity (2B/4B) was not measured (no inference); remarks on difficulty are expectations only.

## Summary table

| source_id | task | family | n converted (raw pool) | len_match | single |
|---|---|---|---|---|---|
| mib | ioi_s2_flip | ioi | 2000 (10000) | 1.00 | 1.00 |
| mib | ioi_s1_s2_flip | ioi | 2000 (10000) | 1.00 | 1.00 |
| mib | addition_random | arithmetic_addition | 2000 (34440) | 1.00 | 0.00 |
| mib | addition_single_digit | arithmetic_addition | 2000 (137760) | 0.95 | 0.00 |
| mib | addition_carry | arithmetic_addition | 2000 (68880) | 0.93 | 0.00 |
| mib | subtraction_random | arithmetic_subtraction | 2000 (17424) | 1.00 | 0.05 |
| mib | subtraction_single_digit | arithmetic_subtraction | 2000 (69696) | 1.00 | 0.06 |
| mib | subtraction_carry | arithmetic_subtraction | 2000 (34848) | 1.00 | 0.00 |
| mib | copycolors_answer_position | mcqa_copycolors | 1890 | 1.00 | 1.00 |
| mib | copycolors_answer_position_color | mcqa_copycolors | 1890 | 1.00 | 1.00 |
| mib | arc_easy_answer_position | mcqa_arc | 2000 (2251) | 1.00 | 1.00 |
| mib | arc_challenge_answer_position | mcqa_arc | 1119 | 1.00 | 1.00 |
| mib | ravel_template_cf | factual_ravel | 2000 (100347 rows) | 0.19 | 0.42 |
| mib | ravel_attribute_cf | factual_ravel | 2000 | 0.11 | 0.44 |
| mib | ravel_entity_swap (own construction) | factual_ravel | 2000 | 0.50 | 0.48 |
| eap_ig | ioi_eapig_gpt2 | ioi | 1000 | 1.00 | 1.00 |
| eap_ig | ioi_eapig_llama | ioi | 499 | 1.00 | 1.00 |
| eap_ig | gender_bias | gender_pronoun | 986 | 0.96 | 1.00 |
| eap_ig | capital_country | factual_capital | 190 | 0.72 | 0.41 |
| eap_ig | hypernymy_comma | hypernymy | 198 | 0.86 | 1.00 |
| eap_ig | hypernymy_type_of | hypernymy | 198 | 0.49 | 1.00 |
| eap_ig | sva | sva | 2000 (12400) | 1.00 | 1.00 |
| wang_ioi | ioi_wang_s2_flip | ioi | 2000 (generated) | 1.00 | 1.00 |
| docstring | docstring_rest | docstring | 2000 (generated) | 1.00 | 1.00 |
| docstring | docstring_goog | docstring | 2000 (generated) | 1.00 | 1.00 |
| gpt2_syntax_circuits | svn | sva | 106 (200 generated) | 0.43 | 1.00 |
| gpt2_syntax_circuits | ioi | ioi | 24 | 1.00 | 1.00 |

Not converted: eap_ig greater_than, greater_than_price, greater_than_sequence (not fitting tier 1, see below); tigges (no new data); gradiend_mib_ioi (exact duplicate of MIB IOI). Nothing failed.

Cross-cutting uncertainty: the SVA and docstring and Wang-IOI tasks have 100% length match because their source filtered/constructed items to single-token words under GPT-2; the Qwen diagnostic confirms this carries over (1.00), except plural-name SVN cases.

## mib (HF `mib-bench/*`)

Paper: Mueller et al., MIB, ICML 2025 (arXiv 2504.13151); IOI by Wang et al. 2023, ARC by Clark et al. 2018, RAVEL by Huang et al. ACL 2024 (all in `references/mib.bib`). Revisions (HF dataset sha): ioi e5f3468f, arithmetic_addition d56d68b7, arithmetic_subtraction 9eeb7f0f, copycolors_mcqa 682676b0, arc_easy be6999b9, arc_challenge ab946b6c, ravel 53dc7a4d (full shas in the registry); downloaded 2026-10-12; sha256 of every raw file in the registry. Licence from the card front matter: ioi MIT; copycolors, arc_easy, arc_challenge Apache-2.0; arithmetic_addition, arithmetic_subtraction and ravel cards state NO licence (redistribute: unclear; the repo `.gitignore` is not mine to edit, so please decide whether `tasks/mib/addition_*`, `subtraction_*`, `ravel_*` may be committed). The ARC card says Apache-2.0 while Clark et al. released ARC under CC BY-SA 4.0 (not verified); flagged. Only public splits exist (ioi 10,000/10,000/1,000, etc.); tasks are sampled from the train split (copycolors: all splits pooled because it is tiny), leaving dev/test untouched for later hold-out.

**IOI.** MIB ioi has 8 counterfactuals: abc (answerKey -1, no flip), random_names (new names, answers not the clean pair), s1_io_flip (first-clause order only; the answer does not change), s2_io_flip, and four combinations with random names. Two flip with the same two names and were kept as separate tasks: `ioi_s2_flip` (last mention of S replaced by IO, ABA to ABB role exchange; one token differs; this is the Wang/Tigges "hard" corruption) and `ioi_s1_s2_flip` (both first-clause order and repeated name swapped; three tokens differ; BABA vs ABBA-type structure change). Templates (15+) stratify the sample. Family comparison: the MIB templates differ from Wang's 15 BABA/ABBA templates (e.g. "As {A} and {B} left the {place}, {C} gave a {object} to", names/objects/places larger vocabularies), so ioi_s2_flip, ioi_wang_s2_flip and ioi_eapig_* are same-family but different surface forms. Whether to keep more than one for composition tests is your call; they will share networks trivially.

**Arithmetic.** Items: 6 prompt templates ("Q: How much is {x} plus {y}? A: ", "{x}+{y}=", "The sum of {x} and {y} is: " etc.) times operands 1-2 digits. The source `idx` field is not unique (repeats across templates), so ids use the line number. Counterfactual types: random (new operands), ones/tens digit of operand 1 or 2 changed, ones_carry / tens_carry (alters the carry/borrow). Each counterfactual yields a different result by design, except some `tens_carry` and subtraction cases whose result is unchanged (dropped; e.g. subtraction_carry keeps only 8,808 of 34,848). Grouped into 3 tasks per operation: `*_random`, `*_single_digit` (4 counterfactual types pooled, `counterfactual` field kept), `*_carry`. Answer = numeric string, no leading space (the prompt ends in a space or `=`); Qwen splits digits, so answers are 1-3 tokens and clean/corrupted answers have equal token count in only 50-80% of items (a log-prob sum comparison favours shorter strings; consider restricting to equal-length answers at run time). Prompt length mismatch arises when an operand changes digit count (5% in single_digit/carry for addition). Overlaps with group 3 (de Varda, Stolfo, Nikankin) in family; the MIB set is 1-2 digit only.

**MCQA copycolors.** 9 configs (2-10 answer choices) times three public splits (1,890 rows; the same questions appear at multiple choice counts). Counterfactuals noun, color, noun_color leave the correct letter unchanged and were not used; `answerPosition` (rotates the correct option, flips the letter) is `copycolors_answer_position`; `answerPosition_color` additionally changes the colour in the correct option (`copycolors_answer_position_color`). symbol / randomLetter / answerPosition_* variants use another label vocabulary than the clean prompt and were rejected. Note the evaluation compares clean_correct vs one other letter; with up to 10 options "incorrect" is just the counterfactual's letter.

**ARC.** `arc_easy_answer_position`, `arc_challenge_answer_position`: same question and options in a different order, so the letter flips. Other ARC counterfactuals (symbol, randomLetter) change the label vocabulary and are rejected. Likely hard for the 2B model on ARC-Challenge (expectation only; both-correct requires clean and corrupted correct).

**RAVEL (MIB split).** 100,347 train rows of city entities with attribute Country/Continent/Language, prompts in 22 formats (JSON few-shot, "X is in", "People in X speak", ...), plus a `wikipedia` attribute (continuation text, no attribute answer; dropped). Native counterfactuals: `prompt_template_counterfactual` (another entity in another template, same attribute) and `attribute_counterfactual` (another entity and another attribute) are kept as `ravel_template_cf` and `ravel_attribute_cf`, but only for pairs whose prompts end the same way (quote vs word; otherwise the clean answer's leading space would be wrong under the corrupted prompt). Because the templates differ, only 19% / 11% of pairs have equal token length; these two tasks are weak candidates. `ravel_entity_swap` is OUR construction (not in MIB): same template and attribute, entity replaced by another entity with a different value, matched on word count and closest character length (ties by random.Random(42)); length match 50% (entity names). Answers are single tokens in 40-65% of items (multi-word languages/countries). Few-shot JSON prompts contain the demonstration's own answer. RAVEL (full) is group 4's source; the licence of the HF card is unclear.

## eap_ig (hannamw/eap-ig-faithfulness d42fea6a; hannamw/EAP-IG e24bafd3; hannamw/gpt2-greater-than 0e6dc8c2)

Hanna, Pezzelle & Belinkov, COLM 2024 (arXiv 2403.17806, "Have Faith in Faithfulness"). Note the repo README calls it "Automatic Circuit Finding and Faithfulness"; the arXiv/COLM title is the former. Licence: EAP-IG repo MIT (`raw/eap_ig/EAP-IG/LICENSE`); eap-ig-faithfulness and gpt2-greater-than have NO licence file; Tigges et al. also state their greater-than and gendered-pronoun data have no licence. redistribute: unclear. The CSVs are per tokenizer family (`gpt2.csv`, `pythia.csv`), filtered so that answers/lengths suit that tokenizer; GPT-2 ids are decoded to strings with the GPT-2 tokenizer (rev 607a30d7), and asserted against the text where possible. pythia.csv files (different filtered subsets) not converted.

- **ioi** (`ioi_eapig_gpt2` 1000, `ioi_eapig_llama` 499): the CSV has `corrupted` (ABC; no flip) and `corrupted_hard` (S2 replaced by IO name; flips). We use corrupted_hard. This equals the Wang generator frame; `ioi_eapig_llama` comes from `EAP-IG/ioi_llama.csv` (names single-token for Llama), same templates.
- **gender_bias** (986): "The nurse said that" vs "The man said that"; answers " she"/" he" (decoded ids), stereotype-defined correct answer. Flip natural. Differs in profession noun/gendered noun tokens; 96% length match.
- **capital_country** (190, `fact-retrieval-comma`): " Kabul, the capital of" -> " Afghanistan". The source scores only the first GPT-2 token of the country; we use the full name (multi-token allowed), first-token strings in metadata. Only 41% have single-token answer pairs; answer token counts differ in half of the items. Country-capital direction does not exist in the repo and was not built (only the capital-country direction is released).
- **hypernymy** (198 each): items have answer SETS (e.g. {gem, gems}); we take the first listed answer not shared with the other item's set (order of the lists undocumented). `hypernymy_comma` uses the source's `clean`/`corrupted` (", gemstones and other", starts with a comma), `hypernymy_type_of` uses `clean2`/`corrupted2` ("gemstones are a type of"; only 49% length match, 86% for the comma version). Same items, two frames; the paper reports low GPT-2 accuracy (prob_diff about 0.2).
- **sva** (12,400 raw, 2000 sampled, stratified by group x number): 15 structure groups ("ML-*", from Marvin & Linzen 2018; BLiMP files exist in `sva_raw_data` but are not in the CSV). The source metric sums over a large list of singular vs plural verbs; there is no single answer. OUR ADAPTATION: " is"/" are" for every structure (plural clean -> " are"). It fits simple, prep, subj_rel, obj_rel_across and obj_rel_no_comp_across groups well; for the `within` groups and vp_coord the next word is not naturally an auxiliary, so restrict by `group` if desired. Group sizes in the raw file range from 20 (simple_agrmt) to 2400 (prep_anim). Whether to split by structure is left to you (field `group`). Overlaps group 2 (Marks, Newman SVA).
- **greater-than** (greater-than, greater-than-price, greater-than-sequence, and `EAP-IG/greater_than_data.csv`): NOT FITTING tier 1. The "01" corruption makes all end years valid, and no flipping variant exists for the monotone "to the year XX" frame (valid-set of the corrupted prompt is a superset of the clean one; proof sketch in the registry). It fits tier 2 (prompt + a valid and an invalid end year); not produced in this pass.

## wang_ioi (redwoodresearch/Easy-Transformer ea15315d; ACDC copy bc99ace8)

Wang et al., ICLR 2023 (arXiv 2211.00593). Both repos MIT-style (`raw/wang_ioi/*/LICENSE`). The generator is a module with heavy imports; it was parsed with `ast` (constants) and re-implemented, not run. Frozen output `tasks/wang_ioi/ioi_wang_s2_flip.json` (sha256 in registry): seed 42 (`random.Random`), N=2000, all 15 BABA + 15 ABBA templates, 99 names, 8 places, 8 objects, symmetric=False. Corruption = the generator's own ("S2","IO") flip. The generator's `text` includes the final IO name; it is stripped from the prompt (and checked by the test). The paper's ABC corruption does not flip and is not used. Deviation: the original `mixed` mode would use only the first 7 templates of each type; we use all.

## tigges (arXiv 2407.10827, NeurIPS 2024)

No new data: IOI from the Wang generator (70 examples), greater-than and gendered pronouns from Hanna et al., SVA from Newman et al. 2021 (appendix on licences read from the PDF; sha in registry). Nothing converted. The paper's IOI corrupted example ("Mary gave a drink to" for "John gave ...") is exactly the flipping variant used here.

## docstring (acdc/docstring/prompts.py, ACDC commit bc99ace8)

Heimersheim & Janiak 2023 (Alignment Forum post, no arXiv; bib composed by hand from the page metadata, flagged in the file; the author handle "StefanHex" is read as Stefan Heimersheim) via the ACDC code (Conmy et al., NeurIPS 2023 Spotlight). Generator parsed and re-implemented (clean prompt reproduces `docstring_induction_prompt_generator` call order), ACDC parameters (3 matching args, 2 prefix, 1 suffix, 0 doc prefix, 3 description words, 2 per arg), per-item seeds from `random.Random(42)`. Two styles: `docstring_rest` (":param x: ...") and `docstring_goog` ("Args:" block). The source corruptions (random_doc etc.) have no defined answer; OUR flipping corruption documents the two def args that precede the matching block, so the correct next arg becomes the first matching arg (= clean_incorrect). Same description words, equal length. Designed for a 4-layer attention-only model; the induction pattern should be easy for Qwen, but both-correct rates are unknown.

## gpt2_syntax_circuits (Nomber1-1/gpt2-syntax-circuits 763553c5, MIT)

Student project, no paper (no bib). SVN generator (seed 42, as committed): 200 cases; we drop the 94 negated cases (the source marks the plural stem correct for both numbers, so no flip) leaving 106 (README claims 162 unique cases; unexplained discrepancy). Plural names ("Alice and Bob") change token length (43% match). Pronoun/name subjects, regular/irregular verbs, is/are and was/were. IOI: 24 hand-written cases, names swapped throughout, flips. Both tiny (<300 both-correct possible); contribute nothing beyond other sources. Generators executed from the cloned repo after reading them in full (only dataclasses/itertools/random/tqdm imports).

## gradiend_mib_ioi (aieng-lab/gradiend-mib-ioi 070f111e, MIT)

Masked cloze version of MIB IOI (21,000 rows). Verified: all 21,000 masked prompts equal a mib-bench/ioi prompt, and `cf_masked` equals MIB `s2_io_flip_counterfactual` in all 21,000 cases. Duplicate of `mib/ioi_s2_flip`; not converted. Dataset card citation field is "TODO".

## Open questions for you

1. Licences: MIB arithmetic and RAVEL cards have none; eap-ig-faithfulness has none. I converted anyway; the registry says `unclear`. Do these stay out of git?
2. Keep how many IOI variants (mib x2, wang, eapig x2, gpt2sc) and arithmetic variants (3 counterfactual groups x 2 operations)? They are one family each.
3. SVA " is"/" are" adaptation and hypernymy first-answer choice are mine; confirm or choose a different rule.
4. Multi-token answer pairs (arithmetic, RAVEL, capital_country) have unequal token counts in many items; add a run-time filter for equal answer length?
5. Greater-than: do you want a tier-2 version in the second pass?
6. No LLM capacity check was run (Qwen3.5-2B/4B); both-correct rates will decide which tasks survive.
