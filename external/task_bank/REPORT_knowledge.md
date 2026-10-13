# Task bank, group 4 (knowledge, binding, entity tracking): report

Date: 2026-10-12. Written by the group-4 collection agent; saved to this path by the main session because the agent's file-writing tool refused it (text reproduced from the agent's hand-back). Scripts: `scripts/task_bank/fetch_<source>.py`, `convert_<source>.py`, `knowledge_common.py`, `fetch_references_knowledge.py`. Test: `PYTHONPATH=. python tests/verify_task_bank_knowledge.py` (10 sources, 121 tasks, 191,587 items; passes; a full re-run gave identical output hashes). No model inference was run: every statement about difficulty for Qwen3.5-2B/4B is an expectation, not a measurement.

## Summary

- Converted (9 sources, 121 tasks): RAVEL full 14, CounterFact 33 (one per relation), LRE relations 20, function vectors 39, Feng and Steinhardt binding 5, ordering-ID binding 5, Prakash entity tracking (boxes) 1, Tang et al. entity tracking across state changes 3, PragWorld 1.
- Registered, not converted: linear relational concepts (its `data/lre` equals the LRE data, 47 of 47 files as parsed JSON; converted once under `lre_relations`).
- Not fitting tier 1: Tang et al. 1put (gold is a set of two objects in random order) and the Tang et al. pilot files (set-valued gold).
- Length match: clean and corrupted prompts have equal Qwen3.5 length for 100% of items in every task except PragWorld (70.3%). Cap 2,000 items per task, 1,000 per function-vector task.

## 1. Conventions common to all sources

- **Layout.** Raw downloads in `external/task_bank/raw/<source_id>/`: git clones at a pinned commit in `repo/`, plus `MANIFEST.json` with source URL, revision, download date and sha256 of every raw file used. Converted tasks in `tasks/<id>/<task>.json` plus `<task>.ids.txt`. Configs have `force_base_evaluation` true and null prefix, suffix and concatenation.
- **Items.** The whole problem, including demonstrations or context, is in `clean_problem` and `corrupted_problem`. Answers carry a leading space, except RAVEL JSON-style templates ending in a double quote, where the value follows without a space.
- **Pair sampler** (CounterFact, LRE, function vectors, RAVEL). A pair is two facts of one relation, and of one template where a source has several. Their answers must differ, neither may be a string prefix of the other, and the query text must have equal Qwen3.5 token length. Allocation over templates is water-filling. Seed `random.Random('42:<task>')`; orientation is a fair coin. Equal-length matching ties the conversion to the Qwen3.5 tokenizer (shared by 2B and 4B), a departure from the plan's "model-independent".
- **Ambiguity filter.** Queries with several different gold answers in the source are dropped (e.g. "In European Union, an official language is", or an LRE subject with two objects). Duplicates are kept once. Counts are in the registry.
- **Granularity.** One task per relation (CounterFact, LRE), per dataset (function vectors), or per (entity type, attribute) (RAVEL). Valid-pair capacity below 200 (CounterFact, RAVEL, function vectors) or fewer than 75 clean facts (LRE) means not converted. Kept tasks with fewer than 300 items are flagged LOW.
- **Tokenizer.** Qwen/Qwen3.5-2B `tokenizer.json`, revision 15852e8c16360a2fea060d615a32b45270f8a8fc, loaded tokenizer-only. "len match" is the share of items whose clean and corrupted problems have the same token count; "single" is the share whose two answers are both single tokens.
- **References.** ACL Anthology, arXiv and doi.org exports. For ICLR/NeurIPS/ICML papers: the arXiv export plus a hand-added `note` line with the venue (venue strings from the OpenReview search API before its bot challenge, and from the ICML 2026 poster page for Tang et al.). Titles and authors were compared with the arXiv meta tags. The arXiv export of Meng et al. carries year 2023 although the paper is NeurIPS 2022.

## 2. RAVEL full (`ravel`)

**What it is.** Huang, Wu, Potts, Geva, Geiger, ACL 2024 (2024.acl-long.470; arXiv 2402.17700). Five entity types: cities 3,552, Nobel laureates 928, verbs 986, objects 563, occupations 799.

**Version and licence.** `explanare/ravel` commit e421fc9e132b3613b1d0020f856d539217fb30ca, file `data.tgz`. MIT (`raw/ravel/repo/LICENSE`); the HF card `hij/ravel` also says MIT. Redistribute: yes.

**Converted: 14 tasks, 2,000 items each, len match 1.000** (single-token answer share in brackets): ravel_city_country (0.405), ravel_city_continent (0.368), ravel_city_language (0.519), ravel_city_timezone (0.000), ravel_nobel_field (0.584), ravel_nobel_country_of_birth (0.245), ravel_nobel_gender (1.000), ravel_verb_past_tense (0.839), ravel_verb_third_person (0.582), ravel_object_category (0.239), ravel_object_color (1.000), ravel_object_texture (1.000), ravel_occupation_industry (0.343), ravel_occupation_work_location (0.573).

**Adaptation.** An item is two entities of one type with different gold attribute values, rendered in the same prompt template. The benchmark's own pipeline keeps only prompts a model completes correctly and labels with model output; that is model-dependent and was not reproduced. Templates ending in a space, "(", "/" or "UTC" are dropped. Nobel Gender keeps only the 3 templates that mention "gender" (the ~30 "<Name>: for his contributions in chemistry. %s: for" templates elicit a pronoun, not the gold "male"/"female"). City Language values that are comma lists (467 cities) are dropped. "non-living things" is mapped to "non-living thing".

**Not converted.** Latitude, Longitude, Award Year, Birth Year (numeric); Size (gold cm/m/mm/km does not match the spelled-out units in the templates); Definition, Pronunciation, Duty (free text or IPA); Gender Bias (no gold values). Not all ~1,000 templates were audited by hand.

**MIB RAVEL** (`mib-bench/ravel`, revision 53dc7a4dbe84276567b895d8fb608b3f169b9276) is a derived release of the same data, restricted to cities and Continent, Country and Language, with 100,347 train, 15,950 val and 1,000 test rows and per-row counterfactuals of three kinds built by RAVEL's pipeline; its card names no licence. Not duplicated here; content overlap is large but was not diffed at row level. This conversion adds Timezone and four further entity types.

## 3. CounterFact (`counterfact`)

**What it is.** Meng, Bau, Andonian, Belinkov, NeurIPS 2022 (arXiv 2202.05262). 21,919 records, 34 Wikidata relations, 3 to 16 templates each.

**Version and licence.** Official file https://rome.baulab.info/data/dsets/counterfact.json, sha256 d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f (the file is unversioned); `kmeng01/rome` pinned at 0874014cd9837e4365f3e6f3c71400ef11509e04. Code MIT, no licence stated for the data: redistribute no. The HF mirrors (`azhx/counterfact` c01c413f…, `NeelNanda/counterfact-tracing` c945b082…) were looked at and not used.

**Converted: 33 tasks (`cf_<Pid>_<label>`), len match 1.000.** Labels read from the Wikidata API. Capped at 2,000: P17, P19, P20, P27, P30, P37, P101, P103, P106, P127, P131, P136, P159, P176, P178, P190, P276, P364, P413 (single 0.639), P449, P495, P740, P937, P1303, P1412. Fewer than 2,000: P39 (1,629), P108 (1,904), P140 (1,984), P407 (1,134), P641 (1,776), P138 (520). LOW: P36 capital (233), P463 member of (217). Not converted: P264 record label (15 valid pairs). Single-token answer shares 0.64 to 1.00.

**Adaptation.** Subject swap within one relation and one template, using the true object (`target_true`); `target_new` is unused because a false object has no ground truth. Zero-shot prompt, answer " " + object. Caveat: multi-valued relations (languages spoken, official language, member of) may have correct answers other than the recorded one.

## 4. LRE relations (`lre_relations`) and linear relational concepts (`linear_relational_concepts`)

**What it is.** Hernandez et al., ICLR 2024 (arXiv 2308.09124). Data repo `evandez/relations`, commit 1b9ec3cf2b8368e42bde7e80fcef384312a6ec07: 47 relations (factual 26, commonsense 8, linguistic 6, bias 7), 11,089 facts. MIT; redistribute yes.

**Linear relational concepts.** Chanin et al., NAACL 2024 (2024.naacl-long.85), `chanind/linear-relational-concepts` commit ec5db9ad942b84b998536a97a621dab6c75733e7, MIT. All 47 `data/lre/*` files equal the LRE files as parsed JSON (`person_native_language` sits under bias/ there and factual/ in the LRE repo). Registered as `not_converted_duplicate_of:lre_relations`. DATASETS.md claims "21 relations / 9,462 samples"; the repo has 26 factual relations and those counts could not be matched.

**Converted: 20 tasks, 1,342 to 2,000 items, len match 1.000** (single in brackets): company_ceo (0.000), company_hq (0.995), landmark_in_country (1.000), landmark_on_continent (1.000), person_father (0.000), person_mother (0.000), person_native_language (1.000), person_occupation (0.908), person_plays_instrument (1.000), person_plays_position_in_sport (0.652), person_plays_pro_sport (1.000), person_university (0.000), product_by_company (1.000), star_constellation (0.186), superhero_archnemesis (0.001), superhero_person (0.000), adj_superlative (0.202), verb_past_tense (0.977), word_first_letter (1.000), word_last_letter (1.000).

**Adaptation.** Subject swap with different objects, same template, equal query-line length. Both prompts share the same 5 demonstrations, drawn from other facts of the relation, one per line. Only `prompt_templates` are used. Double-escaped unicode in `country_capital_city` is decoded. Small relations reuse subjects heavily; `n_distinct_subjects_in_items` is in the registry (91 for person_university, 82 for superhero_person).

**Not converted: 27 relations** with fewer than 75 clean facts: all 7 bias relations, all commonsense relations, the small factual ones (country_capital_city 24, country_currency, country_language, country_largest_city, city_in_country, food_from_country, band_lead_singer, pokemon_evolutions, presidents' years), adj_antonym (fewer than 75 after the ambiguity filter) and adj_comparative. A threshold of 50 would add 4 commonsense relations; a first run at that threshold gave 372 to 838 items from only 50 to 60 subjects.

## 5. Function vectors (`function_vectors`)

**What it is.** Todd et al., ICLR 2024 (arXiv 2310.15213). `ericwtodd/function_vectors` commit fb9eac7b6dc707ea1475a717379916007fe448d5. MIT; redistribute yes (the word lists come from third-party sources, not audited).

**Converted: 39 tasks (`fv_<dataset>`), 1,000 items each, len match 1.000.** Abstractive (27; single in brackets): antonym (0.801), synonym (0.746), capitalize (0.470), capitalize first/last/second letter, lowercase first/last letter, next_capital_letter (all 1.000), word_length (0.000), next_item (0.856), prev_item (0.859), country_capital (0.131), country_currency (0.000), landmark_country (1.000), park_country (0.581), national_parks_state (0.555), person_instrument (1.000), person_occupation (0.905), person_sport (1.000), product_company (1.000), english_french (0.273), english_german (0.122), english_spanish (0.331), present_past (0.973), singular_plural (0.437), sentiment (1.000). Extractive (12, the `_5` variants): adjective_v_verb, alphabetically_first, alphabetically_last, animal_v_object, choose_first_of, choose_last_of, choose_middle_of, color_v_animal, concept_v_object, fruit_v_animal, object_v_concept, verb_v_adjective (single 0.17 to 0.94).

**Adaptation.** Paper format `Q: x\nA: y\n\n`, 5 demonstrations instead of 10, shared by both prompts. The clean prompt queries x1 and the corrupted prompt x2, with y2 != y1 and equal token length. The paper's own corruption (shuffled demonstration labels) does not flip the answer and was not used. Cap 1,000 because this source contributes 39 tasks. Small tasks reuse queries (singular_plural 205 raw queries, country_capital 197).

**Not converted.** ag_news, commonsense_qa, squad_val, conll2003_* (long inputs, or MCQA handled by the MIB group); all are tier-1 feasible by the same query swap. The `_3` extractive variants are near duplicates; `INCLUDE_3 = True` in the script converts them.

## 6. Binding ID, Feng and Steinhardt (`binding_iclr`)

**What it is.** Feng and Steinhardt, ICLR 2024 (arXiv 2310.17191). `jiahai-feng/binding-iclr` commit fb7608dd4c255ac37d7e95c5c8bf45b5512901eb. The repo ships generators and word lists, not frozen datasets. No top-level LICENSE file (only the vendored `transformer_lens/LICENSE`): redistribute no.

**Converted: 5 tasks (seed 42), 2,000 items each, len match 1.000, single 1.000.** binding_capitals (`<Name> lives in the capital city of <Country>.`), binding_parallel_capitals (one sentence "A, B and C live in the capital cities of X, Y and Z respectively."), binding_fruits, binding_shapes, binding_onehop (capitals context plus `The person living in <Capital> likes <color>.`). sha256 prefixes of the frozen JSON: capitals 3db12f35, fruits 4e267323, onehop d94f827e, parallel f0d1162d, shapes 2083ad6c.

**Adaptation.** The repo's templates were re-implemented (the original code needs LLaMA weights and torch and was not run). n in {2,3,4} entities. The corrupted prompt swaps the attributes of the queried entity and one other entity; the query is unchanged. The prompts are token permutations, so equal length is guaranteed. Word lists restricted to single Qwen3.5 tokens with a leading space: names 98 of 100, country-capital pairs 50 of 248, colours 26 of 27, fruits 32, shapes 17, so attributes repeat heavily.

**Not converted.** BIOSBIAS (56 MB of personal biographies, licence unclear, free text) and the MCQ and ICL-SST formats.

## 7. Ordering-ID binding (`oi_subspace`)

**What it is.** Dai, Heinzerling, Inui, EMNLP 2024 (2024.emnlp-main.967). DATASETS.md gave arXiv 2409.05448 with an unverified author list; it resolves to this paper. The GitHub repo (`cl-tohoku/OI-Subspace`, commit e70f4717d40d399974167f386235051014b0c783) holds only README and LICENSE; the data is the ACL Anthology software attachment `…/attachments/2024.emnlp-main.967.software.zip` (unversioned; sha256 in the registry). MIT for the repo; the attachment states no licence; redistribute yes as an assumption, flagged. `bi_data/dataset.jsonl` is byte-identical to Prakash's boxes data (sha256 c2631807e1900a3f66c323bc6df2b3ae88da1a31f4ce9ee6c2424dbe6d3702b9) and is converted once, under `entity_tracking_prakash`.

**Converted: 5 tasks (oi_sell, oi_apply, oi_move, oi_bring, oi_push), 2,000 items each, len match 1.000, single 1.000.** The five verb files with random common-word objects. Prompt is the context plus " Person <Name> <verb phrase> the"; the answer is " <object>". The person names of the queried clause and one other clause are exchanged. Rows whose context repeats a name or an object are dropped. The five tasks are structurally the same task, so they are near duplicates for the composition test.

## 8. Entity tracking, boxes, Prakash et al. (`entity_tracking_prakash`)

**What it is.** Prakash et al., ICLR 2024 (arXiv 2402.14811). `Nix07/finetuning` commit 6d6c0980bf073d305b915669c33533576a70aa77, `data/dataset.jsonl`. No licence: redistribute no. Kim and Schuster's password-protected data is not used.

**Converted: 1 task, boxes_binding, 2,000 items stratified on the position of the queried box, len match 1.000, single 1.000.** The file has 15,400 rows (2,200 contexts x 7 queries), all with numops = 0, so there are no state changes. Prompt is the context plus " Box X contains the". The box labels of the queried box and one other box are exchanged in the context. The authors' desiderata counterfactuals in `data_utils.py` do not flip the answer and were not used.

**Model capacity.** The paper uses 7B models. With no state changes this is a plain lookup, so Qwen3.5-2B/4B are expected to find it much easier than the original task. Not rejected, not tested.

## 9. Entity tracking across state changes (`entity_tracking_mi`)

**What it is.** Tang, Zhao, Franco, Wijaya, Mueller, Schuster, Kim, ICML 2026 (arXiv 2605.30233); authors and venue verified (DATASETS.md had them unverified). `PootieT/entity-tracking-mi` commit 21b1304872708fb85b21cf86730bd7df3d73da73. MIT; redistribute yes, but the generator is modified from Kim and Schuster's code, whose licence was not checked.

**How the data was produced.** The repo ships a generator plus 3 pilot files, not the paper's data (which is on Google Drive and was not fetched). The generator was read in full (local file I/O only) and run with the paper's own `.sh` parameter sets (1remove, 1move, 1put_1fixObj) and `--seed 42` instead of their 2255, from a patched copy (two `random.sample(set)` calls sorted, as Python 3.11 requires), with `PYTHONHASHSEED=0` and `python -s -P` (`-I` would ignore the hash seed). Output is identical on rerun.

**Converted: 3 tasks, len match 1.000, single 1.000.** et_state_noop 2,000 items, et_state_1remove 1,043, et_state_1move 792.

**Adaptation.** Rows with a single-object gold content only; the prompt is cut at "Box <q> contains the". The corrupted prompt keeps the context and queries another box with a different single object (the generator's own `rand_query_id` counterfactual, restricted to unambiguous boxes). For the operation tasks, the clean box is a box changed by the operation that still holds exactly one object.

**Not fitting tier 1.** 1put (set-valued gold, random order) and the 3 pilot files (noop_remove, reintroduce_removed_object, shared_label_object_remove).

**Model capacity.** The paper tests 2B to 70B models. With one operation and 7 boxes, 2B may be near the both-correct threshold and 4B above it. A guess; not rejected.

## 10. PragWorld (`pragworld`)

**What it is.** Vashistha, Bibhuti, Naik, Tutek, Aditya, AAAI 2026 (DOI 10.1609/aaai.v40i39.40618; arXiv 2511.13021). DATASETS.md gave no authors or repo; both now resolved. Repo `SachinVashisth/PRAGWORLD` commit c20b079a2882b8a933247c85e78df3e47ab95cec, MIT. Seed dialogues come from GRICE and CICERO, whose licences are unchecked. Redistribute yes, flagged.

**Converted: 1 task, pragworld_flip, 219 items (135 manual, 84 synthetic), below the 300 target; len match 0.703; single 1.000 (" Yes" / " No").** An item pairs an original and a perturbed conversation whose yes/no gold answers differ. The answers are used, not the `Perturbation Effect` label (the two disagree on 4 manual rows). Rows with notsure, cannot answer or typo answers are dropped, as are answer-invariant perturbations. Orientation is a seeded coin. 102 of 412 manual perturbations rewrite the question rather than the dialogue. Labels are unverified. A yes/no answer passes both-correct by chance with probability 0.5 per prompt, so it needs a stronger filter than other tasks. Roughly 30% of items will be dropped by the token-alignment filter.

## 11. Family comparison

- **Capital.** CounterFact P36 (zero-shot, 233 items, subjects are historical states and regions); `fv_country_capital` (5-shot, 197 countries, multi-token capitals for 87% of items); LRE `country_capital_city` (24 facts, not converted); `binding_capitals` (a context lookup, not factual recall); the EAP-IG capital-country task (group 1). Family `factual_capital` covers CounterFact P36 and `fv_country_capital`.
- **Country.** CounterFact P17 (universities and towns), LRE `landmark_in_country`, RAVEL `city_country` and `nobel_country_of_birth`, `fv_landmark_country` and `fv_park_country` share `factual_country`.
- **Continent.** CounterFact P30, LRE `landmark_on_continent`, RAVEL `city_continent` share `factual_continent`.
- **Language.** CounterFact P37, P103, P364, P407, P1412, LRE `person_native_language` and RAVEL `city_language` share `factual_language` (coarse).
- **Occupation, sport, instrument, product-company.** Present in CounterFact (P106, P641, P1303, P176), LRE and function vectors. The function-vector and LRE lists share the same underlying pairs (the first person-instrument, person-sport and product-company entries are identical). Families `factual_occupation`, `factual_sport`, `factual_instrument`, `factual_product_company`.
- **Position in sport.** CounterFact P413 and LRE `person_plays_position_in_sport` share `factual_sport_position`.
- **Past tense.** LRE (76 facts, 5-shot), `fv_present_past` (293) and RAVEL (986 verbs, template style) share `morphology_past_tense`.
- **Lookup from context.** `boxes_binding`, `et_state_noop`, `oi_*` and `binding_*` all retrieve the attribute bound to a queried entity; `entity_tracking_boxes_state` (`et_state_1remove`, `et_state_1move`) adds an operation. Held-out family analyses may need a coarser grouping than these labels.

## 12. Uncertainties

1. **Licences.** Keep out of git: `tasks/counterfact/` (29 MB), `tasks/binding_iclr/` (11 MB), `tasks/entity_tracking_prakash/` (1.5 MB). Flagged assumptions: `oi_subspace`, `pragworld`, `entity_tracking_mi`, function-vector word lists.
2. **Size.** About 170 MB of JSON across the nine sources: LRE 51 MB, function vectors 43 MB, CounterFact 29 MB, RAVEL 21 MB, binding 11 MB, OI 9 MB, entity tracking 5 MB.
3. **Task count.** 121 tasks from one group would dominate the bank; no model-dependent selection was done.
4. **Prompt style.** Zero-shot (CounterFact, RAVEL) versus 5-shot (LRE, function vectors; 10 shots in the paper) is not unified.
5. **Tokenizer dependence.** Equal-length pairing uses the Qwen3.5 tokenizer and would not carry over to another tokenizer.
6. **Flips.** All flips except PragWorld are our constructions.
7. **Difficulty.** No inference was run. Likely hard: CounterFact long-tail relations, RAVEL timezone, LRE person_father, person_mother and company_ceo, entity tracking with an operation at 2B.
8. **RAVEL.** Gold values used, not model labels; templates not fully audited; the MIB split not diffed row by row.
9. **Gold answers.** CounterFact keeps one gold answer for multi-valued relations; function vectors drop only inputs with several outputs within the file.
10. **DATASETS.md corrections.** LRE has 47 relations and 11,089 facts, at `evandez/relations`; PragWorld has a repo and authors; Tang et al. authors and venue verified; OI authors are Dai, Heinzerling and Inui, and its data is only in the Anthology attachment; CounterFact's 21,919 records confirmed; the LRC counts do not match the repo data.
11. **References.** DBLP and OpenReview refused automated access, so venue `note` lines were added by hand to the arXiv exports; re-export from OpenReview.
