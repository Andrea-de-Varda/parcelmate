# Task bank: gathering the localisation datasets (plan, 2026-10-12)

How the datasets catalogued in [DATASETS.md](DATASETS.md) (deep-research output) are collected, converted and documented, so every task used in the paper traces to an exact source version and a verified reference. Context: [PAPER_PLAN.md](PAPER_PLAN.md).

## 1. Layout

```
external/task_bank/
  registry/<source_id>.yaml  one entry per source (section 2); committed (per-source files so parallel agents never edit the same file)
  references/<source_id>.bib verified BibTeX for that source's papers and datasets; committed (merged into references.bib at the end)
  REPORT_<group>.md          each agent's report: what was fetched, adapted, rejected, and why
  raw/<source_id>/           pristine downloads, never edited; not committed (sha256 in registry)
  tasks/<source_id>/<task>.json   converted items in the LLM_Modularity format (section 3); committed
  configs/<source_id>.json   task configs (prefix, concatenation, suffix, force_base_evaluation); committed
scripts/task_bank/
  fetch_<source_id>.py       download at a pinned revision, write sha256 to the registry
  convert_<source_id>.py     raw -> tasks/, deterministic, no model needed
tests/verify_task_bank.py    counts, field completeness, spot checks against raw
```

The existing vendored LLM_Modularity data (`external/llm_modularity/`) gets a registry entry retroactively and is left where it is.

## 2. Provenance: the registry

One YAML entry per source, filled by the fetch and convert scripts, not by hand where avoidable:

- `source_id`, `title`, `bibkey` (paper) and `bibkey_data` (dataset or repo, when cited separately)
- `url`, `revision` (git commit SHA or Hugging Face revision hash, never a branch name), `download_date`, `sha256` of each raw file
- `licence` as read from the LICENSE file or dataset card at that revision, with the file path; `redistribute: yes/no` (no for password-protected or unlicensed data: such sources stay out of git entirely, including converted items)
- for generators: `generator_commit`, `seed`, parameters, and the frozen output's sha256
- `tier` (1 counterfactual, 2 answer contrast, 3 activation contrast, N neuron list), `family` (section 4), `domain`
- `n_items_raw`, `n_items_converted`, and every conversion choice that drops or changes items (e.g. "multi-token capitals removed", "ARC answer-order counterfactuals used as is")
- `verified_by`, `verified_date`, and notes

**References.** BibTeX entries are exported from the canonical page (ACL Anthology, DBLP, OpenReview, arXiv), never typed by hand, and checked against the paper page for title, authors, venue and year. Every UNVERIFIED field in DATASETS.md must be resolved or the source is marked `status: unverified` and kept out of the analysis. Where a dataset was used through a derived release (e.g. MIB's IOI split rather than the Wang et al. generator), both the original paper and the release are recorded, so the paper can cite both.

## 3. Format

Converted items use the LLM_Modularity JSON fields that [parcelmate/patching.py](../parcelmate/patching.py) already reads (`clean_problem`, `corrupted_problem`, `clean_correct`, `clean_incorrect`, `corrupted_correct`, `corrupted_incorrect`), plus `source_id`, `source_item_id` and source metadata (e.g. `has_carry`, `n_terms`). The patching code then runs on any tier-1 task unchanged. Tier-2 items leave the corrupted fields empty; tier-3 items carry a `condition` label instead. Neuron lists are stored as `(layer, neuron, label)` tables with the model name and revision.

Every tier-1 item must flip symmetrically: the corrupted prompt's correct answer is the clean prompt's incorrect answer and vice versa, because the both-correct criterion requires the corrupted prompt to prefer `clean_incorrect`. Where a source's own corruption does not flip the answer (e.g. the IOI "ABC" corruption, the greater-than "01" year), the converter builds a flipping variant if one exists (e.g. swapping which name is repeated in IOI) and records the adaptation; if none exists, the task is recorded as not fitting tier 1, with the reason.

**Item cap (decided 2026-10-12): at most 2,000 items per task.** Converters write a deterministic sample (seed 42, stratified on the source's own metadata where available) and record the sampled source ids; the rest of the pool stays reproducible from the raw files, so more items can be drawn if a model's filters leave too few. Conversion is model-independent. Model-dependent filtering happens at run time and is recorded per model: the token-length alignment filter and the both-correct criterion (at least 60% of at least 300 items). Converters also report, as a diagnostic, the share of items whose clean and corrupted prompts have equal length under the Qwen3.5 tokenizer (2B and 4B share it; loaded from `tokenizer.json` with the `tokenizers` library).

## 4. Task families and duplicates

Several tasks appear in more than one source (IOI in Wang et al., MIB and EAP-IG; subject-verb agreement in EAP-IG, Marks et al., Tigges et al., CausalGym and LLM_Modularity; arithmetic in MIB, Stolfo et al., Nikankin et al. and de Varda et al.). Near-identical tasks would share networks trivially and inflate the composition test. Each task therefore gets a `family` label, and analyses that assume independent tasks hold out whole families. Variants are not collapsed by default (decided 2026-10-12): agents compare the versions within a family (templates, structures, answer sets, corruption type) and record how they differ, so that sufficiently different versions are kept as separate tasks.

## 5. First-pass scope

Decided 2026-10-12: every source in DATASETS.md is examined, not deferred by default; suitability is judged per source and recorded. Model capacity is judged against Qwen3.5-4B as well as 2B (e.g. entity tracking). Preprints are included. Excluded up front only: Kim and Schuster boxes (password-protected, not to be redistributed) and InterpBench (synthetic models). The de Varda et al. arithmetic dataset is added ([repo](https://github.com/Andrea-de-Varda/arithmetic-dataset), [arXiv 2609.04463](https://arxiv.org/abs/2609.04463); generator, 10,000 problems, numeric/English/Spanish/Italian/word-problem formats with matched ids, counterfactuals by flipping every operator, MIT). The first pass covers tier 1; tiers 2 and 3 and neuron lists follow in a second pass, though agents note tier-2/3 potential when they see it.

## 6. Work split for parallel agents

Agents run on Sonnet (decided 2026-10-12). Each fetches its sources at pinned revisions, writes registry entries and BibTeX, writes and runs the converter and its test, and writes its report. No model inference; downloads, generators and tokenizer-only checks are fine locally.

1. Circuit benchmarks: MIB (IOI, arithmetic +/-, MCQA, ARC-Easy, ARC-Challenge, RAVEL split), the EAP-IG task set (greater-than, gender bias, capital-country, SVA, hypernymy), the IOI generator (Wang et al.), Tigges et al., docstring, gpt2-syntax-circuits, GRADIEND MIB-IOI.
2. Syntax: CausalGym, Marks et al. SVA variants, Newman et al. SVA templates, CLAMS (as a counterfactual source by number swap).
3. Arithmetic: de Varda et al., Nikankin et al., Stolfo et al., format-invariant arithmetic (arXiv 2607.16693).
4. Knowledge, binding and entity tracking: RAVEL (full), CounterFact, LRE relations, function-vector tasks, linear relational concepts, Feng and Steinhardt binding, ordering-ID binding, Prakash et al. entity tracking, entity tracking across state changes (arXiv 2605.30233), PragWorld.
5. Second pass (later): tiers 2 and 3 and neuron lists.

A final pass merges the registry and references, assigns families, and records the bank in [LOG.md](LOG.md).

## 7. Curation after the first pass (decided 2026-10-12)

- **Git.** `tasks/` is not committed (241 MB; several sources have no redistribution licence). `tasks_manifest.tsv` (committed) holds the sha256 of every task file; `scripts/task_bank/manifest.py --check` verifies a copy (e.g. after rsync to the cluster). Converters are deterministic, so `tasks/` can also be rebuilt from the pinned raw sources.
- **Evaluation list.** `eval_tasks.txt`, generated by `scripts/task_bank/make_eval_list.py`, lists the 240 (source, task) pairs evaluated; the 9 exclusions and their reasons are comments in the file. Excluded: the gpt2_syntax_circuits hobby repo; the unmatched CausalGym `agr_sv_num_*` tasks (the `_matched` versions keep all 600 items); the public-generator numeric/English/Spanish/Italian arithmetic tasks, replaced by the paper's own data.
- **de Varda et al. arithmetic.** New source `devarda_paper_arithmetic`: the paper's counterfactual files (5,000 problems x 4 formats, ids aligned; private repo arithmetic-inconsistencies, commit 007b269, copied from Andrea's local clone), one-shot prompts with the paper's exemplars, negative-result problems skipped (3,537 of 5,000 remain), 2,000 items = all 1,402 non-negative problems of the paper's attribution fold plus 598 seed-42 fills. The paper's data have no word-problem formats; the two word-problem tasks stay in `devarda_arithmetic`, where the corrupted story is not a minimal edit (open).
- **EAP-IG SVA.** Restricted to structures where " is"/" are" is the natural next word (the `within` and vp_coord groups dropped), because the remaining pool still exceeds 2,000.
- **Kept as is.** PragWorld (for now); CausalGym mirror pairs (not deduplicated, since every CausalGym task is below 2,000 items); the 8 CausalGym tasks with zero Qwen length match (tracked, since other models tokenize differently).
- **Families.** One primary version per family, chosen after the evaluation pass; the others are kept as replications.
- **Evaluation pass.** `parcelmate.bin.patch_eval --bank` (accuracy only) on Qwen3.5-2B and 4B, `scripts/launch_patching.sh generate_bank|submit_bank`, results in `results/patching_bank/`.
