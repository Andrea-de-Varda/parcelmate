# Datasets and neuron lists for localising task-relevant MLP neurons: a tier-sorted catalogue

The best place to start is the LLM_Modularity release by Han, Andreas, Fedorenko and de Varda (2026). It gives you 46 counterfactual-pair tasks with flipped answers across language, formal, physical and social reasoning, and it already localises MLP neurons. Add the MIB Hugging Face datasets (IOI, arithmetic, MCQA, ARC, RAVEL), CausalGym, and the Hanna et al. EAP-IG task set, and you have a solid tier 1 core. That core covers syntax, arithmetic, factual recall, gender bias, hypernymy and multiple choice, with ready-made clean/corrupted pairs.

## TL;DR

- **Tier 1 is richer than it looks.** I found about 35 usable counterfactual-pair task sets. The strongest are LLM_Modularity (46 tasks, 49,430 pairs, the sum of its four domain counts), MIB (6 tasks on Hugging Face), CausalGym (SyntaxGym tasks, 17.4k rows), the six EAP-IG tasks (IOI, Gender-Bias, Greater-Than, Capital-Country, SVA, Hypernymy), Marks et al.'s four subject-verb agreement variants, entity tracking and binding. Most are templated, so lengths match when the swapped words are single tokens.
- **For tiers 2 and 3**, the key sets are BLiMP (67 paradigms × 1,000 pairs), MultiBLiMP (125k+ pairs, 101 languages, CC-BY-4.0) and CLAMS (multilingual agreement). For activation contrasts, use the AlKhamissi et al. language/ToM/multiple-demand localiser stimuli, the LAPE language-neuron pipeline, and Gurnee et al.'s sparse probing feature datasets.
- **Neuron lists with exact indices exist for:**
  - Gurnee et al. universal neurons (GPT-2, Pythia-160M dataframes)
  - Tang et al. language-specific neurons (LLaMA-2, BLOOM, Mistral and others, as `.neuron.pth` files; the ACL 2024 paper says these neurons sit mainly in the models' top and bottom layers)
  - Nikankin et al. arithmetic heuristic neurons (code repo; Llama3)
  - LLM_Modularity per-task attribution tensors (Hugging Face, about 6 GB)

  Voita et al. (dead/n-gram/positional) did not release a neuron list that I could find.

## Key Findings

- **LLM_Modularity is the closest match to your design.** Each of its 46 tasks has "original" and "alternative" inputs "matched in surface form", whose correct continuation flips. It scores MLP neurons with attribution patching and keeps the top 0.1%. Its per-domain pair counts are:
  - Language: 8 tasks, 8,877 pairs
  - Formal: 20 tasks, 19,941 pairs
  - Physical: 9 tasks, 9,200 pairs
  - Social: 9 tasks, 11,412 pairs

  It ran on six instruction-tuned models of 24B to 123B parameters, plus a GPT-2 control. That gives you a published MLP-neuron localisation to compare your bottom-up networks against.
- **MIB is the cleanest standard benchmark.** It has public Hugging Face splits:

  | Task | Train | Validation | Test (public/private) |
  |---|---|---|---|
  | IOI | 10,000 | 10,000 | 10,000/10,000 |
  | MCQA | 110 | 50 | 50/50 |
  | Arithmetic + | 34,400 | 4,920 | 4,920/4,920 |
  | Arithmetic − | 17,400 | 2,484 | 2,484/2,484 |
  | ARC Easy | 2,251 | 570 | 1,188/1,188 |
  | ARC Challenge | 1,119 | 299 | 586/586 |

  The private test halves are not downloadable.
- **Watch for unequal token counts in CausalGym.** Its own card says the base/source spans "may have an unequal number of tokens". Plan to align by span, not by raw position.
- **Several classic tasks are generators, not files.** IOI (Wang et al.), greater-than and docstring all fall in this group. That is good for making more items, but you will need to fix a seed and freeze your own copy.
- **Data leakage is a live concern for a few sets.** Kim & Schuster ship their entity-tracking boxes data as a password-protected zip so it doesn't end up in training data. BLiMP and CounterFact have been public since 2020 and 2022, so assume modern models have seen them.

## Main table

Legend: CF = has corrupted/counterfactual prompt with flipped answer. "Gen" = template generator available. UNVERIFIED = I could not confirm it on a primary page this session.

| # | Dataset / sub-task | Paper (authors, venue, year, link) | Domain | Data link | Licence | Tier | CF? | Answers | Length-matched / generator | Size, sub-tasks, languages | Original models | Unit type, released? | Known problem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | MIB IOI | Mueller, Geiger, Wiegreffe et al., ICML 2025, arXiv 2504.13151 | syntax/semantics | HF `mib-bench/ioi` | MIT (per derived card `aieng-lab/gradiend-mib-ioi`) | 1 | Yes | Correct and incorrect names, single token | Templated, matched; Gen via MIB code | 21k rows on HF (10k train/10k val/10k public test), EN | GPT-2 small, Qwen-2.5 0.5B, Gemma-2 2B, Llama-3.1 8B (secondary summary) | Edges/nodes (circuit track) and features (causal variable track); leaderboard, example circuits on HF | Private test split held back |
| 2 | MIB Arithmetic (+) | same | arithmetic | HF `mib-bench/arithmetic_addition` | UNVERIFIED | 1 | Yes | Number answers; tokenisation is model dependent | Templated | 40.4k rows | same | same | Multi-digit answers split into several tokens in some tokenisers |
| 3 | MIB Arithmetic (−) | same | arithmetic | HF `mib-bench/arithmetic_subtraction` | UNVERIFIED | 1 | Yes | same | Templated | 20.9k rows | same | same | same |
| 4 | MIB MCQA (copy colours) | same | reasoning (MCQ) | HF `mib-bench/copycolors_mcqa` | UNVERIFIED | 1 | Yes | Answer letter, single token | Templated | 1.89k rows (110 train) | same | same | Small train split |
| 5 | MIB ARC-Easy | same; ARC by Clark et al. | science QA | HF `mib-bench/arc_easy` | UNVERIFIED | 1 | Yes (answer-order counterfactuals) | Letter, single token | Not matched by design | 4.01k rows | same | same | Real-world QA, so weak models score low |
| 6 | MIB ARC-Challenge | same | science QA | HF `mib-bench/arc_challenge` | UNVERIFIED | 1 | Yes | Letter | Not matched | 2k rows | same | same | Hard for small models |
| 7 | MIB RAVEL (city split) | same; RAVEL by Huang et al. | factual knowledge | HF `mib-bench/ravel` | UNVERIFIED | 1 | Yes (entity swap) | Attribute values, often multi-token | Templated prompts, entities vary in length | 117k rows; country/continent/language of cities | same | Residual features | Attribute answers vary in length |
| 8 | IOI original | Wang, Variengien, Conmy, Shlegeris, Steinhardt, ICLR 2023, arXiv 2211.00593 | syntax/semantics | GitHub `redwoodresearch/Easy-Transformer` → `easy_transformer/ioi_dataset.py` (also in ACDC repo) | MIT | 1 | Yes (ABC and other corruptions) | Names, single token | Matched; Gen (15 templates) | Generated, any N; EN | GPT-2 small | Attention heads; circuit described in paper | Old fork of TransformerLens; no static file |
| 9 | Greater-than | Hanna, Liu, Variengien, NeurIPS 2023, arXiv 2305.00586 | arithmetic/comparison | GitHub `hannamw/gpt2-greater-than` (generator); CSV `greater_than_data.csv` in `hannamw/EAP-IG` | UNVERIFIED | 1 | Yes ("01" year corruption) | Two-digit year tokens; prob-diff over many tokens | Matched; Gen | 10,000 prompts in paper | GPT-2 small | Heads plus MLPs; circuit in paper | Metric is a probability sum, not a single wrong token |
| 10 | Docstring | Heimersheim & Janiak, Alignment Forum, 2023 (no arXiv) | code | Generator in ACDC repo (`acdc/docstring/`, path UNVERIFIED); original Colab | UNVERIFIED | 1 | Yes | Argument name, single token | Matched; Gen | Generated | 4-layer attention-only model | Heads | No paper or static file; Colab-dependent |
| 11 | CausalGym | Arora, Jurafsky, Potts, ACL 2024, arXiv 2402.12560 | syntax (SyntaxGym) | HF `aryaman/causalgym`; templates in GitHub `aryamanarora/causalgym` | UNVERIFIED | 1 | Yes (base/src pairs) | Single-token contrasts per task | Partly: spans aligned by index but "may have an unequal number of tokens"; Gen | 17.4k rows, train/dev/test, multiple SyntaxGym tasks, EN | Pythia 14M to 6.9B | Residual directions (DAS, probes); not a neuron list | Token misalignment across spans |
| 12 | LLM_Modularity Language | Han, Andreas, Fedorenko, de Varda, preprint 2026, arXiv 2608.13567 | syntax/semantics (anaphor, det-noun, SVA, NPI, hypernymy, wug) | GitHub `Pengrui-Han/LLM_Modularity` `data/` | MIT | 1 | Yes | Flipped continuations | Surface-form matched by design | 8 tasks, 8,877 pairs | Six instruction-tuned LLMs (24B to 123B), GPT-2 control | MLP neurons (top 0.1%); attribution tensors on HF `barryhpr/LLM_Modularity_attribution` | Project page points to a different repo name (`LLM_Modularity_Final`) |
| 13 | LLM_Modularity Formal | same | arithmetic, logic, code | same | MIT | 1 | Yes | same | same | 20 tasks, 19,941 pairs | same | same | same |
| 14 | LLM_Modularity Physical | same | physical reasoning | same | MIT | 1 | Yes | same | same | 9 tasks, 9,200 pairs | same | same | same |
| 15 | LLM_Modularity Social | same | theory of mind | same | MIT | 1 | Yes | same | same | 9 tasks, 11,412 pairs | same | same | same |
| 16 | Gender-Bias (gendered pronouns) | Hanna, Pezzelle, Belinkov, COLM 2024, arXiv 2403.17806 | social/bias | GitHub `hannamw/eap-ig-faithfulness` | UNVERIFIED | 1 | Yes ("The nurse/doctor said that") | she/he, single token | Matched | Paper uses about 1,000 items | GPT-2 small (plus others) | Edges (EAP-IG); circuits in repo | Bias task, so "correct" is stereotype-defined |
| 17 | Capital-Country / Country-Capital | same | factual knowledge | same | UNVERIFIED | 1 | Yes (France/Italy swap) | City/country names, often single token | Matched when names are single tokens | Not stated in sources seen | same | Edges | Multi-token names must be filtered |
| 18 | SVA (EAP-IG version) | same | syntax | same | UNVERIFIED | 1 | Yes (keys/key) | is/are, single token | Matched | Not stated | same | Edges | Plain EAP was unfaithful on this task |
| 19 | Hypernymy | same | lexical semantics | same | UNVERIFIED | 1 | Yes (roses/diamonds → flowers/gems) | Multi-answer sets | Partly | Not stated | same | Edges | Low baseline: prob diff 0.21 |
| 20 | Gendered-Pronoun, IOI, Greater-Than, SVA across training | Tigges, Hanna, Yu, Biderman, 2024, arXiv 2407.10827 | mixed | Reuses items 8, 9, 16, 18 (SVA from Newman et al. 2021, MIT) | MIT for IOI and SVA | 1 | Yes | Single token | Matched | 4 tasks | Pythia 70M to 2.8B (all checkpoints) | Heads/edges | No new data |
| 21 | SFC subject-verb agreement (4 variants: simple, across RC, within RC, across PP) | Marks, Rager, Michaud, Belinkov, Bau, Mueller, ICLR 2025, arXiv 2403.19647 | syntax | GitHub `saprmarks/feature-circuits` (data folder) | UNVERIFIED | 1 | Yes (noun number swap) | Verb forms filtered to single tokens | Matched; based on Newman et al. 2021 templates | 4 structures; separate discovery/eval splits | Pythia-70M, Gemma-2-2B | SAE features plus neuron baselines; circuits on feature-circuits.xyz | Built for Pythia tokeniser |
| 22 | Entity tracking (boxes) | Prakash, Shaham, Haklay, Belinkov, Bau, ICLR 2024, arXiv 2402.14811 | entity tracking | GitHub `Nix07/finetuning` | UNVERIFIED | 1 | Yes | Object names, single token by design | Matched | 500 eval tasks in paper | Llama-7B, Vicuna, Goat, FLoat | Heads; circuit in paper | Small models fail the task |
| 23 | Entity tracking boxes (original) | Kim & Schuster, ACL 2023, arXiv 2305.02363 | entity tracking | GitHub `sebschu/entity-tracking-lms` `data/boxes-dataset-v1.zip` (pw `iamnotaLM`) | UNVERIFIED | 2 (needs pairing) | Partly | Contents of a box, multi-token | Gen script included | Count UNVERIFIED | GPT-3/3.5, Flan-T5, T5 | Behavioural only | Password zip; do not re-upload |
| 24 | Entity tracking across state changes | 2026, arXiv 2605.30233 (authors UNVERIFIED) | entity tracking | GitHub `PootieT/entity-tracking-mi` | UNVERIFIED | 1 (likely) | Likely | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | Mechanistic, PUT/REMOVE ops | Very new; details unchecked |
| 25 | Binding ID tasks | Feng & Steinhardt, ICLR 2024, arXiv 2310.17191 | variable binding | GitHub `jiahai-feng/binding-iclr` (generates data) | UNVERIFIED | 1 | Yes (swap bindings) | Attribute tokens | Matched; Gen | Several binding tasks | Pythia, LLaMA | Residual binding subspace | Needs LLaMA weights locally |
| 26 | Ordering-ID binding | Dai et al. (UNVERIFIED author list), EMNLP 2024, arXiv 2409.05448 | variable binding | GitHub `cl-tohoku/OI-Subspace` | UNVERIFIED | 1 | Yes | Single-token entities | Matched | UNVERIFIED | UNVERIFIED | Residual subspace | Small release |
| 27 | RAVEL (full) | Huang, Wu, Potts, Geva, Geiger, ACL 2024, arXiv 2402.17700 | factual knowledge | GitHub `explanare/ravel` | Sources MIT/Apache/CC-0 | 1 | Yes | Attribute values, multi-token | Templated | 5 entity types, at least 500 instances and 4 attributes each | Llama2-7B | Residual features (MDAS) | Many prompts need filtering by model accuracy |
| 28 | CounterFact | Meng et al., NeurIPS 2022, arXiv 2202.05262 (UNVERIFIED) | factual knowledge | HF `azhx/counterfact`, `NeelNanda/counterfact-tracing`; official via `kmeng01/rome` | Code MIT; data UNVERIFIED | 1 (pair by subject swap) | Partly (has target_true and target_new) | Mostly single-token objects | Not matched (subject lengths vary) | 21,919 records | GPT-2 XL, GPT-J | MLP layers (causal tracing) | Mirrors are unofficial |
| 29 | Relations (LRE) | Hernandez, Sen Sharma, Haklay, Meng, Wattenberg, Andreas, Belinkov, Bau, "Linearity of Relation Decoding in Transformer Language Models", ICLR 2024, arXiv 2308.09124 | factual, commonsense, linguistic, bias | GitHub repo UNVERIFIED | UNVERIFIED | 1 (pair within relation) | Partly | Objects, often multi-token | Not matched | 47 relations, 4 categories, about 10k facts | GPT-J, Llama | Linear maps | Some relations are tiny (19 items) |
| 30 | Function vector tasks | Todd, Li, Sen Sharma, Mueller, Wallace, Bau, ICLR 2024, arXiv 2310.15213 | ICL tasks (antonyms, translation, capitals...) | GitHub `ericwtodd/function_vectors` (dataset folder) | UNVERIFIED | 2 (input/output pairs; build CF by swapping tasks) | Partly | Mostly single-word | Not matched | 40+ tasks | GPT-J, Llama 2, others | Attention heads (FV) | Answers can be multi-token |
| 31 | Arithmetic CMA | Stolfo, Belinkov, Sachan, EMNLP 2023, arXiv 2305.15054 | arithmetic | GitHub `alestolfo/lm-arithmetic` | UNVERIFIED | 1 | Yes (operand swap) | Number tokens | Templated | UNVERIFIED | GPT-J, Pythia, others (UNVERIFIED) | MLP/attention layers | Factual control needs LAMA download |
| 32 | Arithmetic heuristic prompts | Nikankin, Reusch, Mueller, Belinkov, ICLR 2025, arXiv 2410.21272 | arithmetic | GitHub `technion-cs-nlp/llm-arithmetic-heuristics` | UNVERIFIED | 1 | Yes | Single-token results | Templated | 4 operators | Llama3-8B, Llama3-70B (plus others) | A sparse set of single MLP neurons in mid to late layers (paper); 91% of 3,200 top neurons per operator classified | Limited to results that fit in a single token |
| 33 | Format-invariant arithmetic | 2026, arXiv 2607.16693 (authors UNVERIFIED) | arithmetic, symbols/text/code | GitHub `SharathHN/format-invariant-arithmetic` | UNVERIFIED | 1 (likely) | Likely | UNVERIFIED | UNVERIFIED | 3 formats | UNVERIFIED | MLP neurons | New, unchecked |
| 34 | BLiMP | Warstadt et al., TACL 2020, arXiv 1912.00582 | syntax, morphology, semantics | HF `nyu-mll/blimp`; GitHub `alexwarstadt/blimp` | UNVERIFIED | 2 (shared-prefix subset), 3 | No | Good/bad sentences; final-word subset single token | Gen (data_generation code) | 67 paradigms × 1,000 = 67k pairs, EN | n-gram, LSTM, GPT-2, TXL | None | Many pairs differ mid-sentence, so filter |
| 35 | MultiBLiMP 1.0 | Jumelet, Weissweiler, Bisazza, TACL 2025, arXiv 2504.02768 | multilingual agreement | HF `jumelet/multiblimp` (`<lang>/data.tsv`) | CC-BY-4.0 | 2 | No | `sen` / `wrong_sen` | Not by design | 125k+ pairs, 101 languages, 6 phenomena | Many LLMs | None | Repo layout recently changed |
| 36 | CLAMS | Mueller, Nicolai, Petrou-Zeniou, Talmina, Linzen, ACL 2020, arXiv 2005.00187 | multilingual SVA | GitHub `aaronmueller/clams` (en/fr/de/he/ru evalsets) | Apache-2.0 | 2 (can make CF by number swap) | Partly | Verb forms | Gen grammars included | 5 languages; count UNVERIFIED | LSTMs, mBERT | None | Old LSTM-era focus |
| 37 | Zorro | Huebner et al., CoNLL 2021 (UNVERIFIED) | child-directed syntax | Link UNVERIFIED | UNVERIFIED | 2 | No | Sentence pairs | Gen | 23 paradigms, 13 phenomena | BabyBERTa | None | Link not checked |
| 38 | LLM language localiser | AlKhamissi, Tuckute, Bosselut, Schrimpf, NAACL 2025, arXiv 2411.02280 | language vs non-words | GitHub `bkhmsi/llm-localization` `stimuli/` | UNVERIFIED | 3 | n/a | n/a | n/a | Fedorenko sentences vs non-word lists | 18 LLMs | Units (hidden dims) per layer; code produces masks | Stimuli downloaded by script |
| 39 | LLM theory-of-mind localiser | same | social | same, `--network theory-of-mind` | UNVERIFIED | 3 | n/a | n/a | n/a | False-belief vs false-photo | same | same | Paper reports mixed results |
| 40 | LLM multiple-demand localiser | same | formal/executive | same, `--network multiple-demand` | UNVERIFIED | 3 | n/a | n/a | n/a | Hard vs easy | same | same | Mixed results |
| 41 | LAPE language neurons | Tang, Luo, Huang, Zhang, Wang, Zhao, Wei, Wen, ACL 2024, arXiv 2402.16438 | multilingual | GitHub `RUCAIBox/Language-Specific-Neurons` | UNVERIFIED | 3 plus neuron list | n/a | n/a | n/a | 7 languages, about 100M Wikipedia tokens each | LLaMA-2 7/13/70B, BLOOM-7B, OPT-6.7B, Mistral-7B, Phi-2 | MLP neurons, `.neuron.pth` released | Paper says the neurons sit mainly in the top and bottom layers |
| 42 | Language Arithmetics | Gurgurov et al., 2025, arXiv 2507.22608 | multilingual | GitHub `d-gurgurov/Language-Neurons-Manipulation` | UNVERIFIED | 3 | n/a | n/a | n/a | Many languages | Llama-3.1 and others | MLP neurons (LAPE) | Neuron files release UNVERIFIED |
| 43 | Language-specific neurons and transfer | 2025, arXiv 2503.17456 | multilingual | GitHub `csalt-research/LangSpecificNeurons` | UNVERIFIED | 3 | n/a | n/a | n/a | UNVERIFIED | UNVERIFIED | MLP neurons | Finds neurons do not help transfer |
| 44 | Sparse probing feature datasets | Gurnee et al., TMLR 2023, arXiv 2305.01610 | POS, code language, language ID, data source and more | GitHub `wesg52/sparse-probing-paper` (`probing_datasets/`); ready sets in `wesg52/llm-context-neurons` | UNVERIFIED | 3 | n/a | n/a | n/a | About 10 categories, 100+ binary features (e.g. 1,438 POS seqs, 5,397 code seqs) | Pythia 70M to 6.9B | Neurons | Mostly build scripts, not files |
| 45 | Code-language neurons | 2025, arXiv 2512.19980 | code (C++, Java, Python, Go, JS) | Link UNVERIFIED | UNVERIFIED | 3 | n/a | n/a | n/a | 5 languages | Llama-3.1-8B | Neurons | LAPE found only 0.75% of neurons |
| 46 | Task-specific neurons (CGVST) | EMNLP 2024 (aclanthology 2024.emnlp-main.403) | 8 NLP tasks | GitHub link in paper, UNVERIFIED | UNVERIFIED | 3 | n/a | n/a | n/a | 8 tasks | UNVERIFIED | Neurons | Link unchecked |
| 47 | Copy-bias neurons | Ali et al., 2024, arXiv 2410.01288 | ICL copying | Synthetic data, link UNVERIFIED | UNVERIFIED | 2 | No | UNVERIFIED | Gen | UNVERIFIED | Several LLMs | Neurons (IG) | Unchecked |
| 48 | SAEBench evals | Karvonen et al., 2025, arXiv 2503.09532 | RAVEL, probing and more | Link UNVERIFIED | UNVERIFIED | 1/3 | Partly | Mixed | Mixed | Several evals | Gemma-2, Pythia | SAE latents | Built for SAEs |
| 49 | InterpBench | Gupta et al., 2024, arXiv 2407.14494 | synthetic circuits | HF `mib-bench/interpbench` (models) | UNVERIFIED | 1 (synthetic) | Yes | Mixed | Gen | Tracr tasks | Semi-synthetic transformers | Known ground-truth circuits | Not natural LLMs |
| 50 | gpt2-syntax-circuits SVA | Hobby repo, no paper | syntax | GitHub `Nomber1-1/gpt2-syntax-circuits` | UNVERIFIED | 1 | Yes | is/are | Matched | 162 unique cases | GPT-2 small | Heads | Unreviewed |
| 51 | GRADIEND MIB-IOI | aieng-lab, 2025/2026 | syntax | HF `aieng-lab/gradiend-mib-ioi` | MIT | 2 | No (masked cloze) | IO vs subject | Same as MIB | MIB splits | Encoders | Gradient features | Derivative of MIB |
| 52 | PragWorld | 2025, arXiv 2511.13021 | pragmatics/world model | Link UNVERIFIED | UNVERIFIED | 2 (likely) | Partly (minimal alterations) | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | Unchecked |
| 53 | Linear relational concepts | Chanin et al., NAACL 2024, arXiv 2311.08968 | factual/commonsense | GitHub `chanind/linear-relational-concepts` (reuses LRE data) | UNVERIFIED | 2 | No | Objects | n/a | Factual 21 relations / 9,462 samples; commonsense 7 / 337 | UNVERIFIED | Concept directions | Reuses row 29 |

### Other candidates I could not check this session (all UNVERIFIED)

You probably know these already. I list them for breadth, but I did not open their pages, so treat every detail as UNVERIFIED:
- SyntaxGym itself (Gauthier et al. 2020, Hu et al. 2020), the source of CausalGym
- Marvin & Linzen (2018) targeted syntactic evaluation
- Linzen et al. (2016) agreement
- Newman et al. (2021) SVA templates, which Tigges et al. say are MIT-licensed
- CrowS-Pairs, WinoBias and Winogender (gender/bias minimal pairs)
- ROME "known 1000" facts
- ToMi and BigToM (theory of mind)
- Kim, Schuster & Toshniwal (2024, arXiv 2405.21068) on code pretraining and entity tracking
- The allenai/freq factual-frequency release (arXiv 2504.12459)

## Top 15 for tier 1, ranked

1. **LLM_Modularity (46 tasks).** It has flipped pairs, surface matching, MLP-neuron attribution and four domains, which makes it a direct comparison point for functional networks.
2. **MIB IOI.** Large, clean, single-token answers and length-matched, with leaderboard circuits on four common models including Qwen-2.5 and Gemma-2.
3. **MIB Arithmetic (+ and −).** 61.3k rows of templated counterfactuals (40.4k addition plus 20.9k subtraction on Hugging Face `mib-bench`), though check number tokenisation per model.
4. **CausalGym.** Many syntax tasks in one file with a generator, but align by span because tokens can differ.
5. **SFC subject-verb agreement (Marks et al.).** It covers four structural variants with single-token verbs, plus neuron and SAE circuits on Pythia-70M and Gemma-2-2B.
6. **Greater-than.** A matched, generator-based task with a well-studied MLP role; you can grab the CSV from the EAP-IG repo.
7. **Gender-Bias pronouns.** A single-token she/he contrast, and a social-domain task.
8. **Capital-Country.** A simple factual-recall counterfactual that you can filter to single-token names.
9. **Nikankin arithmetic prompts.** They come with a published MLP neuron story (heuristic neurons) on Llama3.
10. **Entity tracking (Prakash et al.).** Single-token objects with a known circuit, but it needs 7B-scale models.
11. **Binding ID tasks (Feng & Steinhardt).** These make variable binding with swaps easy and the generator is included.
12. **RAVEL / MIB-RAVEL.** Huge entity-attribute counterfactual set; filter to single-token answers.
13. **MIB MCQA.** Small, but it isolates the answer-letter mechanism, which works as a control task.
14. **Hypernymy.** A lexical-semantics task, but it has multi-answer sets, so use a probability-sum metric.
15. **Docstring.** The only clean code counterfactual here, but it is generator-only and was designed for a tiny attention-only model.

## Released neuron or unit lists with indices

- **Gurnee et al., "Universal Neurons in GPT2 Language Models" (TMLR, arXiv 2401.12181).** `github.com/wesg52/universal-neurons`, `dataframes/neuron_dfs`, has per-neuron stats for GPT2-small, GPT2-medium and Pythia-160M. These include layer/neuron ids plus universality and functional-role labels: prediction, suppression, entropy, unigram, position and more. Only 1,253 of 98,304 neurons in GPT2-medium-a pass the universality threshold; comparing neuron activations over 100 million tokens across five random seeds, the paper finds 1 to 5% of neurons are universal.
- **Tang et al. LAPE (ACL 2024).** `github.com/RUCAIBox/Language-Specific-Neurons` has `*.neuron.pth` files indexed as `neuron[lang][layer]` (en, zh, fr, es, vi, id, ja). They cover LLaMA-2 7B/13B/70B, BLOOM-7B, OPT-6.7B, Mistral-7B and Phi-2.
- **Nikankin et al. (ICLR 2025).** `github.com/technion-cs-nlp/llm-arithmetic-heuristics` has the code to get the top MLP neurons per operator and the heuristic labels for Llama3-8B and 70B. Whether ready-made index files are committed is UNVERIFIED.
- **LLM_Modularity (2026).** HF `barryhpr/LLM_Modularity_attribution` holds per-task, per-neuron attribution tensors, about 6 GB, laid out as `results/<model>/<domain>/<task>/`. You can take the top 0.1% to get exact indices for each of the 46 tasks.
- **AlKhamissi et al. (NAACL 2025).** `bkhmsi/llm-localization` produces language/ToM/MD unit masks for 18 LLMs from its stimuli. These are computed by running the code, not shipped as lists (UNVERIFIED whether masks are cached).
- **Voita et al. (Findings ACL 2024, arXiv 2309.04827).** I found no code or neuron list. Voita's site links only the paper for this entry, and the models are OPT 125m to 66b, not your target families.
- **Confidence regulation neurons (Stolfo, Wu et al., NeurIPS 2024).** These cover entropy and token-frequency neurons. The release location is UNVERIFIED.

## Caveats

- **Many licences are unconfirmed.** Only MIB-IOI (MIT), LLM_Modularity (MIT), MultiBLiMP (CC-BY-4.0), CLAMS (Apache-2.0) and the IOI/ROME code (MIT) were confirmed.
- **"Length-matched" here means matched by template.** It only holds in your tokeniser if the swapped words are single tokens there, so re-check for Qwen and Llama-3, whose tokenisers differ from GPT-2.
- **Several recent arXiv ids (2026 preprints) had unconfirmed author lists.** I list them with UNVERIFIED tags rather than guessing.
- **The LLM_Modularity paper is a 2026 preprint, not peer reviewed.** Its project page and README name two different repos. Use `Pengrui-Han/LLM_Modularity`, which is the one I confirmed has `data/` and an MIT licence.