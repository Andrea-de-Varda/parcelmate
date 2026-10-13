# Paper plan: task-free networks against task-localised circuits (started 2026-10-12)

Working document for the paper framing adopted on 2026-10-12, after the sparsification framing was dropped ([LITERATURE_SPARSIFICATION.md](LITERATURE_SPARSIFICATION.md)). Decisions are also recorded in the Decisions section of [LOG.md](LOG.md).

**Citation status.** Papers with a link were found and checked at the level of abstracts or search snippets on 2026-10-12. Papers marked *(unverified)* are cited from memory and must be checked before they go into the paper.

## Story

1. NLP finds functional components mostly top-down: a task is chosen, then the units that implement it are localised (patching, circuit discovery). The bottom-up methods that exist (SAEs, transcoders, ICA, MoEfication) find features or clusters within a layer, not systems that span the network. Top-down localisation can only find components that someone thought to look for.
2. Following neuroscience, we compute task-free functional connectivity between all MLP neurons and parcellate it into networks, after removing token identity and position (the residual connectome).
3. Claims: (a) the networks recover the organisation of circuits found by task localisation, across many published task sets; (b) they reproduce known training-dynamics patterns (supporting result, not the headline); (c) they reveal networks that no task localiser targets ("dark networks").
4. Secondary framing: within-dataset versus across-dataset connectome similarity (0.95 against 0.63 for the Pythia-70m residual connectome) as a stable intrinsic architecture plus state-dependent reconfiguration, after Cole et al. 2014 *(unverified)*. The brain parallel is motivation, not the focus.

## Decisions so far

- 2026-10-12: residual-connectome networks are the main object; ordinary networks are a comparison.
- 2026-10-12: units stay MLP neurons. Head-level circuits are compared by running our own neuron-level attribution on the published tasks' prompts; attention heads are not added to the connectome.
- 2026-10-12: Qwen3.5-2B is the first model; results are to be replicated across several models.
- 2026-10-12: the task bank is to be as large as possible (a separate deep-research search is collecting datasets with data links).

## Open: what data the localisation can use

Our attribution patching ([parcelmate/patching.py](../parcelmate/patching.py), after Han et al.) needs, per item: a clean prompt, a corrupted prompt that flips the preferred answer, and a correct/incorrect answer pair scored under both prompts; clean and corrupted prompts must tokenize to the same length (the alignment filter), and attribution is taken at the last prompt token. Options for widening the data are listed in the conversation record and summarised here once decided.

## Characterising networks bottom-up (brainstorm, to be prioritised)

1. **Ablation profile.** Mean- or resample-ablate each network over a large mixed corpus, against size- and layer-matched random sets, and keep per-token effects. Read them by dataset/domain, token frequency, word-start versus continuation, position, part of speech, and the intact model's surprisal and entropy. Measure the prediction change, not only loss on the correct token (KL, entropy change: calibration networks show up only here). Measure which output tokens gain or lose probability, and the network's direct vocabulary effect (output weights through the unembedding). Describe the most damaged contexts automatically and test the description on held-out text, after Bills et al. 2023 ([OpenAI](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)), with the faithfulness caveats of [Huang et al. 2023](https://arxiv.org/abs/2309.10312).
2. **Activation side.** Each network's mean timecourse as an fMRI network signal. (a) Encoding models on a large annotation bank (part of speech, dependency depth, surprisal, entropy, position, named entities, numbers, bracket and quote depth, code syntax, sentence boundaries). (b) Enrichment against labelled dictionaries: correlate network timecourses with SAE features that already carry automatic labels, in the manner of gene-set enrichment; this also tests whether a network is one feature or a bundle of many. (c) Temporal receptive windows: network reliability on text scrambled at word, sentence and paragraph level, after [Lerner et al. 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3089381).
3. **Steering.** Amplify a network during generation and describe the change (qualitative only).
4. **Connectome organisation.** Network-of-networks structure; principal gradient of the connectome, after Margulies et al. 2016 *(unverified)*.
5. **Operational definition of a dark network.** (i) reliable across split halves, ideally across datasets; (ii) ablation matters beyond matched random sets; (iii) no task in the task bank loads on it; (iv) its timecourse is poorly explained by the annotation bank relative to its own reliability ceiling. Expected first hits are housekeeping networks (attention sinks and BOS handling, massive activations, position, calibration), which no task localiser targets; the neuroscience precedent is the default mode network, found as task-negative activity (Shulman et al. 1997; Raichle et al. 2001; both *unverified*).

## Comparing circuits and networks when a task spans many networks

A task may legitimately recruit several networks. The test is whether the circuit is a compact and predictable combination of networks, not whether it is one network.

1. **Loading profile per task.** Project the continuous attribution map (not only the top 0.1%) onto the networks; report how many networks hold 80% of the attribution mass, against layer-matched random sets.
2. **Composition test.** Tasks that share known components should share networks (for example IOI, gendered pronouns and entity tracking; greater-than and arithmetic; agreement across relative clauses and across prepositional phrases, whose circuits overlap in [Marks et al. 2025](https://arxiv.org/abs/2403.19647)). Overlap is predicted from task structure, and the null partition should fail.
3. **Task-based against rest-based decomposition.** Decompose the attribution maps of all tasks (NMF or ICA) and match the components to the task-free networks, after [Smith et al. 2009](https://pubmed.ncbi.nlm.nih.gov/19620724/), who recovered resting-state networks from thousands of BrainMap activation maps. Candidate centrepiece.
4. **Two independent causal matrices.** Network-ablation deficits (networks x tasks) against attribution loadings (networks x tasks).
5. **Prediction.** Left-out-task prediction (as now), plus predicting a new task's attribution map from the connectome and a few examples, against patching on those few examples alone, after Tavor et al. 2016 *(unverified)*.
6. **Resolution.** Sweep k (20 to 400) and find where circuit-network correspondence peaks.

## Task and circuit sources identified so far

- Han et al., LLM_Modularity (ours; four domains).
- MIB ([Mueller et al., ICML 2025](https://arxiv.org/abs/2504.13151)): IOI, arithmetic, MCQA, ARC; GPT-2, Qwen-2.5-0.5B, Gemma-2-2B, Llama-3.1-8B.
- ACDC tasks ([Conmy et al., NeurIPS 2023](https://arxiv.org/abs/2304.14997)): IOI, greater-than, docstring.
- CausalGym ([Arora et al., ACL 2024](https://arxiv.org/abs/2402.12560)): SyntaxGym minimal pairs on Pythia.
- Published neuron-level findings to locate in the networks: arithmetic heuristic neurons ([Nikankin et al., ICLR 2025](https://arxiv.org/abs/2410.21272)); universal, prediction, suppression and entropy-related neurons ([Gurnee et al., TMLR 2024](https://arxiv.org/abs/2401.12181)); sparse-probing neurons ([Gurnee et al., TMLR 2023](https://arxiv.org/abs/2305.01610)); dead, n-gram and positional neurons ([Voita et al. 2024](https://arxiv.org/abs/2309.04827)); language-specific neurons ([Tang et al., ACL 2024](https://aclanthology.org/2024.acl-long.309)); language-selective units ([AlKhamissi et al., NAACL 2025](https://arxiv.org/abs/2411.02280)).
- Other circuit-discovery methods for triangulation: circuit-tracer attribution graphs (Gemma-2-2B, Llama-3.2-1B; [Anthropic](https://www.anthropic.com/research/open-source-circuit-tracing)), SAE feature circuits, DAS.
