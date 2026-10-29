# Intrinsic timescales of MLP units (plan, 2026-10-28)

Status: **implemented 2026-10-28 (step 0 done; Qwen3.5-2B run not yet submitted).** Step 0 results and the implementation choices are in LOG.md Iteration 47. Decisions are recorded in LOG.md Decisions and Iteration 47.

## 1. Question

In the brain, regions differ in their intrinsic timescale: how long their activity stays correlated with itself. Sensory regions are fast and higher-order regions are slow. We ask whether MLP units in a language model differ in the same way, and whether those differences line up with the networks and with the patching circuits.

Hypotheses:
- **H1 (depth):** τ rises with layer. Expected; a sanity check rather than a finding.
- **H2 (networks):** networks differ in τ beyond what their layer composition predicts.
- **H3 (circuits):** circuits differ from layer-matched random units in mean τ and in τ variance. Harder tasks (lower model accuracy) recruit a wider mix of timescales, so their circuits have higher τ variance.

## 2. The measure

**Autocorrelation at lag k.** For one unit, the correlation between its activation at token t and its activation at token t + k. Only pairs of tokens in the same window are used, so a pair never crosses a window boundary. Each unit's mean and variance come from all its tokens in the sample (per dataset, per half), not from each window. As a result, a difference in a unit's level between windows (for example, between topics) stays in the autocorrelation as a slow component.

**τ_int (primary).** The sum of the autocorrelations from lag 1 to lag L = 256. It is in tokens, and roughly means "how many tokens the unit's activation stays correlated with itself". A unit that resets at every token scores about 0. A unit whose autocorrelation decays exponentially with a time constant of 20 tokens scores about 20. It assumes no shape for the decay.

**Secondary summaries:** lag-1 autocorrelation, and an exponential fit with an offset, A · (exp(-k/τ) + B), as in the primate literature. Both are reported descriptively, with the share of units whose fit fails.

**Which signal.** Units are the post-nonlinearity MLP neurons, all of them. The autocorrelation is computed on the residuals of the token-and-position fit (Iterations 38-41: activation = a(token type) + b(position) + residual). This removes the autocorrelation that comes from repeated token types and from position-locked transients. Units that token and position explain exactly are dropped, as in the residual connectome.

**Covariates kept per unit:** firing rate and variance of the residual. τ could track either of them, so both enter the network and circuit models.

## 3. Material

**Model:** Qwen3.5-2B first (147,456 units). 4B and other models later.

**Datasets: wikitext and bookcorpus only, unshuffled, as contiguous text.** The existing connectome sampling cannot be reused. `get_dataset` shuffles the dataset rows and then packs them into 1024-token windows. A row is one line (wikitext), one sentence (bookcorpus), one article (agnews) or one post (tldr17), so a packed window is a series of unrelated fragments, and its autocorrelation would measure the packing rather than the model.
- Wikitext: lines are regrouped into articles (at the level-1 `= Title =` headings). Each window is the first 1024 tokens of a randomly chosen article, so position is position in the document. Articles shorter than 1024 tokens are skipped.
- Bookcorpus: rows are consecutive sentences. Each window is 1024 contiguous tokens starting at a randomly chosen sentence. Step 0 checks that the `rojagtap/bookcorpus` mirror keeps sentences in book order.
- **Agnews and tldr17 are left out for now.** Turning off the shuffle does not help them: consecutive rows are unrelated articles or posts. A window would hold about 20 agnews articles back to back without a separator, and autocorrelation across those joins would measure how units respond to a topic switch, not how long they integrate. One document per window would avoid this, but agnews articles are about 50 tokens, so they cannot support lags near 256. They can be added later with a short maximum lag.

**Amount:** per dataset, 100 windows (102,400 tokens) for half A, 100 for half B, and a disjoint set of reference windows for the token-and-position fit. The three sets are drawn without overlap. The ACF at lag 256 then rests on about 77,000 token pairs per unit per dataset per half.

**Pooling:** the autocorrelation is averaged over the two datasets within each half. Per-dataset values are kept as a secondary result. Analyses use the mean of the two halves, and the halves serve as replication.

## 4. Conditions

- **Intact:** the windows above.
- **Word-shuffled:** the same windows with whole words permuted within each window (`shuffle_words`, Iteration 32). The bag of words in each window is preserved and the order is destroyed.

The two conditions separate two sources of slowness. Under shuffling, a unit can still look slow because the window's words share a topic. Only under intact text can it also be slow because it integrates ordered context. τ_int(intact) − τ_int(shuffled) is therefore the part of a unit's timescale that depends on word order. Each condition gets its own token-and-position fit, made on its own reference windows in the same way.

There is no separate null for τ itself. The circular-shift null keeps each unit's autocorrelation by construction, so it is the reference for the network test (C), not for the τ values.

## 5. Analyses

**A. Reliability of τ (gate).** Spearman correlation of τ_int across units between half A and half B, per condition; also between wikitext and bookcorpus. If the split-half correlation is below 0.7, we stop and reconsider before B-D.

**B. Depth.** τ_int (intact, shuffled, difference) by layer: distribution per layer, median and spread.

**C. Networks.** Networks are the pooled residual networks of Iteration 43 (k = 100), fitted on the four-dataset connectome.
- For each network, its mean τ_int minus the mean that its layer composition predicts. The prediction is Σ_l (share of the network's units in layer l) × (mean τ_int of all live units in layer l). This difference is the network's *layer-adjusted τ*. It is the figure: one value per network.
- Test statistic: the variance of layer-adjusted τ across networks (how much networks differ).
- Reference 1: unit labels permuted within layer (1,000 permutations), which keeps each network's size and layer composition.
- Reference 2: the same statistic on the circular-shift null partition. That partition keeps each unit's autocorrelation, and LOG Iterations 4-6 showed that differences in autocorrelation alone produce clusterable structure in |r|. Real networks above reference 1 but not above reference 2 would mean the alignment with τ is that clustering effect, not co-activation structure.
- Repeated with firing rate and variance regressed out of τ within layer, and for the shuffled condition and the difference.
- Cross-check: rank correlation between τ_int and node strength, real and null, as a link to the rich-club results (Iteration 45).

**D. Circuits, task by task.** The circuits are those of Iteration 44: the top 0.1% of neurons by attribution, for every task that passes at 2B.
- For each task: the mean and the variance of τ_int over the circuit's units.
- Each is compared with 1,000 layer-matched random draws (same number of units in each layer as the circuit), giving a z-score and a percentile for each task. Layer matching matters even more for the variance: a circuit spread over more layers would have a higher τ variance just because τ changes with depth.
- Reported per task (table and figure), coloured by domain. An aggregate line (median z over tasks, sign test) is reported alongside.
- Difficulty: the model's both-correct accuracy on the task. Spearman correlation across tasks between accuracy and the variance z-score (prediction: negative), and between accuracy and the mean z-score. Caveat: tasks enter only at accuracy ≥ 0.6, so the accuracy range is truncated.
- Domain is used for visual and qualitative inspection only.

**Deferred:** a causal check of τ by replacing distant context, ablating slow against fast units, Pythia checkpoints, and dependency length as a control for "complex" against "long-range". The last will matter if D finds the accuracy effect.

## 6. Implementation

- `parcelmate/timescales.py`. Window sampling as in section 3. The ACF is accumulated on the GPU: for each window and each lag k ≤ L, the sum of z_t · z_{t+k} over within-window pairs and their count. Nothing is stored per token.
- Reuse: `get_timecourses` hooks for MLP units, `residual.py` for the token-and-position fit, `shuffle_words_within` for the shuffled condition, the network and circuit loaders of Iterations 43-44, and the layer-matched sampling of the concentration test.
- `parcelmate/bin/timescales.py` (GPU: ACFs) and the analysis script (CPU: A-D), plus a config generator.
- Tests in `tests/`: τ_int recovers known time constants on synthetic AR(1) series; no pair crosses a window boundary; the shuffled condition preserves each window's bag of words; dead units are excluded; layer-matched draws reproduce the circuit's layer counts.
- Output: per unit, ACF (L lags) per condition, dataset and half, as float32. About 150 MB per condition, dataset and half at 2B.
- Cost estimate: one forward pass over 2 datasets × 3 window sets × 2 conditions, about 1.2M tokens at 2B, about 1 GPU-hour. A-D are CPU and small.

## 7. Step 0 (before writing the analysis code)

1. Confirm that the bookcorpus mirror keeps sentences in book order (read a few hundred consecutive rows).
2. Wikitext article length distribution in tokens: how many articles reach 1024 tokens.
