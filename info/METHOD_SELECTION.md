# Method selection: what was varied, what it bought, and the final choice

Written 2026-09-21, after the confirmation run (LOG.md Iteration 21). This is the compact record of the 55 parcellation arms scored on GPT-2 between 2026-09-10 and 2026-09-15. It answers three questions: which pipeline is chosen and why, which independent dimensions were varied, and what generally worked. Numbers are means over the four prose domains (wikitext, bookcorpus, agnews, tldr17). The full history is in [LOG.md](LOG.md); the method background in [PARCELLATION_DESIGN.md](PARCELLATION_DESIGN.md).

## 1. The final pipeline

1. **Units:** post-GELU MLP neurons (832 per layer, 12 layers, 9,984 units), not residual-stream dimensions.
2. **Connectome:** |r| between unit timecourses over about 400k tokens per domain, Fisher-transformed, diagonal zeroed.
3. **Features:** each unit's row is z-scored (its profile), then only its top 10% of partners is kept and the row is re-standardized.
4. **Reduction:** PCA to 100 components, no whitening.
5. **Clustering:** Lloyd k-means (k-means++ init), k = 100, 200 restarts, aligned with the Hungarian algorithm and averaged into a consensus.
6. **Estimation:** always within a single domain. Domain generality is measured by fitting on one domain and scoring on another, never by pooling.

Confirmed performance (arm `vmf_sparse_pca100_lloyd100_n200`): within-domain ARI 0.71 (attainable ceiling about 0.80), co-association r 0.94, across-domain ARI 0.11, within-domain fidelity r 0.26 above its null, across-domain fidelity r 0.10 above its null, hubness AMI 0.08. Cost about 2.5 min per connectome on 8 cores.

## 2. How arms were judged, in plain terms

Two things are asked of a partition. **Reliability**: fit it on one half of the data and on the other half, and see whether the two agree (adjusted Rand index between the two label sets). **Fidelity**: fit it on one half, replace every connection by its block mean, and see how well that predicts the other half (Pearson r; R2 within domain, where scales match). Each is asked twice: within a domain (two halves of the same text) and across domains (fit on wikitext, score on bookcorpus, and so on for all 12 ordered pairs).

Each metric has a failure mode the other catches. Reliability alone rewards partitions that are stable for trivial reasons, for instance sorting units by how strongly connected they are. Fidelity alone rewards partitions that explain the overall level of connectivity, which again is what sorting by strength does. So every number is read against a **null partition**: the same pipeline run on circularly shifted timecourses (all cross-unit structure destroyed, each unit's own statistics kept), then evaluated on the real data. Real minus null is the credit the clustering earns beyond knowing per-unit properties. A **hubness AMI** (agreement between the partition and a sorting of units by strength) says how much of a partition is that trivial structure.

The selection criterion (LOG.md Iteration 16) is **across-domain fidelity above the null**, with within-domain reliability as a secondary constraint. Within-domain numbers alone were shown to be bought largely by hubness, and hubness costs transfer.

## 3. The dimensions that were varied

Ten independent choices were changed across the 55 arms. Each row says what was tried and what the data said.

| # | Dimension | Values tried | What happened |
|---|---|---|---|
| 1 | **Unit definition** | residual-stream dimension at each layer boundary; MLP post-GELU neuron | Residual units form chains (the same dimension at adjacent layers correlates at 0.87), and the best residual arms were largely following those chains (dimension AMI up to 0.68). MLP units have no chains, and every raw metric improved at equal method. **MLP chosen.** |
| 2 | **Connectome quantity** | \|r\|; \|z\| = \|r\| divided by its circular-shift noise scale | \|z\| made the null a floor but changed the data rather than the comparison; once the null partition was evaluated on real data, \|z\| gave the same ranking as \|r\| and a hair worse numbers. **\|r\| kept; \|z\| retired.** |
| 3 | **What the clustering sees** (the transform of each unit's row) | binarize top 10% per row (inherited, with and without the S2 bug); binarize with one global threshold; dense Fisher magnitudes; top-10% Fisher magnitudes; z-scored profile; z-scored profile sparsified to its top 10% | Any transform that keeps a unit's overall connection strength (dense Fisher, global threshold, sparse Fisher) produced a hubness partition: the most reliable arms in the set (ARI 0.76-0.79) and the worst transfer (across Δ 0.05-0.08). Z-scoring removes strength and asks only about pattern: transfer Δ 0.10-0.22, hubness AMI 0.07-0.10. Sparsifying the z-scored profile added a small, unanimous gain. **Standardized, sparsified.** |
| 4 | **Dimensionality reduction** | none; PCA-200 whitened; PCA-200, PCA-100, PCA-20 unwhitened | Whitening was the single most destructive setting: it flattened the consensus (median membership 0.20) and gave ARI 0.11. Unwhitened PCA-100 matched full profiles on every metric at 1/25 the cost. PCA-20 traded reliability for within-domain fit. **PCA-100 unwhitened.** |
| 5 | **Clustering algorithm** | MiniBatchKMeans; full Lloyd k-means; Ward agglomerative; Ward-initialised Lloyd; spatial ICA with winner-take-all; block-model coordinate descent (per restart, or once on the consensus; raw, double-centred, degree-corrected) | MiniBatch was under-converged. Ward looked best on residual units only because it follows chains; on MLP units it was the least reliable (ARI 0.24 vs 0.61 for Lloyd) at equal fidelity. ICA's component maps reproduce well (0.66) but its hard labels do not (0.28). The block-model polish is the largest fidelity lever in the project (within r Δ 0.26 to 0.37, across 0.10 to 0.18) but cuts reliability by 0.13-0.22, adds hubness (AMI 0.37 raw), and its optimum is not identifiable across restarts. **Lloyd consensus, no polish.** |
| 6 | **Number of networks k** | 50, 100, 150, 200 | Fidelity rises with k in every family (across Δ 0.08, 0.10, 0.11 at k = 50, 100, 200 on MLP Lloyd). Lloyd reliability falls with k (0.68, 0.62, 0.55). Neither trend flattens by 200. **k = 100 as a judgement**, to be revisited on the second model. |
| 7 | **Restarts and consensus** | 20 MiniBatch; 40 or 200 Lloyd; Hungarian alignment vs co-association (evidence accumulation) consensus | Reliability of every Lloyd arm sat on its restart-split ceiling, so the optimizer, not the data, was limiting. 200 restarts raised the ceiling from 0.65 to 0.77 and reliability by 0.06 at no fidelity cost. Co-association consensus did not help. **200 restarts, Hungarian.** |
| 8 | **Estimation design** | one domain; pooled connectome of 2, 3 or all 4 domains (T4) | Pooling raised transfer into a held-out domain by 0.02 and label agreement by 0.03, but as a fraction of the (higher) ceiling the gain is what more tokens buy. **Excluded by design: all clustering within one domain** (Andrea, 2026-09-15). |
| 9 | **Reference for the metrics** (an evaluation choice, not a pipeline choice) | pipeline on null data vs pipeline on real data; null partition on real data; random partition of the same k | Comparing real-data R2 with null-data R2 let a structureless matrix out-score real data, because \|r\| of noise is a block-friendly strength field. Evaluating the null partition on the real data put both on one denominator. **Null partition on real data.** |
| 10 | **Fidelity measure** | R2; Pearson r | R2 rewards predicting the overall level, which hubness supplies; r scores only the pattern. R2 is undefined across domains. **Compare r with r; report R2 within domain alongside.** |

Held fixed throughout, by argument rather than test: taking |r| rather than signed r (PARCELLATION_DESIGN.md 2.1), the 10% sparsity level, about 400k tokens per domain in 4 samples, GPT-2, the four prose domains (whitespace and codeparrot are degenerate and excluded from every mean), hard argmax labels as the reported object.

## 4. What generally worked, and what did not

**Worked.**
- Removing per-unit strength from the features (z-scoring). Everything that transfers across domains descends from this step.
- Converging the optimizer (Lloyd instead of MiniBatch) and giving it enough restarts. Reliability was optimizer-limited until 200 restarts.
- Unwhitened PCA as a pure speed lever.
- Changing the unit to MLP neurons. It removed the chain confound and improved every raw number.
- Pairing reliability with fidelity, and both with a null partition on the real data. Every reversal in the ranking came from a metric being read alone.

**Did not work, or worked for the wrong reason.**
- Whitening. It made the inherited pipeline nearly unreproducible.
- Keeping magnitudes (dense Fisher, global threshold, sparse Fisher, block-model objective). High reliability and high within-domain R2, all bought with hubness, all poor at transfer.
- Ward. Its residual-stream results, the best raw numbers in the project, were chain-following.
- ICA as a partition method. Good maps, poor labels.
- The block-model polish as part of the pipeline. It is the right tool if fidelity is the only goal, and the wrong one when the partition must reproduce.
- Co-association consensus, PCA-20 denoising, surrogate normalization. Each was a reasonable idea that measured as neutral or worse.
- Every attempt to raise across-domain label agreement. No feature, optimizer, consensus, k or estimator moved it past 0.17; the final arm sits at 0.11. This is the open problem, and the uncompressed connectomes of two domains agree at only r 0.51, so much of it is in the data.

## 5. Why this arm and not the runner-up

The only serious alternative is the same pipeline with a raw block-model polish. It wins on fidelity by a wide margin (across-domain Δ 0.18 against 0.10). It loses on everything else: within-domain ARI 0.49 against 0.71, across-domain ARI 0.08 against 0.11, and a hubness AMI of 0.37 against 0.08, meaning a third of its structure is units sorted by connection strength. Since the networks are meant to be things that reproduce and transfer, and since the polish can be applied afterwards to any partition if a paper needs the fidelity number, the unpolished consensus is the object to carry forward.

## 6. What the second model has to settle

- Whether the small margins that decided sparse profiles (+0.03 ARI, +0.007 across-domain ARI) survive on data not used for selection.
- k, which was left at 100 by judgement.
- Whether across-domain label agreement stays near 0.1 on a different model, which would make it a property of the method rather than of GPT-2.
