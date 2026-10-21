# Stage 1 of the network-neuroscience analyses: rich club in Qwen3.5-2B and 4B (plan, revised 2026-10-20)

Status: **implemented 2026-10-20 (Iteration 45); Pythia-70m pilot submitted, Qwen runs wait for the pilot.** Implementation notes are in section 10. Background and the neuroscience literature are in [HUBS_RICH_CLUB_AND_PARTICIPATION.md](HUBS_RICH_CLUB_AND_PARTICIPATION.md). Revised twice on 2026-10-20 after Andrea's review. Decisions so far, recorded in LOG.md Decisions:
- the pooled residual connectome only, with no analysis per dataset;
- edge weights |r| only;
- nulls N1 and N3 only (no rewiring);
- hubs ranked by strength, with degree secondary;
- tiles stored rather than streamed;
- a Pythia-70m pilot first.

## 1. Question and hypotheses

Are the most strongly coupled MLP neurons coupled mainly to each other?

- **H1 (rich club):** the units with the highest strength are more strongly coupled among themselves than their strengths alone predict.
- **H2 (no rich club):** coupling among the top units is what their strengths predict, or less.

## 2. Material

**Models:** a pilot on Pythia-70m (final checkpoint, step 143,000), then Qwen3.5-2B (147,456 MLP units) and Qwen3.5-4B (294,912). All units.

**Connectome:** the pooled residual connectome of Iteration 43 (`qwen3.5-{2b,4b}-resid-pool4`). It is the same connectome the networks compared with the patching circuits were fitted on. Its structure:
- The four prose datasets are wikitext, bookcorpus, agnews and tldr17; codeparrot is excluded.
- Each dataset contributes two samples of 102,400 tokens. Sample 1 goes to half A, sample 2 to half B.
- One token-and-position fit is made on reference windows of all four datasets and removed from every sample before z-scoring.
- Each sample is correlated on its own. The half's connectome is the Fisher mean over its four samples, one per dataset: tanh of the mean of arctanh(r). So each half contains every dataset, and the datasets are averaged at the level of correlations, not concatenated as timecourses.
- Edge weight: w_ij = |r_ij| of that half connectome, without the diagonal, pairs within a layer included.
- The steps that only prepare the clustering input (profile Fisher transform, z-scoring, top-10% sparsification) are not applied.

**The pilot needs a new run.** Pythia-70m has no pooled residual connectome: its residual runs were single-dataset, and their tiles are purged. The pooled residual path exists only for tiled storage, so the pilot runs the same code path as Qwen at a tiny size (10,240 live units after the layer-0 rule).

**Replication unit:** the two halves. There are no per-dataset analyses.

## 3. Measures

**Strength:** s_i = Σ_j w_ij. Binary degree k_i at global densities of 0.1%, 1% and 5% of all edges is secondary.

**Rich sets:** R(n) = the n units with the highest strength, for n on a log grid from 50 to N/5.

**Hubs are selected on one half and evaluated on the other.** Selecting the top-strength units and measuring their mutual coupling on the same data is biased when some units share variance in one half only (for example a document or a burst present in one sample): they are strong in that half because they co-vary there. (On iid noise the bias is negligible, because one edge is 1/N of a unit's strength; the tests showed this, see section 10.) So R(n) is defined on half A and every coupling measure is computed on half B, and then the reverse (B to A). This is the replication the experiment rests on.

**Club-level curve.** ρ(n) = W_B(n) / E_N1,B[W(n)], where:
- W_B(n) = Σ_{i<j in R_A(n)} w^B_ij is the weight among the hubs in the other half;
- N1 is the configuration expectation computed from the other half's strengths: E[w_ij] = s_i s_j / 2W.

**Hub-level statistic** (the replication unit for inference). For each hub i in R_A(n):

c_i(n) = Σ_{j in R_A(n), j≠i} w^B_ij / Σ_{j in R_A(n), j≠i} E_N1,B[w_ij]

This is how much more hub i couples to the other hubs than N1 predicts. c_i > 1 means hub i belongs to the club.

## 4. Nulls

- **N1, configuration (analytic).** It keeps each unit's strength. It rules out a single shared signal: if r_ij = a_i a_j, the strongest units correlate strongly with each other, but N1 predicts exactly that, so ρ = 1.
- **N3, circular-shift connectome.** The same pooled residual run computes a null connectome in which each unit's timecourse is shifted by a random offset. That keeps each unit's statistics but destroys co-activation. The whole analysis (hubs selected on null half A, evaluated on null half B, normalised with N1) is repeated on it.

## 5. Inference and pre-stated criterion

**Inference.** For each n, compare the distribution of log c_i over the real hubs with the distribution over the N3 hubs, using a one-sided Mann-Whitney test, plus the share of hubs with c_i > 1 in each. The hubs are not independent, because they share the club's edges. So the p-values are read as descriptive, and the replication across the two directions (A to B, B to A) carries the weight.

**H1 is claimed for a model if, over a contiguous range of n, all of the following hold:**
- ρ(n) > 1 in both directions (A to B and B to A);
- the real hubs' c_i exceed the N3 hubs' c_i (one-sided Mann-Whitney, p < 0.05 in both directions);
- the real ρ(n) exceeds the N3 ρ(n).

Otherwise the result is H2.

## 6. Characterising the club (descriptive)

1. **Reliability of membership:** Jaccard overlap of R_A(n) and R_B(n), against random sets of the same size.
2. **Composition:**
   - layers of club members, against units of similar strength;
   - firing rate of members;
   - overlap with the units that the token-position fit explains most.
3. **Relation to the networks** (Grayson et al. 2014's question), using the existing pooled residual partitions (k = 100) of each model:
   - each unit's strength towards each network;
   - the participation coefficient of club members against non-members of similar strength;
   - the number of networks the club spans.
   These participation scores also feed the later connector-hub ablation.
4. **2B against 4B:** ρ(n) as a function of n/N.

## 7. Computation

**Storing, not streaming.** Andrea asked whether streaming gives exactly the same result as storing. It does not:
- the streamed values skip the float16 rounding of the stored tiles, so they differ from them by about 1e-3 in r;
- the binary thresholds would come from a histogram rather than an exact sort.

Since only one connectome per model is now needed, storing is affordable, and it has three advantages:
- it is the same object the circuit networks were fitted on;
- it needs no new connectivity code: the `resid-pool4` configs are rerun with purging off;
- the tiles can be reused for the later stages (participation, connector-hub ablation).

**Footprint:**
- Pythia-70m: about 0.8 GB (2 halves x real and null).
- Qwen3.5-2B: 172 GB.
- Qwen3.5-4B: 696 GB.
- Total about 870 GB on the climblab share (27 TB free on 2026-10-20). Decision 8.1 concerns the earlier 3 TB cap.

**Reproduction check.** The rerun uses the same config and seed (the pipeline is bit-for-bit reproducible, S7). Before the rich-club analysis, the stored pooled partitions are re-scored on the recomputed tiles. The split-half R2 of the connectome (2B: 0.928) and the held-out fidelity must match the stored metrics. If they do not, the tiles differ from those the networks were fitted on, and we stop.

**The analysis itself:** a CPU job per model that streams the stored tiles (as `score_big` does). Pass 1 computes the strengths and network strengths; pass 2 computes, for each n, the sums over R_A(n) needed for ρ and c_i on the other half. Real and null trees are processed alike.

**Jobs:**
- Pythia-70m: one connectivity job and one analysis job.
- 2B and 4B: one connectivity job each (2 h 18 at a 130 GB peak for 2B, measured; 4B at 320 GB requested, as in Iteration 43), then one CPU analysis job each.
- Submitted only after the pilot, with the footprint reported.

**Verification (tests/, before any cluster run).**
- The streamed W(n), c_i and strengths equal a brute-force computation on a small dense matrix.
- On synthetic graphs:
  - a planted rich club gives ρ > 1 and c_i > 1;
  - a rank-1 correlation matrix gives ρ ≈ 1;
  - a random weighted graph gives ρ ≈ 1;
  - variance shared by a subset of units in one half only: selecting and evaluating on that half gives ρ > 1, evaluating on the other half gives ρ ≈ 1. This test documents the selection bias that the cross-half design removes.

## 8. Open decisions

1. **The 3 TB cap:** does the cumulative write cap still apply? This stage writes about 870 GB. The null tiles could be purged right after the rich-club analysis, keeping only the real tiles for the later stages (about 435 GB).
2. **The pilot's purpose:** check the code and the cross-half design end to end. Its results are reported but do not count toward H1.

## References (checked against abstracts or records; full texts not read)

- Colizza, Flammini, Serrano and Vespignani, "Detecting rich-club ordering in complex networks", Nature Physics 2, 110-115, 2006 ([arXiv physics/0602134](https://arxiv.org/abs/physics/0602134)).
- Opsahl, Colizza, Panzarasa and Ramasco, "Prominence and control: the weighted rich-club effect", Physical Review Letters 101, 168702, 2008 ([arXiv 0804.0417](https://arxiv.org/abs/0804.0417)).
- van den Heuvel and Sporns, "Rich-club organization of the human connectome", Journal of Neuroscience 31(44), 15775-15786, 2011 ([PMC6623027](https://pmc.ncbi.nlm.nih.gov/articles/PMC6623027)).
- Grayson et al., "Structural and functional rich club organization of the brain in children and adults", PLoS One 9(2), e88297, 2014 ([PMC3915050](https://pmc.ncbi.nlm.nih.gov/articles/PMC3915050)).
- Zalesky, Fornito and Bullmore, "On the use of correlation as a measure of network connectivity", NeuroImage 60(4), 2096-2106, 2012 (doi 10.1016/j.neuroimage.2012.02.001).

## 10. Implementation (2026-10-20, Iteration 45)

Code: [parcelmate/richclub.py](../parcelmate/richclub.py) (streamed measures), [parcelmate/bin/richclub.py](../parcelmate/bin/richclub.py) (both trees, inference, criterion, outputs in `<tree>/metrics/richclub/`), [scripts/make_richclub_configs.py](../scripts/make_richclub_configs.py) (configs in `configs/richclub/`), [scripts/check_tiles_reproduce.py](../scripts/check_tiles_reproduce.py), [scripts/launch_richclub.sh](../scripts/launch_richclub.sh). Verification: [tests/verify_iter45_richclub.py](../tests/verify_iter45_richclub.py), 23 checks, all pass.

Choices made in implementation (to confirm):
- **"Contiguous range of n" = at least 3 consecutive points** of the 20-point log grid (50 to N/5). The verdict file reports the longest run and its n range, so a different threshold can be applied without rerunning.
- **Binary degree** is counted at exact |r| thresholds for 0.1%, 1% and 5% of live off-diagonal pairs (exact because the stored values are float16: the histogram is over their bit patterns). Degree rankings break ties by strength, then by unit index. The criterion is computed for every ranking; strength is the primary one.
- **Dead units** (zero rows: units the token-position fit explains exactly, e.g. Pythia's layer 0) are excluded from rankings and from the live-pair count. They add nothing to any strength.
- **Participation** uses the selection half's partition and the evaluation half's weights, the same cross-half logic as the hubs. Club members are compared with the units ranked just below them (ranks n to 2n).
- **Deferred from section 6.2:** firing rate and overlap with the units the token-position fit explains most. Neither is stored in the tiles (they hold residual statistics only), so they need a forward pass; they will be added with the later stages.
- **Selection-bias test corrected:** the plan expected pure iid noise to show the bias; it does not (ρ in-sample 0.975 at n = 20, within sampling noise). The test now plants variance shared within one half, which gives in-sample ρ = 3.9 and cross-half ρ = 1.000.

Tiles live in `results/richclub/<model>-resid-pool4-tiles{,_null}/connectivity/`. For Qwen the stored Iteration 43 partitions are copied into the new trees and re-scored; the rich-club job runs only if every stored score is reproduced within 2e-3.
