# Stable networks across datasets, and what the connectome is made of

A summary of the work since 2026-10-07, when Andrea set five items: (1) a cross-dataset consensus from half the stored restarts, checked against the other half; (2) graded stability metrics; (3) find the components behind "the connectome is stable within a dataset but not across", and remove them to get a connectome stable within and across datasets in trained models but not in untrained ones; (4) leave codeparrot out of stability definitions; (5) redo the patching comparison on the consensus networks. It also includes the diagnosis that motivated the items (2026-10-06 and 07). Full record in `info/LOG.md`, Iterations 36-38; every number here is in a table under `figures/` or `results/`.

## In short

- **The problem was in the partition, not in the connectome.** Connectomes of different prose datasets correlate about 0.5 (GPT-2) or 0.3 (Qwen3.5-2B), but hard k = 100 partitions of them agree at ARI 0.1 or less. Hard partitions are fragile: random noise that leaves a connectome correlated 0.5 with the original produces the same drop.
- **The ordinary connectome is a token-identity connectome.** It equals the connectome of each token type's mean activation, weighted by how often each type occurs. That makes it equally reproducible in untrained models, and makes datasets disagree in proportion to their word frequencies.
- **The residual connectome is what Andrea asked for.** It is what remains of each activation after subtracting what the current token, the previous token and the position predict. In trained Pythia-70m it agrees within a dataset at 0.98 and across datasets at 0.81 (ordinary connectome: 0.56); in the untrained model it is less reliable and less shared (0.74 within, 0.58 across). Its networks (k = 100) agree across datasets at ARI 0.38, against 0.17 for ordinary networks, and only after training. The pattern holds in Pythia-160m, more weakly.
- **A cross-dataset consensus of the existing Qwen partitions replicates** (ARI 0.41 between independent data halves, null 0.18). But it does not match the patching circuits better than single-dataset networks do.

## 1. The diagnosis (before the five items)

![diagnosis](fig1_diagnosis.png)

**A. What survives across datasets** (GPT-2 final arm, 9,984 MLP units, 4 prose datasets; `scripts/diag_partition_fragility_1.py`). "Within" compares half A and half B of one dataset; "across" compares half A of one dataset with half B of another.

| | within | across | what is compared |
|---|---|---|---|
| connectome | 0.99 | 0.51 | Pearson r between the two |r| matrices (3 million sampled neuron pairs) |
| connectome, strength removed | 0.99 | 0.46 | the same after double-centring (below) |
| neuron profiles the pipeline clusters | 0.97 | 0.47 | per neuron, Pearson r between its two profiles (its row after Fisher transform, row z-score and top-10% sparsification); median over neurons |
| partitions | 0.62 | 0.10 | adjusted Rand index between the two k = 100 partitions (0 = chance, 1 = identical) |

*Double-centring.* From every entry of the |r| matrix subtract the two neurons' mean |r| (their overall strength) and add back the grand mean: M′ᵢⱼ = Mᵢⱼ − sᵢ − sⱼ + ḡ. It centres rows and columns (nothing is divided) and leaves only whether two neurons are more or less coupled than their overall strengths predict. It tests whether the across-dataset similarity is just "the same neurons are hubs everywhere": strengths do agree across datasets (r = 0.71) and explain about a fifth of the variance of |r|, but removing them leaves 0.46 of the 0.51.

About half of the similarity survives all the way to the profiles the pipeline clusters; the partitions keep much less. Note that ARI is on a different scale from r, so the drop in the last row is not a like-for-like loss; panel B puts the two on the same footing. Ablating sparsification or z-scoring, clustering raw |r|, or using k = 20 does not change the across/within ratio of the partitions.

**B. Hard partitions are fragile** (`scripts/diag_partition_fragility_2.py`). Random noise added to one connectome, reducing its correlation with the original to 0.9, 0.7 and 0.5, drops partition ARI to about 0.41, 0.26 and 0.15. The cross-dataset point (0.51, 0.10) sits on that curve. Re-clustering the same matrix with another seed gives 0.45. So label identity is a steep function of connectome similarity, and low cross-dataset ARI does not mean networks are dataset-specific. On graded measures, labels transfer as much as the connectome does: one dataset's partition, applied to another dataset, keeps 45% of the own partition's advantage over the null partition, and pairs of units in the same network stay together 10 times more often than chance.

**C. Qwen3.5-2B, best-match overlap (Dice) of networks.** Each prose dataset reproduces its own networks (0.72-0.84) but not those of other datasets (0.04-0.10). Codeparrot does not even reproduce itself (0.14; its connectome's split-half correlation is 0.55, against 0.96-0.99 for prose). An earlier "stable networks" analysis based on identical labels (Iteration 36) therefore understated the shared structure, especially because its "4 of 5 datasets" criteria effectively required the unreliable codeparrot. Circuits did not favour the units with identical labels everywhere (0.7-1.2x chance), so stability by label identity was dropped as a filter.

## 2. Item 1: cross-dataset consensus from the stored restarts

![consensus](fig2_consensus.png)

**Method** (`parcelmate/bin/restart_consensus.py`). Every dataset's partition is fitted within that dataset, as always; each keeps its 200 k-means restarts. The consensus treats all restarts of all four prose datasets (800 labelings) as votes on which units belong together: the co-association matrix (share of labelings in which two units share a network), clustered into k = 100. At 147k units that matrix has 22 billion entries, so its leading 100 eigenvectors are obtained from the labelings directly (truncated SVD of the one-hot labels, mathematically equivalent), then k-means. Each unit gets a confidence, the average share of its consensus network's members that share its label across the labelings.

**Result, Qwen3.5-2B.**
- **Restarts 1-100 against 101-200 (your check): ARI 0.61** (null partition 0.58). The consensus does not depend on which restarts built it. But both halves of this split use the same data, which is why the null partition passes it too. The single-dataset analogue is 0.81.
- **Half A against half B of the data: ARI 0.41** (null 0.18). The halves are independent tokens, so this is the replication that supports claims. It is far above the ~0.03 at which single-dataset partitions agree across datasets; a single dataset reaches 0.66 within itself. Unit confidence correlates 0.86 between data halves.
- **Mean confidence 0.17** (10th-90th percentile 0.04-0.33). A consensus network is a set of units that tend to co-occur across datasets, not a network present intact in any single one.
- GPT-2 gives the same pattern (0.52 and 0.46; null 0.37 and 0.15).

## 3. Item 2: graded stability metrics

Kept as the standard: **co-membership** (if two units share a network in one dataset, how often do they share one in another, against chance) and **label-transfer fidelity** (one dataset's partition, block means refitted on another dataset's half A, scored on its half B, as a share of the target's own partition's advantage over the null partition). Neither treats a split or merged network as a failure.

## 4. Item 3: why the connectome changes across datasets, and the residual connectome

### The question

The ordinary connectome is almost perfectly reproducible within a dataset (r = 0.99) but only half-shared across datasets (r ≈ 0.5), and this is already true in an untrained model. An untrained model has learned nothing, so whatever makes datasets differ there must come from the text itself, not from language processing. The goal was to remove that part and keep a connectome that (a) is reliable within a dataset, (b) is shared across datasets, and (c) needs training to show both.

### The idea: split each activation into what the input predicts and what is left

Take one MLP neuron and one token in context. Its activation can be written as a sum:

**activation = (typical activation for this token) + (shift due to the previous token) + (shift due to the position in the window) + residual**

- The first three terms are averages, each estimated on a separate reference portion of text (pooled over datasets), never on the text being analysed. *Typical activation for a token* is the neuron's mean activation over every occurrence of that token type, for example every occurrence of " the".
- The **residual** is what the neuron does on this particular occasion beyond those averages: its response to the context.
- A connectome is then the correlation, across tokens, between neurons' activations. The **ordinary connectome** correlates the full activations; the **residual connectome** correlates only the residuals.

### Why the ordinary connectome differs across datasets

If every activation is replaced by its token's typical activation, the connectome barely changes: it is essentially a *token-identity connectome*. Two neurons correlate because they respond to the same token types, and how strongly is weighted by how often those token types occur in the text. Different datasets use different words at different rates (wikitext has numbers and markup, Reddit has "I" and informal spelling), so they weight the token types differently and give different connectomes, even through random weights. One check confirms this: reweighting tokens so that every dataset has the same word-frequency profile raises the untrained model's across-dataset agreement from 0.53 to 0.87. After training it only reaches 0.64, so in a trained model something beyond word frequency differs between datasets.

### Result

![residual](fig3_residual.png)

Pythia-70m (12,288 MLP neurons), four prose datasets, 81,920 tokens per half, one run for both panels. Grey = untrained, blue = trained (final checkpoint); plain bars = within a dataset, hatched = across datasets (mean over dataset pairs).

| | untrained, within / across | trained, within / across |
|---|---|---|
| **A. connectome (r)** | | |
| ordinary (control) | 0.99 / 0.53 | 0.99 / 0.56 |
| token removed (intermediate) | 0.90 / 0.68 | 0.99 / 0.70 |
| residual (kept) | 0.74 / 0.58 | **0.98 / 0.81** |
| **B. networks (ARI, k = 100)** | | |
| ordinary (control) | 0.43 / 0.09 | 0.51 / 0.17 |
| token removed (intermediate) | 0.16 / 0.05 | 0.50 / 0.13 |
| residual (kept) | 0.23 / 0.17 | **0.54 / 0.38** |

How to read it:
1. **Ordinary (control).** Perfectly reliable whether or not the model is trained, and half-shared across datasets in both cases. Training makes almost no difference, which is the problem: this connectome mostly reflects the input.
2. **Token removed (intermediate).** Removing the token's typical activation makes the connectome more shared across datasets (0.56 → 0.70). But the untrained model gains just as much (0.53 → 0.68), so this step removes dataset differences without isolating anything learned. Its networks also do not transfer (ARI 0.13).
3. **Residual (kept).** Also removing the previous token and the position changes the picture. The untrained model becomes less reliable (0.74) and less shared (0.58): what was left there was mostly a positional pattern that the random network produces identically for any text. The trained model stays reliable (0.98) and becomes the most shared (0.81). Its networks agree across datasets at ARI 0.38, 70% of their within-dataset agreement, against 34% for ordinary networks. In the untrained model the residual networks are unreliable (0.23). All three criteria (a)-(c) are met.

Over training the residual's across-dataset agreement rises from 0.58 (initialisation) to 0.72 (step 1,000) to 0.81 (final). In Pythia-160m (fewer tokens per half, so lower values overall) the ordering is the same: the trained residual is 0.88 within and 0.65 across, against 0.93 and 0.48 for the ordinary connectome, and the untrained residual is unreliable (0.36 within).

### What remains open

- Even the residual connectome is less shared across datasets (0.81) than within (0.98). Restricting to words common to all datasets, or matching their frequencies, did not close this gap. It may reflect genuine differences in how the model processes different genres.
- Early in training (step 64, the phase where dimensionality collapses), the residual connectome is shared across datasets even more than at the end (0.87). So cross-dataset agreement alone does not show that a connectome reflects language processing; the contrast with the untrained model and the training trajectory are what support that.
- Other components were tested and not kept, because they made the untrained model look as stable as the trained one (removing slow drift within documents) or reflected architecture rather than language (a gain shared by all neurons of a layer). They are documented in `info/LOG.md`, Iteration 38.

## 5. Item 4: codeparrot

Excluded from the consensus and from stability definitions (unreliable even within itself at 2B). It is kept in the other analyses with that caveat; code also has the lowest cross-dataset agreement under every connectome variant.

## 6. Item 5: patching circuits against the consensus networks

![consensus circuits](fig4_consensus_circuits.png)

The consensus labels were written as an ordinary partition (`parcelmate/bin/consensus_tree.py`), so every circuit test ran unchanged. Qwen3.5-2B, 27 tasks. Each panel shows real networks (blue) against the null-partition consensus (grey), next to the four single-dataset sets for comparison.
- **A. Tasks sharing circuit neurons share networks** (non-shared units): 0.46-0.48 against 0.29-0.34.
- **B. Held-out enrichment** (a new task's share in its domain's top-5 networks, against chance), mean over domains: 2.3x against 1.9x. By domain: Language 2.73x (null 2.02x), Physical 2.66x (1.95x), Social 1.69x (1.22x), Formal 2.12x (2.04x).
- **C. Graded test on all units:** 44-48% of tasks explained beyond layer, against 7-22%; real beats null for 78-85% of tasks (p ≤ 0.007).
- **Enriched task × network pairs** (1%): 112-115 against 60-70.
- **The paired concentration test fails** (22-37% of tasks more concentrated on real than null), as it did on the pooled networks.

The consensus networks carry the circuits about as well as single-dataset networks, not better. The null consensus is a stronger competitor than single-dataset null partitions, because averaging many null partitions removes their noise and keeps what they share: neuron-level properties.

## What this suggests

1. **Parcellate the residual connectome.** It is the object that is reliable, shared across datasets, and dependent on training, and its networks transfer across datasets more than twice as well. Next steps:
   - compute it for Qwen3.5-2B (one extra pass per dataset to estimate per-token-type means: about 1 GPU-hour each), then the usual clustering;
   - compare those networks with the patching circuits.
2. **Measure stability with graded metrics** (co-membership, label transfer), never by identical labels.
3. **The consensus of existing partitions is reproducible but adds nothing for the circuits.** It is not worth pursuing further unless built on residual connectomes.
4. **Open:** the remaining within/across gap of the residual connectome (0.98 vs 0.81); whether the 160m effect strengthens with more tokens; the step-64 peak.

## Cluster state (2026-10-08)

- 4B: bookcorpus, agnews and tldr17 are done; codeparrot is clustering, followed automatically by the cross-dataset cohesion analysis (D), scoring, and the 4B circuit comparison.
- The 4B agnews connectivity failed on a 40 GB GPU, which I caught five days late; it was fixed (memory-adaptive tiles) and rerun successfully.
- No other failures.
- Disk: 1.8 TB under Andrea's directory (4B connectivity awaiting purge), 35 TB free on the share.

## Files

| what | where |
|---|---|
| figures | this folder, made by `figures/make_stability_report.py` |
| diagnosis | `scripts/diag_partition_fragility_{1,2}.py`, `figures/diag_partition_*.csv` |
| stability (identical labels, Iteration 36) | `parcelmate/stability.py`, `parcelmate/bin/stability.py`, `results/qwen35/stability_2b{,_prose}/` |
| consensus | `parcelmate/bin/restart_consensus.py`, `figures/consensus_2b_prose_summary.csv` |
| components and residual connectome | `scripts/explore_connectome_components.py`, `figures/explore_connectome_run{1..5}*.csv` |
| consensus vs circuits | `parcelmate/bin/consensus_tree.py`, `results/qwen35/consensus_2b_prose_tree/circuits/` |
| tests | `tests/verify_iter18_stability.py` (15 checks) |
