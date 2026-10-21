# parcelmate: summary of the approach and the results (2026-10-12)

A short overview of what the repository contains and what it has found. The full record of every decision, run and number is [LOG.md](LOG.md) (Iterations 0-42). The detailed reports are [plots/stability_and_residual/README.md](../plots/stability_and_residual/README.md) and [plots/circuits_vs_networks/README.md](../plots/circuits_vs_networks/README.md).

## 1. The approach: functional connectivity for language models

parcelmate applies resting-state fMRI methods to the neurons of a language model. Each MLP neuron is treated as a voxel, and its activation over tokens of ordinary text is its timecourse. Neurons are correlated with each other to give a connectome, and the connectome is clustered into networks. No task, label or gradient is involved: the networks come only from which neurons co-activate on natural text.

**The pipeline**, selected on GPT-2 by comparing about 55 variants against pre-stated rules (Iterations 1-21):
1. All MLP neurons (post-activation) are the units: 12,288 in Pythia-70m, 36,864 in Pythia-160m, 147,456 in Qwen3.5-2B and 294,912 in Qwen3.5-4B.
2. The connectome is the |r| between all pairs of units, Fisher-averaged over four samples of about 100k tokens.
3. Each unit's profile is Fisher-transformed and z-scored, then sparsified to its top 10% of partners.
4. The profiles are reduced with PCA to 100 components.
5. k-means is run with 200 restarts, the restarts are aligned with Hungarian matching, and k = 100.
6. Every partition is fitted within one dataset: wikitext, bookcorpus, agnews or tldr17 (Reddit). These four prose datasets are the project's datasets. Codeparrot (Python) was run earlier but is excluded because its networks are not reliable (2B within-dataset ARI 0.07); results that include it are historical.

**How the networks are validated.**
- *Reliability:* agreement (ARI) between the partitions of two independent halves of the text.
- *Fidelity:* how well the partition predicts the held-out connectome.
- *Generality:* the same measures across datasets.
- *Null partition:* every measure is read against the same pipeline run on circularly shifted timecourses. That keeps each neuron's own statistics (mean, variance, firing rate, autocorrelation) but destroys co-activation, so anything the real networks show beyond the null comes from co-activation and not from properties of single neurons.

**Infrastructure.**
- Dense GPU correlation for models up to about 37k units.
- An out-of-core path for the Qwen models: fp16 row tiles, about 174 GB per half at 4B, purged after use.
- Pythia checkpoint runs, descriptive training-dynamics measures and attribution patching.
- Verification suites in `tests/`, with one or more per iteration.

**The residual connectome** (Iterations 38-41). The ordinary connectome turned out to be largely a *token-identity connectome*: two neurons correlate because they respond to the same token types, weighted by how often each type occurs in the dataset. That makes it equally reproducible in untrained models, and makes datasets disagree in proportion to their word frequencies. The residual connectome removes this first. For every neuron, activation = a(token type) + b(position) + residual is fitted by joint least squares on separate reference text, and only the residuals are correlated. Neurons that token and position explain exactly are dropped; in Pythia this is the whole first MLP layer, because of its parallel blocks.

## 2. The networks exist, and are mostly dataset-specific

- **Reliable within a dataset.** Final pipeline on GPT-2: split-half ARI 0.71, restart-split ceiling 0.80. Pythia-70m: ARI 0.71-0.80 at every checkpoint. Qwen: ARI 0.58-0.72, against 0.01-0.03 for the null partition.
- **Weakly shared across datasets as hard labels.** Across-dataset ARI is about 0.10-0.22 in every model, and no feature, optimiser, consensus or estimator tried moved it past 0.17 on GPT-2.
- **The low ARI reflects partition fragility more than an absence of shared structure.** Connectomes of different datasets correlate about 0.5, and noise that leaves a connectome correlated 0.5 with the original drops a hard k = 100 partition to the same ARI.
- **Networks stay cohesive elsewhere.** A 4B network fitted on one dataset is still cohesive on another dataset's connectome in 92% of cases (null partition 61%, random 49%), but keeps only about 20% of its own cohesion there.

## 3. Training organises the MLP into sparser, more segregated networks

Pythia-70m and 160m were each analysed at eleven checkpoints, from initialisation to step 143,000, on four prose datasets (Iterations 22-28). Numbers are 70m unless stated; 160m replicates every feature.

| | initialisation | final (step 143,000) |
|---|---|---|
| neuron firing rate (median share of tokens) | 0.50 (160m 0.50) | 0.17 (160m 0.16) |
| connections held by each neuron's top 1% of partners | 0.036 | 0.047 (160m 0.048) |
| held-out modularity of the networks (null partition) | 0.014 (0.000) | 0.029 (0.003); 160m 0.028 (0.004) |
| layers per network (effective number) | 5.9 of 6 | 4.1 (160m 11.8 → 8.2 of 12) |
| token-class coherence of the networks (null) | 0.38 (0.16) | 0.47 (0.23) |

**What changes with training.**
1. **Activation sparsity.** Neurons fire on half of all tokens until step 256, as random pre-activations imply, then on 30% at step 1,000 and 17% at the end. This is the largest single change, and it coincides with the steepest part of the loss curve.
2. **Sparser connectivity.** After a transient phase at step 64, when the whole population co-varies along a few directions (dimensionality falls from 131 to 41 and mean |r| rises), mean |r| falls to about half its initial value. Each neuron's connections concentrate on fewer, stronger partners.
3. **More segregated networks.** Held-out modularity doubles while the null partition's stays near zero. Networks become layer-local and token-selective: early on they prefer punctuation and function words, later content-word starts and then word continuations.
4. **Late convergence.** The final connectome emerges late: similarity to it is 0.01 through step 16, 0.24 at step 4,000 and 0.72 at step 64,000. Partitions reorganise completely at step 64 and only converge from step 16,000.

**What the residual connectome adds** (Pythia-70m, Iteration 42; 160m is running).
- **The ordinary networks are a poor index of learning.** A random model's networks are as reproducible as the trained model's (within ARI 0.78 at initialisation), because they are token-identity networks.
- **The residual networks are built by training.** Their split-half ARI rises from 0.19 untrained to 0.76 trained, and their across-dataset ARI from 0.03 to 0.19. Their connectome's split-half similarity rises from 0.76 to 0.99.
- **Segregation is stronger on the residual connectome:**
  - modularity triples (0.008 → 0.024);
  - concentration on the top 1% of partners rises more (0.037 → 0.053);
  - networks become strongly layer-local: 4.6 → 2.8 effective layers of 5, against 4.4 for the null partition.
- **Token-class coherence is learned.** It starts at the null partition's level and rises above it (0.38 against 0.21).
- **Timing.** On the residual connectome, segregation tracks the loss and the sparsification closely (steps 256-4,000). On the ordinary connectome, modularity lags the loss.

**How far this supports "sparser networks through training".**
- **Supported.** Three effects hold at both sizes and against the null partition:
  - the MLP becomes activation-sparse;
  - each neuron's coupling concentrates on fewer partners;
  - the population divides into more modular, more layer-local and more token-selective networks.
  - On the residual connectome, the networks themselves only become reproducible through training.
- **Caveats.**
  1. All of this is correlational and comes from two small models of one family.
  2. Absolute modularity is small (0.03), although it is about ten times the null partition's.
  3. Dimensionality rises overall but falls between steps 64,000 and 143,000 (368 to 260), which is unexplained.
  4. The residual analysis is complete only for 70m, and its layer counts have a ceiling of 5 against 6 (layer 0 is dropped), so each is read against its own null partition.

## 4. Do the networks match circuits found by attribution patching?

Circuits were defined on Qwen3.5-2B and 4B as the top 0.1% of neurons by attribution, for 27 and 38 minimal-pair tasks in four domains: language, formal reasoning, social reasoning and physical reasoning (Iterations 23, 30-36, 40, 42).
- **Circuits are not networks.** A circuit spreads over about 45 of the 100 networks, against about 50 for layer-matched random neurons.
- **The networks carry the circuits' organisation.** A left-out task puts its circuit into the networks its sibling tasks use most, at 2.0-2.8 times chance on every dataset and both sizes; the null partition reaches 1.2-1.6. Tasks that share circuit neurons send their *other* neurons to similar networks: ρ 0.26-0.46 at 2B and 0.53-0.69 at 4B, against lower values for the null partition.
- **Social reasoning is the weakest domain everywhere.** Its tasks share little circuitry even on the patching side.
- **Residual networks match better on half the data.** At 2B, networks from the residual connectome raise held-out enrichment to 4.5 and 3.9 times chance on wikitext and tldr17, against 2.2 and 2.6 for ordinary networks. On agnews and bookcorpus the results are mixed.

## 5. Status and open items (2026-10-12)

- Pythia-160m residual: 7 of 11 checkpoints are done; the other 4 were resubmitted at 128 GB after running out of memory.
- Qwen3.5-4B scoring on four datasets was resubmitted with 96 h after a timeout. Its 1.3 TB of connectivity is purged when it ends.
- The paired real-against-null concentration test has not been run for the four newer 4B network sets.
- The residual connectome was not run on Qwen3.5-4B.
- Open questions:
  - what keeps the trained residual connectome dataset-specific (across-dataset r 0.63 against 0.95 within);
  - whether k should scale with the number of units;
  - why dimensionality falls late in training.
