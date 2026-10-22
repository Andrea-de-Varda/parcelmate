# Are the networks a circuit spans one larger network?

This page reports LOG.md Iteration 46 for Qwen3.5-2B (4B pending). It follows the per-task concentration result ([../circuits_vs_networks/](../circuits_vs_networks/), Iteration 44): circuits found by attribution patching fall into far fewer networks than random sets of neurons do. A typical circuit still spans many networks, about 25 at the median and 10-15 for the most concentrated tasks. This page asks what those networks are to each other.

## In short

The networks a circuit spans are distinct networks, not pieces of one larger network that our choice of 100 networks has cut apart. Between the networks a circuit spans, coupling is on average about half as strong as within one network, much like any pair of networks. No task comes close to "one network".

Compared with other networks at the same depth in the model, the networks of most circuits (140 of 193) are no more coupled to each other than chance. Where they are, the domain matters:

- Arithmetic and factual-knowledge circuits sit in sets of networks that are somewhat more interlinked than layer-matched networks.
- Agreement circuits (subject-verb, determiner-noun, reflexives) do the opposite. Their networks are less coupled to each other than layer-matched networks are, and less than a typical pair of networks. These tasks recruit several separate networks.
- Filler-gap dependencies, NPI licensing and hypernymy are the clearest exceptions in language: their networks are among the most cohesive of any task.

How few networks a circuit spans says nothing about how coupled those networks are (Spearman ρ = -0.01 across tasks).

## 1. The question

A circuit is the top 0.1% of a task's neurons by attribution (147 of the model's 147,456 MLP neurons). Each neuron belongs to one of 100 networks fitted on task-free text, so a circuit has a share in each network. Its effective number of networks summarises how spread those shares are. Circuits score far below random neuron sets, but a language circuit still spans about 10 to 25 networks. Two readings of that number are possible (Figure 1).

![concept](concept.png)

**Figure 1.** Two ways a circuit (red neurons) can span three networks (circles). **A.** The networks are distinct: neurons are coupled within a network and only weakly across networks. The task recruits several separate systems together. **B.** The three networks are strongly coupled to each other and behave as one larger network (dashed box). Fitting 100 networks has cut a single system into three pieces, and "three networks" is an artefact of granularity.

The two readings make different predictions about one quantity: **how strongly the circuit's networks are coupled to each other**. Under B they are coupled almost as strongly as neurons within one network. Under A they are coupled like any two networks, or less.

## 2. Data

- **Model and neurons.** Qwen3.5-2B, all 24 × 6,144 = 147,456 MLP neurons (post-nonlinearity).
- **Networks.** The 100 networks of the pooled residual connectome (Iteration 43): four prose datasets (wikitext, bookcorpus, agnews, tldr17); activation minus a token-type and position fit; two halves of independent text (half A, half B), each fitted to its own partition. The networks are the same ones used in the concentration result.
- **Connectome.** The stored connectome of the rich-club stage (Iteration 45), which recomputed the same tiles and keeps them. Before use, the 36 stored network scores were recomputed on these tiles: they agree to within 3.9 × 10⁻¹⁰, so the connectome is identical to the one the networks were fitted on.
- **Tasks and circuits.** The 193 tasks that passed the inclusion rule at 2B (164 from the task bank, 29 from LLM_Modularity), with the circuits of the concentration analysis (same attribution, same top-0.1% rule).

## 3. Method

### 3.1 Coupling between two networks

The coupling between networks i and j is the mean |r| over all pairs of a neuron in i and a neuron in j. Within a network (i = j), it is the mean over its distinct pairs.

*Why |r|.* The networks were fitted on |r|, and a strong negative correlation is as strong a coupling as a positive one (the project's convention since 2026-10-20).

Raw coupling is small and differs between networks: median within-network coupling is 0.032, between-network 0.015. To make "as tight as one network" a fixed reference point, coupling is **normalised by within-network coupling**:

$$\tilde B_{ij} = \frac{B_{ij}}{\sqrt{B_{ii}\,B_{jj}}}.$$

$\tilde B = 1$ means two networks are as tightly coupled as each is internally, the signature of reading B. Networks also differ in size (29 to 5,786 neurons, median 1,190). Normalising per pair keeps a large network from dominating, since each coupling is already a mean over pairs.

*Why the other half.* The partition of half A is evaluated on the connectome of half B, and the partition of half B on half A. A partition fitted on one half groups together neurons that happen to correlate in that half, so its within-network coupling measured on the same half is inflated by fitting noise. Measured on the other half, the coupling is an out-of-sample estimate. The inflation turned out small (median within-network coupling 0.0334 on the fitting half, 0.0324 on the other), but the cross-half estimate is correct by construction. The two directions serve as a replication of each other.

![coupling](coupling_2b.png)

**Figure 2.** Coupling between the 100 networks (half-A networks on half-B connectome). **A.** Normalised coupling, networks ordered by their mean layer. **B.** Coupling falls with the distance in depth between two networks (each gray dot is a pair; black: mean per 2-layer bin). Networks within 2 layers of each other couple at 0.58 on average; networks more than 6 layers apart at 0.45. **C.** Distribution over all pairs. On the real networks a typical pair couples at 0.49 of within-network strength: the networks are distinct. On the null partition (same pipeline on circularly shifted timecourses, which keep each neuron's statistics but destroy co-activation) pairs couple at 0.97: those "networks" are interchangeable, as expected.

### 3.2 How cohesive are a circuit's networks?

Let $p_i$ be the share of the circuit's neurons in network i (summing to 1). The circuit's **cohesion** is the share-weighted mean coupling over pairs of distinct networks it touches:

$$C = \frac{\sum_{i \neq j} p_i\, p_j\, \tilde B_{ij}}{\sum_{i \neq j} p_i\, p_j}.$$

*Why weight by shares (Andrea, 2026-10-21).* A circuit with 40% of its neurons in one network and one neuron each in five others is mostly about the first network. Weighting pairs by $p_i p_j$ lets networks count in proportion to how much of the circuit they hold, so a stray neuron cannot dominate. Counting every touched network equally was the alternative.

### 3.3 The null: the same profile, given to other networks

The question is whether these networks are more of one thing than other networks would be. The null therefore keeps the circuit's share profile exactly (same number of networks, same weights) and assigns it to other networks. That keeps how concentrated the circuit is fixed and changes only which networks it is on.

*Why not random neurons (Andrea, 2026-10-20).* The concentration test compared circuits with random neuron sets. Reused here, that null would ask whether the circuit's neurons pick more coupled networks than random neurons land in. That mixes the circuit's spread with the identity of its networks. Permuting networks isolates identity.

*Why match on layer (Andrea, 2026-10-21).* Networks are layer-local, and coupling depends strongly on depth (Figure 2B, Spearman ρ = -0.45 between layer distance and coupling). Circuits favour particular layers. An unconstrained null would hand the circuit's profile to networks spread across all depths, which couple less for that reason alone. Any circuit concentrated in a few layers would then look cohesive. Each network's depth is its neurons' mean layer. The 100 networks are cut into 10 bins of 10 networks by depth, and identities are permuted only within bins: a network is replaced by another network at a similar depth. There are 1,000 permutations per task and direction.

The one-sided p is the share of permutations whose cohesion is at least the circuit's, counting the circuit itself: p = (1 + #{null ≥ observed}) / 1,001.

![example](example_2b.png)

**Figure 3.** The test for one task (filler-gap, PP gap; half-A networks, half-B coupling). **A.** Above the axis, the circuit's share in each network (networks ordered by depth; dotted lines separate the 10 depth bins). Below, the same profile after one layer-matched permutation: the bars move to other networks of the same bins. **B.** Cohesion of 1,000 permutations (gray) and of the circuit (red). **C.** The same for specificity (section 3.4).

*A limit of this null.* A bin holds 10 networks. If a circuit covers most networks of a bin, permutations can only reshuffle its shares among those same networks. The null is then close to the circuit itself, which makes the test conservative for circuits packed into a few layers.

### 3.4 Specificity: a set of hubs is not one network

Some networks are coupled strongly to every other network (the rich-club stage found strongly coupled hubs). A circuit that happens to use several such hub networks would have high cohesion without its networks forming a group of their own. **Specificity** compares the coupling among the circuit's networks with their coupling to networks in general:

$$S = \frac{C}{\sum_i p_i E_i}, \qquad E_i = \frac{1}{k-1}\sum_{j \neq i} \tilde B_{ij}.$$

S > 1 means the circuit's networks are coupled to each other more than to the other networks.

*Why this control (Andrea, 2026-10-21).* The alternative was to match replacement networks on their overall coupling strength as well as on depth. The ratio is simpler and reads directly. Specificity is tested against the same layer-matched permutations. Section 4.3 explains why it must be read against that null rather than against 1.

### 3.5 Verification

[tests/verify_iter46_cohesion.py](../../tests/verify_iter46_cohesion.py), 11 checks, all passing:

- cohesion and specificity equal brute-force double loops;
- within-bin permutations keep every network in its bin;
- the coupling matrices computed from tiled float16 halves equal dense block means of |r|;
- with no planted structure, p < 0.05 in 4.0% of 300 random circuits (the nominal 5%);
- a planted super-network (six networks coupled at 0.7 against a background of 0.2) is detected: p = 0.009, specificity 2.67;
- **six hub networks coupled to everyone also beat the layer-matched null on cohesion (p = 0.002), but their specificity is 1.00.** Cohesion alone cannot separate a super-network from a set of hubs; specificity can.

### 3.6 Null partition

The whole analysis is repeated on the null partition (networks fitted on circularly shifted timecourses). There, coupling between networks is near 1 everywhere (Figure 2C), so cohesion should carry no structure. This checks that the procedure does not manufacture an effect.

## 4. Results

The two directions (half-A networks on half-B coupling, and the reverse) agree almost perfectly: r = 0.989 between the two cohesion values across tasks. Values below are means of the two directions. A task counts as above the null when p < 0.05 in both directions.

### 4.1 The circuit's networks are distinct networks

Cohesion has a median of 0.56 across the 193 tasks. The 10th to 90th percentiles span 0.45 to 0.60, and the maximum is 0.64. One network would score 1.0, and a typical pair of networks 0.49. Every circuit's networks are coupled to each other at roughly half of within-network strength, so none of them behaves like a single network. Reading B of Figure 1 does not hold for any task.

### 4.2 Compared with networks at the same depth

![domains](domains_2b.png)

**Figure 4.** Every task, grouped by domain and sorted by cohesion. **A.** Cohesion of the circuit's networks (filled) and of the layer-matched networks (open, mean of 1,000 permutations). Coloured: p < 0.05 in both directions. Dashed: a typical pair of networks (0.49). **B.** Specificity, same layout. Dashed: 1.

Against layer-matched networks, 53 of 193 tasks are more cohesive (p < 0.05 in both directions). Medians are 0.56 for the circuits against 0.52 for the null.

| Domain | Tasks | Cohesion: circuit | Cohesion: layer-matched | Above null |
|---|---|---|---|---|
| Syntax | 47 | 0.46 | 0.50 | 15% |
| Morphology & lexical semantics | 16 | 0.56 | 0.54 | 19% |
| Arithmetic | 21 | 0.58 | 0.52 | **81%** |
| Logic, code & multiple choice | 12 | 0.51 | 0.52 | 8% |
| Factual knowledge | 47 | 0.58 | 0.54 | **51%** |
| Entity tracking & binding | 18 | 0.54 | 0.53 | 6% |
| In-context string tasks | 20 | 0.58 | 0.57 | 0% |
| Physical reasoning | 5 | 0.50 | 0.54 | 0% |
| Social reasoning | 7 | 0.53 | 0.51 | 0% |

Medians per domain; "above null" = share of tasks with p < 0.05 in both directions.

**Syntax splits in two.**

- **Agreement tasks are below the null.** Subject-verb, determiner-noun and reflexive agreement, from every source (CausalGym, CLAMS, Newman et al., Marks et al., EAP-IG, LLM_Modularity), span networks that are less coupled to each other than layer-matched networks (by about 0.05; reflexives by less, 0.01-0.03). 36 of the 47 syntax tasks fall below even a typical pair of networks. These circuits recruit several separate networks.
- **Long-distance licensing is cohesive.** Filler-gap dependencies (+0.095) and NPI licensing (+0.055 to +0.065, both in CausalGym and LLM_Modularity) are among the most cohesive tasks of any domain.

Hypernymy (three versions from two sources) is the counterpart within the lexical tasks: +0.075 to +0.094.

**Arithmetic and factual knowledge** are the two domains where cohesive sets of networks are the rule rather than the exception.

### 4.3 Specificity

Specificity ranges from 0.96 to 1.25 (median 1.10). It is above its layer-matched null in 66 tasks (p < 0.05 in both directions), with the same domain pattern as cohesion (Figure 4B).

**Specificity must be read against its null, not against 1.** The layer-matched networks themselves have a specificity of 1.065 at the median. Networks at a similar depth are coupled to each other more than to networks in general (Figure 2B), so any depth-matched set exceeds 1. The planted test had no depth structure, so 1 was the right reference there. In the model, the right reference is the null.

### 4.4 Controls

![controls](controls_2b.png)

**Figure 5.** **A.** Cohesion minus its layer-matched null for every task, on the real networks (coloured) and on the null partition (gray); bars are medians. **B.** Cohesion against the effective number of networks (the concentration measure), one dot per task, coloured by domain.

**The null partition.** On the null partition, 53 tasks also reach p < 0.05 in both directions, but only 16 of them are among the 53 on the real networks. Their effects are tiny: the median difference from the layer-matched null is +0.005, against +0.014 on the real networks, whose differences range from -0.07 to +0.10 (Figure 5A). Coupling on the null partition barely varies (Figure 2C), so even minute, consistent differences reach significance there. Two consequences follow:

- p-values alone can mislead in this analysis; effect sizes belong beside them.
- Arithmetic is "cohesive" on the null partition too (86% of its tasks), though by small amounts. That fits the concentration result, where arithmetic circuits concentrated on the null partition as well: part of arithmetic's structure reflects which kinds of neurons are recruited (their own statistics), not co-activation alone.

**Concentration and cohesion are unrelated.** Spearman ρ = -0.01 between a task's effective number of networks and the cohesion of those networks (Figure 5B). The most concentrated circuits (in-context string tasks, about 12-15 networks) are not the most cohesive relative to depth, and the agreement circuits combine high concentration with low cohesion.

## 5. What this shows, and what it does not

**It shows that:**

- the effective number of networks a circuit spans counts distinct networks: their mutual coupling is about half the within-network level, like any pair of networks, so the count is not an artefact of fitting 100 networks;
- relative to networks at the same depth, most circuits use networks that are no more interlinked than chance;
- agreement circuits use networks that are less interlinked than chance, while filler-gap, NPI and hypernymy circuits, arithmetic circuits and about half of the factual circuits use networks that are more interlinked.

**It does not show that:**

- **the networks of cohesive circuits form a separate system.** Even the most cohesive circuits (0.64) are far from 1.
- **the differences are large.** The domain differences in cohesion are a few hundredths on a 0-1 scale, consistent across the two directions and in a direction the null partition does not share, but modest.
- **anything about another granularity.** The analysis uses the 100-network partition. It asks whether its networks behave as one; it does not refit coarser partitions. A granularity curve (merging networks by their coupling and recomputing concentration at each level) was discussed as a complement but not run.
- **the same at 4B.** The 4B coupling job is ready to run once the 4B rich-club scoring finishes.

**Open decision.** Given section 4.4, should a task count as cohesive only when its effect also exceeds a minimum size? An example is cohesion at least 0.02 above its null in both directions, about the upper end of the null partition's effects.

## 6. Files

| What | Where |
|---|---|
| Method (cohesion, specificity, null) | [parcelmate/cohesion.py](../../parcelmate/cohesion.py) |
| Coupling matrices on the stored tiles (cluster) | [parcelmate/bin/network_coupling.py](../../parcelmate/bin/network_coupling.py), [scripts/launch_coupling.sh](../../scripts/launch_coupling.sh) |
| Per-task test (local) | [parcelmate/bin/circuit_cohesion.py](../../parcelmate/bin/circuit_cohesion.py) |
| Results, one row per task × tree × direction | [figures/circuit_cohesion_2b.csv](../../figures/circuit_cohesion_2b.csv) |
| Figures on this page | [figures/make_cohesion_figures.py](../../figures/make_cohesion_figures.py) |
| Verification | [tests/verify_iter46_cohesion.py](../../tests/verify_iter46_cohesion.py) |
| Coupling matrices | `results/richclub/qwen3.5-2b-resid-pool4-tiles{,_null}/metrics/coupling/coupling.npz` |
| Record of decisions | [info/LOG.md](../../info/LOG.md), Iteration 46 |
