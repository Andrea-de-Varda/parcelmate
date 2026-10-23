# Are the networks a circuit spans one larger network?

LOG.md Iteration 46, Qwen3.5-2B (4B pending). Circuits found by attribution patching span far fewer networks than random neurons do ([../circuits_vs_networks/](../circuits_vs_networks/)), but still about 10-25 each. This page asks what those networks are to each other.

## In short

- **Not one network.** A circuit's networks couple to each other at about half the strength found within one network (median 0.56, where 1 = one network). No task comes close to 1.
- **More interlinked than other networks, in most domains.** In 128 of 193 tasks, the circuit's networks are more coupled than networks at the same depth (Wilcoxon across tasks, p = 1.5 × 10⁻⁵). This holds almost without exception for factual knowledge, arithmetic, entity tracking and in-context string tasks.
- **Syntax is the exception.** Agreement circuits span networks *less* coupled than chance. Filler-gap and NPI licensing are the opposite, among the most cohesive tasks of any domain.

*Correction to the first version.* It counted only tasks whose own test reached p < 0.05 (53 of 193) and called the rest "no effect". A single task's test is too weak for that (section 2.4); the evidence is the consistency across tasks.

## 1. The question

![concept](concept.png)

**Figure 1.** A circuit (red) spanning three networks. **A.** Distinct networks, weakly coupled to each other. **B.** One larger network cut into three by the choice of 100 networks.

The two readings differ in one quantity: **how strongly the circuit's networks are coupled to each other**. Under B, almost as strongly as within one network.

**Data.** All 147,456 MLP neurons of Qwen3.5-2B. The 100 networks of the pooled residual connectome (Iteration 43: four prose datasets, token and position effects removed, independent halves A and B). The 193 tasks and circuits (top 0.1% of neurons) of the concentration analysis. The connectome is the stored copy from the rich-club stage, which reproduces the original scores to within 4 × 10⁻¹⁰.

## 2. Method

### 2.1 Coupling between two networks

$B_{ij}$ is the mean |r| between a neuron of network i and a neuron of network j. **In words:** how strongly two neurons from these networks correlate on average, ignoring sign. $B_{ii}$ is the same within one network.

Networks differ in how tight they are internally, so coupling is normalised:

$$\tilde B_{ij} = \frac{B_{ij}}{\sqrt{B_{ii}\,B_{jj}}}$$

**In words:** coupling between two networks as a fraction of how tight each is inside. 1 = as coupled as within one network; 0.5 = half as strong.

Coupling is measured on the half the networks were *not* fitted on (half-A networks on half-B data, and the reverse). A partition groups neurons that happen to correlate in its own half, which inflates coupling measured there. The two directions also replicate each other (r = 0.989 across tasks).

![coupling](coupling_2b.png)

**Figure 2.** **A.** Normalised coupling between the 100 networks, ordered by depth. **B.** Coupling falls with the distance in depth between networks (0.58 within 2 layers, 0.45 beyond 6). **C.** A typical pair of real networks couples at 0.49: the networks are distinct. On the null partition (shifted timecourses, no co-activation) pairs couple at 0.97: those networks are interchangeable.

### 2.2 Cohesion of a circuit's networks

With $p_i$ the share of the circuit's neurons in network i:

$$C = \frac{\sum_{i \neq j} p_i\, p_j\, \tilde B_{ij}}{\sum_{i \neq j} p_i\, p_j}$$

**In words:** take two circuit neurons from different networks and look up how coupled their networks are; C is the average over all such pairs. Networks holding more of the circuit contribute more pairs, so one stray neuron cannot dominate (share weighting, Andrea's choice). C is on the same scale: 1 = one network.

### 2.3 The reference: the same profile on other networks

The circuit's share profile is kept exactly and moved to other networks of **similar depth**, 1,000 times (Figure 3). The mean C of these permutations is the reference $C_{\text{null}}$.

- *Why move networks, not draw random neurons.* The question is whether these networks are more of one thing than other networks. Random neurons spread over more networks than the circuit, mixing spread with network identity.
- *Why similar depth.* Coupling depends strongly on depth (Figure 2B). Circuits favour certain layers, so an unconstrained reference would make any layer-concentrated circuit look cohesive. Networks are sorted by mean layer into 10 bins of 10, and a network is only replaced by one from its own bin.

![example](example_2b.png)

**Figure 3.** One task (filler-gap, PP gap). **A.** The circuit's shares by network (top) and after one depth-matched permutation (bottom). **B, C.** Cohesion and specificity of 1,000 permutations (gray) and of the circuit (red).

**Specificity** guards against hub networks, which couple strongly to everything and would raise C without forming a group:

$$S = \frac{C}{\sum_i p_i E_i}, \qquad E_i = \frac{1}{k-1}\sum_{j \neq i} \tilde B_{ij}$$

**In words:** $E_i$ is how strongly network i couples to an average other network. S compares the circuit's networks' coupling to each other with their coupling to everyone. S above its reference means they form a group, not just a set of well-connected networks. It is read against its reference rather than 1, because depth-matched sets already exceed 1 (median 1.065).

### 2.4 Statistics

- **Per task.** The one-sided permutation p is valid (4% false positives on random circuits) but weak. A task's reference varies by 0.024 (SD), more than the typical effect (+0.014), so most real effects cannot reach significance alone. Per-task p is therefore not used to classify tasks.
- **Across tasks.** For each task, d = C − C_null. Per domain, we count tasks with d > 0 and test d against 0 with a Wilcoxon signed-rank test. Tasks from one source share materials, so these p-values summarise consistency rather than give an exact error rate.

**Verification** ([tests/verify_iter46_cohesion.py](../../tests/verify_iter46_cohesion.py), 11 checks): formulas match brute force; coupling matches dense computation; a planted super-network is detected (S 2.67); planted hub networks raise C but give S = 1.00, which is why S is needed.

## 3. Results

![domains](domains_2b.png)

**Figure 4.** Every task, by domain. **A.** Cohesion of the circuit's networks (filled; coloured if above its reference, gray if below) and of the reference (open). Above each domain: tasks above the reference, and the Wilcoxon p. Dashed: a typical pair of networks. **B.** Specificity, same layout.

| Domain | Tasks | Circuit C | Reference | Above reference | p | Specificity above |
|---|---|---|---|---|---|---|
| Factual knowledge | 47 | 0.58 | 0.54 | 46 | < .001 | 47 |
| Arithmetic | 21 | 0.58 | 0.52 | 18 | < .001 | 20 |
| Entity tracking & binding | 18 | 0.54 | 0.53 | 18 | < .001 | 18 |
| In-context string tasks | 20 | 0.58 | 0.57 | 18 | < .001 | 20 |
| Social reasoning | 7 | 0.53 | 0.51 | 5 | .38 | 5 |
| Morphology & lexical | 16 | 0.56 | 0.54 | 8 | .18 | 14 |
| Logic, code & MC | 12 | 0.51 | 0.52 | 4 | .47 | 10 |
| Physical reasoning | 5 | 0.50 | 0.54 | 1 | – | 1 |
| Syntax | 47 | 0.46 | 0.50 | 10 | .002 (below) | 16 |

Medians per domain.

- **Distinct networks.** Cohesion ranges from 0.45 to 0.60 (10th-90th percentile), maximum 0.64. Reading B holds for no task.
- **Four domains are consistently above their reference**, in cohesion and in specificity. These circuits use distinct networks that form a loosely coupled group. The effect is largest for arithmetic (+0.064) and factual knowledge (+0.042), and small for entity tracking (+0.016) and string tasks (+0.010).
- **Syntax is consistently below.** 37 of 47 circuits span networks less coupled than depth-matched ones (median −0.045). This comes from agreement tasks (subject-verb, determiner-noun, reflexive) in every source. The exceptions are filler-gap (+0.095) and NPI licensing (+0.055 to +0.065), among the most cohesive tasks overall; hypernymy plays the same role among lexical tasks (+0.075 to +0.094).
- **The other domains are mixed.**

![controls](controls_2b.png)

**Figure 5.** **A.** d per task on the real networks (coloured) and the null partition (gray). **B.** Cohesion against the effective number of networks.

- **Null partition.** Its d values are tiny (median +0.005). The real effects are much larger where it matters: factual +0.042 vs +0.000, arithmetic +0.064 vs +0.014; overall d is larger on the real networks in 119 of 193 tasks (p = 0.007).
- **Concentration and cohesion are unrelated** (Spearman ρ = −0.01). Agreement circuits are highly concentrated but not cohesive.

## 4. Limits

- **Modest effects.** Domain differences are a few hundredths on a 0-1 scale: consistent, but small, and the most cohesive set (0.64) is far from one network.
- **No per-task verdicts.** Statements about single tasks rest on effect sizes and agreement between related tasks, not on per-task significance.
- **One granularity.** Only the 100-network partition is tested; merging networks into coarser levels was discussed but not run.
- **4B pending.** The 4B coupling job is ready once its rich-club scoring finishes.

## Files

| What | Where |
|---|---|
| Method | [parcelmate/cohesion.py](../../parcelmate/cohesion.py) |
| Coupling matrices (cluster) | [parcelmate/bin/network_coupling.py](../../parcelmate/bin/network_coupling.py), [scripts/launch_coupling.sh](../../scripts/launch_coupling.sh) |
| Per-task test | [parcelmate/bin/circuit_cohesion.py](../../parcelmate/bin/circuit_cohesion.py) |
| Results | [figures/circuit_cohesion_2b.csv](../../figures/circuit_cohesion_2b.csv) |
| Figures | [figures/make_cohesion_figures.py](../../figures/make_cohesion_figures.py) |
| Tests | [tests/verify_iter46_cohesion.py](../../tests/verify_iter46_cohesion.py) |
| Decisions | [info/LOG.md](../../info/LOG.md), Iteration 46 |
