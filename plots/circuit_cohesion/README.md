# Are the networks a circuit spans one larger network?

LOG.md Iteration 46, Qwen3.5-2B (4B pending). Circuits found by attribution patching span far fewer networks than random neurons do ([../circuits_vs_networks/](../circuits_vs_networks/)), but still about 10-25 each. This page asks what those networks are to each other.

## In short

- **Not one network.** A circuit's networks couple to each other at about half the strength found within one network (median 0.56, where 1 = one network). No task comes close to 1.
- **More interlinked than other networks, in most domains.** In 128 of 193 tasks, the circuit's networks are more coupled than networks at the same depth (Wilcoxon across tasks, p = 1.5 × 10⁻⁵). In 151 of 193 they are also a group of their own rather than generally well-connected networks. Both hold almost without exception for factual knowledge, arithmetic, entity tracking and in-context string tasks.
- **Syntax is the exception.** Agreement circuits span networks *less* coupled than chance. Filler-gap and NPI licensing are the opposite, among the most cohesive tasks of any domain.

*Correction to the first version.* It counted only tasks whose own test reached p < 0.05 (53 of 193) and called the rest "no effect". A single task's test is too weak for that (section 2.6); the evidence is the consistency across tasks.

## 1. The question and the data

![concept](concept.png)

**Figure 1.** A circuit (red) spanning three networks. **A.** Distinct networks, weakly coupled to each other. **B.** One larger network cut into three by the choice of 100 networks.

The two readings differ in how strongly the circuit's networks are coupled to each other. Under B, almost as strongly as within one network.

**Data.** All 147,456 MLP neurons of Qwen3.5-2B. The 100 networks of the pooled residual connectome (Iteration 43: four prose datasets, token and position effects removed, independent halves A and B). The 193 tasks and circuits (top 0.1% of neurons) of the concentration analysis. The connectome is the stored copy from the rich-club stage, which reproduces the original scores to within 4 × 10⁻¹⁰.

## 2. Method

### 2.1 Coupling between two networks

$B_{ij}$ is the mean |r| between a neuron of network i and a neuron of network j: how strongly two neurons from these networks correlate on average, ignoring sign. $B_{ii}$ is the same for two neurons of one network. Because networks differ in how tight they are inside, coupling is normalised:

$$\tilde B_{ij} = \frac{B_{ij}}{\sqrt{B_{ii}\,B_{jj}}}$$

This is the coupling between two networks as a fraction of how tight each is inside: 1 = as coupled as within one network, 0.5 = half as strong.

Coupling is measured on the half the networks were *not* fitted on (half-A networks on half-B data, and the reverse). A partition groups neurons that happen to correlate in its own half, which would inflate coupling measured there. The two directions replicate each other (r = 0.989 across tasks).

![coupling](coupling_2b.png)

**Figure 2.** **A.** Normalised coupling between the 100 networks, ordered by depth. **B.** Coupling falls with the distance in depth between networks (0.58 within 2 layers, 0.45 beyond 6). **C.** A typical pair of real networks couples at 0.49: the networks are distinct. On the null partition (shifted timecourses, no co-activation) pairs couple at 0.97: those networks are interchangeable.

### 2.2 First metric: cohesion

**Cohesion asks: how strongly are the circuit's networks coupled to each other?**

With $p_i$ the share of the circuit's neurons in network i:

$$C = \frac{\sum_{i \neq j} p_i\, p_j\, \tilde B_{ij}}{\sum_{i \neq j} p_i\, p_j}$$

Take two of the circuit's neurons that sit in different networks, and look up how coupled their two networks are. C is the average of that over all such pairs. A network that holds more of the circuit contributes more pairs, so a single stray neuron cannot dominate (share weighting, Andrea's choice).

C is on the scale of $\tilde B$, so it answers the question of Figure 1 directly: near 1, the networks behave as one; near a typical pair (0.49), they are as distinct as any two networks.

### 2.3 Second metric: specificity

**Specificity asks: are the circuit's networks coupled to *each other* more than to the rest of the model?**

First, each network's general sociability, $E_i$: its average coupling to all other networks,

$$E_i = \frac{1}{k-1}\sum_{j \neq i} \tilde B_{ij}.$$

The circuit's networks have an average sociability $\sum_i p_i E_i$, weighted by the circuit's shares. This is the cohesion they would show if they were coupled to each other no more than to anyone else. Specificity is the cohesion they actually show divided by that expectation:

$$S = \frac{C}{\sum_i p_i E_i}.$$

S = 1: the circuit's networks are just as coupled to each other as to everything else. S > 1: they are preferentially coupled to each other, a group of their own.

### 2.4 Why two metrics

Each metric alone can mislead.

- **Cohesion alone cannot tell a group from a set of hubs.** Some networks couple strongly to every network (the rich-club stage found such hubs). A circuit that uses several of them has high cohesion even though its networks form no group: they are equally coupled to everything. In the planted test below, six hub networks gave high cohesion but specificity of exactly 1.00. Specificity is what exposes them.
- **Specificity alone cannot say how close to one network the set is.** It is a ratio, so it discards the absolute level. A set of networks that are weakly coupled to everything, but slightly less weakly to each other, has S > 1 while being nowhere near one network. Only cohesion, on its 0-1 scale, answers the question of Figure 1.

So cohesion measures *how tight* the circuit's networks are, and specificity measures *whether that tightness is their own*. A loosely coupled group scores above its reference on both.

### 2.5 The reference: the circuit's profile on networks of similar depth

Each metric is compared with a reference built from the circuit itself. The circuit's share profile (its bar chart of shares over networks) is kept exactly, and each bar is moved to a different network of similar depth. C and S are recomputed. This is repeated 1,000 times, and the means are the references $C_{\text{null}}$ and $S_{\text{null}}$. How concentrated the circuit is never changes; only which networks it sits on.

**How "similar depth" is defined.** Matching is done on networks, not on circuits, and it is approximate.
- Each network's depth is the mean layer of its neurons (a network has 29 to 5,786 neurons spread over a few neighbouring layers, so this is a real number such as 9.4).
- The 100 networks are sorted by depth and cut into 10 bins of 10 consecutive networks.
- A network is only ever replaced by another network from its own bin.

No two networks have exactly the same depth, so each bin spans a range of depths. Six bins are narrow, 0.5 to 1.9 layers (layers 9.0-12.3 and 16.0-19.3, where networks are dense). Four are wider where networks are sparser: layers 0.0-5.8 (5.7 layers), 5.8-8.7 (2.9), 12.4-15.4 (3.0) and 19.4-22.5 (3.1). In practice a network moves by 0.76 layers on average under a permutation (median 0.42; 90% of moves under 2.1 layers). Coupling falls by about 0.02 per layer of distance (Figure 2B), so these moves change coupling by roughly 0.01 on average, less than most of the domain effects below.

- *Why move networks, not draw random neurons.* The question is whether these networks are more of one thing than other networks. Random neurons spread over more networks than the circuit does, mixing spread with network identity.
- *Why match on depth.* Coupling falls with depth distance (Figure 2B), and circuits favour certain layers. An unconstrained reference would put the circuit's profile on networks at all depths, which couple less for that reason alone, and any layer-concentrated circuit would look cohesive.

![example](example_2b.png)

**Figure 3.** One task (filler-gap, PP gap). **A.** The circuit's shares by network (top; dotted lines separate the 10 depth bins) and after one permutation (bottom): each bar moves within its bin. **B.** Cohesion of 1,000 permutations (gray) and of the circuit (red). **C.** The same for specificity.

*Reading specificity.* Networks at similar depth are coupled to each other more than to networks in general, so a depth-matched set already has S above 1 (median 1.065). Specificity is therefore read against its reference, not against 1.

*A limit.* If a circuit covers most networks of a bin, permutations can only reshuffle shares among those same networks, so the reference stays close to the circuit. This makes the comparison conservative for circuits packed into a few layers.

### 2.6 Statistics

- **Per task.** A one-sided permutation p (the share of permutations at least as high as the circuit). It is valid (4% false positives on random circuits) but weak: a task's reference varies by 0.024 (SD), more than the typical effect (+0.014), so most real effects cannot reach significance alone. Per-task p is not used to classify tasks.
- **Across tasks.** For each task, d = C − C_null (and likewise for S). Per domain, we count tasks with d > 0 and test d against 0 with a Wilcoxon signed-rank test. Tasks from one source share materials, so these p-values summarise consistency rather than give an exact error rate.

**Verification** ([tests/verify_iter46_cohesion.py](../../tests/verify_iter46_cohesion.py), 11 checks): both formulas match brute force; coupling matches a dense computation; a planted super-network is detected on both metrics (S 2.67); planted hubs raise C but give S = 1.00 (section 2.4).

## 3. Results

![domains](domains_2b.png)

**Figure 4.** Every task, by domain. **A.** Cohesion of the circuit's networks (filled; coloured if above its reference, gray if below) and of the reference (open). Above each domain: tasks above the reference, and the Wilcoxon p. Dashed: a typical pair of networks. **B.** Specificity, same layout. Dashed: 1.

| Domain | Tasks | Cohesion: circuit / reference | Above | p | Specificity above | p |
|---|---|---|---|---|---|---|
| Factual knowledge | 47 | 0.58 / 0.54 | 46 | < .001 | 47 | < .001 |
| Arithmetic | 21 | 0.58 / 0.52 | 18 | < .001 | 20 | < .001 |
| Entity tracking & binding | 18 | 0.54 / 0.53 | 18 | < .001 | 18 | < .001 |
| In-context string tasks | 20 | 0.58 / 0.57 | 18 | < .001 | 20 | < .001 |
| Social reasoning | 7 | 0.53 / 0.51 | 5 | .38 | 5 | .30 |
| Morphology & lexical | 16 | 0.56 / 0.54 | 8 | .18 | 14 | .004 |
| Logic, code & MC | 12 | 0.51 / 0.52 | 4 | .47 | 10 | .034 |
| Physical reasoning | 5 | 0.50 / 0.54 | 1 | – | 1 | – |
| Syntax | 47 | 0.46 / 0.50 | 10 | .002 (below) | 16 | .20 |

Medians per domain; "above" = tasks above their reference; Wilcoxon p over the domain's tasks.

### 3.1 Cohesion

- **No circuit's networks behave as one network.** Cohesion ranges from 0.45 to 0.60 (10th-90th percentile), maximum 0.64.
- **Most circuits' networks are tighter than depth-matched networks:** 128 of 193 (p = 1.5 × 10⁻⁵), and 153 lie above a typical pair. The consistent domains are factual knowledge, arithmetic, entity tracking and string tasks. The effect is largest for arithmetic (median +0.064) and factual knowledge (+0.042), and small for entity tracking (+0.016) and string tasks (+0.010).
- **Syntax is consistently below.** 37 of 47 circuits span networks less coupled than depth-matched ones (median −0.045). This comes from agreement tasks (subject-verb, determiner-noun, reflexive) in every source. The exceptions are filler-gap (+0.095) and NPI licensing (+0.055 to +0.065), among the most cohesive tasks overall; hypernymy plays the same role among lexical tasks (+0.075 to +0.094).

### 3.2 Specificity

- **The tightness is the networks' own:** 151 of 193 circuits have specificity above their reference (p = 2.3 × 10⁻¹⁵). In the four consistent domains this holds for 105 of 106 tasks, so their cohesion does not come from hub networks.
- **Two domains are specific without being cohesive.** Morphology and lexical tasks (14 of 16) and logic and code tasks (10 of 12) are above their reference in specificity but not in cohesion. Their networks prefer each other over the rest of the model, but are not tighter overall than depth-matched networks. This is the case specificity was added to separate.
- **Syntax is not specific either** (16 of 47 above).

### 3.3 Controls

![controls](controls_2b.png)

**Figure 5.** **A.** Cohesion minus its reference per task, on the real networks (coloured) and the null partition (gray). **B.** Cohesion against the effective number of networks.

- **Null partition.** Its differences are tiny (median +0.005). The real effects are much larger where it matters: factual +0.042 vs +0.000, arithmetic +0.064 vs +0.014. Overall, the difference is larger on the real networks in 119 of 193 tasks (p = 0.007).
- **Concentration and cohesion are unrelated** (Spearman ρ = −0.01). Agreement circuits are highly concentrated but not cohesive.

## 4. Limits

- **Modest effects.** Domain differences are a few hundredths on a 0-1 scale: consistent, but small. The most cohesive set (0.64) is far from one network.
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
