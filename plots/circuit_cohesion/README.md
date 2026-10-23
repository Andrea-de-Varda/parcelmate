# Do the networks a circuit spans prefer each other?

LOG.md Iteration 46, Qwen3.5-2B (4B pending). Circuits found by attribution patching span far fewer networks than random neurons do ([../circuits_vs_networks/](../circuits_vs_networks/)), but still about 10-25 each. This page asks what those networks are to each other.

## In short

- **Not one network.** A circuit's networks are coupled to each other at about half the strength found within one network (median 0.56, where 1 = one network). No task comes close to 1, so the networks a circuit spans are genuinely distinct.
- **But in most domains they prefer each other.** In 147 of 193 tasks, a circuit's networks are more coupled to each other than to other networks at the same depth (median 5% more; Wilcoxon across tasks, p = 8 × 10⁻¹⁶). This holds for every task, or nearly every task, in arithmetic (+10%), in-context string tasks (+8%), entity tracking (+7%) and factual knowledge (+7%).
- **Syntax is the exception.** Most syntax circuits (35 of 47) span networks that are *less* coupled to each other than to networks at the same depth. Agreement tasks drive this. Filler-gap (+31%) and NPI licensing (+9 to +14%) go the other way and are among the strongest preferences of any task.

## 1. The question

![concept](concept.png)

**Figure 1.** A circuit (red) spanning three networks. **A.** Distinct networks, weakly coupled to each other. **B.** One larger network, cut into three by the choice of 100 networks.

A circuit spans many networks. Are those networks pieces of one larger network (B)? And if not, do they at least form a group, more connected to each other than to the rest of the model?

**Data.** All 147,456 MLP neurons of Qwen3.5-2B. The 100 networks of the pooled residual connectome (Iteration 43: four prose datasets, token and position effects removed, independent halves A and B). The 193 tasks and circuits (top 0.1% of neurons) of the concentration analysis. The connectome is the stored copy from the rich-club stage, which reproduces the original network scores to within 4 × 10⁻¹⁰.

## 2. Method

### 2.1 Coupling between two networks

The coupling between networks i and j is the mean |r| between a neuron of i and a neuron of j, normalised by how tight each network is inside:

$$\tilde B_{ij} = \frac{B_{ij}}{\sqrt{B_{ii}\,B_{jj}}}$$

Here $B_{ij}$ is the mean |r| between neurons of the two networks and $B_{ii}$ the mean |r| within network i. On this scale, 1 means the two networks are as coupled to each other as each is inside: effectively one network. 0.5 means half as strong.

Coupling is measured on the half the networks were *not* fitted on: half-A networks on half-B data, and the reverse. A partition groups neurons that happen to correlate in its own half, which would inflate coupling measured there. The two directions serve as replications.

![coupling](coupling_2b.png)

**Figure 2.** **A.** Coupling between the 100 networks, ordered by depth. **B.** Coupling falls with the distance in depth between two networks: 0.58 within 2 layers, 0.45 beyond 6 layers. **C.** A typical pair of real networks couples at 0.49, so the networks are distinct. On the null partition (shifted timecourses, no co-activation) pairs couple at 0.97: those networks are interchangeable.

### 2.2 Is the set one network? Cohesion

With $p_i$ the share of the circuit's neurons in network i, the circuit's **cohesion** is the average coupling between its networks:

$$C = \frac{\sum_{i \neq j} p_i\, p_j\, \tilde B_{ij}}{\sum_{i \neq j} p_i\, p_j}$$

Take two of the circuit's neurons that sit in different networks and look up how coupled their networks are. C is the average of that over all such pairs. A network holding more of the circuit contributes more pairs, so a single stray neuron cannot dominate. C is on the 0-1 scale above, so it answers the first question directly: near 1, the networks behave as one.

Cohesion is descriptive. It is not compared with anything.

### 2.3 Do the networks prefer each other? The preference ratio

**The preference ratio asks whether the circuit's networks are more coupled to each other than to the rest of the model.** It is a single number:

> preference = coupling among the circuit's networks ÷ coupling of the same networks to networks outside the circuit at the same depth

Concretely, every pair of the circuit's networks (i, j) is compared with an equivalent pair in which j is replaced by outsiders. These are the 3 networks outside the circuit whose depth is closest to j's, where depth is the mean layer of a network's neurons. The ratio sums over all pairs, with the same share weights as cohesion:

$$R = \frac{\sum_{i \neq j} p_i\, p_j\, \tilde B_{ij}}{\sum_{i \neq j} p_i\, p_j\, O_{ij}}, \qquad O_{ij} = \text{mean coupling of } i \text{ to the 3 outsiders nearest } j \text{ in depth}.$$

- R = 1: the circuit's networks are no more coupled to each other than to comparable outsiders.
- R = 1.10: they are 10% more coupled to each other.
- R < 1: they are less coupled to each other than to outsiders.

![method](method_2b.png)

**Figure 3.** The ratio for one task (filler-gap, PP gap). **A.** Networks by depth (vertical position is jitter). The circuit's networks are red, sized by share. One pair of circuit networks, i and j, is shown with the comparison: i's coupling to the 3 outsiders nearest j's depth. **B.** Every pair of the circuit's networks: its coupling (y) against its comparison (x), sized by weight. Pairs above the diagonal are more coupled than their comparison. The ratio is the weighted mean of y over the weighted mean of x: 1.32 here.

**Why outsiders at the same depth.** Coupling falls with depth distance (Figure 2B), and circuits are concentrated in certain layers. If the comparison were all outside networks, any circuit packed into a few layers would look like a group, simply because its networks are close in depth. The data show this. Against all outsiders, the ratio correlates strongly with how spread out in depth a circuit is (ρ = −0.72; Figure 5A), and the null partition exceeds 1 in 192 of 193 tasks. Matching each comparison to the partner's depth removes most of this (ρ = −0.22).

**Why this also handles hub networks.** Some networks are strongly coupled to everything. Network i appears in both the numerator and the denominator of each pair. A network coupled at the same level to every network therefore gets the same coupling to its partners as to their outsiders, and scores 1. Networks that are each strongly coupled to everything, and even more to each other, score above 1. That is correct: they are coupled to each other more than to the rest of the model.

**Why 3 outsiders.** One outsider would make the comparison noisy; many would reach far in depth. With 3, an outsider sits a median 0.38 layers from the partner it replaces (90% within 1.4 layers). A circuit touches 44 networks at the median, which leaves 56 outsiders to choose from.

### 2.4 Statistics

The ratio has a natural zero point (1), so no permutation reference is needed. For each task, R is averaged over the two directions; they agree across tasks with r = 0.91. Per domain, we count tasks with R > 1 and test R − 1 against 0 with a two-sided Wilcoxon signed-rank test across tasks. Tasks from one source share materials, so these p-values summarise consistency rather than give an exact error rate. Single tasks are not classified one by one.

**Verification** ([tests/verify_iter46_cohesion.py](../../tests/verify_iter46_cohesion.py), 12 checks, all pass):
- both formulas match brute-force loops;
- on synthetic coupling that depends only on depth, layer-concentrated circuits score R = 1.005, while the ratio against all outsiders is inflated (1.29);
- a planted group scores 1.27 and planted avoidance 0.72;
- hubs coupled equally to every network score 1.000;
- the coupling matrices match a dense computation.

## 3. Results

### 3.1 Not one network

Cohesion has a median of 0.56. Its 10th-90th percentiles are 0.45-0.60 and its maximum is 0.64. Every circuit's networks are coupled to each other at roughly half of within-network strength, so none behaves like one network (reading B of Figure 1 holds for no task).

### 3.2 Most circuits' networks prefer each other; syntax circuits do not

![domains](domains_2b.png)

**Figure 4.** The preference ratio of every task, by domain, sorted. Filled: real networks, coloured when above 1 and gray when below. Open: the same circuit on the null partition. Above each domain: tasks above 1, and the Wilcoxon p.

| Domain | Tasks | Median ratio | Above 1 | p | Null partition (median) |
|---|---|---|---|---|---|
| Arithmetic | 21 | 1.10 | 21 | < .001 | 1.02 |
| In-context string tasks | 20 | 1.08 | 20 | < .001 | 1.01 |
| Entity tracking & binding | 18 | 1.07 | 18 | < .001 | 1.00 |
| Factual knowledge | 47 | 1.07 | 45 | < .001 | 1.03 |
| Morphology & lexical | 16 | 1.03 | 13 | .013 | 1.02 |
| Logic, code & MC | 12 | 1.02 | 11 | .005 | 1.01 |
| Social reasoning | 7 | 1.03 | 5 | .47 | 1.00 |
| Physical reasoning | 5 | 0.99 | 2 | – | 1.01 |
| Syntax | 47 | 0.96 | 12 | .015 (below) | 1.00 |

Over all 193 tasks, the median ratio is 1.05, with 147 tasks above 1 (p = 8 × 10⁻¹⁶).

- **Four domains are consistently above 1.** In arithmetic, in-context string tasks, entity tracking and factual knowledge, essentially every circuit spans networks that are 7-10% more coupled to each other than to equivalent outsiders. Taken together with section 3.1, these circuits use distinct networks that form a loosely connected group.
- **Morphology, lexical semantics and logic/code are weakly above 1** (2-3%, but consistent).
- **Syntax is below 1.**
  - 35 of 47 circuits span networks less coupled to each other than to outsiders (median 0.96).
  - This comes from agreement tasks (subject-verb, determiner-noun, reflexive), and holds in every source that has them. These circuits recruit networks that are, if anything, more separate than usual.
  - The exceptions are long-distance dependencies: filler-gap (1.31, 1.32) and NPI licensing (1.09-1.14, in both CausalGym and LLM_Modularity).
- **Physical and social reasoning** have too few tasks for a conclusion.

### 3.3 Controls

![controls](controls_2b.png)

**Figure 5.** **A.** Why outsiders must be depth-matched. The ratio against all outsiders (gray) rises sharply for circuits concentrated in depth; against depth-matched outsiders (coloured by domain), much less. **B.** The preference ratio against the effective number of networks.

- **Null partition.** The same circuits on the null partition score close to 1 (median 1.010). They sit slightly above 1 in most tasks, probably a small depth residue the 3-nearest comparison does not fully remove. The real networks are higher in 136 of 193 tasks (Wilcoxon p = 1.4 × 10⁻¹⁰). The difference is clear where the effects are: arithmetic 1.10 vs 1.02, factual 1.07 vs 1.03, entity tracking 1.07 vs 1.00, string tasks 1.08 vs 1.01.
- **Concentration and preference are unrelated** (ρ = 0.12). How many networks a circuit spans does not predict whether those networks prefer each other: agreement circuits are concentrated but their networks do not prefer each other.

## 4. Limits

- **Modest effects.** The ratios are a few to ten percent above 1. They are consistent across tasks and halves, and clearly above the null partition, but they describe loose groups, not tight systems.
- **Approximate depth matching.** Outsiders are matched to the nearest available depths, not exact ones. The null partition's small offset above 1 (median 1.010) is the best estimate of what remains.
- **One granularity.** Only the 100-network partition is tested.
- **4B pending.** The 4B coupling job is ready once its rich-club scoring finishes.

**History.** A first version of this page used two metrics (cohesion and a specificity ratio) against permutations of networks within depth bins. It then classified tasks by per-task significance, which was underpowered and led to a wrong conclusion. The single preference ratio replaces both metrics and the permutation reference (LOG.md Iteration 46).

## Files

| What | Where |
|---|---|
| Method | [parcelmate/cohesion.py](../../parcelmate/cohesion.py) |
| Coupling matrices (cluster) | [parcelmate/bin/network_coupling.py](../../parcelmate/bin/network_coupling.py), [scripts/launch_coupling.sh](../../scripts/launch_coupling.sh) |
| Per-task values | [parcelmate/bin/circuit_cohesion.py](../../parcelmate/bin/circuit_cohesion.py) → [figures/circuit_cohesion_2b.csv](../../figures/circuit_cohesion_2b.csv) |
| Figures | [figures/make_cohesion_figures.py](../../figures/make_cohesion_figures.py) |
| Tests | [tests/verify_iter46_cohesion.py](../../tests/verify_iter46_cohesion.py) |
| Decisions | [info/LOG.md](../../info/LOG.md), Iteration 46 |
