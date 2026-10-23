# Rich club in the functional connectome of language models

This page reports the first stage of the network-neuroscience analyses (LOG.md Iteration 45; plan in [info/RICH_CLUB_PLAN.md](../../info/RICH_CLUB_PLAN.md)). It asks whether the most strongly coupled MLP neurons of a language model are coupled mainly to each other. Pythia-70m is the pilot; Qwen3.5-2B and Qwen3.5-4B are the main models.

## In short

All three models have a rich club: their most strongly coupled neurons are more coupled to each other than their overall coupling predicts. This replicates across two independent halves of the data. It also survives a null in which every neuron keeps its own statistics but co-activation is destroyed. That null has a sizeable club of its own, so the real excess over it is what counts. The excess is large in Pythia-70m, modest in Qwen3.5-2B (1-12%, depending on the club's size) and somewhat larger in Qwen3.5-4B (1-17%).

The models build their club differently. In Pythia-70m the extra coupling is between hubs in the same layer: across layers, the real hubs couple no more than the null's. In both Qwen models the club also holds across layers, and it spreads over most of the networks.

The models also differ in how hubs spread their strongest links over the networks. In the Qwen models, the more strongly coupled a unit is, the more networks its links reach: the top hubs come close to placing their links without regard to network, while weaker units keep theirs within fewer networks. In Pythia-70m every unit's links are concentrated in few networks, and the hubs most of all.

## 1. The question

Picture the network formed by a model's MLP neurons. Each neuron is a node, and two neurons are linked by how strongly their activations correlate over text. Some neurons are linked strongly to many others. We call these neurons **hubs**. A system with hubs can be organised in two broad ways, shown in Figure 1.

![concept](concept.png)

**Figure 1.** Two networks with the same hubs (black). **A.** Each hub anchors its own group of neurons, and the hubs barely link to each other. Communication between groups goes through ordinary neurons. **B.** The hubs are densely linked to each other and form a core through which the groups can communicate.

If the hubs are densely interconnected, as in B, they form a **rich club**. The term comes from network science (Colizza et al. 2006): the "rich" (well-connected) nodes associate with each other. In the human brain, the best-connected regions form such a club in the anatomical wiring (van den Heuvel and Sporns 2011). A club also appears in resting-state functional connectivity, though there its members connect mainly within their own systems (Grayson et al. 2014). The question here is whether LLM neurons show the same organisation.

There is one subtlety. Hubs have strong links to many neurons, so they will also have fairly strong links to each other, purely because they have strong links to everyone. The question is therefore not whether hubs are linked, but whether they are linked *more than their overall linkage predicts*. The two hypotheses are:

- **H1 (rich club):** hubs are more strongly coupled to each other than their overall coupling predicts.
- **H2 (no rich club):** hubs are coupled to each other only as much as their overall coupling predicts, or less.

Section 3 makes "predicts" precise.

## 2. The connectome

Every MLP neuron (post-nonlinearity) is a node, called a *unit* below. Units whose activation is fully explained by the current token and its position carry no information beyond that and are left out. Pythia-70m loses almost all of its first MLP layer this way: its parallel attention and MLP blocks make layer 0 a function of the token alone. The remaining units are called **live**.

| Model | Layers × neurons | Units | Live units |
|---|---|---|---|
| Pythia-70m (final checkpoint) | 6 × 2,048 | 12,288 | 10,264 |
| Qwen3.5-2B | 24 × 6,144 | 147,456 | 147,456 |
| Qwen3.5-4B | 32 × 9,216 | 294,912 | 294,912 |

The links come from the **pooled residual connectome** (LOG.md Iteration 43), the same connectome the networks compared with patching circuits were fitted on. Each unit's activation is recorded over text from four prose datasets: wikitext, bookcorpus, agnews and tldr17 (Reddit).

Two neurons can correlate simply because both respond to the same frequent tokens. That co-activation reflects token identity rather than the neurons working together. To remove it, a joint least-squares fit of activation = a(token type) + b(position) is estimated on separate reference text and subtracted from every activation. Only the remaining residual co-activation is correlated.

Each dataset contributes two samples of 102,400 tokens, and each sample is correlated on its own. **Half A** combines the first sample of each dataset: tanh of the mean of the arctanh-transformed correlations (the standard way to average correlations). **Half B** combines the second samples the same way. The two halves share no text, so every result can be measured twice, independently. The weight of the link between units i and j is the absolute correlation $w_{ij} = |r_{ij}|$: a strong negative correlation is as strong a coupling as a positive one. The transformations applied before clustering networks (Fisher transform, z-scoring, keeping each unit's top 10% of partners) are not used here; the graph is the connectome itself.

## 3. Method

### 3.1 How strongly is a unit coupled?

A hub is a unit that is strongly coupled overall. There are two standard ways to measure this, shown in Figure 2A.

![definitions](definitions.png)

**Figure 2.** **A.** One unit and its links, with weights. Its **strength** is the sum of all its link weights. Its **degree** is the number of links that pass a threshold. **B.** The participation coefficient (section 3.7) for two units with the same eight links. On the left all partners are in one network; on the right they are spread evenly over four.

The **strength** of unit $i$ is the sum of its link weights:

$$s_i = \sum_{j \neq i} w_{ij}.$$

It counts every link, weak or strong, so a unit with many moderate correlations can be as strong as one with a few large ones.

The **degree** of unit $i$ is the number of its links above a threshold $t$: $k_i = \#\{j : w_{ij} \geq t\}$. Weak links are ignored. The threshold is set by **density**, the fraction of all possible links that are kept. At a density of 1%, $t$ is the correlation exceeded by exactly 1% of all unit pairs.

Strength is the primary measure. Degree at densities of 0.1%, 1% and 5% is a robustness check: it asks whether the result depends on counting weak links.

### 3.2 Hubs and clubs

Units are ranked by strength. The **club of size $n$**, written $R(n)$, is the set of the $n$ strongest units; its members are the **hubs**.

There is no natural cut-off for how strong a unit must be to count as a hub. So the analysis is repeated for 20 club sizes, log-spaced from 50 units up to one fifth of the live units. A rich club is a statement about how coupling among the top units changes as the cut-off becomes stricter.

### 3.3 How much coupling should hubs have by chance?

To test whether hubs are coupled to each other *more than their strength predicts*, we need that prediction. Imagine each unit kept its strength but spread it over partners at random, giving each partner a share proportional to the partner's own strength. A unit $j$ then receives a fraction $s_j / 2W$ of unit $i$'s strength, where $2W = \sum_k s_k$ is the total strength of the network. The expected weight between $i$ and $j$ is

$$E[w_{ij}] = \frac{s_i\, s_j}{2W}.$$

We call this the **strength baseline**. It is what "predicted by strength" means throughout this page. In matrix terms, with the |r| matrix and its diagonal set to zero, it is (sum of row $i$) × (sum of column $j$) / (sum of the whole matrix). That is the same expectation as for a cell of a contingency table under independence.

It already expects hubs to be well linked to each other, because $s_i s_j$ is large when both units are strong. It rules out the trivial club. If every correlation were produced by one shared factor ($r_{ij} = a_i a_j$), the strongest units would correlate most with each other, but the baseline predicts this exactly. The verification suite confirms it: ρ = 1.002 for such a matrix.

The club as a whole is measured by its **rich-club coefficient**, the total weight among its members divided by the total the baseline expects:

$$\rho(n) = \frac{\sum_{i<j \,\in\, R(n)} w_{ij}}{\sum_{i<j \,\in\, R(n)} s_i s_j / 2W}.$$

The numerator adds up the observed weight of every link between two club members; the denominator adds up what the baseline expects for the same links. ρ = 1 means the hubs are coupled to each other exactly as much as their strengths predict (H2). ρ = 1.5 means 50% more (H1).

**Where this coefficient comes from.** It is the weighted rich-club coefficient of Serrano (2008). There, nodes are selected by strength, and the weight among them is normalised by its expectation under a null that keeps every node's strength. In the uncorrelated limit that expectation is given analytically as proportional to $s_i s_j$:

$$E[w_{ij}] = \frac{s_i s_j}{2W - \langle s^2\rangle / \langle s\rangle}.$$

Our baseline differs only by the small correction $\langle s^2\rangle/\langle s\rangle$ in the denominator, which accounts for excluding self-links and is of order $1/N$ of the total at our sizes. The analytic form, rather than averaging over randomly rewired networks as most brain studies do (e.g. van den Heuvel and Sporns 2011, 1,000 degree-preserving rewirings), is the appropriate choice for our connectome. Zlatic et al. (2009) show that degree-preserving rewiring fails on dense graphs. In a complete graph no link can be swapped, so the "random" networks are the original one and ρ = 1 whether or not a club exists. Our connectome is complete: every pair of units has a weight. Alstott et al. (2014) review the weighted rich-club coefficients and warn that many studies mixed the contributions of topology and weights by using controls that shuffle both. A complete graph has no topology to shuffle, so ρ here is a purely weighted rich club. The hub coefficient, the cross-half design and the circular-shift null below are our additions; none of them changes the coefficient.

The same ratio for a single hub is its **hub coefficient**:

$$c_i(n) = \frac{\sum_{j \,\in\, R(n),\, j\neq i} w_{ij}}{\sum_{j \,\in\, R(n),\, j\neq i} s_i s_j / 2W}.$$

This is hub $i$'s observed weight to the other club members over the weight the baseline expects. For example, suppose hub $i$'s links to the other 99 hubs of a 100-unit club sum to 12, while the baseline expects 8. Then $c_i = 1.5$: this hub gives the other hubs 50% more coupling than its strength and theirs predict. ρ summarises the whole club; the $c_i$ give one value per hub and are what the statistical test in section 3.6 is applied to.

### 3.4 Choosing hubs on one half, measuring them on the other

Choosing the strongest units and measuring their mutual coupling on the same data is biased. Suppose some units happen to share variance in one sample only, for example because they all respond to one document in it. They become strong in that sample precisely because they co-vary there, and their mutual coupling is then inflated in the same data.

The fix is to choose the hubs on half A and measure every coupling, and the baseline, on half B (A→B), then the reverse (B→A). Variance shared within one half cannot carry over to the other. In the verification suite, a planted case of this kind gives ρ = 3.8 when hubs are chosen and measured on the same half, and ρ = 1.00 across halves. The two directions are two independent replications of the whole analysis.

### 3.5 A second null: units that vary slowly

The strength baseline accounts for each unit's strength, but not for other properties of individual units that could create a club by chance. The most important is how slowly a unit's activation changes from token to token (its **autocorrelation**). The correlation between two unrelated but slowly varying signals fluctuates more widely than that between two fast ones: a slow signal has fewer independent stretches, so chance alignments do not average out. Because the weights are absolute correlations, wider fluctuations mean larger weights. And the inflation is largest when *both* units are slow. Slow units therefore end up coupled to each other more than their strengths predict, which looks like a rich club made only of slow units.

The **circular-shift null** controls for this. Each unit's activation timecourse is shifted in time by a random offset, with the end wrapped round to the start. This keeps each unit's mean, variance and autocorrelation, but breaks any alignment between units, so it destroys co-activation. The null connectome is built with exactly the same pipeline, and the whole analysis is repeated on it. A real club must exceed the strength baseline and also the club this null produces.

Figure 3 puts sections 3.2 to 3.5 together.

![method](method.png)

**Figure 3.** **A.** Units are ranked by strength on half A, and the club is the top $n$. Its members' links are then read on half B and compared with the strength baseline for the same pairs. **B.** The club-level (ρ) and hub-level ($c_i$) ratios. **C.** The circular-shift null: each timecourse is shifted by a random offset, which keeps its own statistics and destroys co-activation. The whole analysis is repeated on the result.

### 3.6 How H1 is tested

The test is run separately for each club size $n$ and each direction (hubs chosen on A and measured on B, and the reverse). For a given size and direction there are three checks:

1. **The club beats the strength baseline:** ρ(n) > 1.
2. **The real hubs beat the null's hubs:** the hub coefficients $c_i$ of the real club are compared with those of the club found in the circular-shift null, using a one-sided Mann-Whitney test on $\log c_i$. The test asks whether a randomly chosen real hub tends to have a larger $c_i$ than a randomly chosen null hub; the check passes if p < 0.05.
3. **The real club beats the null's club:** the real ρ(n) is larger than the null ρ(n).

A club size **passes** only if all three checks hold in both directions, so six conditions in all. **H1 is accepted** for a model if at least three consecutive club sizes pass. This rule was fixed before any result was seen. Requiring consecutive sizes guards against one size passing by chance, and requiring both directions makes every passing size an independent replication.

The three checks are not redundant in principle, but in practice the second does the work. Check 3 implies check 1 whenever the null's ρ is above 1, which it always is here. And in every model the real ρ exceeded the null ρ at every club size. So every size that failed did so on the Mann-Whitney test: real and null hubs overlapped too much for the club-level difference to be trusted.

The shaded regions in the result figures mark the club sizes that pass. In every model the passing sizes form a single unbroken run, so the shading is exactly the range over which H1 holds.

Two caveats apply to the p-values. They are not corrected for the 20 sizes × 2 directions tested. And hubs share the club's links, so they are not independent observations, which makes the p-values optimistic. They are therefore read as descriptive. The evidence rests on the pattern replicating in both independent halves over a contiguous range of club sizes, not on any single p-value.

### 3.7 Do hubs bridge networks? The participation coefficient

Whether hubs form a club is a separate question from whether they connect different networks. Each unit belongs to one of the k = 100 networks found by the clustering pipeline. A hub can have all its links inside its own network (a **provincial hub**), or spread them across many networks (a **connector hub**). Grayson et al. (2014) found that the members of the human functional rich club are largely provincial.

The **participation coefficient** (Guimerà and Amaral 2005) measures how evenly a unit's links are spread over networks:

$$P_i = 1 - \sum_{s=1}^{K} \left(\frac{\kappa_{is}}{k_i}\right)^2 .$$

Here $k_i$ is the unit's number of links and $\kappa_{is}$ the number of them that go to network $s$, so $\kappa_{is}/k_i$ is the share of its links in network $s$. The sum of squared shares is large when the links are concentrated: it is 1 when all links go to one network, giving $P = 0$ (Figure 2B, left). It is small when they are spread out: with links divided evenly over $m$ networks each share is $1/m$, the sum is $1/m$, and $P = 1 - 1/m$. Four networks give 0.75 (Figure 2B, right); spreading over all 100 networks gives the maximum, 0.99. P measures how widely a unit's links are distributed over networks, not how many links it has.

P is computed on a thresholded graph, keeping the strongest 1% of all links (with 2% and 10% as robustness checks). On the full weighted graph every unit has some weight to every network, so every unit's P is close to the maximum. The networks come from the half used to choose the hubs, and the links from the other half, following the same cross-half logic.

**Two problems, and how they are handled.**

The first is that P depends on how many links a unit has. A unit with 3 links can reach at most 3 networks, and a unit with 3,000 can come close to the maximum by chance alone. Hubs have many more links than other units, so comparing their P with the median unit's mixes being a hub with having many links. The reference used here is therefore **random spread**: the participation a unit would have if its $k_i$ links went to $k_i$ randomly chosen other units, regardless of network. Under random spread the share of links landing in network $s$ follows a hypergeometric distribution, with mean $p_s$, the fraction of the unit's possible partners that lie in network $s$. Its expected participation is

$$E[P_i] = \Big(1 - \sum_s p_s^2\Big)\Big(1 - \frac{M - k_i}{k_i\,(M - 1)}\Big),$$

where $M$ is the number of possible partners (all other live units). The first factor is the participation of perfectly proportional spread; the second corrects for the unit having only $k_i$ links, and goes to 1 as $k_i$ grows. The verification suite checks this formula against simulation. Each unit is read against its own expectation, so units with few and many links are compared fairly.

The second problem is that with 100 networks P is compressed near its maximum. Random spread over 100 equal networks gives P = 0.99. A unit sending 10% of its links into its own network, ten times what chance predicts, still has P = 0.982. Differences in P therefore look negligible even when they are large. The comparison is made on a readable scale instead: the **effective number of networks reached**, $1/(1 - P)$. That is 1 if all links go to one network and $m$ if they are spread evenly over $m$. The unit's value is divided by its random-spread value, $1/(1 - E[P_i])$. A ratio of 1 means the unit places its links regardless of network; 0.5 means its links reach half as many networks as chance predicts.

Rather than comparing the club with an arbitrary group of other units, the ratio is shown as a function of strength rank across all live units, with the club marked. This shows whether hubs differ from weaker units, and whether there is any break at the top or simply a trend.

### 3.8 Verification and reproduction

[tests/verify_iter45_richclub.py](../../tests/verify_iter45_richclub.py) runs 28 checks, all of which pass:
- the streamed computations equal a direct calculation on small matrices;
- a planted club is recovered (60 of 60 hubs, ρ = 5.6, null ρ = 1.00);
- a single-factor matrix and random weights give ρ ≈ 1;
- the cross-half case of section 3.4 and the layer case of section 4.2 behave as described.

For Qwen the connectome was recomputed for this analysis, because the original tiles had been deleted after scoring. The stored partitions, re-scored on the new tiles, reproduce all 36 stored scores (maximum difference 3.9 × 10⁻¹⁰ at 2B and 2.3 × 10⁻¹⁰ at 4B). So the new connectome is identical to the one the networks were fitted on.

## 4. Results

### 4.1 Is there a rich club?

![rho curves](rho_curves.png)

**Figure 4.** The rich-club coefficient ρ as a function of club size, for the real connectome (blue) and the circular-shift null (gray), in both directions (solid: hubs chosen on A, measured on B; dashed: the reverse). ρ = 1 (black line) is the strength baseline. Shading: club sizes that pass all checks of section 3.6.

| | Pythia-70m | Qwen3.5-2B | Qwen3.5-4B |
|---|---|---|---|
| H1 criterion holds for club sizes | 50 to 939 (16 of 20) | 70 to 29,491 (19 of 20) | 50 to 58,982 (20 of 20) |
| ρ, club of 1% of live units (real / null) | 2.10 / 1.58 ($n$ = 109) | 1.37 / 1.26 ($n$ = 1,436) | 1.42 / 1.23 ($n$ = 3,002) |
| median $c_i$ at 1% (real / null) | 2.18 / 1.56 | 1.35 / 1.25 | 1.39 / 1.22 |
| Mann-Whitney p at 1% (A→B, B→A) | 4 × 10⁻²⁰, 3 × 10⁻²³ | 1 × 10⁻¹⁰², 7 × 10⁻¹²⁰ | 3 × 10⁻²⁴³, 2 × 10⁻²⁶⁴ |
| criterion with hubs ranked by degree | holds at all sizes | holds at all sizes | holds at all sizes (at 5% density from 73 units) |

All three models meet the criterion. The two directions agree almost exactly, and ranking hubs by degree instead of strength gives the same answer. The hubs themselves are stable: the club chosen on half A and the club chosen on half B are largely the same units. Their overlap (Jaccard index, shared members over all members) is 0.90 to 0.97 in Pythia-70m, 0.85 to 0.93 in Qwen3.5-2B and 0.88 to 0.96 in Qwen3.5-4B, against at most 0.11 for random sets of the same size.

Figure 5 shows the comparison behind check 2 for a club of 1% of the live units: every hub's coefficient, real against null.

![hub level](hub_level.png)

**Figure 5.** Each dot is one hub's $\log_2 c_i$ (0 = the strength baseline), for the real club (blue) and the club of the circular-shift null (gray), both directions pooled. Boxes show all hubs; at most 300 dots are drawn per box.

Almost every hub, real or null, lies above 0, which is the null's own club (below). In all three models the real hubs lie above the null hubs, most clearly in Pythia-70m.

The circular-shift null is far from ρ = 1. For the smallest clubs its ρ is 1.6 to 1.7 in every model, and its hubs are different units: of the top 50, none are shared with the real hubs in Pythia-70m or Qwen3.5-4B, and 2 in Qwen3.5-2B. This is the club of slow units anticipated in section 3.5. Against the strength baseline alone, the real club would therefore look much larger than it is.

The real excess over the null is largest for small clubs and shrinks as the club grows. In Pythia-70m it disappears at about 1,100 units (11% of live units). In both Qwen models it persists up to the largest size tested (one fifth of the live units), where it has narrowed to about 1%. In Qwen3.5-4B it is largest for clubs of 150 to 3,000 units, at 15 to 17% above the null.

### 4.2 Does the club survive when hubs in the same layer are left out?

Units in the same layer correlate more than units in different layers, because they read the same point of the residual stream. Hubs are also unevenly spread over layers; in Pythia-70m, 55% of the 500 strongest units sit in layer 4. So the club could be nothing more than strong units of one layer correlating with each other.

To test this, the whole analysis of section 3 is repeated using only pairs of hubs in different layers. ρ and every $c_i$ are computed on those pairs alone, each against the strength baseline for the same pairs, and the H1 criterion of section 3.6 is applied unchanged. This is a strict test. A club made only of within-layer coupling gives ρ *below* 1 on cross-layer pairs (0.46 in the verification suite), because the baseline spreads the layer's extra strength over every partner while the coupling stays inside the layer. The test was added after the Pythia pilot showed the concentration in layer 4, so it is exploratory.

![layer split](layer_split.png)

**Figure 6.** ρ on hub pairs in different layers only, real (blue) against null (gray), mean of the two directions. The faint blue line is the real ρ on all pairs (Figure 4), for reference. Shading: club sizes that pass all checks on cross-layer pairs.

| club of 1% of live units | Pythia-70m | Qwen3.5-2B | Qwen3.5-4B |
|---|---|---|---|
| share of hub pairs in different layers | 0.61 | 0.91 | 0.90 |
| ρ on cross-layer pairs, real / null | 1.68 / 1.58 | 1.34 / 1.26 | 1.36 / 1.23 |
| Mann-Whitney p, cross-layer $c_i$ (A→B, B→A) | 0.92, 0.96 | 2 × 10⁻⁶⁸, 2 × 10⁻⁸⁴ | 2 × 10⁻¹⁸⁹, 5 × 10⁻²⁰⁹ |
| H1 criterion on cross-layer pairs holds for | no club size | 70 to 10,770 | 50 to 19,307 |

In both Qwen models the club survives. Restricted to hubs in different layers, which are about 90% of all hub pairs, the real ρ is barely below its all-pairs value and stays above the null for clubs up to about 10,000 (2B) and 19,000 (4B) units.

In Pythia-70m it does not. On cross-layer pairs the real hubs couple no more than the null's hubs, so its club is a within-layer effect. That fits the layer composition: 39% of its hub pairs share a layer, against 9 to 10% in the Qwen models.

### 4.3 Where are the hubs?

Knowing that a club exists does not say what role it can play. Two properties of its members bear on that.

**Depth.** In the brain the rich club sits mainly in association cortex (precuneus, superior frontal and superior parietal cortex), far from the sensory input (van den Heuvel and Sporns 2011). The analogous question in a language model is where along its depth the hubs sit: near the input, in the middle, or near the output. That indicates whether the strongly interconnected core works at the level of the incoming tokens or late in the computation, close to the prediction.

**Networks.** The k = 100 networks found by the clustering pipeline are groups of units that couple more among themselves than with the rest. If all hubs fell in one network, the rich club would simply be that network, unusually cohesive and already captured by the parcellation. If the hubs are spread over many networks, their strong mutual coupling links networks that are otherwise separate. That is the integrating role attributed to rich clubs: a core through which systems communicate (Figure 1B). The spread is measured as the **effective number of networks** spanned by the club: the exponential of the entropy of its members' network labels. It is 1 if all hubs are in one network, and $m$ if they are spread evenly over $m$ networks.

![composition](composition.png)

**Figure 7.** **A.** Layer distribution of the 500 strongest units, real (blue) and circular-shift null (light gray). **B.** Effective number of the k = 100 networks spanned by the club, as a function of club size. Null hubs are labelled with the null's own partition.

In the Qwen models the hubs sit late. In Qwen3.5-2B none of the top 500 are below layer 6, and the share rises to 21% in the last layer. In Qwen3.5-4B, 93% are in the upper half (layers 16 to 31) and 28% in the last layer. In Pythia-70m they sit in layers 3 and 4 (82%) of six. The null's hubs are mostly different units (none of the top 50 shared in Pythia-70m and 4B, 2 in 2B), concentrated in a few layers.

In the Qwen models the club spans many networks from the start. The top 50 hubs fall in an effective 25 networks (2B) and 17 (4B), and the top 1% in an effective 55 and 51. The null's hubs crowd into 2 to 4 networks at every size; a likely reason is that the null's partition, like its club, groups together units that vary slowly. Pythia-70m's club stays in few networks (effective 6.4 at 1%), barely more than its null.

So in the Qwen models the club is late in depth and spread across the networks: a cross-network core, not one network. Pythia-70m's is a set of strongly coupled units within two layers and a few networks.

### 4.4 Do hubs spread their links across networks?

![participation](participation.png)

**Figure 8.** Effective number of networks a unit's strongest links reach, $1/(1-P)$, divided by the value expected under random spread, $1/(1-E[P])$. Median per strength-rank bin, both directions averaged; rank 1 is the strongest unit. Thick line: graph of the strongest 1% of links; thin lines: 2% and 10%. Shading: the club of 1% of live units.

| graph of the strongest 1% of links | Pythia-70m | Qwen3.5-2B | Qwen3.5-4B |
|---|---|---|---|
| networks reached, club of 1% (median; random spread) | 9.0 (67) | 45 (64) | 44 (64) |
| ratio to random spread, club of 1% | 0.13 | 0.71 | 0.68 |
| ratio to random spread, all live units | 0.27 | 0.47 | 0.43 |

In the Qwen models participation rises steadily with strength. The weakest units' links reach 40% of the networks random spread predicts. Units around rank 1,000 reach 65%, the strongest few 85 to 97%, and on the 10% graph the top units slightly exceed random spread. There is no break at the edge of the club: hubs are the high end of a continuous trend. The more strongly a unit is coupled, the less its links respect network boundaries. So in the Qwen models hubs lean towards connectors relative to other units, though even they reach somewhat fewer networks than chance on the strongest links.

Pythia-70m shows the opposite. Every unit's links are concentrated, and the hubs most of all: the club's links reach 9 networks where random spread predicts 67, a ratio of 0.13, against 0.25 to 0.34 for the weaker units. Its hubs are provincial, which fits section 4.2, where its club was a within-layer effect confined to a few networks.

## 5. What this shows, and what it does not

All three models have a rich club that cannot be explained by three things:
- the units' strengths (the strength baseline);
- their individual statistics, autocorrelation in particular (the circular-shift null);
- noise in choosing the hubs (choosing on one half, measuring on the other).

The club differs in character between Pythia-70m and the Qwen models (sections 4.2 and 4.3). Pythia-70m's is a set of strongly coupled units within layers 3 and 4, confined to a few networks. In Qwen3.5-2B and 4B the club spans layers and most networks, sits mostly in the upper half of the model, and is slightly stronger relative to the null at 4B. That is closer to the integrative core described for the brain's anatomical wiring than to the provincial functional club of Grayson et al. (2014). Pythia-70m differs from the Qwen models in size, depth, architecture and training data, so the contrast cannot be attributed to scale alone. Within the Qwen family, the two sizes agree.

The effect in the Qwen models is modest relative to the null: 1 to 12% above it at 2B and 1 to 17% at 4B, largest for clubs of a few hundred to a few thousand units. The null's own club is large. A graph measure that compares only against the strength baseline would therefore attribute to co-activation what is in part a property of single units. Any further graph analysis on these connectomes needs the circular-shift comparison.

The participation results point the same way. In the Qwen models hubs spread their strongest links over more networks than weaker units do, while in Pythia-70m they keep them within fewer. That mirrors the human functional rich club, whose members connect mainly within their own systems (Grayson et al. 2014); Pythia-70m resembles it and the Qwen models do not.

Two limits apply to the inference. The p-values are descriptive because hubs are not independent; the replication across halves carries the evidence. And the layer analysis was added after the pilot, so it is exploratory.

## 6. Files

The measures are computed by [parcelmate/richclub.py](../../parcelmate/richclub.py). [parcelmate/bin/richclub.py](../../parcelmate/bin/richclub.py) runs them on the real and null connectomes, applies the criterion and writes the outputs. [scripts/launch_richclub.sh](../../scripts/launch_richclub.sh) runs the cluster chain, and [scripts/check_tiles_reproduce.py](../../scripts/check_tiles_reproduce.py) is the reproduction check. The configs are `configs/richclub/<model>-resid-pool4-tiles.yml`: the connectivity settings of Iteration 43, with the tiles kept for the later stages.

Results are in `results/richclub/<model>-resid-pool4-tiles/metrics/richclub/`:
- `curves.csv`: ρ and hub statistics per club size;
- `layers.csv`, `participation.csv`;
- `participation_random.csv`: observed participation against random spread, by strength rank (written by [parcelmate/bin/richclub_participation.py](../../parcelmate/bin/richclub_participation.py) from the stored per-unit values);
- `verdict.json`: the criterion;
- `units_{real,null}.npz` and `hubs_{real,null}.npz`: per-unit and per-hub values.

In the code and output files the strength baseline is called N1 and the circular-shift null N3, and the null tree is labelled `null`. Read `curves.csv` with pandas' `keep_default_na=False`, or that label is read as missing. `python figures/make_richclub_figures.py` draws every figure on this page as SVG with editable text, plus PNG.

## References

- Colizza, Flammini, Serrano and Vespignani (2006). Detecting rich-club ordering in complex networks. *Nature Physics* 2, 110-115.
- van den Heuvel and Sporns (2011). Rich-club organization of the human connectome. *Journal of Neuroscience* 31(44), 15775-15786.
- Grayson, Ray, Carpenter, Iyer, Costa Dias, Stevens, Nigg and Fair (2014). Structural and functional rich club organization of the brain in children and adults. *PLoS One* 9(2), e88297.
- Guimerà and Amaral (2005). Functional cartography of complex metabolic networks. *Nature* 433, 895-900.
- Serrano (2008). Rich-club vs rich-multipolarization phenomena in weighted networks. *Physical Review E* 78, 026101.
- Zlatic, Bianconi, Díaz-Guilera, Garlaschelli, Rao and Caldarelli (2009). On the rich-club effect in dense and weighted networks. *European Physical Journal B*, doi:10.1140/epjb/e2009-00007-9.
- Alstott, Panzarasa, Rubinov, Bullmore and Vértes (2014). A unifying framework for measuring weighted rich clubs. *Scientific Reports* 4, 7258.
