# Rich club in the functional connectome of language models

This page reports the first stage of the network-neuroscience analyses (LOG.md Iteration 45; plan in [info/RICH_CLUB_PLAN.md](../../info/RICH_CLUB_PLAN.md)). It asks whether the most strongly coupled MLP neurons of a language model are coupled mainly to each other. Pythia-70m is the pilot; Qwen3.5-2B and Qwen3.5-4B are the main models.

## In short

All three models have a rich club: their most strongly coupled neurons are more coupled to each other than their overall coupling predicts. This replicates across two independent halves of the data. It also survives a null in which every neuron keeps its own statistics but co-activation is destroyed. That null has a sizeable club of its own, so the real excess over it is what counts. The excess is large in Pythia-70m, modest in Qwen3.5-2B (1-12%, depending on the club's size) and somewhat larger in Qwen3.5-4B (1-17%).

The models build their club differently. In Pythia-70m the extra coupling is between hubs in the same layer: across layers, the real hubs couple no more than the null's. In both Qwen models the club also holds across layers, and it spreads over most of the networks.

In no model are club members special bridges between networks. They spread their links across networks about as much as neurons of similar coupling strength do.

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

**Figure 2.** **A.** One unit and its links, with weights. Its **strength** is the sum of all its link weights. Its **degree** is the number of links that pass a threshold. **B.** The participation coefficient (section 3.6) for two units with the same eight links. On the left all partners are in one network; on the right they are spread evenly over four.

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

We call this the **strength baseline**. It is the configuration model of network science, and it is what "predicted by strength" means throughout this page.

It already expects hubs to be well linked to each other, because $s_i s_j$ is large when both units are strong. It rules out the trivial club. If every correlation were produced by one shared factor ($r_{ij} = a_i a_j$), the strongest units would correlate most with each other, but the baseline predicts this exactly. The verification suite confirms it: ρ = 1.002 for such a matrix.

The club as a whole is measured by its **rich-club coefficient**, the total weight among its members divided by the total the baseline expects:

$$\rho(n) = \frac{\sum_{i<j \,\in\, R(n)} w_{ij}}{\sum_{i<j \,\in\, R(n)} s_i s_j / 2W}.$$

The numerator adds up the observed weight of every link between two club members; the denominator adds up what the baseline expects for the same links. ρ = 1 means the hubs are coupled to each other exactly as much as their strengths predict (H2). ρ = 1.5 means 50% more (H1).

The same ratio for a single hub is its **hub coefficient**:

$$c_i(n) = \frac{\sum_{j \,\in\, R(n),\, j\neq i} w_{ij}}{\sum_{j \,\in\, R(n),\, j\neq i} s_i s_j / 2W}.$$

This is hub $i$'s observed weight to the other club members over the weight the baseline expects. For example, suppose hub $i$'s links to the other 99 hubs of a 100-unit club sum to 12, while the baseline expects 8. Then $c_i = 1.5$: this hub gives the other hubs 50% more coupling than its strength and theirs predict. ρ summarises the whole club; the $c_i$ give one value per hub and are what the statistical test in section 3.5 is applied to.

### 3.4 Choosing hubs on one half, measuring them on the other

Choosing the strongest units and measuring their mutual coupling on the same data is biased. Suppose some units happen to share variance in one sample only, for example because they all respond to one document in it. They become strong in that sample precisely because they co-vary there, and their mutual coupling is then inflated in the same data.

The fix is to choose the hubs on half A and measure every coupling, and the baseline, on half B (A→B), then the reverse (B→A). Variance shared within one half cannot carry over to the other. In the verification suite, a planted case of this kind gives ρ = 3.8 when hubs are chosen and measured on the same half, and ρ = 1.00 across halves. The two directions are two independent replications of the whole analysis.

### 3.5 A second null: units that vary slowly

The strength baseline accounts for each unit's strength, but not for other properties of individual units that could create a club by chance. The most important is how slowly a unit's activation changes from token to token (its **autocorrelation**). The correlation between two unrelated but slowly varying signals fluctuates more widely than that between two fast ones: a slow signal has fewer independent stretches, so chance alignments do not average out. Because the weights are absolute correlations, wider fluctuations mean larger weights. And the inflation is largest when *both* units are slow. Slow units therefore end up coupled to each other more than their strengths predict, which looks like a rich club made only of slow units.

The **circular-shift null** controls for this. Each unit's activation timecourse is shifted in time by a random offset, with the end wrapped round to the start. This keeps each unit's mean, variance and autocorrelation, but breaks any alignment between units, so it destroys co-activation. The null connectome is built with exactly the same pipeline, and the whole analysis is repeated on it. A real club must exceed the strength baseline and also the club this null produces.

Figure 3 puts sections 3.2 to 3.5 together.

![method](method.png)

**Figure 3.** **A.** Units are ranked by strength on half A, and the club is the top $n$. Its members' links are then read on half B and compared with the strength baseline for the same pairs. **B.** The club-level (ρ) and hub-level ($c_i$) ratios. **C.** The circular-shift null: each timecourse is shifted by a random offset, which keeps its own statistics and destroys co-activation. The whole analysis is repeated on the result.

### 3.6 Testing H1

The hub coefficients of the real hubs are compared with those of the null's hubs using a one-sided Mann-Whitney test on $\log c_i$. It asks whether a randomly chosen real hub tends to have a larger $c_i$ than a randomly chosen null hub. Hubs share the club's links, so they are not independent observations. The p-values are therefore read as descriptive, and the evidence rests on the result replicating in both directions.

The criterion for H1 was fixed before any result. For at least three consecutive club sizes, all of the following must hold in both directions:
1. ρ > 1;
2. the real hubs' $c_i$ exceed the null hubs' (p < 0.05);
3. the real ρ exceeds the null ρ.

### 3.7 Do hubs bridge networks? The participation coefficient

Whether hubs form a club is a separate question from whether they connect different networks. Each unit belongs to one of the k = 100 networks found by the clustering pipeline. A hub can have all its links inside its own network (a **provincial hub**), or spread them across many networks (a **connector hub**). Grayson et al. (2014) found that the members of the human functional rich club are largely provincial.

The **participation coefficient** (Guimerà and Amaral 2005) measures how evenly a unit's links are spread over networks:

$$P_i = 1 - \sum_{s=1}^{K} \left(\frac{\kappa_{is}}{k_i}\right)^2 .$$

Here $k_i$ is the unit's number of links and $\kappa_{is}$ the number of them that go to network $s$, so $\kappa_{is}/k_i$ is the share of its links in network $s$. The sum of squared shares is large when the links are concentrated: it is 1 when all links go to one network, giving $P = 0$ (Figure 2B, left). It is small when they are spread out: with links divided evenly over $m$ networks each share is $1/m$, the sum is $1/m$, and $P = 1 - 1/m$. Four networks give 0.75 (Figure 2B, right); spreading over all 100 networks gives the maximum, 0.99. P measures how widely a unit's links are distributed over networks, not how many links it has.

P is computed on a thresholded graph, keeping the strongest 1% of all links (with 2% and 10% as robustness checks). On the full weighted graph every unit has some weight to every network, so every unit's P is close to the maximum and the measure cannot discriminate. The networks come from the half used to choose the hubs, and the links from the other half, following the same cross-half logic. Club members are compared with the units ranked just below them in strength (ranks $n+1$ to $2n$). The comparison then asks whether being a hub matters, not just being strong.

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

**Figure 4.** The rich-club coefficient ρ as a function of club size, for the real connectome (colour) and the circular-shift null (gray), in both directions (solid: hubs chosen on A, measured on B; dashed: the reverse). ρ = 1 (black line) is the strength baseline. Shading: club sizes where the H1 criterion holds.

| | Pythia-70m | Qwen3.5-2B | Qwen3.5-4B |
|---|---|---|---|
| H1 criterion holds for club sizes | 50 to 939 (16 of 20) | 70 to 29,491 (19 of 20) | 50 to 58,982 (20 of 20) |
| ρ, club of 1% of live units (real / null) | 2.10 / 1.58 ($n$ = 109) | 1.37 / 1.26 ($n$ = 1,436) | 1.42 / 1.23 ($n$ = 3,002) |
| median $c_i$ at 1% (real / null) | 2.18 / 1.56 | 1.35 / 1.25 | 1.39 / 1.22 |
| Mann-Whitney p at 1% (A→B, B→A) | 4 × 10⁻²⁰, 3 × 10⁻²³ | 1 × 10⁻¹⁰², 7 × 10⁻¹²⁰ | 3 × 10⁻²⁴³, 2 × 10⁻²⁶⁴ |
| criterion with hubs ranked by degree | holds at all sizes | holds at all sizes | holds at all sizes (at 5% density from 73 units) |

All three models meet the criterion. The two directions agree almost exactly, and ranking hubs by degree instead of strength gives the same answer.

The circular-shift null is far from ρ = 1. For the smallest clubs its ρ is 1.6 to 1.7 in every model, and its hubs are different units: of the top 50, none are shared with the real hubs in Pythia-70m or Qwen3.5-4B, and 2 in Qwen3.5-2B. This is the club of slow units anticipated in section 3.5. Against the strength baseline alone, the real club would therefore look much larger than it is.

The real excess over the null is largest for small clubs and shrinks as the club grows. In Pythia-70m it disappears at about 1,100 units (11% of live units). In both Qwen models it persists up to the largest size tested (one fifth of the live units), where it has narrowed to about 1%. In Qwen3.5-4B it is largest for clubs of 150 to 3,000 units, at 15 to 17% above the null.

### 4.2 Is the club just a layer effect?

Units in the same layer correlate more than units in different layers: they read the same point of the residual stream. The networks the pipeline finds are partly layer-local. The networks share 0.26 (Qwen3.5-2B) and 0.30 (Qwen3.5-4B) adjusted mutual information with layer membership, against 0.13 for the null's networks.

The hubs are also unevenly spread over layers. In Pythia-70m, 55% of the 500 strongest units sit in layer 4. A club whose members share a layer could therefore arise simply because units of the same layer correlate, not because hubs as such couple to each other. To tell these apart, each hub's coupling to the other hubs is split in two: coupling to hubs in its own layer, and coupling to hubs in other layers. The ratio to the strength baseline is computed separately for each ($\rho_\text{same}$ and $\rho_\text{other}$), and the H1 criterion is applied to the other-layer part.

A club built only of within-layer coupling gives $\rho_\text{other}$ *below* 1. The extra coupling stays inside the layer, while the baseline spreads the hubs' extra strength over every partner. The verification suite gives 0.46 for such a club. So $\rho_\text{other} > 1$, and above the null, shows coupling between hubs beyond what layer membership produces. This test was added after the Pythia pilot showed the concentration in layer 4, so it is exploratory.

![layer split](layer_split.png)

**Figure 5.** ρ computed separately on hub pairs in the same layer (filled circles) and in different layers (open squares), real (colour) and null (gray), averaged over the two directions. Shading: club sizes where the H1 criterion holds on other-layer pairs.

| club of 1% of live units | Pythia-70m | Qwen3.5-2B | Qwen3.5-4B |
|---|---|---|---|
| $\rho_\text{same}$, real / null | 2.76 / 1.58 | 1.62 / 1.26 | 1.94 / 1.23 |
| $\rho_\text{other}$, real / null | 1.68 / 1.58 | 1.34 / 1.26 | 1.36 / 1.23 |
| Mann-Whitney p on other-layer $c_i$ (A→B, B→A) | 0.92, 0.96 | 2 × 10⁻⁶⁸, 2 × 10⁻⁸⁴ | 2 × 10⁻¹⁸⁹, 5 × 10⁻²⁰⁹ |
| share of hub pairs in the same layer, real / null | 0.39 / 0.68 | 0.09 / 0.11 | 0.10 / 0.09 |
| H1 criterion on other-layer pairs holds for | no club size | 70 to 10,770 | 50 to 19,307 |

The models answer differently. In Pythia-70m the excess over the null is a within-layer effect. Same-layer hub pairs are coupled 2.76 times the baseline. Across layers, real hubs couple about as strongly as the null's hubs, and slightly less for clubs above about 350 units. When hubs are ranked by degree at 0.1% or 1% density, the other-layer criterion is met for the smallest clubs only (50 to 109 units).

In the Qwen models about 90% of hub pairs are in different layers. These cross-layer pairs exceed both the baseline and the null, for clubs from 70 to 10,770 units in 2B and from 50 to 19,307 in 4B. Same-layer pairs remain the most strongly coupled (in 4B, 1.94 times the baseline at 1%), but the club is not a layer artefact. With hubs ranked by degree at 0.1% or 1% density, the 4B cross-layer criterion holds at every club size.

Figure 6 shows the same comparison hub by hub, for a club of 1% of the live units.

![hub level](hub_level.png)

**Figure 6.** Each dot is one hub's $\log_2 c_i$ (0 = the strength baseline), real (colour) against null (gray), both directions pooled. Left pair: coupling to all other hubs. Right pair: coupling to hubs in other layers only. Boxes show all hubs; at most 300 dots are drawn per box.

Almost every hub, real or null, lies above the baseline, which is the null's own club again. In both Qwen models the real hubs lie above the null hubs on both measures. In Pythia-70m they do so only when same-layer pairs are included.

### 4.3 Where are the hubs, and are they the same in both halves?

![composition](composition.png)

**Figure 7.** **A.** Layer distribution of the 500 strongest units, real (colour, black edge) and null (gray). **B.** How many of the k = 100 networks the club spans, as an effective number (the exponential of the entropy of its members' network labels: 1 if all members are in one network, 100 if they are spread evenly over all of them). **C.** Overlap (Jaccard index, shared members over total members) between the club chosen on half A and the club chosen on half B, against random sets of the same size (dotted).

In Pythia-70m the hubs sit in layers 3 and 4 (82% of the top 500); the null's hubs sit mostly in layer 5. In Qwen3.5-2B the hubs occupy the upper two thirds of the model. None of the top 500 are below layer 6, and their share rises to 21% in the last layer. Qwen3.5-4B is similar: 93% of the top 500 are in the upper half (layers 16 to 31) and 28% in the last layer. Its null's hubs are also concentrated there (23% in the last layer), but they are different units (43 of the top 500 shared).

The clubs also differ in how many networks they reach. Qwen3.5-2B's top 50 hubs fall in 29 networks (effective number 25), and the top 1% in an effective 55. Qwen3.5-4B's top 50 fall in 22 to 23 networks (effective 17), and the top 1% in an effective 51. In both Qwen models the null's hubs crowd into 2 to 4 networks. Pythia-70m's club stays in few networks (effective number 6.4 at 1%), barely more than its null.

In every model the same units are hubs in both halves: Jaccard 0.90 to 0.97 in Pythia-70m, 0.85 to 0.93 in Qwen3.5-2B and 0.88 to 0.96 in Qwen3.5-4B, against at most 0.11 for random sets. Unit strengths correlate at r = 0.999, 0.997 and 0.996 across halves.

### 4.4 Do hubs bridge networks?

![participation](participation.png)

**Figure 8.** Participation coefficient of the club of 1% of live units (filled), of the units ranked just below it (open, ranks $n+1$ to $2n$), and of the median unit (tick), on graphs keeping the strongest 1%, 2% and 10% of links and on the full weighted graph.

| graph of the strongest 1% of links, club of 1% | club | ranked just below | median unit |
|---|---|---|---|
| Pythia-70m | 0.890 | 0.881 | 0.864 |
| Qwen3.5-2B | 0.977 | 0.975 | 0.956 |
| Qwen3.5-4B | 0.976 | 0.973 | 0.958 |

Club members spread their links over networks slightly more widely than the median unit, and almost exactly as widely as the units just below them in strength. They are neither provincial hubs nor distinctive connector hubs: they spread their links like other strongly coupled units. The 2% and 10% graphs agree. On the full weighted graph, as expected, every unit is close to the maximum. In the null, every unit has P ≈ 0.98, as random links imply.

## 5. What this shows, and what it does not

All three models have a rich club that cannot be explained by three things:
- the units' strengths (the strength baseline);
- their individual statistics, autocorrelation in particular (the circular-shift null);
- noise in choosing the hubs (choosing on one half, measuring on the other).

The club differs in character between Pythia-70m and the Qwen models. Pythia-70m's is a set of strongly coupled units within layers 3 and 4, confined to a few networks. In Qwen3.5-2B and 4B the club spans layers and most networks, sits mostly in the upper half of the model, and is slightly stronger relative to the null at 4B. That is closer to the integrative core described for the brain's anatomical wiring than to the provincial functional club of Grayson et al. (2014). Pythia-70m differs from the Qwen models in size, depth, architecture and training data, so the contrast cannot be attributed to scale alone. Within the Qwen family, the two sizes agree.

The effect in the Qwen models is modest relative to the null: 1 to 12% above it at 2B and 1 to 17% at 4B, largest for clubs of a few hundred to a few thousand units. The null's own club is large. A graph measure that compares only against the strength baseline would therefore attribute to co-activation what is in part a property of single units. Any further graph analysis on these connectomes needs the circular-shift comparison.

Two limits apply to the inference. The p-values are descriptive because hubs are not independent; the replication across halves carries the evidence. And the layer analysis was added after the pilot, so it is exploratory.

## 6. Files

The measures are computed by [parcelmate/richclub.py](../../parcelmate/richclub.py). [parcelmate/bin/richclub.py](../../parcelmate/bin/richclub.py) runs them on the real and null connectomes, applies the criterion and writes the outputs. [scripts/launch_richclub.sh](../../scripts/launch_richclub.sh) runs the cluster chain, and [scripts/check_tiles_reproduce.py](../../scripts/check_tiles_reproduce.py) is the reproduction check. The configs are `configs/richclub/<model>-resid-pool4-tiles.yml`: the connectivity settings of Iteration 43, with the tiles kept for the later stages.

Results are in `results/richclub/<model>-resid-pool4-tiles/metrics/richclub/`:
- `curves.csv`: ρ and hub statistics per club size;
- `layers.csv`, `participation.csv`;
- `verdict.json`: the criterion;
- `units_{real,null}.npz` and `hubs_{real,null}.npz`: per-unit and per-hub values.

In the code and output files the strength baseline is called N1 and the circular-shift null N3, and the null tree is labelled `null`. Read `curves.csv` with pandas' `keep_default_na=False`, or that label is read as missing. `python figures/make_richclub_figures.py` draws every figure on this page as SVG with editable text, plus PNG.

## References

- Colizza, Flammini, Serrano and Vespignani (2006). Detecting rich-club ordering in complex networks. *Nature Physics* 2, 110-115.
- van den Heuvel and Sporns (2011). Rich-club organization of the human connectome. *Journal of Neuroscience* 31(44), 15775-15786.
- Grayson, Ray, Carpenter, Iyer, Costa Dias, Stevens, Nigg and Fair (2014). Structural and functional rich club organization of the brain in children and adults. *PLoS One* 9(2), e88297.
- Guimerà and Amaral (2005). Functional cartography of complex metabolic networks. *Nature* 433, 895-900.
