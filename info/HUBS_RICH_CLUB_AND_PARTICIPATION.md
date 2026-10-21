# Two network-neuroscience hub measures to test in LLMs (2026-10-15)

Brainstorm note, not yet an analysis plan. It records two separate ideas from network neuroscience, what each would mean for the parcelmate connectome, and the methodological constraints. Citations were checked on 2026-10-15 against abstracts, PubMed Central records or arXiv pages; full texts were not read.

Terminology used throughout:
- **Degree** (or strength, in a weighted graph): how much a unit is connected, regardless of where those connections go.
- **Hub**: a unit with high degree. Where a different definition is meant, it is named explicitly.
- **Network**: one of the k = 100 groups of our parcellation.

## 1. The rich club: are the most connected units connected mainly to each other?

**The idea.** Take the units with the highest degree. A rich club exists if they are more densely connected among themselves than chance predicts. The question is about the set of hubs as a whole, and it does not use any partition into networks.

**Why it is not trivial.** Two outcomes are possible a priori. The hubs may each sit at the centre of their own region of the graph and barely link to one another, or they may form a tightly connected core that most long-range paths pass through. The second is a specific architecture (integration through a central backbone); the first is not.

**How it is measured.** For a degree threshold k, the rich-club coefficient φ(k) is the fraction of possible edges among the units with degree above k that actually exist. φ(k) rises with k even in random graphs, so it is normalised by the same quantity in random graphs that keep every unit's degree but shuffle its partners: φ_norm(k) = φ(k) / φ_random(k). A rich club exists where φ_norm(k) > 1. Weighted versions compare the weights among hubs with the strongest weights in the whole graph. Method and null: Colizza, Flammini, Serrano and Vespignani, "Detecting rich-club ordering in complex networks", Nature Physics 2, 110-115, 2006 ([arXiv physics/0602134](https://arxiv.org/abs/physics/0602134)).

**Neuroscience findings.**
- van den Heuvel and Sporns, "Rich-club organization of the human connectome", Journal of Neuroscience 31(44), 15775-15786, 2011 ([PMC6623027](https://pmc.ncbi.nlm.nih.gov/articles/PMC6623027)). Structural connectivity (diffusion imaging, 21 adults): 12 strongly interconnected hub regions, including precuneus, superior frontal and superior parietal cortex, hippocampus, putamen and thalamus, interpreted as a backbone for global integration.
- Grayson, Ray, Carpenter, Iyer, Costa Dias, Stevens, Nigg and Fair, "Structural and functional rich club organization of the brain in children and adults", PLoS One 9(2), e88297, 2014 ([PMC3915050](https://pmc.ncbi.nlm.nih.gov/articles/PMC3915050)). Both structural and resting-state functional connectivity have a rich club, in similar places. In structural connectivity the rich-club regions link the major networks; in functional connectivity they connect mostly within their own systems. The functional rich club is stronger in adults than in children; the structural one does not differ.

**LLM test.** On the functional connectome: are the units with the highest strength more strongly coupled to each other than in degree-preserving rewired connectomes? Extensions:
- the same measure on a weight-based (structural) connectome, to mirror the structural-functional contrast of Grayson et al. 2014;
- across Pythia checkpoints, to mirror their developmental contrast.

**Constraints.**
- Run on the residual connectome. On the ordinary connectome, massive-activation, BOS and position units are likely hubs that correlate with each other for trivial reasons.
- Run on the dense weighted |r| matrix. Our sparsified graph keeps each unit's top 10% of partners, which gives every unit the same out-degree and makes degree-based rich-club measures nearly meaningless.
- No rich-club analysis of transformer neurons was found in one search (2026-10-15). Not conclusive.

## 2. Participation coefficient: connector hubs and provincial hubs

**The idea.** Given a partition into networks, ask for each unit where its connections go. The participation coefficient is P_i = 1 - Σ_s (κ_is / k_i)², where k_i is the unit's degree and κ_is is how much of it goes to network s. P_i is close to 1 when the connections spread evenly across networks and 0 when they all stay in the unit's own network. Defined by Guimerà and Amaral, "Functional cartography of complex metabolic networks", Nature 433, 895-900, 2005 ([arXiv q-bio/0502035](https://arxiv.org/abs/q-bio/0502035)), together with a within-network degree z-score.

**Degree and participation are independent.** A unit can have many connections that all stay inside its own network: high degree, low participation, a *provincial hub*. A unit with high participation is a *connector hub*: its connections reach many networks. Analogy: a person with 500 friends, all in their own village, is very well connected but links nothing; a person with 30 friends spread across ten villages links the villages. Guimerà and Amaral found in metabolic networks that low-degree connector nodes are more conserved across organisms than provincial hubs.

**Neuroscience findings.**
- Power, Schlaggar, Lessov-Schlaggar and Petersen, "Evidence for hubs in human functional brain networks", Neuron 79(4), 2013 ([PMC3838673](https://pmc.ncbi.nlm.nih.gov/articles/PMC3838673)). In functional connectivity, ranking regions by degree mostly picks parts of large systems: a region in a large network correlates with many members of that network. They identify hubs by participation coefficient instead, and find a different, distributed set of regions.
- Warren et al. (full author list not yet checked), "Network measures predict neuropsychological outcome after brain injury", PNAS 111(39), 14247-14252, 2014 ([PMC4191760](https://pmc.ncbi.nlm.nih.gov/articles/PMC4191760)). 19 patients with focal lesions to six hubs selected by high participation coefficient and system density had severe, widespread cognitive deficits. 11 patients with lesions to two control hubs selected by high degree had narrower deficits.

**LLM test.** Rank units by participation coefficient and by within-network degree, then ablate matched sets of high-participation units, high-degree low-participation units, and layer-matched random units. Measure the loss change and how many tasks in the circuit bank are affected. Prediction from Warren et al. 2014: ablating connector hubs damages more tasks than ablating provincial hubs.

**Why this one tests the parcellation itself.** Participation is computed relative to our networks. If the networks are arbitrary, units that bridge them should not be causally special; if ablating them causes widespread damage, that is causal evidence that the partition is meaningful. The rich club does not use the partition and gives no such evidence.

**Constraints.**
- Same residual-connectome and dense-matrix requirements as the rich club.
- Depends on the partition and on k, so it should be checked across k.
- Our connectome uses |r|. Whether participation should be computed on |r| or on signed r (positive edges only) is an open decision.

## How the two relate

They answer different questions and can come apart. The rich club asks whether the most connected units connect to each other; participation asks whether a unit's connections cross network boundaries. A rich club whose members all have low participation is possible, and is what Grayson et al. 2014 report for human functional connectivity: the hubs are coupled to each other, but each is anchored in its own system. In structural connectivity the rich-club regions instead bridge systems.
