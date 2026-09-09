# Parcellation method: design decisions and alternatives

Scope: how the connectivity matrix is turned into networks, which of those steps are free choices rather than necessities, and what the alternatives are. Complements [LOG.md](LOG.md), which records bugs, fixes and the dated edit history — this file records *method* design and is the input to the paper's methods section. Empirical numbers below were measured on `results/alldomains` (GPT-2, seven domains, 4 x 100k tokens per domain) on 2026-09-08.

## 1. What the current method does

Each hidden unit is a "voxel" (GPT-2: 13 layers x 768 dims = 9,984 units, the residual stream at every layer boundary) and each token is a "timepoint" (T = 400,000). The pipeline computes a unit-by-unit Pearson correlation matrix per text domain, then converts it to networks in four steps:

1. **Absolute value.** `R = |r|`, so anticorrelation counts as connection.
2. **Local binarization.** For each unit, keep its top 10% of partners (999 of 9,983), set to 1, everything else 0. Each unit is now a binary "fingerprint" of who it couples to.
3. **PCA.** Reduce each 9,984-long fingerprint to 200 dimensions, with `whiten=True`.
4. **k-means.** MiniBatchKMeans, k = 50, 100 restarts, Hungarian-aligned and averaged into a soft membership table (9,984 x 50).

Downstream, networks that form a reciprocal-best-match clique across all domains are called domain-general, and each can be lesioned to test its causal contribution.

## 2. The five non-obvious choices

None of these is wrong, but each is a substantive claim, and each is a place where a different lab would have chosen differently. 2.1 is now settled (2026-09-08); the rest remain open.

**2.1 Taking |r| discards the sign — SETTLED, keep |r|.** The network literature's default advice is to keep the sign, because negative edges are expected to fall *between* communities and folding them in destroys that information (Masuda et al. 2025); fMRI handles this with signed modularity rather than absolute values (Rubinov & Sporns 2011). We are deliberately not following that advice, for a reason specific to this setting: the advice assumes sign is interpretable, and in fMRI it is. A BOLD deactivation is a physical event with a direction, which is why task-negative structure such as the default-mode network is a finding rather than an artefact. A residual-stream dimension has no comparable privileged direction — its sign is closer to a coordinate convention than to a physical quantity — so an anticorrelation between two units does not carry the interpretive weight that an anticorrelation between two brain regions does. Taking |r| makes the analysis invariant to that convention, which is the desirable behaviour. Masuda et al. themselves make the parallel point in the other direction: a negative fMRI edge "does not necessarily imply that these regions are connected by inhibitory synapses".

*Limitation to state in the paper, not to fix:* under |r| the method cannot distinguish two units that rise and fall together from two that trade off against each other. Both are called coupled. If a specific claim later depends on that distinction, it needs the signed matrix, which is preserved on disk and can be revisited without recomputation.

**2.2 Binarizing discards the weights — the local thresholding itself is defensible.** Worth separating these two. Ranking edges *per node* rather than globally is the well-supported part: Hamann et al. (2016) find local filtering beats global thresholding for every property class, and specifically preserves community structure best; Chen et al. (2024) benchmark 12 sparsifiers on graphs up to 685k nodes and find kNN-style local selection is the single best sparsifier for preserving Louvain partitions. Discarding the magnitudes afterwards is the part with no support.

| measurement (wikitext) | value |
|---|---|
| partners kept per unit | 999 of 9,983 (10%) |
| range of kept edges, all set to 1 | \|r\| = 0.025 to 1.000 |
| strongest *dropped* edge | \|r\| = 0.203 |
| fraction of *kept* edges weaker than that | 95.7% |
| kept edges that are one-directional (i keeps j, j drops i) | 29.0% |

The last row is a consequence of local thresholding on an exactly symmetric matrix, and is intrinsic to the approach rather than a defect. The 95.7% figure is the cost of binarizing on top of it.

**2.3 PCA truncation and whitening.** This is the choice most likely to surprise a reader, and the one most often misunderstood, so it is worth stating slowly.

*The units themselves are never transformed.* Clustering operates on a table whose rows are units and whose columns are that unit's connections. PCA compresses the *columns*, not the rows. Unit (layer 3, dim 271) goes in as row 2575 and comes out as row 2575 with a cluster label, and the final parcellation has exactly one membership row per real unit. That is why a lesion can target an actual dimension. Nothing is ever clustered in a rotated space of units, and the parcellation is not "in PCA space" in the sense people usually fear.

*What is compressed is each unit's description of itself.* Before clustering, a unit is described by a 9,984-long list: "here are the 999 units I am most strongly coupled to". Call that its fingerprint. k-means decides which units belong together by comparing fingerprints — two units join the same network when their fingerprints look alike. PCA replaces each 9,984-number fingerprint with a 200-number summary, and k-means then compares the summaries instead. The unit is unchanged; the *description used to judge whether two units are similar* is what changed. By analogy: you are still the same person whether you are described by your answers to 9,984 questions or by 200 summary scores derived from them, but who counts as "similar to you" can change a great deal depending on which summaries were kept and how they are weighted.

Two consequences follow, and both are consequential here.

**Truncation.** Keeping 200 summaries out of a possible 9,984 discards more than half the variation in the fingerprints. Any two units that differ *only* in the discarded directions become indistinguishable to the clustering step.

| measurement (wikitext) | value |
|---|---|
| variance of fingerprint space retained by 200 PCs | 45.3% |
| variance discarded before clustering | 54.7% |
| PC1 alone / PC1-10 / PC1-50 | 2.8% / 12.3% / 26.6% |
| largest-to-smallest retained eigenvalue | 38x |

Discarding half the variation would be reasonable if the discarded half were noise, but the spectrum is nearly flat — the single strongest direction accounts for only 2.8%, and it takes 50 directions to reach 26.6%. There is no low-dimensional structure to isolate, so the truncation is not separating signal from noise; it is removing a large arbitrary chunk of an essentially full-rank object.

**Whitening.** `whiten=True` then rescales all 200 summaries to have equal spread before k-means measures distances. In effect it declares that every retained direction is equally important for deciding network membership. Concretely, the 200th direction carries 0.074% of the variation and the 1st carries 2.8%, yet after whitening they contribute equally to whether two units are judged similar — a 38-fold reweighting in favour of the weakest directions. This is a real statistical claim (formally, it turns Euclidean distance into Mahalanobis distance), and it is currently asserted by a default argument rather than argued for.

The recommended fix in section 7 removes this step entirely rather than tuning it. Cosine similarity between standardized profiles is exactly the correlation between those two profiles, so profile similarity can be computed in full — all 9,984 columns, every direction at its natural weight — without any compression step to truncate or reweight.

**2.4 k = 50 is fixed, with no selection criterion.** See section 5 — the field's position is that no credible criterion exists.

**2.5 Pearson is assumed to be the right edge.** Liu et al. (2025) benchmark 239 pairwise connectivity statistics in fMRI and find covariance, precision and distance-based measures each have desirable properties Pearson lacks. Partial correlation would additionally address transitivity: pairwise r cannot separate direct from third-unit-mediated association, which is a known problem for correlation networks (Masuda et al. 2025).

## 3. Where this sits in the literature

The architecture has strong precedent and should not be abandoned. Yeo et al. (2011), the canonical fMRI parcellation, represents each of 18,715 vertices by its correlation profile to 1,175 sampled ROIs, L2-normalizes onto a hypersphere, and clusters with a von Mises-Fisher mixture — that is, spherical k-means on an N x d matrix, with no N x N matrix ever formed. Clustering connectivity profiles is what the field actually does; the pipeline is a version of it. What differs from the canonical form is precisely the four choices in section 2.

This matters for framing: the fix is to become *more* canonical, not to switch to graph community detection. Cosine similarity between L2-normalized z-scored profiles is exactly their correlation, so the Yeo formulation needs no absolute value, no binarization and no PCA.

## 4. Scaling: two hard limits

**4.1 The dense matrix breaks around 8B parameters — and does not have to be formed.**

| model | units N | dense N x N (float32) | profile N x 2000 |
|---|---|---|---|
| GPT-2 | 9,984 | 0.4 GB | 0.1 GB |
| GPT-2-XL | 78,400 | 24.6 GB | 0.6 GB |
| Qwen3-8B | 151,552 | 91.9 GB | 1.2 GB |
| Qwen3-32B | 332,800 | 443.0 GB | 2.7 GB |
| 70B-class | 663,552 | 1,761.2 GB | 5.3 GB |

Because `C = ZZ'` for z-scored timecourses Z (N x T), the matrix is a Gram matrix and never needs materializing: any product is `Cv = Z(Z'v)` at O(NT), exact top-k neighbours come from a tiled GEMM, and the nonzero eigenvalues of the N x N matrix equal those of the T x T matrix `Z'Z` (800 MB at T = 10^4). Published fMRI precedent for exactly this trick: Wink et al. (2012), voxel-wise centrality at 10^5-10^6 nodes without storing the matrix.

Compute is then not the constraint. rapids-singlecell reports the full kNN-plus-Leiden pipeline on 1M cells in 26 seconds on one GPU (preprint, see section 9), two to three orders of magnitude above N = 400k.

**4.2 The real constraint is tokens, not compute.** A correlation matrix estimated from T timepoints has rank <= T, and under Marchenko-Pastur the noise bulk extends to (1 + sqrt(q))^2 with q = N/T. At the current T = 400,000:

| model | units N | q = N/T | verdict |
|---|---|---|---|
| GPT-2 | 9,984 | 0.02 | fine |
| Llama-3.2-1B | 34,816 | 0.09 | fine |
| GPT-2-XL | 78,400 | 0.20 | marginal |
| Qwen3-8B | 151,552 | 0.38 | marginal |
| Qwen3-32B | 332,800 | 0.83 | mostly noise |
| 70B-class | 663,552 | 1.66 | singular |

Reaching a comfortable q = 0.1 needs ~1.5M tokens for Qwen3-8B, ~3.3M for Qwen3-32B and ~6.6M for a 70B model — 4x, 8x and 17x the current budget. **Report q for every model published.** If a run is stuck at q > 0.5, use rotationally-invariant estimators or Ledoit-Wolf shrinkage rather than the raw sample matrix (Bun et al. 2017; Ledoit & Wolf 2004).

## 5. On choosing k: the field's position is that you cannot

This is the direct answer to "is there an algorithm that finds the number of clusters". Essentially no, outside a generative model.

Schaefer et al. (2018) state in their Discussion that BIC-type metrics and stability analysis "are unlikely to estimate a truly optimal number of clusters", that stability-estimated k "might partially reflect the size of the dataset", and conclude that "it is unclear if there is a correct number of brain parcels" — then ship 400/600/800/1000 resolutions. Thirion et al. (2014) show cross-validated likelihood and bootstrap reproducibility disagree by an order of magnitude on the same data (accuracy favours 3,000-7,000 parcels, reproducibility peaks near 200). Eickhoff et al. (2015) name the underlying cluster validity problem: "clustering algorithms will always find subregions... whether these truly exist in nature or not". Arslan et al. (2018) benchmark 34 parcellation methods and find none wins on all evaluation axes.

The one principled exception is minimum description length inside a stochastic block model, which buys automatic model selection by committing to a generative model of the graph.

**Practical consequence.** Do not select or defend a single k. Run a range of granularities, report the main result at each, and show the conclusion survives. Precedent for sweeping rather than fixing: Power et al. (2011) ran Infomap across tie densities from 10% down to 2%. A hard methodological rule from Schaefer et al. (2018): when comparing two parcellations, always match the number of parcels, or use a null of random parcellations with matched size distributions.

## 6. Options assessed

| method | auto-k | signed | scales to 30B | maintained | verdict |
|---|---|---|---|---|---|
| current (\|r\|, binarize, PCA, k-means) | no | no | no | — | keep architecture, fix choices |
| **vMF / spherical k-means on profiles** | no (sweep) | yes | **yes** | yes | **default** |
| **nested weighted SBM (graph-tool)** | **yes (MDL)** | **yes, natively** | no (days at 10^4) | yes | **anchor at GPT-2 scale** |
| Leiden + CPM on a kNN graph | gamma instead of k | yes (two-layer) | yes | yes | scalable cross-check |
| OSLOM | alpha instead of k | no | no | no (~2012) | not recommended |
| MacMahon-Garlaschelli RMT null | no | yes | no (needs dense modularity matrix) | no (2015 MATLAB) | cite conceptually |
| Hoffmann et al. (no observed edges) | yes | n/a | unverified | yes | ambitious variant |

**Why the SBM is worth running even though it does not scale.** Modularity, Infomap, Leiden and vMF all define communities as internally dense by construction. The SBM does not. If some unit groups are core-periphery or disassortative, every method except the SBM will misassign them. Brain precedent: Betzel et al. (2018), Faskowitz et al. (2018). Zhang & Peixoto (2020) additionally let you *test* whether assortativity is the dominant pattern rather than assuming it, and report that modularity maximization "systematically overfits both in artificial as well as in empirical examples".

**Why not OSLOM.** It is not signed, it trades k for a tolerance parameter alpha rather than eliminating a free parameter, Palowitch et al. (2018) show it becomes increasingly anti-conservative as N grows (assigning background nodes to communities), and it has been unmaintained since roughly 2012 — CDlib declined to integrate it because only a subprocess wrapper exists. Defensible as a secondary robustness check, not as a backbone.

## 7. Recommendation

Three stages, in priority order.

**Stage 1 — cheap, and removes the binarization and the PCA at once.** Keep the profile-clustering architecture and keep `|r|` (decision 2.1). Fisher-transform as `arctanh(|r|)` to stabilize variance, keeping the magnitudes rather than binarizing them; standardize and L2-normalize each unit's profile; cluster with spherical k-means or a von Mises-Fisher mixture; sweep k and report at several granularities. Cosine similarity between standardized profiles is exactly the correlation between those profiles, so profile similarity is computed in full with no truncation and no reweighting — the PCA step disappears rather than being tuned. This is Yeo et al. (2011) adapted, so it is *more* faithful to the fMRI analogy the project is built on, not less.

Note that Stage 1 keeps only the *local* part of the current sparsification and discards the binarization. Whether to retain a top-10% profile mask at all, or to feed dense magnitude profiles, is a sub-decision worth testing both ways: Hamann et al. (2016) and Chen et al. (2024) support local selection, but neither studied profile clustering.

**Stage 2 — principled sparsification level.** If a sparse graph is needed (for Leiden, or for scale), build a symmetric kNN graph directly from Z by tiled GEMM, sweeping k in {10, 15, 30, 50, 100}. Choose the level by module-based cross-validation (Neuman et al. 2022) rather than an arbitrary percentile. Note the genuine tension in the literature on k: von Luxburg (2007) suggests k ~ log n, while Maier et al. (2009) find k must be "of the order n" for reliable cluster identification.

**Stage 3 — principled anchor.** Fit a weighted nested degree-corrected SBM in graph-tool at GPT-2 scale (`rec_types=["real-normal"]` takes continuous weights natively -- signed or, as here, the Fisher-transformed magnitudes; sample with merge-split MCMC, not single-node moves), and show it agrees with the vMF/Leiden result. That licenses the cheap method at large scale, and is a good paper narrative in itself. Budget: Melo et al. (2024) needed about a week for 5,261 nodes and 500k edges.

**Risk to state explicitly.** Sparsifying too aggressively pushes the graph below the detectability threshold, where no algorithm can recover planted communities (Decelle et al. 2011; Krzakala et al. 2013).

## 8. Ruled out, with reasons

- **Backboning (disparity filter, Polya urn, noise-corrected).** All three test edges against nulls that assume non-negative apportioned flows such as traffic or counts. A correlation is signed, bounded, and a Gram entry with forced transitivity; node strength has no interpretation as a divisible quantity. They also operate on a materialized edge list, which is what we are avoiding. No published validation on correlation matrices was found.
- **PMFG / TMFG.** Fixed at 3(N-2) edges, i.e. mean degree about 6, which at N = 4e5 is likely below the detectability threshold. PMFG's construction also requires sorting all N(N-1)/2 edges.
- **Effective-resistance spectral sparsification.** Undefined for signed weights (the Laplacian is not PSD, so effective resistance is not a metric); the edge budget O(n log n / eps^2) is not competitive with kNN at this n; and its guarantee is the Laplacian quadratic form, not community structure. Chen et al. (2024) measured 990 s to compute effective resistances on a 132k-node, 40M-edge graph.
- **Gradients as a replacement for parcellation.** Kong et al. (2023) found principal gradients need 40-60 dimensions to match hard parcellations, not the 1-3 usually reported. Note also that BrainSpace's default `sparsity=0.9` is itself an unswept arbitrary threshold.

## 9. Open decisions

1. Adopt the Stage 1 change (Fisher-transformed `|r|` profiles, no binarization, no PCA, vMF) as the plan of record? This changes all existing parcellations, so results before and after are not comparable.
2. Within Stage 1, keep a top-10% profile mask or cluster dense magnitude profiles? Test both; the supporting evidence for local selection was not gathered on profile clustering.
3. Which k range to report, and whether to add partial correlation as a robustness check on the edge definition.
4. Token budget for models above ~8B, given section 4.2. This is a data-collection decision that gates the whole large-model direction.

*Settled 2026-09-08:* keep `|r|` rather than signed weights (see 2.1), on the grounds that a residual-stream dimension's sign is a coordinate convention rather than a physical direction, unlike a BOLD deactivation.

Open question with no literature answer: no published work assembles implicit Gram products, RMT filtering, kNN construction and community detection for correlation graphs at N ~ 10^5-10^6. Building it would be a modest methodological contribution worth claiming.

## 10. References

Verified by fetching the publisher or arXiv page unless marked otherwise.

**Correlation networks and thresholding.** Masuda, Boyd, Garlaschelli & Mucha (2025), *Introduction to correlation networks: Interdisciplinary approaches beyond thresholding*, Physics Reports 1136:1-39. MacMahon & Garlaschelli (2015), *Community detection for correlation matrices*, Phys. Rev. X 5:021006. Kojaku & Masuda (2019), *Constructing networks by filtering correlation matrices: a null model approach*, Proc. R. Soc. A 475:20190578. Neuman, Jonsson, Calatayud & Rosvall (2022), *Cross-validation of correlation networks using modular structure*, Applied Network Science 7:75. Melo, Pallares & Ayroles (2024), *Reassessing the modularity of gene co-expression networks using the Stochastic Block Model*, PLOS Comput. Biol. 20(7):e1012300.

**Stochastic block models.** Peixoto (2018), *Nonparametric weighted stochastic block models*, Phys. Rev. E 97:012306. Peixoto (2014), *Hierarchical block structures and high-resolution model selection in large networks*, Phys. Rev. X 4:011047. Peixoto (2014), *Efficient Monte Carlo and greedy heuristic for the inference of stochastic block models*, Phys. Rev. E 89:012804. Peixoto (2020), *Merge-split Markov chain Monte Carlo for community detection*, Phys. Rev. E 102:012305. Zhang & Peixoto (2020), *Statistical inference of assortative community structures*, Phys. Rev. Research 2:043271. Peixoto (2023), *Descriptive vs. Inferential Community Detection in Networks*, Cambridge Elements. Betzel, Medaglia & Bassett (2018), *Diversity of meso-scale architecture in human and non-human connectomes*, Nat. Commun. 9:346. Faskowitz, Yan, Zuo & Sporns (2018), *Weighted Stochastic Block Models of the Human Connectome across the Life Span*, Sci. Rep. 8:12997.

**Modularity, Leiden, signed networks.** Traag, Waltman & van Eck (2019), *From Louvain to Leiden: guaranteeing well-connected communities*, Sci. Rep. 9:5233. Traag, Van Dooren & Nesterov (2011), *Narrow scope for resolution-limit-free community detection*, Phys. Rev. E 84:016114. Traag & Bruggeman (2009), *Community detection in networks with positive and negative links*, Phys. Rev. E 80:036115. Fortunato & Barthelemy (2007), *Resolution limit in community detection*, PNAS 104(1):36-41. Good, de Montjoye & Clauset (2010), *Performance of modularity maximization in practical contexts*, Phys. Rev. E 81:046106. Gomez, Jensen & Arenas (2009), *Analysis of community structure in networks of correlated data*, Phys. Rev. E 80:016114. Rubinov & Sporns (2011), *Weight-conserving characterization of complex functional brain networks*, NeuroImage 56(4):2068-2079. Lancichinetti & Fortunato (2012), *Consensus clustering in complex networks*, Sci. Rep. 2:336. Jeub, Sporns & Fortunato (2018), *Multiresolution Consensus Clustering in Networks*, Sci. Rep. 8:3259.

**OSLOM and its critique.** Lancichinetti, Radicchi, Ramasco & Fortunato (2011), *Finding Statistically Significant Communities in Networks*, PLOS ONE 6(4):e18961. Palowitch, Bhamidi & Nobel (2018), *Significance-based community detection in weighted networks*, JMLR 18(188):1-48.

**fMRI parcellation.** Yeo et al. (2011), *The organization of the human cerebral cortex estimated by intrinsic functional connectivity*, J. Neurophysiol. 106(3):1125-1165. Schaefer et al. (2018), *Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic Functional Connectivity MRI*, Cerebral Cortex 28(9):3095-3114. Thirion, Varoquaux, Dohmatob & Poline (2014), *Which fMRI clustering gives good brain parcellations?*, Front. Neurosci. 8:167. Eickhoff, Thirion, Varoquaux & Bzdok (2015), *Connectivity-based parcellation: Critique and implications*, Hum. Brain Mapp. 36(12):4771-4792. Arslan et al. (2018), *Human brain mapping: A systematic comparison of parcellation methods for the human cerebral cortex*, NeuroImage 170:5-30. Bellec et al. (2010), *Multi-level bootstrap analysis of stable clusters in resting-state fMRI*, NeuroImage 51(3):1126-1139. Power et al. (2011), *Functional Network Organization of the Human Brain*, Neuron 72(4):665-678. Craddock et al. (2012), *A whole brain fMRI atlas generated via spatially constrained spectral clustering*, Hum. Brain Mapp. 33(8):1914-1928. Kong et al. (2023), *Comparison between gradients and parcellations for functional connectivity prediction of behavior*, NeuroImage 273:120044. Liu et al. (2025), *Benchmarking methods for mapping functional connectivity in the brain*, Nature Methods 22:1593-1602.

**Sparsification and scale.** Chen, Ye, Vedula, Bronstein, Dreslinski, Mudge & Talati (2024), *Demystifying Graph Sparsification Algorithms in Graph Properties Preservation*, PVLDB 17(3):427-440. Hamann, Lindner, Meyerhenke, Staudt & Wagner (2016), *Structure-preserving sparsification methods for social networks*, Social Network Analysis and Mining 6:22. Satuluri, Parthasarathy & Ruan (2011), *Local graph sparsification for scalable clustering*, SIGMOD 2011:721-732. von Luxburg (2007), *A tutorial on spectral clustering*, Statistics and Computing 17:395-416. Maier, Hein & von Luxburg (2009), *Optimal construction of k-nearest neighbor graphs for identifying noisy clusters*, Theoretical Computer Science 410(19):1749-1764. Decelle, Krzakala, Moore & Zdeborova (2011), *Asymptotic analysis of the stochastic block model for modular networks and its algorithmic applications*, Phys. Rev. E 84:066106. Krzakala et al. (2013), *Spectral redemption in clustering sparse networks*, PNAS 110(52):20935-20940. Johnson, Douze & Jegou (2019), *Billion-scale similarity search with GPUs*, IEEE Trans. Big Data 7(3):535-547. Wink, de Munck, van der Werf, van den Heuvel & Barkhof (2012), *Fast eigenvector centrality mapping of voxel-wise connectivity in fMRI*, Brain Connectivity 2(5):265-274.

**Random matrix theory.** Laloux, Cizeau, Bouchaud & Potters (1999), *Noise dressing of financial correlation matrices*, Phys. Rev. Lett. 83(7):1467-1470. Plerou et al. (2002), *Random matrix approach to cross correlations in financial data*, Phys. Rev. E 65:066126. Bun, Bouchaud & Potters (2017), *Cleaning large correlation matrices: tools from Random Matrix Theory*, Physics Reports 666:1-109. Ledoit & Wolf (2004), *A well-conditioned estimator for large-dimensional covariance matrices*, J. Multivariate Analysis 88(2):365-411.

**Threshold-free alternative.** Hoffmann, Peel, Lambiotte & Jones (2020), *Community detection in networks without observing edges*, Science Advances 6(4):eaav1478.

**Preprints, not peer-reviewed** — cite with care: Bhattacharya et al. (2025), *Comparative Evaluation of Assumption Lean Community Detection Methods for Human Connectome Networks*, bioRxiv 2025.11.13.688333 (reports that silhouette, Calinski-Harabasz, modularity and NMI all failed to identify an optimal number of communities on real connectome data). Dicks et al. (2026), *GPU-accelerated single-cell analysis at scale with rapids-singlecell*, arXiv:2603.02402 (the 1M-cell kNN-plus-Leiden timing). Neuman, Smiljanic & Rosvall (2025), arXiv:2510.15013. Vu-Le et al. (2025), arXiv:2508.03843.

## 11. Not verified — do not cite without checking

Marchenko & Pastur (1967) original reference; Ledoit-Peche nonlinear shrinkage; PMFG's O(N^3) and TMFG's exact complexity claims; graph-tool's O(E log^2 N) per-sweep complexity and any concrete node/edge ceiling; the Leiden paper's own benchmark table; `leidenalg`'s current negative-weight API name (**verify before writing code**); cuGraph's 500M-edge figure; Infomap's scalability numbers; Jbabdi et al. (2009); Lashkari et al.; Caparelli et al. (2025); Bassett et al. (2013) Chaos DOI; Xiong et al. TAPER volume and pages; the `pyRMT` and `fast_tmfg` packages; Coscia's backboning code URL.

Two specific cautions. The Yeo et al. (2011) stability wording and the Power et al. (2011) density ranges quoted above came from automated full-text extraction rather than direct reading — verify the exact wording before quoting in a manuscript. And arXiv:2605.14258 surfaced in search results as building signed correlation graphs over transformer units with top-k edges and signed Leiden; the abstract does not confirm this and describes Jacobian eigendecomposition instead, so do not cite it for that claim without reading the full paper.
