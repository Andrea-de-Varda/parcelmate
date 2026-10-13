# Novelty check: the sparsification and segregation results (2026-10-12)

Question: how novel are the training-dynamics results in [SUMMARY.md](SUMMARY.md) section 3 (activation sparsity, sparser connectivity, more segregated networks, late convergence)? This file records the closest published work found and the resulting judgement.

**Verification status.** The papers below were found by web search on 2026-10-12. Titles, authors, venues and the claims attributed to them were checked against abstracts, search snippets, or the arXiv HTML pages (Zhang et al. 2024; Liu et al. 2026; Urbina-Rodriguez et al. 2026; Li et al. 2025). The full texts were not read. Before any of these go into a paper, read the full papers. Two items need it most: the Pythia step boundaries of the three phases in Li et al. 2025, and the Zhang et al. 2024 numbers, which were read off figure descriptions. The search was about a dozen queries, not a systematic review, so "not found" means not found in that search.

## Verdict

- The sparsification story is not novel as a headline. Activation sparsity emerging with training is established, and modular structure emerging during pretraining has a direct precedent.
- The parts that look new are (a) applying a resting-state-style MLP connectome across training checkpoints, with a null that preserves firing rates, and (b) the residual connectome: ordinary co-activation networks are token-identity networks, are as reproducible in an untrained model as in a trained one, and only become a product of training once token and position effects are removed. Point (b) bears directly on the ICA functional-network papers and on MoEfication-style co-activation clustering, none of which use an untrained-model or token-identity control.
- Decision (2026-10-12): the sparsification route is not pursued as the paper's story. See the Decisions section of [LOG.md](LOG.md).

## Finding by finding

### 1. Activation sparsity (firing rate 0.50 to 0.17): known

- Li et al., "The Lazy Neuron Phenomenon: On Emergence of Activation Sparsity in Transformers", ICLR 2023 ([arXiv 2210.06313](https://arxiv.org/abs/2210.06313)). Trained transformers are very sparse (about 3.0% non-zero MLP activations in T5-Base, 6.3% in ViT-B16); larger models are sparser; the sparsity appears with random labels, random inputs and unlimited data, and is not caused by dead neurons.
- Zhang et al., "Exploring the Benefit of Activation Sparsity in Pre-training", ICML 2024 ([arXiv 2410.03440](https://arxiv.org/abs/2410.03440)). ReLU GPT, BERT and T5 (base size) pretrained on a Pile subset, checkpoints every 4,000 steps. Sparsity starts near 0.5 (symmetric initialisation), rises to about 0.9 by about 20k steps, then plateaus. This is the same curve as ours, in different models.
- Voita, Ferrando and Nalmpantis, "Neurons in Large Language Models: Dead, N-gram, Positional", Findings of ACL 2024 ([arXiv 2309.04827](https://arxiv.org/abs/2309.04827)). OPT 125M to 66B: many dead neurons in early layers, many token and n-gram detectors, more of both with scale.
- Not found: a published per-checkpoint sparsity curve for Pythia's GELU MLPs. A minor gap. A possibly new angle is the alignment of the sparsity timing with the steepest part of the loss curve.

### 2. Each neuron's coupling concentrates on fewer partners; late convergence: partly anticipated, and exposed to a confound

- Zhang et al. 2024 (above) also group co-activated neurons (MoEfication) and compare groupings across checkpoints by ARI. Sparsity stabilises first (about 20k steps); the co-activation pattern stabilises later (consecutive-checkpoint ARI about 0.9 after about 50k steps), and distant checkpoints stay dissimilar. This is close to our late-convergence finding.
- Confound: lower mean |r| and stronger concentration on top partners may follow mechanically from sparser firing. The circular-shift null preserves each neuron's firing rate, so modularity above the null is protected, but SUMMARY.md reports no null value for the top-1% concentration measure. A null or firing-rate-matched control for that measure is needed before it is claimed.

### 3. More modular and segregated networks: qualitative precedent exists

- Zhang et al., "Emergent Modularity in Pre-trained Transformers", Findings of ACL 2023 ([arXiv 2305.18390](https://arxiv.org/abs/2305.18390)). The closest precedent. Neurons are functionally specialised and cluster into functional "experts" that emerge during pretraining; perturbing an expert affects its function. Timing differs from ours: the coarse modules stabilise early (about 15% of training) and individual neurons later (about 75%), described as coarse-to-fine.
- Filan et al., "Clusterability in Neural Networks", 2021 ([arXiv 2103.03386](https://arxiv.org/abs/2103.03386)). Modularity of the weight graph (spectral clustering, shuffled-weight baseline) rises as accuracy rises during training.
- Golechha et al., "Studying Cross-cluster Modularity in Neural Networks", 2025 ([arXiv 2502.02470](https://arxiv.org/abs/2502.02470)). Pretrained models form highly enmeshed clusters, and an explicit clusterability loss is needed to make them modular. This is in tension with the previous two papers and with our result.
- Liu et al. (Tsinghua; Xie and Song corresponding), "Training-Driven Representational Geometry Modularization Predicts Brain Alignment in Language Models", 2026 ([arXiv 2602.07539](https://arxiv.org/abs/2602.07539)). Pythia 70M to 1B, 19 checkpoints: layers split into two modules by entropy and curvature after about step 4k. A layer-level geometry result, not a neuron-level one.
- Neuroscience parallels. Baum et al., "Modular Segregation of Structural Brain Networks Supports the Development of Executive Function in Youth", Current Biology 2017 ([arXiv 1608.03619](https://arxiv.org/abs/1608.03619)): brain modules segregate with age (structural networks, ages 8-22). Fair et al., "Functional Brain Networks Develop from a 'Local to Distributed' Organization", PLoS Computational Biology 2009 ([PMC2671306](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2671306/)): communities shift from anatomical proximity in children to functional relationships in adults. Our networks move the opposite way (more layer-local with training).

### 4. Collapse at step 64 and late fall in dimensionality: a likely published explanation

- Li, Agrawal, Ghosh, Teru, Santoro, Lajoie and Richards, "Tracing the Representation Geometry of Language Models from Pretraining to Post-training", 2025 ([arXiv 2509.23024](https://arxiv.org/abs/2509.23024)). Pythia (160M to 12B) and OLMo show three phases: an early collapse of representations onto dominant directions (warmup), an "entropy-seeking" expansion, and a late "compression-seeking" phase in which effective rank (RankMe) falls. This maps onto our step-64 transient (dimensionality 131 to 41) and onto the late fall (368 to 260) listed as unexplained in SUMMARY.md. Their measure is the effective rank of representations, not the dimensionality of the MLP connectome, and their Pythia step boundaries have not been checked against ours.

### 5. fMRI-style functional networks on LLM neurons: precedents exist, none over training found

- Liu et al., "Brain-Inspired Exploration of Functional Networks and Key Neurons in Large Language Models", 2025 ([arXiv 2502.20408](https://arxiv.org/abs/2502.20408)). FastICA on MLP activations treated as fMRI-like signals; recurring functional networks, usually under 2% of neurons, whose inhibition degrades performance. The same group's follow-ups apply the networks to pruning ([arXiv 2508.05239](https://arxiv.org/abs/2508.05239)) and to model fingerprinting ("FNF", [arXiv 2601.22692](https://arxiv.org/abs/2601.22692)). Code: [WhatAboutMyStar/LLM_ACTIVATION](https://github.com/WhatAboutMyStar/LLM_ACTIVATION).
- Urbina-Rodriguez et al. (incl. Luppi, Shanahan, Mediano), "A Brain-like Synergistic Core in LLMs Drives Behaviour and Learning", 2026 ([arXiv 2601.06851](https://arxiv.org/abs/2601.06851)). Integrated information decomposition (ΦID) over attention heads; the synergistic core emerges with training in Pythia-1B and looks random at the earliest checkpoints.
- Not found: a resting-state-style MLP connectome tracked across training checkpoints.

### 6. Token identity and untrained models (related to the residual connectome)

- "Transformers Are Born Biased", 2026 ([arXiv 2602.05927](https://arxiv.org/abs/2602.05927)). Randomly initialised transformers already show strong token preferences, which persist through training.
- CCN 2024 paper on language-selective units in untrained models ([PDF](https://2024.ccneuro.org/pdf/348_Paper_authored_CCN_Authors.pdf)). Untrained models reach high brain alignment, attributed mainly to token aggregation. Authors not yet recorded; check before citing.
- AlKhamissi et al., "The LLM Language Network", NAACL 2025 ([arXiv 2411.02280](https://arxiv.org/abs/2411.02280)). Language-selective units localised with neuroscience localisers; ablating them, but not random units, causes language deficits.
