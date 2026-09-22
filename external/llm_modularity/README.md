# Vendored task data from LLM_Modularity

`data/` and `config/` are copied verbatim from https://github.com/Pengrui-Han/LLM_Modularity at commit e3ac7fb (2026-09-21), MIT licence (LICENSE alongside). They define the 46 minimal-pair tasks (four domains: Lan, MD, phys, ToM) used to locate task circuits by attribution patching. parcelmate re-implements that repository's evaluation and neuron attribution in plain PyTorch (`parcelmate/patching.py`, LOG.md Iteration 23); the data are not modified.
