# Superseded configs

Kept so the runs they produced stay reproducible from the repo; not for new work. The live experiment is [configs/ladder.yml](../ladder.yml).

- `reliability.yml` -- Iteration 8 (LOG.md), job 17366628, `results/reliability`. Five arms on |r|, scored against the pipeline-on-shifted-data null. Its scores are `figures/scores.csv` until the ladder run replaces them. Superseded because that null sits on a different denominator from the real data (Iteration 9) and because normalization was a property of the whole experiment rather than an arm.
- `reliability_norm.yml` -- Iteration 11, jobs 17368677/816/817, `results/reliability_norm`. Same five arms, all on |z| via presence-triggered normalization. Its scores are `figures/scores_norm.csv`. Superseded by `ladder.yml`, where |z| is one explicit rung and the primary comparison is the null partition on real data (Iteration 12). Its connectivity tree (r plus `surrogate_var`) is what `results/ladder` hard-links.
