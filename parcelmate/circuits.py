"""Do attribution-patching task circuits and connectivity networks find the same units?
(LOG.md Iteration 30.)

Design agreed with Andrea on 2026-09-23. A task's **circuit** is the top 0.1% of ALL MLP
neurons by positive attribution, exactly as LLM_Modularity's `run_overlap.py` selects them
(`circuit_indices`); 1% is the robustness check. Our side is the 100 networks of one text
domain and half (`parcellation_<domain>_<half>.h5`), and every test is repeated per domain
and half, on the real partition and, as a reference, on the null partition.

  1. concentration (the headline)  does the circuit fall into fewer of our networks than a
                                   random neuron set of the same size AND the same layer
                                   distribution would? Both methods favour certain layers
                                   and our networks are partly layer-local, so the layer
                                   match is what makes the null reasonable.
  2. enrichment                    which networks hold more of the circuit than that null
                                   predicts; Benjamini-Hochberg over task x network pairs.
  3. shared structure              do tasks that share circuit neurons (the original's
                                   overlap ratio) also sit close together in network space
                                   (cosine of the network profiles of the units they do NOT
                                   share, minus its layer-matched expectation; permutation
                                   p), and are same-domain tasks closer than cross-domain
                                   ones? The circular all-unit version is kept for the record.

At the level of the four task domains (Iteration 31): `domain_overlap_test` (step A: do a
domain's tasks share more circuit units with each other than with other domains' tasks?)
and `domain_attribution` (step B: one map per domain, the mean over its tasks, each task
rescaled to mean |a| = 1 first), whose top units then go through tests 1 and 2.

Units are addressed by their flat index layer * width + neuron, the layout of the
attribution arrays (n_layers x width); the partitions' `coordinates` (layer, neuron) are
mapped onto it explicitly, so no ordering is assumed.
"""

import numpy as np
from scipy import stats

from parcelmate.metrics import hard_labels

# ---------------------------------------------------------------------------- circuits

def circuit_indices(attribution, pct=0.1):
    """Flat indices of the task circuit: the top `pct`% of all units by positive score.

    Mirrors LLM_Modularity's `get_sorted_indices(sign='positive')` + `get_top_neurons`:
    only strictly positive units are ranked, top_k = max(1, int(total * pct / 100)) is taken
    from that ranking, so a task with fewer positive units than top_k keeps all of them.
    """
    a = np.asarray(attribution, dtype=np.float64).ravel()
    top_k = max(1, int(a.size * pct / 100.0))
    pos = np.flatnonzero(a > 0)
    order = pos[np.argsort(-a[pos], kind='stable')]
    return order[:top_k]


def overlap_matrix(circuits):
    """The original's overlap ratio: |C_i & C_j| / |C_i| (symmetric when sizes are equal)."""
    sets = [set(map(int, c)) for c in circuits]
    n = len(sets)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = len(sets[i] & sets[j]) / len(sets[i]) if sets[i] else 0.0
    return M


def flat_labels(parcellation, coordinates, n_layers, width):
    """Network label of every unit, indexed by flat index layer * width + neuron."""
    labels = hard_labels(parcellation)
    coords = np.asarray(coordinates).astype(np.int64)
    flat = coords[:, 0] * width + coords[:, 1]
    assert len(flat) == n_layers * width and len(np.unique(flat)) == len(flat), \
        'the partition does not cover every (layer, neuron) exactly once'
    out = np.empty(n_layers * width, dtype=np.int64)
    out[flat] = labels
    return out


# ---------------------------------------------------------------------------- the null

def layer_matched_draws(circuit, width, n_layers, n_draws, rng):
    """Random unit sets with the circuit's size and per-layer counts, (n_draws, |C|) flat."""
    circuit = np.asarray(circuit)
    per_layer = np.bincount(circuit // width, minlength=n_layers)
    out = np.empty((n_draws, len(circuit)), dtype=np.int64)
    for d in range(n_draws):
        pos = 0
        for layer in np.flatnonzero(per_layer):
            m = per_layer[layer]
            out[d, pos:pos + m] = layer * width + rng.choice(width, size=m, replace=False)
            pos += m
    return out


# ---------------------------------------------------------------------------- test 1

def concentration(labels_of_set, k):
    """How few networks a unit set falls into: entropy (nats), its exponential, top shares."""
    counts = np.bincount(labels_of_set, minlength=k).astype(np.float64)
    p = counts / counts.sum()
    nz = p[p > 0]
    H = float(-(nz * np.log(nz)).sum())
    top = np.sort(p)[::-1]
    return dict(entropy=H, effective_networks=float(np.exp(H)), max_share=float(top[0]),
                top3_share=float(top[:3].sum()))


def concentration_test(circuit, labels_flat, k, width, n_layers, n_draws=1000, rng=None):
    """Observed concentration against the layer-matched null.

    Returns the observed values, the null mean and sd of the entropy, z = (obs - mean) / sd
    (negative = more concentrated than chance), and the one-sided p that a layer-matched
    random set is at least as concentrated, (1 + #{null <= obs}) / (1 + n_draws). Also the
    draws' counts per network, which test 2 reuses.
    """
    rng = rng or np.random.RandomState(0)
    obs = concentration(labels_flat[circuit], k)
    draws = layer_matched_draws(circuit, width, n_layers, n_draws, rng)
    lab = labels_flat[draws]                                      # (n_draws, |C|)
    counts = np.stack([np.bincount(row, minlength=k) for row in lab]).astype(np.float64)
    p = counts / counts.sum(1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        H = -np.where(p > 0, p * np.log(p), 0.0).sum(1)
    out = dict(obs)
    out.update(null_entropy_mean=float(H.mean()), null_entropy_sd=float(H.std(ddof=1)),
               null_effective_networks_mean=float(np.exp(H).mean()),
               entropy_z=float((obs['entropy'] - H.mean()) / H.std(ddof=1)) if H.std() > 0 else float('nan'),
               p_concentrated=float((1 + (H <= obs['entropy'] + 1e-12).sum()) / (1 + n_draws)),
               n_units=int(len(circuit)))
    return out, counts


# ---------------------------------------------------------------------------- test 2

def benjamini_hochberg(p):
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    ranked = p[order] * n / np.arange(1, n + 1)
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1]
    return np.minimum(q, 1.0)


def enrichment(circuit, labels_flat, k, null_counts):
    """Per network: observed count, expected under the null, their ratio, one-sided p."""
    obs = np.bincount(labels_flat[circuit], minlength=k).astype(np.float64)
    exp_ = null_counts.mean(0)
    n = null_counts.shape[0]
    p = (1 + (null_counts >= obs[None, :]).sum(0)) / (1.0 + n)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(exp_ > 0, obs / exp_, np.nan)
    return obs, exp_, ratio, p


# ---------------------------------------------------------------------------- test 3

def _upper(M):
    return M[np.triu_indices(M.shape[0], k=1)]


def _cosine(P):
    P = np.asarray(P, dtype=np.float64)
    n = np.linalg.norm(P, axis=1, keepdims=True)
    Q = np.where(n > 0, P / np.where(n > 0, n, 1.0), 0.0)
    return Q @ Q.T


def partial_spearman(x, y, z):
    """Spearman correlation of x and y controlling for z (ranks, then residual correlation)."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    def resid(a):
        A = np.column_stack([np.ones_like(rz), rz])
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    return float(np.corrcoef(resid(rx), resid(ry))[0, 1])


def structure_test(circuits, labels_flat, k, width, n_layers, task_domains, n_perm=1000, rng=None,
                   n_draws=200):
    """Circuit overlap against network-profile similarity, controlling for layer similarity.

    Returns Spearman(overlap, network similarity), Spearman(overlap, layer similarity), the
    partial Spearman of the first given the second with a permutation p (task labels of the
    network-similarity matrix permuted), and the same-domain minus cross-domain mean network
    similarity with a permutation p over the domain labels (its layer analogue alongside).
    """
    rng = rng or np.random.RandomState(0)
    O = overlap_matrix(circuits)
    O = (O + O.T) / 2.0
    Pn = np.stack([np.bincount(labels_flat[c], minlength=k) for c in circuits])
    Pl = np.stack([np.bincount(np.asarray(c) // width, minlength=n_layers) for c in circuits])
    Sn, Sl = _cosine(Pn), _cosine(Pl)
    o, sn, sl = _upper(O), _upper(Sn), _upper(Sl)
    out = dict(spearman_overlap_network=float(stats.spearmanr(o, sn)[0]),
               spearman_overlap_layer=float(stats.spearmanr(o, sl)[0]))
    obs = partial_spearman(o, sn, sl)
    null = []
    n = len(circuits)
    for _ in range(n_perm):
        perm = rng.permutation(n)
        null.append(partial_spearman(o, _upper(Sn[np.ix_(perm, perm)]), sl))
    null = np.asarray(null)
    out.update(partial_overlap_network_given_layer=obs,
               p_partial=float((1 + (null >= obs).sum()) / (1 + n_perm)))
    doms = np.asarray(task_domains)
    same = _upper(doms[:, None] == doms[None, :])

    def contrast(S, labels):
        s = _upper(labels[:, None] == labels[None, :])
        return float(_upper(S)[s].mean() - _upper(S)[~s].mean())
    c_obs = contrast(Sn, doms)
    c_null = np.array([contrast(Sn, rng.permutation(doms)) for _ in range(n_perm)])
    out.update(domain_contrast_network=c_obs,
               p_domain_contrast=float((1 + (c_null >= c_obs).sum()) / (1 + n_perm)),
               domain_contrast_layer=contrast(Sl, doms),
               n_tasks=int(n), n_same_domain_pairs=int(same.sum()))

    # The measures above are circular: units two circuits share carry the same network label
    # in any partition, so overlap raises network similarity by construction. The version
    # below compares only the units the two circuits do NOT share, and subtracts what
    # layer-matched random sets of those sizes and layers would give, because the cosine of
    # two count vectors grows with the number of units counted (so smaller remainders, i.e.
    # larger overlaps, would otherwise look less similar under any partition).
    X = excess_network_similarity(circuits, labels_flat, k, width, n_layers, n_draws, rng)
    x = _upper(X)
    ok = np.isfinite(x)
    out['spearman_overlap_excess'] = float(stats.spearmanr(o[ok], x[ok])[0])
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(n)
        xp = _upper(X[np.ix_(perm, perm)])
        okp = np.isfinite(xp)
        null.append(stats.spearmanr(o[okp], xp[okp])[0])
    null = np.asarray(null)
    out['p_overlap_excess'] = float((1 + (null >= out['spearman_overlap_excess']).sum()) / (1 + n_perm))

    def contrast_x(labels):
        s = _upper(labels[:, None] == labels[None, :])
        return float(np.nanmean(x[s]) - np.nanmean(x[~s]))
    cx = contrast_x(doms)
    cx_null = np.array([contrast_x(rng.permutation(doms)) for _ in range(n_perm)])
    out.update(mean_excess_similarity=float(np.nanmean(x)),
               domain_contrast_excess=cx,
               p_domain_contrast_excess=float((1 + (cx_null >= cx).sum()) / (1 + n_perm)))
    return out


def excess_network_similarity(circuits, labels_flat, k, width, n_layers, n_draws=200, rng=None):
    """Per task pair: cosine of the network profiles of the NON-shared units (C_i minus C_j
    against C_j minus C_i), minus its mean over layer-matched random sets of the same sizes.

    Zero in expectation under any partition that ignores what the circuits compute; positive
    when our networks group the distinct units that two tasks use. NaN when a remainder is
    empty (one circuit contained in the other).
    """
    rng = rng or np.random.RandomState(0)
    n = len(circuits)
    sets = [np.asarray(c) for c in circuits]
    X = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            a = np.setdiff1d(sets[i], sets[j])
            b = np.setdiff1d(sets[j], sets[i])
            if len(a) == 0 or len(b) == 0:
                continue
            pa = np.bincount(labels_flat[a], minlength=k).astype(np.float64)
            pb = np.bincount(labels_flat[b], minlength=k).astype(np.float64)
            obs = pa @ pb / (np.linalg.norm(pa) * np.linalg.norm(pb))
            da = labels_flat[layer_matched_draws(a, width, n_layers, n_draws, rng)]
            db = labels_flat[layer_matched_draws(b, width, n_layers, n_draws, rng)]
            ca = np.stack([np.bincount(r, minlength=k) for r in da]).astype(np.float64)
            cb = np.stack([np.bincount(r, minlength=k) for r in db]).astype(np.float64)
            exp_ = ((ca * cb).sum(1) / (np.linalg.norm(ca, axis=1) * np.linalg.norm(cb, axis=1))).mean()
            X[i, j] = X[j, i] = obs - exp_
    return X


# ---------------------------------------------------------------------------- domain level

def domain_overlap_test(circuits, task_domains, n_perm=10000, rng=None):
    """Step A (Andrea, 2026-09-23): do tasks of one domain share more circuit units with each
    other than with tasks of other domains?

    Overlap is the original's ratio |C_i & C_j| / |C_i|, symmetrised. Returns one row for all
    domains together (mean within-domain pairs minus mean cross-domain pairs) and one per
    domain (its within pairs minus its pairs with every other domain), each with a one-sided
    permutation p over the task-domain labels.
    """
    rng = rng or np.random.RandomState(0)
    O = overlap_matrix(circuits)
    O = (O + O.T) / 2.0
    o = _upper(O)
    doms = np.asarray(task_domains)
    iu = np.triu_indices(len(doms), k=1)
    names = sorted(set(doms))

    def stats_of(labels):
        a, b = labels[iu[0]], labels[iu[1]]
        same = a == b
        out = {'all': o[same].mean() - o[~same].mean()}
        for d in names:
            w = same & (a == d)
            x = (a != b) & ((a == d) | (b == d))
            out[d] = o[w].mean() - o[x].mean() if w.any() and x.any() else np.nan
        return out, same

    obs, same = stats_of(doms)
    null = [stats_of(rng.permutation(doms))[0] for _ in range(n_perm)]
    rows = []
    for d in ['all'] + names:
        nd = np.array([n[d] for n in null])
        a, b = doms[iu[0]], doms[iu[1]]
        if d == 'all':
            w, x = same, ~same
        else:
            w = same & (a == d)
            x = (a != b) & ((a == d) | (b == d))
        rows.append(dict(task_domain=d, n_tasks=int((doms == d).sum()) if d != 'all' else len(doms),
                         mean_within=float(o[w].mean()), mean_across=float(o[x].mean()),
                         difference=float(obs[d]),
                         p=float((1 + np.sum(nd[np.isfinite(nd)] >= obs[d])) / (1 + np.isfinite(nd).sum()))))
    return rows


def domain_attribution(attributions, task_domains, rescale=True):
    """Step B: one attribution map per task domain, the mean over its tasks.

    rescale=True divides each task's map by its mean absolute value first, so every task
    weighs the same whatever the scale of its scores (Andrea's choice, 2026-09-23); False is
    the plain mean, reported as a check.
    """
    out = {}
    doms = np.asarray(task_domains)
    for d in sorted(set(doms)):
        maps = []
        for a, t in zip(attributions, doms):
            if t != d:
                continue
            a = np.asarray(a, dtype=np.float64)
            if rescale:
                s = np.abs(a).mean()
                a = a / s if s > 0 else a
            maps.append(a)
        out[d] = np.mean(maps, 0)
    return out


# ---------------------------------------------------------------------------- real vs null

def paired_real_null(real_deficit, null_deficit, n_perm=10000, rng=None):
    """Paired test across tasks: is a circuit more concentrated on the real partition than
    on the null partition, each measured against its OWN layer-matched baseline?

    deficit = observed entropy minus the mean entropy of layer-matched random sets on that
    partition (nats; negative = more concentrated than chance). Each partition's baseline
    absorbs its own network sizes and layer layout, so the difference d = real - null is
    what the real networks' co-activation structure adds beyond the per-unit properties the
    null partition also captures (Iteration 33). Returns the mean and median d, the share
    of tasks with d < 0, a one-sided Wilcoxon signed-rank p and a sign-flip permutation p
    on the mean (both for d < 0).
    """
    rng = rng or np.random.RandomState(0)
    d = np.asarray(real_deficit, dtype=np.float64) - np.asarray(null_deficit, dtype=np.float64)
    out = dict(n=int(len(d)), mean_diff=float(d.mean()), median_diff=float(np.median(d)),
               frac_real_more_concentrated=float((d < 0).mean()))
    out['p_wilcoxon'] = float(stats.wilcoxon(d, alternative='less').pvalue) if np.any(d != 0) else 1.0
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (flips * np.abs(d)[None, :]).mean(1)
    out['p_signflip'] = float((1 + (null <= d.mean()).sum()) / (1 + n_perm))
    return out


# ---------------------------------------------------------------------------- graded (all units)

def graded_network_variance(attributions, labels_flat, k, width, n_layers, n_perm=200, rng=None):
    """How much of the variation in attribution across ALL units the networks explain,
    beyond layer (Iteration 35). No top-pct cut, so no circuit size to choose.

    attributions  (T, n_layers * width) flat maps, one row per task (or task domain)

    Per map: ranks of the attribution (signed, so high = supports the correct answer) scaled
    to [0, 1], minus each layer's mean rank; eta2 = between-network sum of squares of these
    residuals over their total sum of squares. Null: the network labels permuted WITHIN each
    layer (every layer keeps its count of units per network), the same layer match as test
    1; one permutation is shared by all maps of a call. Returns per map eta2 of layer, the
    observed network eta2 beyond layer, the null mean and sd, excess = observed - null mean,
    z and the one-sided p (1 + #{null >= obs}) / (1 + n_perm).
    """
    from scipy import sparse
    rng = rng or np.random.RandomState(0)
    A = np.atleast_2d(np.asarray(attributions, dtype=np.float64))
    T, N = A.shape
    assert N == n_layers * width
    R = np.apply_along_axis(stats.rankdata, 1, A) / N            # (T, N)
    layer = np.arange(N) // width
    lm = np.stack([np.bincount(layer, weights=r, minlength=n_layers) for r in R]) / width
    total = ((R - R.mean(1, keepdims=True)) ** 2).sum(1)
    eta_layer = (width * ((lm - R.mean(1, keepdims=True)) ** 2).sum(1)) / total
    E = (R - lm[:, layer]).T                                       # (N, T) residuals
    ss = (E ** 2).sum(0)

    def eta(labels):
        G = sparse.csr_matrix((np.ones(N), (labels, np.arange(N))), shape=(k, N))
        n = np.asarray(G.sum(1)).ravel()
        S = G @ E                                                  # (k, T)
        ok = n > 0
        return ((S[ok] ** 2) / n[ok, None]).sum(0) / ss

    obs = eta(labels_flat)
    null = np.empty((n_perm, T))
    lab = labels_flat.copy()
    for i in range(n_perm):
        for l in range(n_layers):
            sl = slice(l * width, (l + 1) * width)
            lab[sl] = rng.permutation(labels_flat[sl])
        null[i] = eta(lab)
    mu, sd = null.mean(0), null.std(0, ddof=1)
    return [dict(eta2_layer=float(eta_layer[t]), eta2_network=float(obs[t]), null_mean=float(mu[t]),
                 null_sd=float(sd[t]), excess=float(obs[t] - mu[t]),
                 z=float((obs[t] - mu[t]) / sd[t]) if sd[t] > 0 else float('nan'),
                 p=float((1 + (null[:, t] >= obs[t]).sum()) / (1 + n_perm))) for t in range(T)]


# ---------------------------------------------------------------------------- held-out enrichment

def heldout_enrichment(observed, expected, task_domains, n_top=5):
    """Do a task domain's tasks reuse the same networks? (Iteration 35.)

    observed, expected  (n_tasks, k) circuit counts per network and their layer-matched
                        expectation, as `enrichment` returns them, on ONE partition
    For each task: the `n_top` networks with the largest summed excess (observed - expected)
    over the OTHER tasks of its domain; then the held-out task's share of circuit units in
    those networks, its expected share, and their ratio. Leave-one-task-out, so the choice
    of networks never sees the task it is scored on (picking them on the same data inflates
    every domain). NaN for a task alone in its domain.
    """
    O = np.asarray(observed, dtype=np.float64)
    E = np.asarray(expected, dtype=np.float64)
    doms = np.asarray(task_domains)
    out = []
    for i in range(len(O)):
        others = np.flatnonzero((doms == doms[i]) & (np.arange(len(O)) != i))
        if len(others) == 0:
            out.append(dict(share=np.nan, expected_share=np.nan, ratio=np.nan))
            continue
        top = np.argsort(-(O[others] - E[others]).sum(0), kind='stable')[:n_top]
        n = O[i].sum()
        share, exp_share = O[i, top].sum() / n, E[i, top].sum() / n
        out.append(dict(share=float(share), expected_share=float(exp_share),
                        ratio=float(share / exp_share) if exp_share > 0 else np.nan))
    return out
