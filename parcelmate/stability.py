"""Which networks are stable across datasets? (LOG.md Iteration 36.)

Andrea, 2026-10-06: the connectome differs across datasets, partly by noise and partly
systematically; keep the part that is stable. Three readings, all on the stored hard
partitions (no recomputation), every one of them against two references: the split-half
ceiling within a dataset (a network cannot be more stable across datasets than across two
halves of one dataset) and chance given the layer layout (every partition's labels shuffled
within each layer: what the per-layer composition of the networks alone predicts; a
network that is the same subset of its layer everywhere is real structure and counts as
stable, a network that is merely 'layer l' does not). Stability caused by properties of
single neurons (activity, variance) is not removed by this and is judged by the null
partition.

  A. per network    its best match (Dice) in every other dataset, against its match in the
                    other half of its own dataset (ceiling) and against layer-shuffled
                    partitions (chance): generality = (across - chance) / (within - chance),
                    0 at chance, 1 when as stable across datasets as within one.
  B. meta-networks  networks that recur across datasets: mutual best matches between EVERY
                    pair of datasets (no chain, no order) with Dice above the layer-shuffled
                    99th percentile, joined into connected components; a component's span is
                    the number of datasets it reaches. Fitted on the half-A partitions and
                    checked against the half-B ones.
  C. per unit       a consensus partition of the units' label vectors across datasets
                    (k-means on the concatenated one-hot labels, i.e. on co-assignment), and
                    each unit's consensus score: the share of datasets in which it sits with
                    the majority of its consensus network. A stable core is a set of units,
                    so a network can keep a stable core and lose a dataset-specific fringe.

Every reading is computed on the real partitions and, as the neuron-property reference, on
the null partitions (pipeline on circularly shifted data), which are expected to be stable
too: the null partition groups units by their own statistics, which do not depend much on
the text.
"""

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def contingency(a, b, ka, kb):
    """ka x kb counts of units in network i of `a` and network j of `b`."""
    return np.bincount(np.asarray(a) * kb + np.asarray(b), minlength=ka * kb).reshape(ka, kb).astype(np.float64)


def best_dice(a, b, ka, kb):
    """Per network of `a`: the Dice of its best match in `b`, that match, and its containment
    (share of the network inside its best-containing network of `b`)."""
    C = contingency(a, b, ka, kb)
    na, nb = C.sum(1), C.sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        D = 2 * C / (na[:, None] + nb[None, :])
        cont = (C / np.maximum(na[:, None], 1)).max(1)
    D = np.nan_to_num(D)
    return D.max(1), D.argmax(1), cont, D


def layer_shuffle(labels, layer, rng):
    """Labels permuted within each layer: same per-layer composition, no other structure."""
    out = np.array(labels, copy=True)
    for l in np.unique(layer):
        idx = np.flatnonzero(layer == l)
        out[idx] = labels[rng.permutation(idx)]
    return out


def effective_layers(labels, layer, k):
    """exp(entropy) of each network's distribution over layers."""
    L = int(layer.max()) + 1
    C = contingency(labels, layer, k, L)
    p = C / np.maximum(C.sum(1, keepdims=True), 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        H = -np.where(p > 0, p * np.log(p), 0.0).sum(1)
    return np.exp(H)


# ---------------------------------------------------------------------------- A

def network_stability(parts, layer, k, n_shuffle=5, rng=None):
    """A. One row per (tree, dataset, half, network).

    parts  {(tree, dataset, half): hard labels}, all on the same unit order
    """
    rng = rng or np.random.RandomState(0)
    rows = []
    for (tree, d, h), lab in sorted(parts.items()):
        other_h = 'halfB' if h == 'halfA' else 'halfA'
        within, _, within_cont, _ = best_dice(lab, parts[(tree, d, other_h)], k, k)
        others = sorted({dd for (t, dd, hh) in parts if t == tree and dd != d})
        across = {}
        chance = []
        for dd in others:
            vals = [best_dice(lab, parts[(tree, dd, hh)], k, k)[0] for hh in ('halfA', 'halfB')]
            across[dd] = np.mean(vals, 0)
            for _ in range(n_shuffle):
                hh = ('halfA', 'halfB')[rng.randint(2)]
                chance.append(best_dice(lab, layer_shuffle(parts[(tree, dd, hh)], layer, rng), k, k)[0])
        chance = np.asarray(chance)
        cm, cs = chance.mean(0), chance.std(0)
        across_mean = np.mean([across[dd] for dd in others], 0)
        thr = cm + 3 * cs
        n_present = np.sum([across[dd] > thr for dd in others], 0)
        size = np.bincount(lab, minlength=k)
        eff_l = effective_layers(lab, layer, k)
        with np.errstate(invalid='ignore', divide='ignore'):
            gen = (across_mean - cm) / (within - cm)
        for i in range(k):
            if size[i] == 0:
                continue
            row = dict(tree=tree, dataset=d, half=h, network=i, size=int(size[i]),
                       effective_layers=float(eff_l[i]), within=float(within[i]),
                       within_containment=float(within_cont[i]), across_mean=float(across_mean[i]),
                       chance=float(cm[i]), chance_sd=float(cs[i]), generality=float(gen[i]),
                       n_datasets_present=int(n_present[i]), n_other_datasets=len(others))
            for dd in others:
                row['across_' + dd] = float(across[dd][i])
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------- B

def meta_networks(labels_by_dataset, layer, k, n_shuffle=5, q=99, rng=None):
    """B. Networks that recur across datasets, from one partition per dataset.

    Mutual best matches by Dice between every pair of datasets, kept when the Dice exceeds
    the q-th percentile of best-match Dice against layer-shuffled partitions of that pair;
    nodes (dataset, network) joined into connected components. Returns the components and,
    per component, its span (datasets reached), size (networks), and its core: the units in
    the component's networks in at least `span - 1` of its datasets (all of them for a span
    below 3).
    """
    rng = rng or np.random.RandomState(0)
    ds = sorted(labels_by_dataset)
    parent = {(d, i): (d, i) for d in ds for i in range(k)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    edges = []
    for a in range(len(ds)):
        for b in range(a + 1, len(ds)):
            la, lb = labels_by_dataset[ds[a]], labels_by_dataset[ds[b]]
            _, _, _, D = best_dice(la, lb, k, k)
            null = [best_dice(la, layer_shuffle(lb, layer, rng), k, k)[0] for _ in range(n_shuffle)]
            thr = np.percentile(np.concatenate(null), q)
            ab, ba = D.argmax(1), D.argmax(0)
            for i in range(k):
                j = ab[i]
                if ba[j] == i and D[i, j] > thr:
                    edges.append((ds[a], i, ds[b], int(j), float(D[i, j])))
                    parent[find((ds[a], i))] = find((ds[b], int(j)))
    comps = {}
    for node in parent:
        comps.setdefault(find(node), []).append(node)
    out = []
    for nodes in comps.values():
        span = len({d for d, _ in nodes})
        if span < 2:
            continue
        member = np.zeros(len(layer), dtype=np.int64)
        for d, i in nodes:
            member += (labels_by_dataset[d] == i)
        need = span - 1 if span >= 3 else span
        core = np.flatnonzero(member >= need)
        out.append(dict(nodes=sorted(nodes), span=span, n_networks=len(nodes),
                        pure=len(nodes) == span, core=core))
    out.sort(key=lambda m: (-m['span'], -len(m['core'])))
    return out, edges


def match_meta(ma, mb):
    """Best Dice of each meta-network core of `ma` among the cores of `mb` (reliability of B)."""
    out = []
    for m in ma:
        a = set(m['core'].tolist())
        best = 0.0
        for n in mb:
            b = set(n['core'].tolist())
            if a or b:
                best = max(best, 2 * len(a & b) / (len(a) + len(b)))
        out.append(best)
    return np.asarray(out)


# ---------------------------------------------------------------------------- C

def consensus(labels_by_dataset, k, n_init=10, seed=0):
    """C. Consensus partition of the units' label vectors across datasets, and per-unit
    consensus scores.

    k-means on the concatenated one-hot labels (Euclidean distance between two units'
    one-hot vectors = 2 x the number of datasets that separate them, so this clusters
    co-assignment). Score of unit u in consensus network c: the share of datasets in which
    u's label is c's modal label in that dataset.
    """
    ds = sorted(labels_by_dataset)
    n = len(labels_by_dataset[ds[0]])
    blocks = [sparse.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), labels_by_dataset[d])),
                                shape=(n, k)) for d in ds]
    X = sparse.hstack(blocks).tocsr()
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)
    c = km.labels_
    agree = np.zeros(n)
    for d in ds:
        C = contingency(c, labels_by_dataset[d], k, k)
        mode = C.argmax(1)
        agree += labels_by_dataset[d] == mode[c]
    return c, agree / len(ds)


def adjusted_rand(a, b):
    return float(adjusted_rand_score(a, b))


# ---------------------------------------------------------------------------- D (needs connectivity)

def network_cohesion(M):
    """Per network of a k x k block-mean matrix (`stream_block_means`, |r| with self-pairs
    excluded): mean |r| within the network minus its mean |r| to the other networks."""
    M = np.asarray(M, dtype=np.float64)
    k = M.shape[0]
    off = (M.sum(1) - np.diag(M)) / max(k - 1, 1)
    return np.diag(M) - off, np.diag(M), off


def cohesion_across(matrices, partitions):
    """D. Cohesion of every partition's networks on every evaluation matrix, one pass per
    matrix (LOG.md Iteration 36).

    matrices    {eval dataset: matrix} (TiledMatrix or array, |r|)
    partitions  {(fit dataset, tree): hard labels or soft memberships}
    Returns rows (fit, tree, eval, network, size, within, between, cohesion). The row with
    eval == fit is the held-out cohesion within the fit dataset (fit on half A, evaluated on
    half B): the ceiling for the across-dataset rows.
    """
    from parcelmate.metrics import stream_block_means
    rows = []
    for e, R in matrices.items():
        bm = stream_block_means(R, partitions)
        for (d, tree), (M, labels) in bm.items():
            coh, w, b = network_cohesion(M)
            size = np.bincount(labels, minlength=M.shape[0])
            for i in range(M.shape[0]):
                if size[i] < 2:
                    continue
                rows.append(dict(fit=d, tree=tree, eval=e, network=i, size=int(size[i]),
                                 within=float(w[i]), between=float(b[i]), cohesion=float(coh[i])))
    return rows


# ---------------------------------------------------------------------------- stable sets vs circuits

def set_enrichment(circuit, in_set, width, n_layers, n_draws=1000, rng=None):
    """Share of a circuit's units inside a unit set, against layer-matched random sets.

    Returns observed share, expected share (exact: per layer, the set's share of that layer,
    weighted by the circuit's per-layer counts), their ratio, and a one-sided p from
    `n_draws` layer-matched draws (Iteration 36: do circuits favour the stable cores?).
    """
    from parcelmate.circuits import layer_matched_draws
    rng = rng or np.random.RandomState(0)
    circuit = np.asarray(circuit)
    in_set = np.asarray(in_set, dtype=bool)
    per_layer = np.bincount(circuit // width, minlength=n_layers)
    frac = in_set.reshape(n_layers, width).mean(1)
    exp_ = float((per_layer * frac).sum() / len(circuit))
    obs = float(in_set[circuit].mean())
    draws = layer_matched_draws(circuit, width, n_layers, n_draws, rng)
    null = in_set[draws].mean(1)
    return dict(observed=obs, expected=exp_, ratio=obs / exp_ if exp_ > 0 else np.nan,
                p=float((1 + (null >= obs).sum()) / (1 + n_draws)))


def set_graded(attribution, in_set, width, n_layers, n_perm=1000, rng=None):
    """Mean attribution rank (0-1, layer means removed) of units in a set minus outside it,
    with a one-sided p from permuting set membership within each layer."""
    from scipy import stats as _st
    rng = rng or np.random.RandomState(0)
    a = np.asarray(attribution, dtype=np.float64).ravel()
    r = _st.rankdata(a) / a.size
    R = r.reshape(n_layers, width)
    R = (R - R.mean(1, keepdims=True)).ravel()
    S = np.asarray(in_set, dtype=bool).reshape(n_layers, width)

    def diff(m):
        m = m.ravel()
        return R[m].mean() - R[~m].mean() if m.any() and (~m).any() else np.nan
    obs = diff(S)
    null = np.array([diff(np.stack([rng.permutation(row) for row in S])) for _ in range(n_perm)])
    return dict(rank_diff=float(obs), null_sd=float(np.nanstd(null)),
                p=float((1 + (null >= obs).sum()) / (1 + n_perm)))


# ---------------------------------------------------------------------------- restart consensus

def restart_consensus(labelings, k, k_out=None, n_components=100, n_init=10, seed=0):
    """Consensus partition of many hard labelings of the same units (Iteration 38): the
    stored k-means restarts of several datasets, each fitted within its own dataset.

    The co-association matrix (share of labelings in which two units share a label) is
    N x N, too large at 147k units; its leading eigenvectors are those of the one-hot design
    X (units x labelings*k, one 1 per labeling), since co-association = X X^T / n_labelings.
    So: truncated SVD of X (randomized), then k-means with `n_init` restarts on the unit
    scores. Returns the consensus labels and each unit's confidence: the mean, over the
    labelings, of the share of its consensus network's members that share its label there.
    """
    from scipy import sparse
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    L = np.asarray(labelings, dtype=np.int64)            # (n_labelings, N)
    m, N = L.shape
    k_out = k_out or k
    cols = (L + (np.arange(m) * k)[:, None]).T.ravel()
    X = sparse.csr_matrix((np.ones(N * m, dtype=np.float32), (np.repeat(np.arange(N), m), cols)), shape=(N, m * k))
    Z = TruncatedSVD(n_components=min(n_components, m * k - 1), algorithm='randomized', n_iter=5,
                     random_state=seed).fit_transform(X)
    c = KMeans(k_out, n_init=n_init, random_state=seed).fit_predict(Z)
    size = np.bincount(c, minlength=k_out).astype(np.float64)
    conf = np.zeros(N)
    for row in L:
        M = np.zeros((k_out, k))
        np.add.at(M, (c, row), 1.0)
        conf += M[c, row] / size[c]
    return c, conf / m
