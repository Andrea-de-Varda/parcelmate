"""Rich club of the pooled residual connectome (LOG.md Iteration 45, info/RICH_CLUB_PLAN.md).

Are the most strongly coupled MLP units coupled mainly to each other? Edge weights are
w_ij = |r_ij| of a half connectome (the Fisher mean over its samples), diagonal excluded;
the profile transform the clustering uses is not applied.

Hubs are selected on one half and evaluated on the other. Selecting the strongest units
and measuring their coupling on the same data is biased upward (units that are strong
partly by noise share noisy edges); `tests/verify_iter45_richclub.py` documents the bias.

For a ranking of the selection half and the rich set R(n) = its top n units, on the
evaluation half (strengths s, total 2W = sum_i s_i):

  N1 (configuration)  E[w_ij] = s_i s_j / 2W
  club                rho(n) = sum_{i<j in R} w_ij / sum_{i<j in R} E[w_ij]
  hub                 c_i(n) = sum_{j in R, j != i} w_ij / sum_{j in R, j != i} E[w_ij]

Everything is streamed from row blocks, so the same code serves a tiled float16 half
(`bigconn.TiledMatrix`, which returns float32 |r|) and a dense array (`DenseRows`).
"""

import time

import numpy as np

from parcelmate.util import stderr

N_FP16 = 1 << 15            # bit patterns of the non-negative float16 values
DENSITIES = (0.001, 0.01, 0.05)          # binary degree for the secondary hub rankings
PART_DENSITIES = (0.01, 0.02, 0.1)        # thresholded graphs for participation (Andrea 2026-10-20)
DEFAULT_BLOCK = 2048


class DenseRows:
    """A dense matrix behind the `TiledMatrix` interface: rows as float32 |r|."""

    def __init__(self, R):
        self.R = np.asarray(R)
        self.shape = self.R.shape

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        out = np.array(self.R[key], dtype=np.float32)
        np.nan_to_num(out, copy=False)
        return np.abs(out, out=out)

    def close(self):
        pass


def _blocks(M, block, verbose, label):
    N = M.shape[0]
    t0 = time.time()
    n_blocks = int(np.ceil(N / block))
    for bi, s in enumerate(range(0, N, block)):
        e = min(s + block, N)
        rows = M[s:e]
        rows[np.arange(e - s), np.arange(s, e)] = 0.0   # no self-coupling
        yield s, e, rows
        if verbose:
            stderr('\r  %s: block %d/%d (%.0f s)' % (label, bi + 1, n_blocks, time.time() - t0))
    if verbose:
        stderr('\n')


def onehot(labels, k=None):
    labels = np.asarray(labels)
    k = int(labels.max()) + 1 if k is None else k
    H = np.zeros((len(labels), k), dtype=np.float32)
    H[np.arange(len(labels)), labels] = 1.0
    return H


def strength_pass(M, partitions=None, block=DEFAULT_BLOCK, verbose=True, label='strength'):
    """One pass: strengths, the exact histogram of |r| and per-network strengths.

    Returns dict(strength=(N,) float64, hist=(2**15,) int64 counts of every non-negative
    float16 value (off-diagonal entries of every row, both triangles), kappa={name: (N, k)
    float64 strength of each unit towards each network of the partition `name`}).
    The histogram is exact because the stored values are float16: binning by bit pattern
    loses nothing (for a dense float input it bins the float16 rounding).
    """
    N = M.shape[0]
    strength = np.zeros(N, dtype=np.float64)
    hist = np.zeros(N_FP16, dtype=np.int64)
    H = {name: onehot(lab) for name, lab in (partitions or {}).items()}
    kappa = {name: np.zeros((N, h.shape[1]), dtype=np.float64) for name, h in H.items()}
    for s, e, rows in _blocks(M, block, verbose, label):
        strength[s:e] = rows.sum(1, dtype=np.float64)
        bits = rows.astype(np.float16).view(np.uint16).ravel()
        hist += np.bincount(bits, minlength=N_FP16)[:N_FP16]
        hist[0] -= e - s            # the zeroed diagonal
        for name, h in H.items():
            kappa[name][s:e] = rows @ h
    return dict(strength=strength, hist=hist, kappa=kappa)


def fp16_value(bit):
    return float(np.array([bit], dtype=np.uint16).view(np.float16)[0])


def density_thresholds(hist, n_pairs, densities=DENSITIES):
    """Exact |r| thresholds keeping at most a share `density` of the `n_pairs` ordered
    off-diagonal pairs: the smallest float16 value t with count(|r| >= t) <= density * n_pairs.

    Returns {density: (threshold, achieved density)}.
    """
    above = np.cumsum(hist[::-1])[::-1]      # above[b] = count(bit >= b)
    out = {}
    for d in densities:
        target = d * n_pairs
        ok = np.nonzero(above <= target)[0]
        b = int(ok[0]) if len(ok) else N_FP16 - 1
        out[d] = (fp16_value(b), float(above[b]) / n_pairs if b < N_FP16 else 0.0)
    return out


def degree_pass(M, thresholds, partitions=None, part_keys=(), block=DEFAULT_BLOCK, verbose=True,
                label='degree'):
    """Binary degree of every unit at each threshold, and on the thresholded graphs named in
    `part_keys` the number of each unit's edges into each network of every partition.

    Returns (deg={key: (N,) int64}, kappa={(key, name): (N, k) float64}).
    """
    N = M.shape[0]
    deg = {key: np.zeros(N, dtype=np.int64) for key in thresholds}
    H = {name: onehot(lab) for name, lab in (partitions or {}).items()}
    kappa = {(key, name): np.zeros((N, h.shape[1]), dtype=np.float64)
             for key in part_keys for name, h in H.items()}
    for s, e, rows in _blocks(M, block, verbose, label):
        for key, t in thresholds.items():
            above = rows >= np.float32(t)
            deg[key][s:e] = above.sum(1)
            if key in part_keys:
                A = above.astype(np.float32)
                for name, h in H.items():
                    kappa[(key, name)][s:e] = A @ h
    return deg, kappa


def ranking(score, live, tiebreak=None):
    """Live units ordered by decreasing score (then decreasing tiebreak, then index)."""
    idx = np.nonzero(live)[0]
    keys = [idx, -(tiebreak[idx] if tiebreak is not None else np.zeros(len(idx))), -score[idx]]
    return idx[np.lexsort(keys)]


def n_grid(n_live, n_min=50, frac=0.2, points=20):
    """Rich-set sizes on a log grid from n_min to frac * n_live."""
    hi = max(int(frac * n_live), n_min)
    return np.unique(np.round(np.geomspace(n_min, hi, points)).astype(np.int64))


def eval_pass(M, orders, grid, layers=None, block=DEFAULT_BLOCK, verbose=True, label='eval'):
    """For each ranking (`orders[name]`, unit indices by decreasing rank on the selection
    half), the coupling of every hub to the hubs above the cut, on the matrix M.

    Returns {name: (n_max, G) float64}: entry [p, g] is sum_{q < grid[g]} w(order[p], order[q])
    for the hub of rank p (meaningful for p < grid[g]; the self term is the zeroed diagonal).
    With `layers` (N,) it returns {name: (all, same)}, `same` restricted to hubs q in the
    layer of hub p.
    """
    n_max = int(grid[-1])
    tops = {name: np.asarray(o[:n_max]) for name, o in orders.items()}
    pos = {}
    for name, top in tops.items():
        p = np.full(M.shape[0], -1, dtype=np.int64)
        p[top] = np.arange(len(top))
        pos[name] = p
    out = {name: np.zeros((len(top), len(grid)), dtype=np.float64) for name, top in tops.items()}
    same = {name: np.zeros((len(top), len(grid)), dtype=np.float64) for name, top in tops.items()}
    cols = grid - 1
    for s, e, rows in _blocks(M, block, verbose, label):
        for name, top in tops.items():
            p = pos[name][s:e]
            local = np.nonzero(p >= 0)[0]
            if not len(local):
                continue
            sub = rows[local][:, top].astype(np.float64)
            if layers is not None:
                mask = layers[top][None, :] == layers[s + local][:, None]
                same[name][p[local]] = np.cumsum(sub * mask, axis=1)[:, cols]
            np.cumsum(sub, axis=1, out=sub)
            out[name][p[local]] = sub[:, cols]
    if layers is None:
        return out
    return {name: (out[name], same[name]) for name in tops}


def club_stats(numer, order, strength, grid, numer_same=None, layers=None):
    """rho(n) and the hubs' c_i(n) from `eval_pass` output for one ranking.

    numer     (n_max, G) from eval_pass
    order     the ranking (selection half)
    strength  (N,) strengths of the EVALUATION half
    Returns dict(rho=(G,), W=(G,), EW=(G,), c=[(n_g,) per g]). With `numer_same` and
    `layers`, also the same split by layer: rho_same/rho_other and c_same/c_other, each
    hub's coupling to the hubs of its own layer and of the other layers, against the N1
    expectation over the same hubs.
    """
    two_w = float(strength.sum())
    top = np.asarray(order[:int(grid[-1])])
    s_top = strength[top]
    cum_s = np.cumsum(s_top)
    cum_s2 = np.cumsum(s_top ** 2)
    split = numer_same is not None
    if split:
        lay = layers[top]
        n_layers = int(layers.max()) + 1
        onel = np.zeros((len(top), n_layers))
        onel[np.arange(len(top)), lay] = 1
        cum_sl = np.cumsum(onel * s_top[:, None], axis=0)        # strength by layer above each cut
    rho, W, EW, c = [], [], [], []
    rs, ro, cs_, co_ = [], [], [], []
    for g, n in enumerate(grid):
        n = int(n)
        num = numer[:n, g]
        S = cum_s[n - 1]
        exp_i = s_top[:n] * (S - s_top[:n]) / two_w
        w_club = num.sum() / 2.0
        e_club = (S * S - cum_s2[n - 1]) / 2.0 / two_w
        W.append(w_club)
        EW.append(e_club)
        rho.append(w_club / e_club if e_club > 0 else np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            c.append(num / exp_i)
        if split:
            num_s = numer_same[:n, g]
            exp_s = s_top[:n] * (cum_sl[n - 1][lay[:n]] - s_top[:n]) / two_w
            num_o, exp_o = num - num_s, exp_i - exp_s
            with np.errstate(divide='ignore', invalid='ignore'):
                rs.append(num_s.sum() / exp_s.sum() if exp_s.sum() > 0 else np.nan)
                ro.append(num_o.sum() / exp_o.sum() if exp_o.sum() > 0 else np.nan)
                cs_.append(num_s / exp_s)
                co_.append(num_o / exp_o)
    out = dict(rho=np.asarray(rho), W=np.asarray(W), EW=np.asarray(EW), c=c)
    if split:
        out.update(rho_same=np.asarray(rs), rho_other=np.asarray(ro), c_same=cs_, c_other=co_)
    return out


def participation(kappa, strength):
    """Participation coefficient P_i = 1 - sum_s (kappa_is / s_i)^2 (Guimera and Amaral 2005)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        f = kappa / strength[:, None]
        P = 1.0 - (f ** 2).sum(1)
    P[strength <= 0] = np.nan
    return P


def effective_number(labels, k):
    """exp(entropy) of the label distribution: the number of networks a set effectively spans."""
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def longest_run(flags):
    """(start, length) of the longest run of True in a boolean sequence."""
    best, start, cur, cur_start = 0, -1, 0, 0
    for i, f in enumerate(flags):
        if f:
            if cur == 0:
                cur_start = i
            cur += 1
            if cur > best:
                best, start = cur, cur_start
        else:
            cur = 0
    return start, best


def expected_participation(k, labels, live):
    """Participation expected if each unit's k links went to k distinct other live units drawn
    at random, regardless of network (the random-spread reference).

    For unit i in network g with M = L - 1 possible partners, of which m_s in network s
    (m_s = L_s - [s == g]), the share f_s of its k links in network s is hypergeometric with
    E[f_s] = p_s = m_s / M and Var[f_s] = p_s (1 - p_s) (M - k) / (k (M - 1)). So
        E[P_i] = 1 - sum_s E[f_s^2] = (1 - sum_s p_s^2) (1 - (M - k) / (k (M - 1))).
    k (N,) link counts; labels (N,) network of every unit; live (N,) bool. NaN where k = 0
    or the unit is dead.
    """
    k = np.asarray(k, dtype=np.float64)
    labels = np.asarray(labels)
    L_s = np.bincount(labels[live], minlength=int(labels.max()) + 1).astype(np.float64)
    M = float(live.sum() - 1)
    sum_sq = ((L_s ** 2).sum() - 2 * L_s[labels] + 1) / M ** 2      # sum_s p_s^2, own network minus self
    with np.errstate(divide='ignore', invalid='ignore'):
        E = (1 - sum_sq) * (1 - (M - k) / (k * (M - 1)))
    E[(k <= 0) | ~live] = np.nan
    return E


# ---------------------------------------------------------------------- fixed-k participation
#
# Revised 2026-10-22 (Andrea). On a globally thresholded graph a weak unit keeps only its very
# closest partners and a hub thousands, so participation mixes being a hub with how deep into
# its partner list a unit's links reach. Here every unit is read on its own top-k partners (the
# same k for all), against two references: random spread over all live units, and random
# spread within the layers the unit's partners actually come from.

TOPK_FRACS = (0.01, 0.02, 0.1)           # k as a fraction of the live units


def layer_network_table(labels, layers, live, n_layers=None, n_networks=None):
    """counts[l, s] = live units in layer l and network s; sizes[l] = live units in layer l."""
    n_layers = int(layers.max()) + 1 if n_layers is None else n_layers
    n_networks = int(labels.max()) + 1 if n_networks is None else n_networks
    counts = np.zeros((n_layers, n_networks), dtype=np.float64)
    np.add.at(counts, (layers[live], labels[live]), 1.0)
    return counts, counts.sum(1)


def layer_matched_expectation(C, counts, sizes):
    """E[P] if each of a unit's partners were replaced by a random live unit of the SAME layer
    (drawn without replacement within each layer, independently across layers).

    C (b, L) partners of each unit per layer (rows sum to k); counts (L, K), sizes (L,) from
    `layer_network_table`. For layer l with c_l partners, the number landing in network s is
    hypergeometric: mean c_l q_ls, variance c_l q_ls (1 - q_ls) (N_l - c_l) / (N_l - 1), with
    q_ls = counts[l, s] / N_l. Then f_s = (sum over layers) / k and E[P] = 1 - sum_s E[f_s^2].
    The unit itself is not excluded from its own layer (an O(1 / N_l) effect).
    """
    k = C.sum(1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        Q = np.where(sizes[:, None] > 0, counts / sizes[:, None], 0.0)
        fpc = np.where(sizes[None, :] > 1, (sizes[None, :] - C) / (sizes[None, :] - 1), 0.0)
    mean = C @ Q / k                                              # (b, K)
    var = (C * fpc) @ (Q * (1 - Q)) / k ** 2
    return 1.0 - (mean ** 2 + var).sum(1)


def topk_pass(M, ks, labels, layers, live, block=1024, verbose=True, label='top-k'):
    """For every live unit, its top-k partners on M (live units only, self excluded), for each
    k in `ks`: observed participation over `labels`, the random-spread and layer-matched
    expectations, and the shares of partners in the unit's own network and own layer.

    Returns {k: dict(P, E_random, E_layer, own_network, own_layer)}, each (N,) float64, NaN
    for dead units.
    """
    N = M.shape[0]
    ks = sorted(int(k) for k in ks)
    kmax = ks[-1]
    n_layers = int(layers.max()) + 1
    n_networks = int(labels.max()) + 1
    counts, sizes = layer_network_table(labels, layers, live, n_layers, n_networks)
    out = {k: {key: np.full(N, np.nan) for key in ('P', 'E_random', 'E_layer', 'own_network', 'own_layer')}
           for k in ks}
    dead_cols = ~live
    for k in ks:
        out[k]['E_random'][:] = expected_participation(np.full(N, k), labels, live)
    for s, e, rows in _blocks(M, block, verbose, label):
        rows[np.arange(e - s), np.arange(s, e)] = -1.0          # no self (the diagonal is zeroed)
        rows[:, dead_cols] = -1.0
        idx_rows = np.nonzero(live[s:e])[0]
        if not len(idx_rows):
            continue
        R = rows[idx_rows]
        top = np.argpartition(-R, kmax - 1, axis=1)[:, :kmax]   # the kmax strongest, unordered
        vals = np.take_along_axis(R, top, axis=1)
        order = np.argsort(-vals, axis=1, kind='stable')
        top = np.take_along_axis(top, order, axis=1)            # strongest first
        units = s + idx_rows
        for k in ks:
            t = top[:, :k]
            lab = labels[t]
            lay = layers[t]
            F = np.zeros((len(units), n_networks))
            np.add.at(F, (np.repeat(np.arange(len(units)), k), lab.ravel()), 1.0)
            C = np.zeros((len(units), n_layers))
            np.add.at(C, (np.repeat(np.arange(len(units)), k), lay.ravel()), 1.0)
            o = out[k]
            o['P'][units] = 1.0 - ((F / k) ** 2).sum(1)
            o['E_layer'][units] = layer_matched_expectation(C, counts, sizes)
            o['own_network'][units] = (lab == labels[units][:, None]).mean(1)
            o['own_layer'][units] = (lay == layers[units][:, None]).mean(1)
    return out
