"""The residual connectome (LOG.md Iterations 38-39): remove what token identity and position
predict from every unit's activation before correlating.

For every unit, activation = a(token type) + b(position in the window) + residual, fitted by
ordinary least squares on a REFERENCE portion of text (pooled over the run's domains,
disjoint from the analysed samples) and applied to the analysed samples. The fit is joint,
exact and needs one forward pass over the reference: the normal equations of the two-factor
additive model only involve

  S[t] = sum of activations over tokens of type t        (types x units)
  P[k] = sum of activations at position k                (positions x units)
  n[t], m[k] = counts, and C[t, k] = how often type t occurs at position k,

and are solved by alternating a = (S - C b) / n and b = (P - C^T a) / m until they stop
changing (backfitting on sufficient statistics; the identifying constraint is that b has
zero mean over the reference tokens). Token types seen fewer than `min_count` times in the
reference are dropped from the fit and from the analysed samples; the first token of a
window is kept (position 0 has its own b).

Exploration (Iteration 38) showed that this residual connectome, unlike the ordinary one,
is reliable within and shared across datasets only in trained models; adding the previous
token's identity changed nothing (run 6).
"""

import numpy as np
import torch

from parcelmate.util import stderr


class TokenPositionFit:
    """Fitted a(type) and b(position) for every unit, and the vocabulary that was fitted."""

    def __init__(self, look, a, b, counts, n_iter, change, n_ref_tokens, min_count, unexplained=None,
                 dead_threshold=1e-6):
        self.look = look            # (V,) long: row of `a` for every vocabulary id, -1 if dropped
        self.a = a                  # (types, units) float32
        self.b = b                  # (seq_len, units) float32
        self.counts = counts        # (V,) reference count of every vocabulary id
        self.n_iter = n_iter
        self.change = change
        self.n_ref_tokens = n_ref_tokens
        self.min_count = min_count
        # Per unit: share of its reference variance that token type and position leave
        # unexplained (RSS / TSS). A unit below `dead_threshold` is a function of the token and
        # the position alone (Pythia's first MLP, which in the parallel GPT-NeoX block sees only
        # the token embedding): its residual is float rounding, so it is dropped (Iteration 41).
        self.unexplained = unexplained
        self.live = None if unexplained is None else (unexplained > dead_threshold)

    def provenance(self):
        out = dict(residualize='token_position', residual_ref_tokens=int(self.n_ref_tokens),
                   residual_types=int(self.a.shape[0]), residual_min_count=int(self.min_count),
                   residual_backfit_iterations=int(self.n_iter), residual_backfit_change=float(self.change))
        if self.live is not None:
            out['residual_dead_units'] = int((~self.live).sum())
        return out


def _mlp_capture(model):
    from parcelmate.model import mlp_projections
    captured = {}

    def make_hook(layer):
        def hook(module, inputs):
            captured[layer] = inputs[0]
        return hook
    hooks = [p.register_forward_pre_hook(make_hook(l)) for l, p in mlp_projections(model)]
    return hooks, captured


@torch.no_grad()
def fit_token_position(model, input_ids, attention_mask, vocab_size, batch_size=8, min_count=3,
                       max_iter=50, tol=1e-5, device=None, verbose=True, indent=0):
    """Joint least-squares fit of activation = a(type) + b(position), all MLP units.

    input_ids, attention_mask  (windows, seq_len) reference text (any number of domains)
    """
    device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    seq_len = input_ids.shape[1]
    ids_all = input_ids[attention_mask.bool()]
    cnt_all = torch.bincount(ids_all.reshape(-1), minlength=vocab_size)
    keep_ids = torch.nonzero(cnt_all >= min_count).reshape(-1)
    look = torch.full((vocab_size,), -1, dtype=torch.long)
    look[keep_ids] = torch.arange(len(keep_ids))
    look_d = look.to(device)
    U = len(keep_ids)
    hooks, captured = _mlp_capture(model)
    S = P = Q = None
    n = torch.zeros(U, dtype=torch.float64, device=device)
    m = torch.zeros(seq_len, dtype=torch.float64, device=device)
    C = torch.zeros(U, seq_len, dtype=torch.float64, device=device)
    n_tok = 0
    pos_grid = torch.arange(seq_len, device=device)
    for i in range(0, input_ids.shape[0], batch_size):
        ids = input_ids[i:i + batch_size].to(device)
        mask = attention_mask[i:i + batch_size].to(device).bool()
        captured.clear()
        model(input_ids=ids, attention_mask=mask.long())
        X = torch.cat([captured[l] for l in range(len(captured))], -1).float()   # (B, L, units)
        rows = look_d[ids]                                                        # (B, L)
        sel = mask & (rows >= 0)
        x = X[sel]                                                                # (tokens, units)
        r = rows[sel]
        p = pos_grid.expand_as(ids)[sel]
        if S is None:
            S = torch.zeros(U, x.shape[1], dtype=torch.float64, device=device)
            P = torch.zeros(seq_len, x.shape[1], dtype=torch.float64, device=device)
        S.index_add_(0, r, x.double())
        P.index_add_(0, p, x.double())
        Q = (x.double() ** 2).sum(0) if Q is None else Q + (x.double() ** 2).sum(0)
        n += torch.bincount(r, minlength=U).double()
        m += torch.bincount(p, minlength=seq_len).double()
        C.index_put_((r, p), torch.ones_like(r, dtype=torch.float64), accumulate=True)
        n_tok += int(sel.sum())
        del X, x
    for h in hooks:
        h.remove()
    # Backfitting on the sufficient statistics, in float32 (the float64 sums are exact; at
    # 160m a float64 types x units table is 9 GB), b constrained to zero mean over tokens.
    S64, P64 = S, P
    S, P, C, n, m = S.float(), P.float(), C.float(), n.float(), m.float()
    b = torch.zeros_like(P)
    a = S / n[:, None]
    change = float('inf')
    it = 0
    mm = torch.clamp(m, min=1.0)[:, None]
    for it in range(1, max_iter + 1):
        a_new = (S - C @ b) / n[:, None]
        b_new = (P - C.T @ a_new) / mm
        b_new -= (m[:, None] * b_new).sum(0, keepdim=True) / m.sum()
        a_new = (S - C @ b_new) / n[:, None]
        change = float(torch.max(torch.abs(a_new - a)).item())
        a, b = a_new, b_new
        if change < tol:
            break
    if verbose:
        stderr('%sresidual fit: %d reference tokens, %d token types (>= %d occurrences), '
               'backfitting converged in %d iterations (max change %.1e)\n'
               % (' ' * indent, n_tok, U, min_count, it, change))
    model.to('cpu')
    # At the least-squares solution sum(x * fitted) = sum(fitted^2), so RSS = sum x^2 - sum_t a_t S_t
    # - sum_k b_k P_k, exactly, from the float64 sums.
    rss = Q - (a.double() * S64).sum(0) - (b.double() * P64).sum(0)
    tss = Q - P64.sum(0) ** 2 / n_tok
    unexplained = torch.where(tss > 0, rss / torch.clamp(tss, min=1e-300), torch.zeros_like(tss)).clamp(min=0).cpu().numpy()
    fit = TokenPositionFit(look, a.float().cpu(), b.float().cpu(), cnt_all, it, change, n_tok, min_count,
                           unexplained=unexplained)
    if verbose:
        stderr('%s%d of %d units fully explained by token type and position (dropped)\n'
               % (' ' * indent, int((~fit.live).sum()), len(fit.live)))
    del S, P, C, S64, P64
    torch.cuda.empty_cache()
    return fit


def residualize_timecourses(fit, timecourses, input_ids, attention_mask, chunk=8192, device=None):
    """Subtract a(type) + b(position) from a units x tokens array (the layout
    `get_timecourses` returns: tokens in row-major order over windows and positions, masked)
    in place, drop tokens whose type was not fitted, and return the kept view (units x kept
    tokens, the first columns of the same buffer) and the number of tokens dropped."""
    device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    mask = attention_mask.bool()
    seq_len = input_ids.shape[1]
    ids = input_ids[mask]
    pos = torch.arange(seq_len).expand_as(input_ids)[mask]
    rows = fit.look[ids]
    keep = rows >= 0
    assert timecourses.shape[1] == len(ids), 'timecourses and tokens do not match'
    a = fit.a.to(device)
    b = fit.b.to(device)
    w = 0
    for s in range(0, len(ids), chunk):
        e = min(s + chunk, len(ids))
        k = keep[s:e]
        if not bool(k.any()):
            continue
        X = torch.as_tensor(timecourses[:, s:e][:, k.numpy()], device=device)
        X -= (a[rows[s:e][k].to(device)] + b[pos[s:e][k].to(device)]).T
        nk = X.shape[1]
        timecourses[:, w:w + nk] = X.cpu().numpy()
        w += nk
    del a, b
    return timecourses[:, :w], int(len(ids) - w)


def reference_inputs(load, domains, n_analysed, n_reference):
    """Reference windows per domain: the windows that follow the analysed ones in the domain's
    own document stream (same seed and order), so they never overlap the analysed text and the
    analysed text is exactly what a run without residualization sees.

    load(domain, n_tokens) -> (input_ids, attention_mask) with that many tokens."""
    ids_all, mask_all = [], []
    for d in domains:
        ids, mask = load(d, n_analysed + n_reference)
        n_win = int(np.ceil(n_analysed / ids.shape[1]))
        ids_all.append(ids[n_win:])
        mask_all.append(mask[n_win:])
    return torch.cat(ids_all), torch.cat(mask_all)


@torch.no_grad()
def fit_token_position_large(model, input_ids, attention_mask, vocab_size, batch_size=8, min_count=3,
                             max_iter=50, tol=1e-5, unit_chunk=8192, device=None, verbose=True, indent=0):
    """`fit_token_position` for models whose types x units table does not fit twice on a GPU
    in float64 (Qwen3.5: 147k-295k units; Iteration 40). Same least-squares solution:

    - per-type and per-position sums accumulated layer by layer, in float32 (relative error
      about 1e-4 at a million tokens, far below the residuals' scale);
    - the backfitting solved independently for each chunk of `unit_chunk` units (the two-factor
      model is separate per unit, and C, n, m are shared), so only one chunk's tables are
      ever duplicated;
    - a and b are returned in float32 (a float16 a added rounding noise of about 1e-3 of the
      token effects to every residual).
    """
    device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    seq_len = input_ids.shape[1]
    ids_all = input_ids[attention_mask.bool()]
    cnt_all = torch.bincount(ids_all.reshape(-1), minlength=vocab_size)
    keep_ids = torch.nonzero(cnt_all >= min_count).reshape(-1)
    look = torch.full((vocab_size,), -1, dtype=torch.long)
    look[keep_ids] = torch.arange(len(keep_ids))
    look_d = look.to(device)
    U = len(keep_ids)
    hooks, captured = _mlp_capture(model)
    S = P = None
    offsets = None
    n = torch.zeros(U, dtype=torch.float64, device=device)
    m = torch.zeros(seq_len, dtype=torch.float64, device=device)
    C = torch.zeros(U, seq_len, dtype=torch.float64, device=device)
    n_tok = 0
    pos_grid = torch.arange(seq_len, device=device)
    for i in range(0, input_ids.shape[0], batch_size):
        ids = input_ids[i:i + batch_size].to(device)
        mask = attention_mask[i:i + batch_size].to(device).bool()
        captured.clear()
        model(input_ids=ids, attention_mask=mask.long())
        rows = look_d[ids]
        sel = mask & (rows >= 0)
        r = rows[sel]
        p = pos_grid.expand_as(ids)[sel]
        if S is None:
            widths = [int(captured[l].shape[-1]) for l in range(len(captured))]
            offsets = np.concatenate([[0], np.cumsum(widths)])
            S = torch.zeros(U, int(offsets[-1]), dtype=torch.float32, device=device)
            P = torch.zeros(seq_len, int(offsets[-1]), dtype=torch.float32, device=device)
        for l in range(len(offsets) - 1):
            x = captured[l][sel].float()
            S[:, offsets[l]:offsets[l + 1]].index_add_(0, r, x)
            P[:, offsets[l]:offsets[l + 1]].index_add_(0, p, x)
            del x
        n += torch.bincount(r, minlength=U).double()
        m += torch.bincount(p, minlength=seq_len).double()
        C.index_put_((r, p), torch.ones_like(r, dtype=torch.float64), accumulate=True)
        n_tok += int(sel.sum())
        captured.clear()
    for h in hooks:
        h.remove()
    model.to('cpu')
    torch.cuda.empty_cache()
    C, n, m = C.float(), n.float(), m.float()
    mm = torch.clamp(m, min=1.0)[:, None]
    N = S.shape[1]
    a_out = torch.empty(U, N, dtype=torch.float32)
    b_out = torch.empty(seq_len, N, dtype=torch.float32)
    worst, iters = 0.0, 0
    for c0 in range(0, N, unit_chunk):
        c1 = min(c0 + unit_chunk, N)
        Sc, Pc = S[:, c0:c1], P[:, c0:c1]
        b = torch.zeros_like(Pc)
        a = Sc / n[:, None]
        change = float('inf')
        for it in range(1, max_iter + 1):
            a_new = (Sc - C @ b) / n[:, None]
            b_new = (Pc - C.T @ a_new) / mm
            b_new -= (m[:, None] * b_new).sum(0, keepdim=True) / m.sum()
            a_new = (Sc - C @ b_new) / n[:, None]
            change = float(torch.max(torch.abs(a_new - a)).item())
            a, b = a_new, b_new
            if change < tol:
                break
        worst, iters = max(worst, change), max(iters, it)
        a_out[:, c0:c1] = a.cpu()
        b_out[:, c0:c1] = b.cpu()
        del a, b, a_new, b_new
    del S, P, C
    torch.cuda.empty_cache()
    if verbose:
        stderr('%sresidual fit (large): %d reference tokens, %d token types (>= %d occurrences), %d units, '
               'backfitting converged in at most %d iterations (max change %.1e)\n'
               % (' ' * indent, n_tok, U, min_count, N, iters, worst))
    return TokenPositionFit(look, a_out, b_out, cnt_all, iters, worst, n_tok, min_count)
