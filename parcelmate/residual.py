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

    def __init__(self, look, a, b, counts, n_iter, change, n_ref_tokens, min_count):
        self.look = look            # (V,) long: row of `a` for every vocabulary id, -1 if dropped
        self.a = a                  # (types, units) float32
        self.b = b                  # (seq_len, units) float32
        self.counts = counts        # (V,) reference count of every vocabulary id
        self.n_iter = n_iter
        self.change = change
        self.n_ref_tokens = n_ref_tokens
        self.min_count = min_count

    def provenance(self):
        return dict(residualize='token_position', residual_ref_tokens=int(self.n_ref_tokens),
                    residual_types=int(self.a.shape[0]), residual_min_count=int(self.min_count),
                    residual_backfit_iterations=int(self.n_iter), residual_backfit_change=float(self.change))


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
    S = P = None
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
        n += torch.bincount(r, minlength=U).double()
        m += torch.bincount(p, minlength=seq_len).double()
        C.index_put_((r, p), torch.ones_like(r, dtype=torch.float64), accumulate=True)
        n_tok += int(sel.sum())
        del X, x
    for h in hooks:
        h.remove()
    # Backfitting on the sufficient statistics, in float32 (the float64 sums are exact; at
    # 160m a float64 types x units table is 9 GB), b constrained to zero mean over tokens.
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
    fit = TokenPositionFit(look, a.float().cpu(), b.float().cpu(), cnt_all, it, change, n_tok, min_count)
    del S, P, C
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
