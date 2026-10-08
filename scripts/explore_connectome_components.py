"""Exploratory (LOG.md Iteration 38): what makes the connectome stable within a dataset but
not across datasets, and can it be removed?

    python scripts/explore_connectome_components.py --model EleutherAI/pythia-70m \
        --steps 0 1000 143000 --tokens 40960 --out results/explore/run1.csv

For every checkpoint and every dataset, three disjoint portions of `--tokens` tokens: half A
and half B (the two connectomes compared) and a REFERENCE portion, used only to estimate the
statistics that some variants remove (so halves stay independent). Each variant transforms
the activations (tokens x all MLP units, post-nonlinearity) of a half, then the |r|
connectome is compared, over a fixed sample of unit pairs, between half A and half B of the
same dataset ("within") and between half A of one dataset and half B of another ("across").

Candidate components, and the variant that removes each:
  C1 token identity: every token type drives each unit to a typical level, so the
     connectome is a covariance of token-type profiles weighted by the dataset's word
     frequencies.                                            -> tokres (type means removed)
     ... the token-driven part alone                         -> tokmean (only type means)
  C2 word-frequency differences, keeping token effects       -> freqmatch (tokens weighted so
                                                                every half has the same type
                                                                distribution)
  C3 position in the window (first-token effects)            -> posres
  C4 slow, document-level drift (topic, style, formatting)   -> winres (unit means removed per
                                                                64-token chunk)
  C5 outlier tokens dominating correlations                  -> rank (Spearman)
  C6 formatting / punctuation / digits / code symbols        -> wordonly (alphabetic words)
  and combinations. Success: within and across both high in the trained model, and lower
  (in particular across) in the untrained one.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from parcelmate.bigconn import _mlp_hooks
from parcelmate.data import get_dataset
from parcelmate.model import domain_data_kwargs, get_model_and_tokenizer, mlp_projections
from parcelmate.util import derive_seed, stderr

DATASETS = ['wikitext', 'bookcorpus', 'agnews', 'tldr17', 'codeparrot']
PROSE = DATASETS[:4]
SEQ = 1024
CHUNK = 64
LAYER_WIDTH = None   # set from the model
SHARED_MIN = 5


def load_portions(tokenizer, n_tok, seed, take, ref_mult=1):
    out = {}
    for d in DATASETS:
        kw = domain_data_kwargs(d)
        kw['tokenizer'] = tokenizer
        ids, mask = get_dataset(n_tokens=(2 + ref_mult) * n_tok, seq_len=SEQ, take=take.get(d, 20000),
                                seed=derive_seed(seed, 'data', d), verbose=False, **kw)
        ids = ids[mask.sum(1) == SEQ]
        n = n_tok // SEQ
        n_ref = min(ref_mult * n, len(ids) - 2 * n)
        assert n_ref >= n, '%s: only %d full windows' % (d, len(ids))
        out[d] = {'A': ids[:n], 'B': ids[n:2 * n], 'ref': ids[2 * n:2 * n + n_ref]}
        stderr('  %s: %d windows per half, %d reference windows\n' % (d, n, n_ref))
    return out


@torch.no_grad()
def activations(model, ids, dev, bs=8):
    captured = {}
    hooks, n_layers = _mlp_hooks(model, captured)
    xs = []
    for i in range(0, len(ids), bs):
        captured.clear()
        model(input_ids=ids[i:i + bs].to(dev))
        xs.append(torch.cat([captured[l] for l in range(n_layers)], -1).reshape(-1, captured[0].shape[-1] * n_layers).half())
    for h in hooks:
        h.remove()
    return torch.cat(xs)                                            # (tokens, units) fp16 on dev


def connectome_vec(X, pair_i, pair_j, w=None):
    """|r| of the columns of X (tokens x units), optionally token-weighted, at the sampled pairs."""
    X = X.float()
    if w is None:
        Xc = X - X.mean(0)
        C = Xc.T @ Xc
    else:
        w = w.float() / w.sum()
        Xc = X - (w[:, None] * X).sum(0)
        C = (Xc * w[:, None]).T @ Xc
    d = torch.sqrt(torch.clamp(torch.diag(C), min=1e-12))
    v = (C[pair_i, pair_j] / (d[pair_i] * d[pair_j])).abs()
    dead = (torch.diag(C) < 1e-10)
    v[dead[pair_i] | dead[pair_j]] = float('nan')
    return v.cpu().numpy()


def rank_cols(X, step=512):
    out = torch.empty(X.shape, dtype=torch.float16, device=X.device)
    for s in range(0, X.shape[1], step):
        x = X[:, s:s + step].float()
        out[:, s:s + step] = torch.argsort(torch.argsort(x, dim=0), dim=0).half()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='EleutherAI/pythia-70m')
    ap.add_argument('--steps', nargs='+', default=['0', '1000', '143000'])
    ap.add_argument('--tokens', type=int, default=40960)
    ap.add_argument('--pairs', type=int, default=2_000_000)
    ap.add_argument('--min-count', type=int, default=3)
    ap.add_argument('--ref-mult', type=int, default=1, help='reference portion = this many halves')
    ap.add_argument('--variants', nargs='+', default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    _, tok = get_model_and_tokenizer(args.model, revision='step%s' % args.steps[-1])
    data = load_portions(tok, args.tokens, args.seed, {'wikitext': 150000, 'codeparrot': 8000, 'agnews': 60000,
                                                       'tldr17': 40000, 'bookcorpus': 60000}, args.ref_mult)
    stderr('data loaded, %.0f s\n' % (time.time() - t0))

    # token classes for C6 (alphabetic word tokens)
    vocab = tok.convert_ids_to_tokens(list(range(len(tok))))
    is_word = torch.tensor([t.replace('Ġ', '').isalpha() for t in vocab], device=dev)

    # unigram distributions of the reference portions, for the JSD diagnostic
    V = len(tok)
    uni = {d: np.bincount(data[d]['ref'].ravel().numpy(), minlength=V) / data[d]['ref'].numel() for d in DATASETS}

    rows = []
    for step in args.steps:
        model, _ = get_model_and_tokenizer(args.model, revision='step%s' % step)
        model.to(dev).eval()
        X = {(d, p): activations(model, data[d][p], dev) for d in DATASETS for p in ('A', 'B')}
        T = {(d, p): data[d][p].reshape(-1).to(dev) for d in DATASETS for p in ('A', 'B', 'ref')}
        pos = torch.arange(SEQ, device=dev).repeat(args.tokens // SEQ)
        N = X[(DATASETS[0], 'A')].shape[1]
        global LAYER_WIDTH
        LAYER_WIDTH = N // len(mlp_projections(model))
        g = torch.Generator(device='cpu').manual_seed(0)
        pi = torch.randint(0, N, (args.pairs,), generator=g)
        pj = torch.randint(0, N, (args.pairs,), generator=g)
        keep = pi != pj
        pi, pj = pi[keep].to(dev), pj[keep].to(dev)
        stderr('step %s: activations %.0f s, %d units\n' % (step, time.time() - t0, N))

        # Reference statistics, streamed over the reference windows (never stored):
        # per-type means pooled over all datasets (first pass), per-position means, and the
        # mean residual (x - type mean) by PREVIOUS token type (second pass).
        ref_ids = torch.cat([T[(d, 'ref')] for d in DATASETS])
        types, inv = torch.unique(ref_ids, return_inverse=True)
        look = torch.full((V,), -1, dtype=torch.long, device=dev)
        look[types] = torch.arange(len(types), device=dev)
        cnt = torch.zeros(V, device=dev)
        cnt[types] = torch.bincount(inv, minlength=len(types)).float()
        tab = torch.zeros(len(types), N, device=dev)
        pos_sum = torch.zeros(SEQ, N, device=dev)
        n_win = 0
        for d in DATASETS:
            ids = data[d]['ref']
            for i in range(0, len(ids), 8):
                x = activations(model, ids[i:i + 8], dev).float()
                t = ids[i:i + 8].reshape(-1).to(dev)
                tab.index_add_(0, look[t], x)
                pos_sum += x.reshape(-1, SEQ, N).sum(0)
                n_win += x.shape[0] // SEQ
        tab /= torch.clamp(cnt[types], min=1)[:, None]
        pos_mu = pos_sum / n_win
        mu = lambda t: tab[look[t].clamp(min=0)]
        nu_tab = torch.zeros(len(types), N, device=dev)
        nu_cnt = torch.zeros(len(types), device=dev)
        for d in DATASETS:
            ids = data[d]['ref']
            for i in range(0, len(ids), 8):
                x = activations(model, ids[i:i + 8], dev).float().reshape(-1, SEQ, N)
                t = ids[i:i + 8].to(dev)
                r = (x - mu(t.reshape(-1)).reshape(x.shape))[:, 1:].reshape(-1, N)
                prev = look[t[:, :-1].reshape(-1)]
                nu_tab.index_add_(0, prev, r)
                nu_cnt += torch.bincount(prev, minlength=len(types)).float()
        nu_tab /= torch.clamp(nu_cnt, min=1)[:, None]
        nu = lambda t: nu_tab[look[t].clamp(min=0)]
        stderr('  reference statistics: %d types, %d windows, %.0f s\n' % (len(types), n_win, time.time() - t0))
        # Shared vocabulary (iteration 3): token types seen at least SHARED_MIN times in the
        # reference portion of EVERY prose dataset.
        ref_cnt = torch.stack([torch.bincount(data[d]['ref'].reshape(-1).to(dev), minlength=V) for d in PROSE])
        shared = (ref_cnt >= SHARED_MIN).all(0)
        stderr('  shared vocabulary: %d types\n' % int(shared.sum()))
        target_sh = torch.where(shared, ref_cnt.float().sum(0), torch.zeros(V, device=dev))
        target_sh = target_sh / target_sh.sum()
        target = torch.tensor(np.mean([uni[d] for d in PROSE], 0), device=dev, dtype=torch.float32)

        def chunk_res(Y):
            Z = Y.reshape(-1, CHUNK, N)
            return (Z - Z.mean(1, keepdim=True)).reshape(-1, N)

        def residual(name, x, t):
            """Full-length residual and the mask of tokens it is defined for."""
            seen = cnt[t] >= args.min_count
            if name in ('tokres', 'tokres+posres'):
                R, m = x - mu(t), seen
            elif name in ('ctxres', 'ctxres+posres'):
                prev = torch.roll(t, 1)
                R, m = x - mu(t) - nu(prev), seen & (cnt[prev] >= args.min_count) & (pos > 0)
            else:
                raise ValueError(name)
            if name.endswith('+posres'):
                R = R - pos_mu[pos] + pos_mu.mean(0)
            return R, m

        def variant(name, d, p):
            x, t = X[(d, p)].float(), T[(d, p)]
            seen = cnt[t] >= args.min_count
            if name == 'base':
                return x, None
            if name == 'tokres':
                return (x - mu(t))[seen], None
            if name.startswith('cm:'):
                # C7 (iteration 2): a per-token gain shared by all units of a layer (layer
                # norm, residual-stream scale). Remove each token's mean over the units of
                # each layer from whatever the base variant leaves.
                Y, w = variant(name[3:], d, p)
                Z = Y.float().reshape(Y.shape[0], -1, LAYER_WIDTH)
                return (Z - Z.mean(2, keepdim=True)).reshape(Y.shape[0], -1), w
            if name.startswith('sh:') or name.startswith('shfm:') or name.endswith('+winres') and name.startswith('ctx'):
                # Iteration 3 modifiers on a residual variant (tokres, ctxres, with or
                # without +posres): sh: keep only shared-vocabulary tokens; shfm: also reweight
                # them to the pooled shared-type frequencies (weights capped at 5); +winres:
                # remove each unit's mean per 64-token chunk from the residual.
                prefix, base_name = (name.split(':', 1) if ':' in name else ('', name))
                win = base_name.endswith('+winres')
                if win:
                    base_name = base_name[:-len('+winres')]
                R, m = residual(base_name, x, t)
                if win:
                    R = chunk_res(R * m[:, None])
                if prefix in ('sh', 'shfm'):
                    m = m & shared[t] & (pos > 0)
                w = None
                if prefix == 'shfm':
                    tt = t[m]
                    ht = torch.bincount(tt, minlength=V).float()
                    w = torch.clamp(target_sh[tt] / (ht[tt] / len(tt)), max=5.0)
                return R[m], w
            if name == 'tokres20':
                ok = cnt[t] >= 20
                return (x - mu(t))[ok], None
            if name in ('ctxres', 'ctxres+posres', 'ctxres+freqmatch'):
                prev = torch.roll(t, 1)
                first = pos == 0
                ok = seen & (cnt[prev] >= args.min_count) & ~first
                r = x - mu(t) - nu(prev)
                if name == 'ctxres+posres':
                    r = r - pos_mu[pos] + pos_mu.mean(0)
                w = None
                if name == 'ctxres+freqmatch':
                    ht = torch.bincount(t[ok], minlength=V).float()
                    w = torch.clamp(target[t[ok]] / (ht[t[ok]] / ok.sum()), max=20.0)
                return r[ok], w
            if name == 'tokres+freqmatch':
                ht = torch.bincount(t[seen], minlength=V).float()
                w = torch.clamp(target[t[seen]] / (ht[t[seen]] / seen.sum()), max=20.0)
                return (x - mu(t))[seen], w
            if name == 'tokres+posres+freqmatch':
                ht = torch.bincount(t[seen], minlength=V).float()
                w = torch.clamp(target[t[seen]] / (ht[t[seen]] / seen.sum()), max=20.0)
                return (x - mu(t) - pos_mu[pos] + pos_mu.mean(0))[seen], w
            if name == 'tokmean':
                return mu(t)[seen], None
            if name == 'freqmatch':
                ht = torch.bincount(t, minlength=V).float()
                w = target[t] / (ht[t] / len(t))
                w = torch.clamp(w, max=20.0)
                return x, w
            if name == 'posres':
                return x - pos_mu[pos], None
            if name == 'winres':
                return chunk_res(x), None
            if name == 'rank':
                return rank_cols(x), None
            if name == 'wordonly':
                return x[is_word[t]], None
            if name == 'tokres+winres':
                r = chunk_res(x - mu(t) * seen[:, None])
                return r[seen], None
            if name == 'tokres+posres':
                return (x - mu(t) - pos_mu[pos] + pos_mu.mean(0))[seen], None
            if name == 'tokres+rank':
                return rank_cols((x - mu(t))[seen]), None
            if name == 'tokres+wordonly':
                m = seen & is_word[t]
                return (x - mu(t))[m], None
            raise ValueError(name)

        names = args.variants or ['base', 'tokmean', 'tokres', 'tokres20', 'freqmatch', 'posres', 'winres', 'rank',
                                  'wordonly', 'tokres+winres', 'tokres+posres', 'tokres+rank', 'tokres+wordonly',
                                  'tokres+freqmatch', 'tokres+posres+freqmatch', 'ctxres', 'ctxres+posres', 'ctxres+freqmatch']
        for name in names:
            vec = {}
            for d in DATASETS:
                for p in ('A', 'B'):
                    Y, w = variant(name, d, p)
                    vec[(d, p)] = connectome_vec(Y, pi, pj, w)
            def r(a, b):
                ok = np.isfinite(a) & np.isfinite(b)
                return float(np.corrcoef(a[ok], b[ok])[0, 1])
            for d1 in DATASETS:
                for d2 in DATASETS:
                    rows.append(dict(step=step, variant=name, fit=d1, eval=d2,
                                     kind='within' if d1 == d2 else 'across', r=r(vec[(d1, 'A')], vec[(d2, 'B')]),
                                     jsd=float(0.5 * sum(np.sum(np.where(u > 0, u * np.log(u / (0.5 * (uni[d1] + uni[d2]))), 0))
                                                         for u in (uni[d1], uni[d2])))))
            pr = [x for x in rows if x['step'] == step and x['variant'] == name]
            w = np.mean([x['r'] for x in pr if x['kind'] == 'within' and x['fit'] in PROSE])
            a = np.mean([x['r'] for x in pr if x['kind'] == 'across' and x['fit'] in PROSE and x['eval'] in PROSE])
            wc = [x['r'] for x in pr if x['kind'] == 'within' and x['fit'] == 'codeparrot'][0]
            ac = np.mean([x['r'] for x in pr if x['kind'] == 'across' and 'codeparrot' in (x['fit'], x['eval'])])
            stderr('  step %-6s %-16s prose within %.3f across %.3f (ratio %.2f) | code within %.3f, prose<->code %.3f   [%.0f s]\n' % (
                step, name, w, a, a / w, wc, ac, time.time() - t0))
        del X, tab, nu_tab
        torch.cuda.empty_cache()
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    stderr('wrote %s, %.0f s\n' % (args.out, time.time() - t0))


if __name__ == '__main__':
    main()
