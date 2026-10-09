"""Diagnostic (LOG.md Iteration 39): why does the joint least-squares token + position fit give a
less shared connectome than subtracting raw token means and raw position means in sequence?

    python scripts/diag_residual_joint.py --step 143000 --half 40 --ref 160

Pythia-70m, four prose datasets; half A / half B of `--half` windows per dataset, a reference
of `--ref` windows per dataset (pooled) that follows them. Every reference statistic is
streamed (nothing tokens x units is kept for the reference). Variants differ only in what
is subtracted; within / across agreement of the |r| connectome over 1M sampled pairs.
"""

import argparse
import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from parcelmate.data import get_dataset
from parcelmate.model import domain_data_kwargs, get_model_and_tokenizer
from parcelmate.residual import _mlp_capture, fit_token_position
from parcelmate.util import derive_seed, stderr

DS = ['wikitext', 'bookcorpus', 'agnews', 'tldr17']
L = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', default='143000')
    ap.add_argument('--model', default='EleutherAI/pythia-70m')
    ap.add_argument('--half', type=int, default=40)
    ap.add_argument('--ref', type=int, default=160)
    args = ap.parse_args()
    dev = 'cuda:0'
    model, tok = get_model_and_tokenizer(args.model, revision='step%s' % args.step)
    data = {}
    for d in DS:
        kw = domain_data_kwargs(d)
        kw['tokenizer'] = tok
        ids, mask = get_dataset(n_tokens=(2 * args.half + args.ref) * L, seq_len=L, take=150000,
                                seed=derive_seed(42, 'data', d), verbose=False, **kw)
        ids = ids[mask.sum(1) == L]
        data[d] = dict(A=ids[:args.half], B=ids[args.half:2 * args.half], ref=ids[2 * args.half:2 * args.half + args.ref])
    ref = torch.cat([data[d]['ref'] for d in DS])
    fit = fit_token_position(model, ref, torch.ones_like(ref), len(tok), batch_size=8, min_count=3, device=dev)
    model.to(dev).eval()
    hooks, cap = _mlp_capture(model)

    @torch.no_grad()
    def acts(ids):
        cap.clear()
        model(input_ids=ids.to(dev))
        return torch.cat([cap[l] for l in range(len(cap))], -1).float()

    look = fit.look.to(dev)
    a, b = fit.a.to(dev), fit.b.to(dev)
    U, N = a.shape
    # streamed: raw token means mu, raw position means pi, position means of (x - mu) b1,
    # and the reference's mean token profile at each position (comp)
    mu = torch.zeros(U, N, device=dev)
    cnt = torch.zeros(U, device=dev)
    pi = torch.zeros(L, N, device=dev)
    nwin = 0
    for i in range(0, len(ref), 8):
        x = acts(ref[i:i + 8])
        r = look[ref[i:i + 8].to(dev)]
        ok = r >= 0
        mu.index_add_(0, r[ok], x[ok])
        cnt += torch.bincount(r[ok], minlength=U).float()
        pi += x.sum(0)
        nwin += x.shape[0]
    mu /= cnt[:, None]
    pi /= nwin
    b1 = torch.zeros(L, N, device=dev)
    comp = torch.zeros(L, N, device=dev)
    for i in range(0, len(ref), 8):
        x = acts(ref[i:i + 8])
        r = look[ref[i:i + 8].to(dev)].clamp(min=0)
        b1 += (x - mu[r]).sum(0)
        comp += mu[r].sum(0)
    b1 /= nwin
    comp /= nwin
    # the same average of token profiles, but from tokens drawn at random positions of the
    # reference (same number per position): structured like comp, unrelated to position
    flat = look[ref.to(dev).reshape(-1)]
    flat = flat[flat >= 0]
    gen = torch.Generator(device=dev).manual_seed(2)
    comp_rand = torch.stack([mu[flat[torch.randint(0, len(flat), (nwin,), device=dev, generator=gen)]].mean(0)
                             for _ in range(L)])
    b1c = b1 - b1.mean(0)
    stderr('position effects: var across positions (median over units): raw pi %.3g, joint b %.3g, b1 %.3g, '
           'token composition %.3g\n' % (pi.var(0).median(), b.var(0).median(), b1.var(0).median(), comp.var(0).median()))
    # a random per-position pattern with the per-unit spread of the reference's token
    # composition: if subtracting it raises cross-dataset agreement, the sequential gain is
    # an injected shared pattern, not structure in the data
    g = torch.randn(L, N, device=dev, generator=torch.Generator(device=dev).manual_seed(1))
    g = g * (comp - comp.mean(0)).std(0, keepdim=True)
    rs = np.random.RandomState(0)
    pi_i = torch.as_tensor(rs.randint(0, N, 1_000_000), device=dev)
    pj_i = torch.as_tensor(rs.randint(0, N, 1_000_000), device=dev)

    def vec(Y):
        Y = Y - Y.mean(0)
        C = Y.T @ Y
        dg = torch.sqrt(torch.diag(C).clamp(min=1e-12))
        return (C[pi_i, pj_i] / dg[pi_i] / dg[pj_i]).abs().cpu().numpy()

    out = {}
    for d in DS:
        for h in ('A', 'B'):
            ids = data[d][h]
            X = torch.cat([acts(ids[i:i + 8]) for i in range(0, len(ids), 8)])
            idd = ids.to(dev)
            pos = torch.arange(L, device=dev).expand_as(idd)
            r = look[idd]
            ok = r >= 0
            r = r.clamp(min=0)
            V = {
                'token (raw means)': X - mu[r],
                'token + position, sequential (raw means)': X - mu[r] - pi[pos] + pi.mean(0),
                'token + position, one backfit step': X - mu[r] - b1c[pos],
                'token + position, joint least squares': X - a[r] - b[pos],
                'joint + reference token composition by position': X - a[r] - b[pos] - (comp[pos] - comp.mean(0)),
                'joint + RANDOM per-position pattern (same spread)': X - a[r] - b[pos] - g[pos],
                'joint + token composition of RANDOM positions': X - a[r] - b[pos] - (comp_rand[pos] - comp_rand.mean(0)),
            }
            for k, v in V.items():
                out.setdefault(k, {})[(d, h)] = vec(v[ok])
            del X, V
    for k, V in out.items():
        w = np.mean([np.corrcoef(V[(d, 'A')], V[(d, 'B')])[0, 1] for d in DS])
        ac = np.mean([np.corrcoef(V[(d1, 'A')], V[(d2, 'B')])[0, 1] for d1, d2 in itertools.permutations(DS, 2)])
        print('step %-6s %-52s within %.3f  across %.3f' % (args.step, k, w, ac), flush=True)


if __name__ == '__main__':
    main()
