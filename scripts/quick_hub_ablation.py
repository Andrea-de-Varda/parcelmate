"""QUICK-AND-DIRTY (2026-11-20, to be redone properly): mean-ablate hub units of Pythia-70m against
layer-matched random units, and measure (a) loss / perplexity on document text and (b) the task bank.

    python scripts/quick_hub_ablation.py --out results/quick_hub_ablation

Hubs: the top-n live units by strength (mean of the two halves) of the document-coherent pooled
residual connectome (results/richclub/pythia-70m-resid-pool4-docs). Controls: random live units
with the same count per layer as the hubs. Ablation: each unit's post-activation MLP output is
replaced by its mean over reference windows. Text: wikitext articles and bookcorpus blocks
(timescales.load_windows); the first windows give the means, the rest the loss. Caveat: the
windows may overlap the connectome's text. Tasks: the task bank list that passed Qwen3.5-2B
(eval_tasks_qwen35.txt), first --items items of each, clean prompt only; score = log P(correct) -
log P(incorrect), accuracy = share > 0. Pythia-70m is near chance on many of these tasks.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from parcelmate.model import mlp_projections
from parcelmate.patching import (BANK_ROOT, bank_eval_list, encode, format_prompts, load_bank_config,
                                 load_bank_task, sequence_log_probs)
from parcelmate.timescales import load_windows
from parcelmate.util import stderr

UNITS = 'results/richclub/pythia-70m-resid-pool4-docs/metrics/richclub/units_real.npz'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='EleutherAI/pythia-70m')
    ap.add_argument('--sizes', type=int, nargs='+', default=[50, 200, 1000])
    ap.add_argument('--n-random', type=int, default=5)
    ap.add_argument('--n-windows', type=int, default=60)
    ap.add_argument('--n-mean', type=int, default=20)
    ap.add_argument('--seq-len', type=int, default=512)
    ap.add_argument('--items', type=int, default=50)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(dev).eval()
    projs = mlp_projections(model)
    width = projs[0][1].in_features

    # hubs and layer-matched controls
    z = np.load(UNITS)
    coords, live = z['coordinates'], z['live'].astype(bool)
    strength = (z['strength_halfA'] + z['strength_halfB']) / 2
    order = np.flatnonzero(live)[np.argsort(-strength[live])]
    rng = np.random.RandomState(args.seed)
    sets = {'clean': None}
    for n in args.sizes:
        hubs = order[:n]
        sets['hub_n%d' % n] = hubs
        layers = coords[hubs, 0]
        for r in range(args.n_random):
            pick = []
            for l in np.unique(layers):
                pool = np.flatnonzero(live & (coords[:, 0] == l))
                pool = np.setdiff1d(pool, hubs)
                pick.append(rng.choice(pool, (layers == l).sum(), replace=False))
            sets['random_n%d_r%d' % (n, r)] = np.concatenate(pick)
    stderr('hub layers (n=%d): %s\n' % (args.sizes[-1], np.bincount(coords[order[:args.sizes[-1]], 0])))

    # text windows
    wins = [load_windows(d, tok, args.n_windows, args.seq_len, args.seed)[0] for d in ('wikitext', 'bookcorpus')]
    ref = torch.as_tensor(np.concatenate([w[:args.n_mean] for w in wins]), dtype=torch.long)
    evalw = torch.as_tensor(np.concatenate([w[args.n_mean:] for w in wins]), dtype=torch.long)

    # unit means over the reference windows
    sums = [torch.zeros(width, dtype=torch.float64, device=dev) for _ in projs]
    count = [0]

    def acc_hook(l):
        def h(mod, inp):
            x = inp[0]
            sums[l] += x.reshape(-1, x.shape[-1]).double().sum(0)
            if l == 0:
                count[0] += x.shape[0] * x.shape[1]
        return h
    hooks = [p.register_forward_pre_hook(acc_hook(l)) for l, p in projs]
    with torch.no_grad():
        for i in range(0, len(ref), 8):
            model(input_ids=ref[i:i + 8].to(dev))
    for h in hooks:
        h.remove()
    means = [(s / count[0]).float() for s in sums]

    current = {'idx': None}

    def abl_hook(l):
        def h(mod, inp):
            idx = current['idx']
            if idx is None or l not in idx:
                return None
            x = inp[0].clone()
            x[..., idx[l]] = means[l][idx[l]]
            return (x,) + tuple(inp[1:])
        return h
    for l, p in projs:
        p.register_forward_pre_hook(abl_hook(l))

    def set_units(units):
        if units is None:
            current['idx'] = None
            return
        d = {}
        for u in units:
            d.setdefault(int(coords[u, 0]), []).append(int(coords[u, 1]))
        current['idx'] = {l: torch.tensor(v, device=dev) for l, v in d.items()}

    def text_loss():
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(evalw), 8):
                x = evalw[i:i + 8].to(dev)
                lg = model(input_ids=x).logits[:, :-1].float()
                l = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1), reduction='sum')
                tot += l.item()
                n += x[:, 1:].numel()
        return tot / n

    # tasks, encoded once
    tasks = []
    for source, task in bank_eval_list('eval_tasks_qwen35.txt'):
        cfg = load_bank_config(source)
        tc = next((t for t in cfg['tasks'] if t['name'] == task), None) if isinstance(cfg.get('tasks'), list) else cfg.get(task)
        if tc is None:
            stderr('skip %s/%s: no config\n' % (source, task))
            continue
        try:
            data = load_bank_task(source, tc)[:args.items]
            fmt = format_prompts(data, tc, tok)
            ic, pc = encode(tok, fmt['clean_prompts'], fmt['clean_correct'])
            ii, pi = encode(tok, fmt['clean_prompts'], fmt['clean_incorrect'])
        except Exception as e:  # quick test: skip anything that does not load
            stderr('skip %s/%s: %s\n' % (source, task, e))
            continue
        tasks.append((source, task, ic, pc, ii, pi))
    stderr('%d tasks\n' % len(tasks))

    rows = []
    for name, units in sets.items():
        t0 = time.time()
        set_units(units)
        loss = text_loss()
        diffs, accs = {}, {}
        for source, task, ic, pc, ii, pi in tasks:
            d = sequence_log_probs(model, ic, pc, batch_size=64, pad_id=0) - \
                sequence_log_probs(model, ii, pi, batch_size=64, pad_id=0)
            diffs['%s/%s' % (source, task)] = float(d.mean())
            accs['%s/%s' % (source, task)] = float((d > 0).mean())
        rows.append(dict(set=name, n_units=0 if units is None else len(units), loss=loss, ppl=float(np.exp(loss)),
                         task_diff=diffs, task_acc=accs))
        stderr('%-16s loss %.4f ppl %.2f mean task acc %.3f diff %.3f (%.0f s)\n' % (
            name, loss, np.exp(loss), np.mean(list(accs.values())), np.mean(list(diffs.values())), time.time() - t0))
        json.dump(rows, open(os.path.join(args.out, 'results.json'), 'w'))
    stderr('wrote %s\n' % os.path.join(args.out, 'results.json'))


if __name__ == '__main__':
    main()
