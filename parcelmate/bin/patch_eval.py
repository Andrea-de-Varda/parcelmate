"""Evaluate a model on the LLM_Modularity minimal-pair tasks, and optionally attribute.

    python -m parcelmate.bin.patch_eval --model LiquidAI/LFM2.5-350M --out results/patching
    python -m parcelmate.bin.patch_eval --model openai-community/gpt2 --domains Lan --bos none
    python -m parcelmate.bin.patch_eval --model ... --attribution --min-both-correct 0.6

LOG.md Iteration 23. Writes, per task, `<out>/<model>/<domain>/<task>/baselines.json` in
the original repository's schema (plus per-item scores and the BOS choice), and one
summary table `<out>/<model>/accuracy.csv`. With --attribution, tasks that pass the
inclusion rule also get `neuron_attribution.npy` (n_layers x width, float64).

--bos governs raw-text tasks only (chat-template tasks carry their own): `none` is the
original protocol, `bos` prepends the tokenizer's BOS token, `both` runs the two and
writes the BOS variant under `<task>_bos/`.
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from parcelmate.patching import (
    DOMAINS, evaluate_task, load_domain_config, load_task, neuron_attribution,
)
from parcelmate.util import git_commit, stderr


def short_name(model):
    return model.replace('/', '_').replace('.', '-')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--revision', default=None)
    ap.add_argument('--domains', nargs='+', default=list(DOMAINS))
    ap.add_argument('--tasks', nargs='+', default=None, help='restrict to these task names')
    ap.add_argument('--out', default='results/patching')
    ap.add_argument('--bos', choices=('none', 'bos', 'both'), default='both')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--attribution', action='store_true')
    ap.add_argument('--min-both-correct', type=float, default=0.6)
    ap.add_argument('--min-n-both-correct', type=int, default=0,
                    help='minimum both-correct items for attribution (0: accuracy rule only)')
    ap.add_argument('--dtype', choices=('float32', 'bfloat16'), default='float32',
                    help='float32 is the protocol; bfloat16 only where float32 does not fit the GPU')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    kw = {} if args.revision is None else dict(revision=args.revision)
    tokenizer = AutoTokenizer.from_pretrained(args.model, **kw)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()
    model = model.float() if args.dtype == 'float32' else model.to(torch.bfloat16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    stderr('%s on %s, %.0fM parameters, %s\n' % (
        args.model, device, sum(p.numel() for p in model.parameters()) / 1e6, args.dtype))

    root = os.path.join(args.out, short_name(args.model))
    rows = []
    for domain in args.domains:
        config = load_domain_config(domain)
        for task, task_config in config.items():
            if args.tasks and task not in args.tasks:
                continue
            force_base = bool(task_config.get('force_base_evaluation', False))
            variants = [(False, task)]
            if force_base and args.bos == 'bos':
                variants = [(True, task)]
            elif force_base and args.bos == 'both':
                variants = [(False, task), (True, task + '_bos')]
            for bos, name in variants:
                out_dir = os.path.join(root, domain, name)
                os.makedirs(out_dir, exist_ok=True)
                path = os.path.join(out_dir, 'baselines.json')
                if os.path.exists(path) and not args.overwrite:
                    with open(path) as f:
                        b = json.load(f)
                    stderr('%s/%s: cached\n' % (domain, name))
                else:
                    stderr('%s/%s (bos=%s)\n' % (domain, name, bos))
                    data = load_task(domain, task_config)
                    t0 = time.time()
                    b = evaluate_task(model, tokenizer, data, task_config, bos=bos,
                                      batch_size=args.batch_size)
                    b.update(model=args.model, revision=args.revision or '', task=task,
                             domain=domain, dtype=args.dtype, git_commit=git_commit(),
                             seconds=time.time() - t0)
                    with open(path, 'w') as f:
                        json.dump(b, f, indent=1)
                stderr('    clean %.3f  corrupted %.3f  both %.3f (%d/%d aligned of %d)\n' % (
                    b['clean_accuracy'], b['corrupted_accuracy'], b['both_correct_accuracy'],
                    b['n_both_correct'], b['n_after_alignment_filter'], b['n_total']))
                passes = (b['both_correct_accuracy'] >= args.min_both_correct
                          and b['n_both_correct'] >= args.min_n_both_correct)
                rows.append(dict(model=args.model, domain=domain, task=name, bos=int(bos),
                                 n_total=b['n_total'], n_aligned=b['n_after_alignment_filter'],
                                 clean_accuracy=b['clean_accuracy'],
                                 corrupted_accuracy=b['corrupted_accuracy'],
                                 both_correct_accuracy=b['both_correct_accuracy'],
                                 n_both_correct=b['n_both_correct'],
                                 clean_baseline=b['clean_baseline'],
                                 corrupted_baseline=b['corrupted_baseline'],
                                 passes=int(passes)))
                if args.attribution and passes:
                    apath = os.path.join(out_dir, 'neuron_attribution.npy')
                    if os.path.exists(apath) and not args.overwrite:
                        stderr('    attribution cached\n')
                        continue
                    data = load_task(domain, task_config)
                    t0 = time.time()
                    attr = neuron_attribution(model, tokenizer, data, task_config, b, bos=bos,
                                              batch_size=max(1, args.batch_size // 2))
                    np.save(apath, attr)
                    with open(os.path.join(out_dir, 'attribution_meta.json'), 'w') as f:
                        json.dump(dict(shape=list(attr.shape), n_items=b['n_both_correct'],
                                       bos=bos, seconds=time.time() - t0,
                                       git_commit=git_commit()), f, indent=1)
                    stderr('    attribution %s in %.0f s\n' % (attr.shape, time.time() - t0))

    summary = os.path.join(root, 'accuracy.csv')
    with open(summary, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    stderr('\nwrote %s\n' % summary)
    print('%-5s %-38s %4s %6s %6s %6s %6s %s' % ('dom', 'task', 'bos', 'clean', 'corr', 'both', 'n_bc', 'pass'))
    for r in rows:
        print('%-5s %-38s %4d %6.3f %6.3f %6.3f %6d %s' % (
            r['domain'], r['task'], r['bos'], r['clean_accuracy'], r['corrupted_accuracy'],
            r['both_correct_accuracy'], r['n_both_correct'], 'yes' if r['passes'] else ''))


if __name__ == '__main__':
    main()
