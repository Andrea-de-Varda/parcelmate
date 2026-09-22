"""Cross-checkpoint agreement for a Pythia training-dynamics run, plus one long score table.

    python -m parcelmate.bin.score_checkpoints --root results/pythia/pythia-160m

LOG.md Iteration 22. The per-checkpoint score files say whether the networks at each step
are real (reliability and fidelity against the null partition). They do not say WHEN the
networks settle. This adds, for each domain and half:

  ari_consecutive   ARI between the real partitions at step s and the next step s'
                    (fit = s, eval = s').
  ari_to_final      ARI between the real partition at step s and at the last step.

Each has a `pnull` reference built as the within-domain one is (score.py): the mean of
ARI(null partition at s, real at s') and ARI(real at s, null partition at s'). A partition
that carries only per-unit properties would score there.

It also concatenates every checkpoint's scores.csv into `scores_all.csv` with `model` and
`step` columns, so plotting reads one file. Refuses to write if any checkpoint is missing,
unless --allow-partial is given.
"""

import argparse
import csv
import os
import re

import numpy as np

from parcelmate.constants import HALF_NAMES
from parcelmate.metrics import reliability
from parcelmate.bin.score import parc_path
from parcelmate.util import load_h5_data, stderr

STEP_RE = re.compile(r'^step(\d+)$')


def find_steps(root):
    """The step directories under `root` (real trees only), sorted by step."""
    out = []
    for name in os.listdir(root):
        m = STEP_RE.match(name)
        if m and os.path.isdir(os.path.join(root, name)):
            out.append(int(m.group(1)))
    return sorted(out)


def load_labels(root, step, tree_suffix, variant, domain, key):
    path = parc_path(os.path.join(root, 'step%d%s' % (step, tree_suffix)), variant, domain, key)
    if not os.path.exists(path):
        return None
    return load_h5_data(path, verbose=False)['parcellation']


def score_checkpoints(root, steps, variant, domains, allow_partial=False, verbose=True):
    model = os.path.basename(os.path.normpath(root))
    rows, missing = [], []
    final = steps[-1]
    for domain in domains:
        for key in HALF_NAMES:
            real = {s: load_labels(root, s, '', variant, domain, key) for s in steps}
            null = {s: load_labels(root, s, '_null', variant, domain, key) for s in steps}
            for s in steps:
                if real[s] is None:
                    missing.append('step%d/%s/%s/%s real parcellation' % (s, variant, domain, key))
                if null[s] is None:
                    missing.append('step%d_null/%s/%s/%s null parcellation' % (s, variant, domain, key))
            pairs = [(a, b, 'ari_consecutive') for a, b in zip(steps[:-1], steps[1:])]
            pairs += [(s, final, 'ari_to_final') for s in steps if s != final]
            for a, b, metric in pairs:
                if real[a] is None or real[b] is None:
                    continue
                rows.append(dict(model=model, tree='real', variant=variant, metric=metric,
                                 domain=domain, key=key, fit_step=a, eval_step=b,
                                 value=reliability(real[a], real[b])))
                if null[a] is not None and null[b] is not None:
                    rows.append(dict(model=model, tree='pnull', variant=variant, metric=metric,
                                     domain=domain, key=key, fit_step=a, eval_step=b,
                                     value=(reliability(null[a], real[b])
                                            + reliability(real[a], null[b])) / 2.0))
            if verbose:
                stderr('  %-12s %s done\n' % (domain, key))
    return rows, missing


def concat_scores(root, steps):
    """Every checkpoint's scores.csv as one table with model and step columns."""
    model = os.path.basename(os.path.normpath(root))
    rows, missing = [], []
    for s in steps:
        path = os.path.join(root, 'step%d' % s, 'metrics', 'scores.csv')
        if not os.path.exists(path):
            missing.append('step%d/metrics/scores.csv' % s)
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                r = dict(r)
                r['model'] = model
                r['step'] = s
                rows.append(r)
    return rows, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True,
                    help='One model\'s directory, e.g. results/pythia/pythia-160m, holding step<N> '
                         'and step<N>_null trees')
    ap.add_argument('--steps', nargs='+', type=int, default=None,
                    help='Steps to include, in order (default: every step<N> under --root)')
    ap.add_argument('--variant', default='final')
    ap.add_argument('--domains', nargs='+', default=['wikitext', 'bookcorpus', 'agnews', 'tldr17'])
    ap.add_argument('--allow-partial', action='store_true')
    args = ap.parse_args()

    steps = args.steps or find_steps(args.root)
    assert len(steps) >= 2, 'need at least two checkpoints under %s' % args.root
    stderr('Checkpoints: %s\n' % ', '.join(str(s) for s in steps))
    rows, missing = score_checkpoints(args.root, steps, args.variant, args.domains)
    all_rows, missing_scores = concat_scores(args.root, steps)
    missing += missing_scores
    if missing:
        stderr('\n%d missing input(s):\n' % len(missing))
        for m in missing[:40]:
            stderr('  - %s\n' % m)
        if not args.allow_partial:
            raise SystemExit('\nRefusing to write partial results; pass --allow-partial to override.')

    out_dir = os.path.join(args.root, 'metrics')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'checkpoints.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'tree', 'variant', 'metric', 'domain', 'key',
                                          'fit_step', 'eval_step', 'value'])
        w.writeheader()
        w.writerows(rows)
    stderr('Wrote %d rows to %s\n' % (len(rows), out))
    out_all = os.path.join(out_dir, 'scores_all.csv')
    with open(out_all, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'step', 'tree', 'variant', 'metric', 'fit',
                                          'eval', 'value'])
        w.writeheader()
        w.writerows(all_rows)
    stderr('Wrote %d rows to %s\n' % (len(all_rows), out_all))

    # Reading aid: mean over domains and halves of ARI to the final checkpoint, real and null.
    for metric in ('ari_consecutive', 'ari_to_final'):
        print('\n%s (mean over domains and halves)' % metric)
        print('%8s %8s %8s %8s' % ('fit', 'eval', 'real', 'pnull'))
        keys = sorted({(r['fit_step'], r['eval_step']) for r in rows if r['metric'] == metric})
        for a, b in keys:
            vals = {}
            for tree in ('real', 'pnull'):
                xs = [r['value'] for r in rows
                      if r['metric'] == metric and r['tree'] == tree
                      and r['fit_step'] == a and r['eval_step'] == b]
                vals[tree] = np.mean(xs) if xs else float('nan')
            print('%8d %8d %8.3f %8.3f' % (a, b, vals['real'], vals['pnull']))


if __name__ == '__main__':
    main()
