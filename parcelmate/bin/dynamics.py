"""Training-dynamics measures for a checkpointed model (LOG.md Iteration 28).

    # one checkpoint: connectome and activation measures (GPU)
    python -m parcelmate.bin.dynamics checkpoint configs/pythia/pythia-70m_step64.yml \
        --out results/pythia/pythia-70m/dynamics

    # every checkpoint's stored partitions: crispness, size, depth, generality, transitions,
    # birth steps (CPU, no model)
    python -m parcelmate.bin.dynamics partitions --root results/pythia/pythia-70m \
        --out results/pythia/pythia-70m/dynamics

    # after all checkpoints: connectome similarity between checkpoints from the subsamples,
    # and one long table of everything (CPU)
    python -m parcelmate.bin.dynamics combine --out results/pythia/pythia-70m/dynamics

Everything written is group readable and writable (umask 002).
"""

import argparse
import csv
import glob
import os
import re

import h5py
import numpy as np

from parcelmate.cfg import get_cfg
from parcelmate.constants import HALF_NAMES
from parcelmate.dynamics import (
    birth_steps, checkpoint_measures, cross_domain_generality, partition_measures,
    transition_measures,
)
from parcelmate.metrics import hard_labels
from parcelmate.util import stderr

os.umask(0o002)
STEP_RE = re.compile(r'^step(\d+)$')
PROSE = ['wikitext', 'bookcorpus', 'agnews', 'tldr17']


def write_rows(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    stderr('wrote %s (%d rows)\n' % (path, len(rows)))


def load_partition(root, step, tree, variant, domain, key):
    d = os.path.join(root, 'step%d%s' % (step, '' if tree == 'real' else '_null'), variant,
                     'parcellation', 'parcellation_%s_%s.h5' % (domain, key))
    if not os.path.exists(d):
        return None
    with h5py.File(d, 'r') as f:
        return np.asarray(f['parcellation']), np.asarray(f['coordinates'])


def cmd_partitions(args):
    model = os.path.basename(os.path.normpath(args.root))
    steps = sorted(int(STEP_RE.match(n).group(1)) for n in os.listdir(args.root)
                   if STEP_RE.match(n) and os.path.isdir(os.path.join(args.root, n)))
    rows, tracks_rows, missing = [], [], []

    def row(step, domain, key, tree, measure, value):
        rows.append(dict(model=model, step=step, domain=domain, key=key, tree=tree,
                         measure=measure, value=float(value)))

    labels = {}   # (tree, key, domain, step) -> labels
    k = None
    for tree in ('real', 'null'):
        for key in HALF_NAMES:
            for domain in args.domains:
                for step in steps:
                    got = load_partition(args.root, step, tree, args.variant, domain, key)
                    if got is None:
                        missing.append('%s step%d %s %s' % (tree, step, domain, key))
                        continue
                    P, coords = got
                    k = P.shape[1]
                    labels[(tree, key, domain, step)] = hard_labels(P)
                    for name, v in partition_measures(P, coords).items():
                        row(step, domain, key, tree, name, v)
            # Cross-domain generality, per step (same tree, same half).
            for step in steps:
                by_dom = {d: labels[(tree, key, d, step)] for d in args.domains
                          if (tree, key, d, step) in labels}
                if len(by_dom) < 2:
                    continue
                for d, v in cross_domain_generality(by_dom, k).items():
                    for name, val in v.items():
                        row(step, d, key, tree, name, val)
            # Transitions and births, per domain.
            for domain in args.domains:
                have = [s for s in steps if (tree, key, domain, s) in labels]
                for a, b in zip(have[:-1], have[1:]):
                    for name, v in transition_measures(labels[(tree, key, domain, a)],
                                                       labels[(tree, key, domain, b)], k).items():
                        row(b, domain, key, tree, name, v)
                if len(have) >= 2:
                    tr, births, st = birth_steps({s: labels[(tree, key, domain, s)] for s in have}, k)
                    for c in range(tr.shape[0]):
                        for j, s in enumerate(st):
                            tracks_rows.append(dict(model=model, domain=domain, key=key, tree=tree,
                                                    network=c, step=s, best_jaccard=float(tr[c, j]),
                                                    birth_step=int(births[c])))
                    for s in st:
                        row(s, domain, key, tree, 'birth_frac_born_by', float((births <= s).mean()))
    if missing:
        stderr('%d missing partition(s), e.g. %s\n' % (len(missing), missing[:3]))
        if not args.allow_partial:
            raise SystemExit('Refusing to write partial results; pass --allow-partial to override.')
    os.makedirs(args.out, exist_ok=True)
    write_rows(os.path.join(args.out, 'partitions.csv'), rows)
    write_rows(os.path.join(args.out, 'network_tracks.csv'), tracks_rows)


def cmd_checkpoint(args):
    cfg = get_cfg(args.config)
    checkpoint_measures(cfg, args.out, step=args.step, variant=args.variant, n_sub=args.n_sub)


def _upper(M):
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu].astype(np.float64)


def cmd_combine(args):
    files = sorted(glob.glob(os.path.join(args.out, 'subsample_step*_*_*.h5')))
    by = {}
    for p in files:
        m = re.match(r'subsample_step(\d+)_(.+)_(half[AB])\.h5$', os.path.basename(p))
        by.setdefault((m.group(2), m.group(3)), {})[int(m.group(1))] = p
    rows = []
    for (domain, key), paths in sorted(by.items()):
        steps = sorted(paths)
        vecs = {}
        for s in steps:
            with h5py.File(paths[s], 'r') as f:
                vecs[s] = _upper(np.asarray(f['absr']))
        final = steps[-1]
        for a, b in zip(steps[:-1], steps[1:]):
            rows.append(dict(domain=domain, key=key, step=b, measure='similarity_to_previous',
                             value=float(np.corrcoef(vecs[a], vecs[b])[0, 1])))
        for s in steps:
            rows.append(dict(domain=domain, key=key, step=s, measure='similarity_to_final',
                             value=float(np.corrcoef(vecs[s], vecs[final])[0, 1])))
        if key == HALF_NAMES[0]:
            other = by.get((domain, HALF_NAMES[1]), {})
            for s in steps:
                if s in other:
                    with h5py.File(other[s], 'r') as f:
                        vb = _upper(np.asarray(f['absr']))
                    rows.append(dict(domain=domain, key='-', step=s, measure='similarity_between_halves',
                                     value=float(np.corrcoef(vecs[s], vb)[0, 1])))
    write_rows(os.path.join(args.out, 'similarity.csv'), rows)
    # One long table of the per-checkpoint connectome and activation rows.
    allrows = []
    for p in sorted(glob.glob(os.path.join(args.out, 'connectome_step*.csv'))):
        allrows.extend(csv.DictReader(open(p)))
    write_rows(os.path.join(args.out, 'connectome.csv'), allrows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('checkpoint')
    c.add_argument('config')
    c.add_argument('--out', required=True)
    c.add_argument('--step', type=int, default=None)
    c.add_argument('--variant', default='final')
    c.add_argument('--n-sub', type=int, default=4096)
    p = sub.add_parser('partitions')
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--variant', default='final')
    p.add_argument('--domains', nargs='+', default=PROSE)
    p.add_argument('--allow-partial', action='store_true')
    m = sub.add_parser('combine')
    m.add_argument('--out', required=True)
    args = ap.parse_args()
    {'checkpoint': cmd_checkpoint, 'partitions': cmd_partitions, 'combine': cmd_combine}[args.cmd](args)


if __name__ == '__main__':
    main()
