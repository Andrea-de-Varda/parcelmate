"""Connectome consistency without any parcellation (LOG.md Iteration 32).

    python -m parcelmate.bin.connectome_consistency <config.yml> [--purge]

For the config's real tree: the Pearson r over all unit pairs (strict upper triangle)
between two |r| connectomes, computed exactly as `score.py` computes its `(ceiling)` rows
(`load_connectivity` then `fidelity_ceiling`, measure r):

  fidelity_within_r       half A vs half B of one domain
  fidelity_across_halves  half A of the fit domain vs half B of the eval domain

Writes `<output_dir>/metrics/connectome_consistency.csv` with the score table's columns
(tree, variant='(ceiling)', metric, fit, eval, value) plus model, revision and
shuffle_words from the connectivity provenance. At most two matrices are in memory at once.
`--purge` then deletes the real tree's connectivity (the table is all this job keeps).
"""

import argparse
import csv
import os
import shutil

import yaml

from parcelmate.bin.score import conn_path, load_connectivity
from parcelmate.constants import CONNECTIVITY_NAME, HALF_NAMES
from parcelmate.metrics import fidelity_ceiling
from parcelmate.util import read_attrs, stderr

os.umask(0o002)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config')
    ap.add_argument('--purge', action='store_true')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root = cfg['output_dir']
    domains = list(cfg['connectivity']['domains'])
    attrs = read_attrs(conn_path(root, domains[0], HALF_NAMES[0]))
    meta = dict(model=attrs.get('model_name', ''), revision=attrs.get('revision', ''),
                shuffle_words=attrs.get('shuffle_words', ''))
    rows = []
    for fit in domains:
        Ra = load_connectivity(conn_path(root, fit, HALF_NAMES[0]))
        for ev in domains:
            Rb = load_connectivity(conn_path(root, ev, HALF_NAMES[1]))
            metric = 'fidelity_within_r' if ev == fit else 'fidelity_across_halves'
            v = fidelity_ceiling(Ra, Rb, measure='r')
            rows.append(dict(meta, tree='real', variant='(ceiling)', metric=metric, fit=fit, eval=ev, value=v))
            stderr('  %-10s -> %-10s %s %.4f\n' % (fit, ev, metric, v))
            del Rb
        del Ra
    out = os.path.join(root, 'metrics')
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, 'connectome_consistency.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    stderr('wrote %s (%d rows)\n' % (path, len(rows)))
    if args.purge:
        d = os.path.join(root, CONNECTIVITY_NAME)
        shutil.rmtree(d)
        stderr('purged %s\n' % d)


if __name__ == '__main__':
    main()
