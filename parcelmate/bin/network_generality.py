"""D: do networks fitted on one dataset stay cohesive in another dataset's connectome?
(LOG.md Iteration 36.)

    python -m parcelmate.bin.network_generality configs/qwen35/qwen3.5-4b-rest.yml

For every dataset e of the config, one pass over its tiled half B; on it, the block means of
the half-A partitions of every dataset (real, null partition and a random partition), so each
network gets its cohesion (mean |r| within minus mean |r| to other networks) in every
dataset. The fit == eval rows are the held-out ceiling. Run while the real tiles exist, i.e.
before `score purge_connectivity`. Writes <output_dir>/metrics/network_generality.csv.
"""

import argparse
import os

import yaml

from parcelmate.bigconn import TiledMatrix
from parcelmate.bin.score import conn_path, parc_path, random_partition
from parcelmate.bin.stability import write
from parcelmate.metrics import hard_labels
from parcelmate.stability import cohesion_across
from parcelmate.util import derive_seed, load_h5_data, stderr

os.umask(0o002)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config')
    ap.add_argument('--variant', default='final')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root = cfg['output_dir']
    null_root = root.rstrip('/') + '_null'
    datasets = cfg['connectivity']['domains']
    seed = cfg.get('seed', 0) or 0
    parts = {}
    for d in datasets:
        P = load_h5_data(parc_path(root, args.variant, d, 'halfA'), verbose=False)['parcellation']
        Pn = load_h5_data(parc_path(null_root, args.variant, d, 'halfA'), verbose=False)['parcellation']
        k = P.shape[1]
        parts[(d, 'real')] = hard_labels(P)
        parts[(d, 'pnull')] = hard_labels(Pn)
        parts[(d, 'rand')] = random_partition(len(P), k, derive_seed(seed, 'rand_partition', args.variant, d, 'a')).argmax(1)
    rows = []
    for e in datasets:
        stderr('eval %s (half B)\n' % e)
        R = TiledMatrix(conn_path(root, e, 'halfB'))
        rows += cohesion_across({e: R}, parts)
        R.close()
    out = os.path.join(root, 'metrics')
    os.makedirs(out, exist_ok=True)
    write(os.path.join(out, 'network_generality.csv'), rows)


if __name__ == '__main__':
    main()
