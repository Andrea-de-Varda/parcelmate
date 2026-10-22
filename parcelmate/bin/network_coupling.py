"""Network-to-network coupling matrices of stored partitions on stored tiles (LOG.md Iteration 46).

    python -m parcelmate.bin.network_coupling --networks results/richclub/qwen3.5-2b-resid-pool4-tiles

For the real tree (--networks) and its `_null` tree: the block means of |r| (`metrics.
stream_block_means`, self-pairs excluded) of every `final` partition (half A, half B) on every
connectome half, one streamed pass per half with all partitions at once. Written to
`<tree>/metrics/coupling/coupling.npz` with keys `M_<fit>_on_<eval>` (k x k), `labels_<fit>`
(hard labels) and `coordinates`. The cross-half matrices (partition of one half, connectome of
the other) are the ones the cohesion analysis uses; the same-half ones are kept for reference.
The tiles must still exist (the rich-club trees keep them).
"""

import argparse
import glob
import os
import time

import h5py
import numpy as np

from parcelmate.bigconn import TiledMatrix
from parcelmate.constants import CONNECTIVITY_NAME, EXTENSION, HALF_NAMES
from parcelmate.metrics import hard_labels, stream_block_means
from parcelmate.util import stderr

os.umask(0o002)


def run_tree(root, variant, pool):
    parts, coords = {}, None
    for h in HALF_NAMES:
        p = os.path.join(root, variant, 'parcellation', 'parcellation_%s_%s.h5' % (pool, h))
        with h5py.File(p, 'r') as f:
            parts[h] = np.asarray(f['parcellation'])
            c = np.asarray(f['coordinates'])
        assert coords is None or np.array_equal(coords, c), 'partitions disagree on unit order'
        coords = c
    out = {'coordinates': coords}
    for h, P in parts.items():
        out['labels_%s' % h] = hard_labels(P)
    for ev in HALF_NAMES:
        path = os.path.join(root, 'connectivity', '%s_%s_%s%s' % (CONNECTIVITY_NAME, pool, ev, EXTENSION))
        R = TiledMatrix(path)
        with h5py.File(path, 'r') as f:
            assert np.array_equal(np.asarray(f['coordinates']), coords), 'tiles and partitions disagree on unit order'
        t0 = time.time()
        means = stream_block_means(R, parts)
        R.close()
        for fit, (M, _) in means.items():
            out['M_%s_on_%s' % (fit, ev)] = M
        stderr('  %s: block means of %d partitions on %s (%.0f s)\n' % (root, len(parts), ev, time.time() - t0))
    d = os.path.join(root, 'metrics', 'coupling')
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, 'coupling.npz'), **out)
    stderr('wrote %s\n' % os.path.join(d, 'coupling.npz'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--networks', required=True)
    ap.add_argument('--variant', default='final')
    ap.add_argument('--pool', default='pooled')
    args = ap.parse_args()
    for root in (args.networks.rstrip('/'), args.networks.rstrip('/') + '_null'):
        run_tree(root, args.variant, args.pool)


if __name__ == '__main__':
    main()
