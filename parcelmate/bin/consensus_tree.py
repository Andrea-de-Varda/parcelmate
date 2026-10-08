"""Write cross-dataset consensus labels as a partition tree (LOG.md Iteration 38, item 5).

    python -m parcelmate.bin.consensus_tree results/qwen35/consensus_2b_prose/units.npz \
        results/qwen35/consensus_2b_prose_tree

From `bin/restart_consensus.py`'s units.npz: the consensus over all restarts of the half-A
partitions and of the half-B partitions, real and null, written as
<out>{,_null}/final/parcellation/parcellation_consensus_<half>.h5 (one-hot `parcellation`,
`coordinates`, and the unit confidence), the layout `compare_circuits`,
`graded_circuits` and `paired_circuits` read. The text domain is called `consensus`.
"""

import os
import sys

import h5py
import numpy as np

os.umask(0o002)


def main():
    src, out = sys.argv[1], sys.argv[2].rstrip('/')
    z = np.load(src)
    coords = z['coordinates']
    for tree, root in (('real', out), ('null', out + '_null')):
        d = os.path.join(root, 'final', 'parcellation')
        os.makedirs(d, exist_ok=True)
        for half, key in (('halfA', 'A_all'), ('halfB', 'B_all')):
            lab = z['%s_%s_labels' % (tree, key)].astype(np.int64)
            k = int(lab.max()) + 1
            P = np.zeros((len(lab), k), dtype=np.float32)
            P[np.arange(len(lab)), lab] = 1.0
            with h5py.File(os.path.join(d, 'parcellation_consensus_%s.h5' % half), 'w') as f:
                f.create_dataset('parcellation', data=P, compression='gzip')
                f.create_dataset('coordinates', data=coords)
                f.create_dataset('confidence', data=z['%s_%s_confidence' % (tree, key)])
                f.attrs['source'] = os.path.abspath(src)
                f.attrs['n_networks'] = k
            print('wrote', os.path.join(d, 'parcellation_consensus_%s.h5' % half))


if __name__ == '__main__':
    main()
