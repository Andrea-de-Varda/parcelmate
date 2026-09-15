"""Verification for the Iteration 19 additions (the last round: consensus polish, T3, T4).

    PYTHONPATH=. python tests/verify_iter11_last.py

Covers: co-association counts, correlation and reliability against brute force and their
invariances; the co-association consensus (exact k, planted recovery, restarts in different
local optima, soft membership, determinism); `polish_partition`; `run_parcellation` with
stored restart labels, the consensus polish (raw and centred, splits polished too, the
unpolished consensus kept), the co-association consensus, the unchanged per-restart path, and
the new guards; the entropy-based effective network count; `crossfit_yardstick` and
`partition_inertia`; `run_pool_domains` (values, unit statistics, provenance, skipping, and its
refusals) and the `pool_domains` pipeline step on both trees; the scorer's `score.domains` and
`score.across_pairs` and its co-association reliability rows with their partition-null
reference; the yardstick driver end to end and its refusal of a non-standardized arm; the two
run configs; and the launcher's arm lists.
"""

import csv
import inspect
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import yaml
from sklearn.metrics import adjusted_rand_score

from parcelmate.bin.score import score_config
from parcelmate.constants import CONNECTIVITY_NAME, HALF_NAMES
from parcelmate.metrics import (
    blockmodel_refine, center_connectivity, coassociation_correlation, coassociation_counts,
    coassociation_reliability, crossfit_yardstick, hard_labels, partition_inertia, triviality,
)
from parcelmate.model import (
    align_samples, coassociation_consensus, polish_partition, pool_unit_stats, run_parcellation,
    run_pool_domains, sample_parcellations,
)
from parcelmate.util import derive_seed, load_h5_data, read_attrs, save_h5_data

failures = []
n_checks = [0]
ENV = dict(os.environ, PYTHONPATH='.')


def check(name, cond):
    n_checks[0] += 1
    print('%s %s' % ('OK' if cond else 'FAIL', name))
    if not cond:
        failures.append(name)


def raises(fn):
    try:
        fn()
    except AssertionError:
        return True
    return False


def planted_abs_r(n=300, k=5, noise=0.05, seed=0):
    """|r|-like symmetric matrix with k planted blocks; returns (R, labels). As in iter8."""
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, k, n)
    M = rng.uniform(0.05, 0.4, (k, k))
    M = (M + M.T) / 2
    np.fill_diagonal(M, M.diagonal() + 0.3)
    R = M[labels][:, labels] + noise * rng.randn(n, n)
    R = np.abs((R + R.T) / 2)
    np.fill_diagonal(R, 1.0)
    return R, labels


def planted_shared(n, k, noise, structure_seed, noise_seed):
    """Like `planted_abs_r`, but the block structure and the noise have separate seeds, so the
    avg file and the two halves of one domain share their networks and differ in noise only."""
    rs = np.random.RandomState(structure_seed)
    labels = rs.randint(0, k, n)
    M = rs.uniform(0.05, 0.4, (k, k))
    M = (M + M.T) / 2
    np.fill_diagonal(M, M.diagonal() + 0.3)
    R = M[labels][:, labels] + noise * np.random.RandomState(noise_seed).randn(n, n)
    R = np.abs((R + R.T) / 2)
    np.fill_diagonal(R, 1.0)
    return R, labels


def same_partition(P, Q):
    """Equal up to the naming of networks. A Hungarian consensus is only defined up to a
    permutation of its columns: it is seeded by the lowest-inertia restart, and restarts that
    converge to the same labels can differ in inertia in the last bits."""
    return adjusted_rand_score(hard_labels(P), hard_labels(Q)) > 1 - 1e-12


def noisy_restarts(truth, k, n_restarts, noise, seed):
    """Restarts that relabel a random `noise` fraction of units and permute label names."""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_restarts):
        lab = truth.copy()
        flip = rng.rand(len(lab)) < noise
        lab[flip] = rng.randint(0, k, flip.sum())
        out.append(rng.permutation(k)[lab])
    return np.array(out)


# ---------------------------------------------------------------- co-association
rng = np.random.RandomState(0)
S = rng.randint(0, 5, (7, 60))
C = coassociation_counts(S)
brute = sum((s[:, None] == s[None, :]).astype(int) for s in S)
check('coassociation_counts equals the brute-force pair count, uint16, diagonal = restarts',
      C.dtype == np.uint16 and np.array_equal(C, brute) and np.all(np.diag(C) == 7))
S_perm = np.array([rng.permutation(5)[s] for s in S])
check('coassociation_counts ignores label names', np.array_equal(coassociation_counts(S_perm), C))
S2 = rng.randint(0, 4, (9, 60))
iu = np.triu_indices(60, 1)
C2 = coassociation_counts(S2)
ref = np.corrcoef(C[iu].astype(float), C2[iu].astype(float))[0, 1]
check('coassociation_correlation equals Pearson r over the upper triangle (%.4f)' % ref,
      abs(coassociation_correlation(C, C2, block=7) - ref) < 1e-10)
check('coassociation_reliability: 1 for the same ensemble under relabelling',
      abs(coassociation_reliability(S, S_perm) - 1.0) < 1e-10)
rel_ind = coassociation_reliability(rng.randint(0, 10, (20, 400)), rng.randint(0, 10, (20, 400)))
check('coassociation_reliability: ~0 for independent random ensembles (%.3f)' % rel_ind, abs(rel_ind) < 0.05)
check('coassociation_correlation: nan when one matrix is constant off the diagonal',
      np.isnan(coassociation_correlation(coassociation_counts(np.zeros((3, 20), int)), C2[:20, :20])))

truth8 = np.random.RandomState(1).randint(0, 8, 400)
restarts = noisy_restarts(truth8, 8, 40, 0.25, seed=2)
cons = coassociation_consensus(restarts, 8)
ari = adjusted_rand_score(truth8, cons['labels'])
check('coassociation_consensus: exactly k clusters, planted partition recovered (ARI %.3f)' % ari,
      len(np.unique(cons['labels'])) == 8 and ari > 0.99)
check('coassociation_consensus: one-hot parcellation matching its labels',
      cons['parcellation'].shape == (400, 8) and np.array_equal(hard_labels(cons['parcellation']), cons['labels'])
      and np.all(cons['parcellation'].sum(axis=1) == 1))
agree = float((cons['membership'].argmax(axis=1) == cons['labels']).mean())
check('coassociation_consensus: membership argmax agrees with the cut for %.3f of units' % agree, agree > 0.95)
again = coassociation_consensus(restarts, 8)
check('coassociation_consensus is deterministic', np.array_equal(again['labels'], cons['labels'])
      and np.array_equal(again['membership'], cons['membership']))
# Restarts in different local optima: each merges two true networks and splits a third, so
# no two restarts need share a labelling.
kq, size = 6, 60
truth6 = np.repeat(np.arange(kq), size)
rng = np.random.RandomState(11)
optima = []
for _ in range(40):
    lab = truth6.copy()
    a, b, c = rng.choice(kq, 3, replace=False)
    lab[lab == b] = a
    lab[rng.permutation(np.flatnonzero(truth6 == c))[:size // 2]] = b
    optima.append(rng.permutation(kq)[lab])
optima = np.array(optima)
ari_co = adjusted_rand_score(truth6, coassociation_consensus(optima, kq)['labels'])
ari_hu = adjusted_rand_score(truth6, hard_labels(align_samples(optima, np.zeros(40), verbose=False)))
check('restarts in different optima: co-association recovers the networks they share '
      '(ARI %.3f; Hungarian consensus %.3f)' % (ari_co, ari_hu), ari_co > 0.99 and ari_co >= ari_hu - 1e-9)
check('guard: co-association consensus needs two restarts and k <= n',
      raises(lambda: coassociation_consensus(optima[:1], kq)) and raises(lambda: coassociation_consensus(optima, 999)))

# ---------------------------------------------------------------- polish_partition
R, truth = planted_abs_r()
k = truth.max() + 1
n = R.shape[0]
start = truth.copy()
flip = np.random.RandomState(3).rand(n) < 0.3
start[flip] = np.random.RandomState(4).randint(0, k, flip.sum())
start_p = np.eye(k)[start]
pol, sse = polish_partition(R, start_p, k)
ref_labels, ref_sse = blockmodel_refine(R, start, k)
check('polish_partition: one-hot of blockmodel_refine from the hard labels, same SSE',
      np.array_equal(hard_labels(pol), ref_labels) and abs(sse - ref_sse) < 1e-9
      and pol.shape == (n, k) and np.all(pol.sum(axis=1) == 1))

# ---------------------------------------------------------------- triviality entropy
coords = np.stack([np.repeat(np.arange(3), n // 3), np.tile(np.arange(n // 3), 3)], 1).astype(np.int32)
eq = np.arange(n) % 5
giant = np.zeros(n, int)
giant[:4] = [1, 2, 3, 4]
check('entropy_effective_networks: k for equal sizes, near 1 for one giant network plus singletons '
      '(%.2f, %.2f)' % (triviality(np.eye(5)[eq], coords)['entropy_effective_networks'],
                        triviality(np.eye(5)[giant], coords)['entropy_effective_networks']),
      abs(triviality(np.eye(5)[eq], coords)['entropy_effective_networks'] - 5) < 1e-9
      and triviality(np.eye(5)[giant], coords)['entropy_effective_networks'] < 1.2)

# ---------------------------------------------------------------- run_parcellation
tmp = tempfile.mkdtemp()
os.makedirs(os.path.join(tmp, CONNECTIVITY_NAME))
for key in ('avg',) + HALF_NAMES:
    save_h5_data(dict(connectivity=R, coordinates=coords),
                 os.path.join(tmp, CONNECTIVITY_NAME, '%s_dom_%s.h5' % (CONNECTIVITY_NAME, key)), verbose=False)
common = dict(output_dir=tmp, n_networks=k, n_samples=8, clustering='kmeans', binarize_connectivity=False,
              fisher_transform=True, standardize_profiles=True, connectivity_pca_components=None, seed=0,
              verbose=False)
arms = dict(
    base={},
    nostore=dict(store_samples=False),
    polish_raw=dict(blockmodel_refine_labels=True, blockmodel_refine_stage='consensus'),
    polish_deg=dict(blockmodel_refine_labels=True, blockmodel_refine_stage='consensus', blockmodel_center='degree'),
    restarts_deg=dict(blockmodel_refine_labels=True, blockmodel_center='degree'),
    coassoc=dict(consensus='coassociation'),
    coassoc_deg=dict(consensus='coassociation', blockmodel_refine_labels=True,
                     blockmodel_refine_stage='consensus', blockmodel_center='degree'),
)
for name, over in arms.items():
    run_parcellation(variant=name, **dict(common, **over))


def pf(name):
    return os.path.join(tmp, name, 'parcellation', 'parcellation_dom_halfA.h5')


D = {name: load_h5_data(pf(name), verbose=False) for name in arms}
A = {name: read_attrs(pf(name)) for name in arms}
seed_file = derive_seed(0, 'parcellation', '%s_dom_halfA.h5' % CONNECTIVITY_NAME)
ref_samples = sample_parcellations(np.abs(R), n_networks=k, n_samples=8, binarize_connectivity=False,
                                   fisher_transform=True, standardize_profiles=True, clustering='kmeans',
                                   seed=seed_file, verbose=False)['samples']
check('store_samples: restart labels written as int16, equal to the restarts drawn',
      D['base']['samples'].dtype == np.int16 and np.array_equal(D['base']['samples'], ref_samples))
check('store_samples: false writes no labels and leaves the parcellation unchanged',
      'samples' not in D['nostore'] and same_partition(D['nostore']['parcellation'], D['base']['parcellation']))
check('provenance: consensus, blockmodel_refine_stage and store_samples recorded',
      A['base'].get('consensus') == 'hungarian' and A['base'].get('blockmodel_refine_stage') == 'restarts'
      and A['nostore'].get('store_samples') in (False, 'False', 0)
      and A['polish_deg'].get('blockmodel_refine_stage') == 'consensus' and A['coassoc'].get('consensus') == 'coassociation')
check('consensus polish: the restarts and the unpolished consensus are those of the unpolished arm',
      np.array_equal(D['polish_deg']['samples'], D['base']['samples'])
      and same_partition(D['polish_deg']['parcellation_unpolished'], D['base']['parcellation']))
target = center_connectivity(np.abs(R), 'degree')
check('consensus polish (degree): the parcellation is exactly the polish of its own consensus on the centred target',
      np.array_equal(D['polish_deg']['parcellation'],
                     polish_partition(target, D['polish_deg']['parcellation_unpolished'], k)[0])
      and same_partition(D['polish_deg']['parcellation'], polish_partition(target, D['base']['parcellation'], k)[0]))
check('consensus polish (degree): both restart-split consensuses are polished the same way',
      all(same_partition(D['polish_deg'][s], polish_partition(target, D['base'][s], k)[0])
          for s in ('parcellation_split1', 'parcellation_split2')))
check('consensus polish (raw): exactly the polish of its own consensus on raw |r|',
      np.array_equal(D['polish_raw']['parcellation'],
                     polish_partition(np.abs(R).astype(np.float64), D['polish_raw']['parcellation_unpolished'], k)[0]))
raw_vs_deg = adjusted_rand_score(hard_labels(D['polish_raw']['parcellation']), hard_labels(D['polish_deg']['parcellation']))
print('    (raw against degree polish on planted blocks: ARI %.3f)' % raw_vs_deg)
ref_restarts = sample_parcellations(np.abs(R), n_networks=k, n_samples=8, binarize_connectivity=False,
                                    fisher_transform=True, standardize_profiles=True, clustering='kmeans',
                                    blockmodel_refine_labels=True, blockmodel_center='degree',
                                    seed=seed_file, verbose=False)['samples']
check('per-restart refinement unchanged: restarts refined before the consensus, nothing kept unpolished',
      np.array_equal(D['restarts_deg']['samples'], ref_restarts) and 'parcellation_unpolished' not in D['restarts_deg'])
co = coassociation_consensus(D['base']['samples'], k)
check('co-association arm: same restarts, parcellation and membership from coassociation_consensus',
      np.array_equal(D['coassoc']['samples'], D['base']['samples'])
      and np.array_equal(D['coassoc']['parcellation'], co['parcellation'])
      and np.array_equal(D['coassoc']['coassoc_membership'], co['membership']))
check('co-association arm: restart-split consensuses from the two halves of the restarts',
      np.array_equal(D['coassoc']['parcellation_split1'],
                     coassociation_consensus(D['base']['samples'][:4], k)['parcellation'])
      and np.array_equal(D['coassoc']['parcellation_split2'],
                         coassociation_consensus(D['base']['samples'][4:], k)['parcellation']))
check('co-association + polish: polishes the co-association consensus',
      np.array_equal(D['coassoc_deg']['parcellation_unpolished'], D['coassoc']['parcellation'])
      and np.array_equal(D['coassoc_deg']['parcellation'], polish_partition(target, co['parcellation'], k)[0]))
check('co-association arm recovers the planted blocks (ARI %.3f)'
      % adjusted_rand_score(truth, hard_labels(D['coassoc']['parcellation'])),
      adjusted_rand_score(truth, hard_labels(D['coassoc']['parcellation'])) > 0.9)
gdir = tempfile.mkdtemp()
check('guard: consensus stage needs blockmodel_refine_labels',
      raises(lambda: run_parcellation(variant='g', **dict(common, output_dir=gdir, blockmodel_refine_stage='consensus'))))
check('guard: unknown consensus and refine stage rejected',
      raises(lambda: run_parcellation(variant='g', **dict(common, output_dir=gdir, consensus='vote')))
      and raises(lambda: run_parcellation(variant='g', **dict(common, output_dir=gdir, blockmodel_refine_labels=True,
                                                               blockmodel_refine_stage='sweeps'))))
check('guard: centring needs the refinement, at either stage',
      raises(lambda: run_parcellation(variant='g', **dict(common, output_dir=gdir, blockmodel_center='degree'))))
check('guard: co-association consensus of a deterministic clustering rejected',
      raises(lambda: run_parcellation(variant='g_ward', **dict(common, clustering='ward', consensus='coassociation'))))
check('guard: co-association consensus with sample weights rejected',
      raises(lambda: run_parcellation(variant='g', **dict(common, output_dir=gdir, consensus='coassociation',
                                                           weight_samples=True))))

# ---------------------------------------------------------------- crossfit_yardstick
rng = np.random.RandomState(5)
kg, dg, ng = 6, 20, 600
centers = rng.randn(kg, dg) * 3
lab_g = rng.randint(0, kg, ng)
XA = centers[lab_g] + rng.randn(ng, dg)
XB = centers[lab_g] + rng.randn(ng, dg)
res = crossfit_yardstick(XA, XB, lab_g * 2 + 1, labels_eval=lab_g)
check('yardstick: shared structure is kept (ARI %.3f), non-contiguous labels handled, inertia ratio %.3f'
      % (res['ari'], res['inertia_ratio']),
      res['ari'] > 0.95 and res['n_clusters'] == kg and res['inertia_ratio'] < 1.02 and abs(res['reliability'] - 1) < 1e-12)
lab_other = rng.randint(0, kg, ng)
res_ind = crossfit_yardstick(XA, centers[lab_other] + rng.randn(ng, dg), lab_g, labels_eval=lab_other)
check('yardstick: nothing is kept when the halves share no structure (ARI %.3f)' % res_ind['ari'], res_ind['ari'] < 0.1)
brute_inertia = sum(((XA[lab_g == c] - XA[lab_g == c].mean(0)) ** 2).sum() for c in range(kg))
check('partition_inertia equals the brute-force within-cluster sum of squares',
      abs(partition_inertia(XA, lab_g) - brute_inertia) < 1e-6 * brute_inertia)

# ---------------------------------------------------------------- run_pool_domains
ptmp = tempfile.mkdtemp()
pconn = os.path.join(ptmp, CONNECTIVITY_NAME)
os.makedirs(pconn)
npu = 90
pcoords = np.stack([np.repeat(np.arange(3), npu // 3), np.tile(np.arange(npu // 3), 3)], 1).astype(np.int32)


def conn_file(root, domain, key):
    return os.path.join(root, CONNECTIVITY_NAME, '%s_%s_%s.h5' % (CONNECTIVITY_NAME, domain, key))


def write_domain(root, name, seed, coordinates):
    r = np.random.RandomState(seed)
    for i, key in enumerate(('avg',) + HALF_NAMES):
        M = r.uniform(-0.6, 0.6, (npu, npu))
        M = ((M + M.T) / 2).astype(np.float32)
        np.fill_diagonal(M, 1.0)
        save_h5_data(dict(connectivity=M, coordinates=coordinates,
                          unit_means=r.randn(npu).astype(np.float32),
                          unit_stds=r.uniform(0.5, 2, npu).astype(np.float32),
                          n_obs=np.asarray(100 * (seed + 1) + i)),
                     conn_file(root, name, key), verbose=False)


for i, d in enumerate(('d1', 'd2', 'd3')):
    write_domain(ptmp, d, i, pcoords)
pools = {'lodo_d1': ['d2', 'd3'], 'pool_all': ['d1', 'd2', 'd3']}
run_pool_domains(ptmp, pools=pools, verbose=False)
ok_vals = ok_stats = True
for name, members in pools.items():
    for key in ('avg',) + HALF_NAMES:
        got = load_h5_data(conn_file(ptmp, name, key), verbose=False)
        src = [load_h5_data(conn_file(ptmp, m, key), verbose=False) for m in members]
        hand = np.tanh(np.mean([np.arctanh(s['connectivity'].astype(np.float64) * (1 - 1e-3)) for s in src], axis=0))
        ok_vals &= np.allclose(got['connectivity'], hand, atol=1e-5) and np.array_equal(got['coordinates'], pcoords)
        mu, sd = pool_unit_stats([s['unit_means'] for s in src], [s['unit_stds'] for s in src],
                                 [int(s['n_obs']) for s in src])
        ok_stats &= (np.allclose(got['unit_means'], mu) and np.allclose(got['unit_stds'], sd)
                     and int(got['n_obs']) == sum(int(s['n_obs']) for s in src))
check('pools: each key is the Fisher mean of the members\' same key, units unchanged', ok_vals)
check('pools: unit means, standard deviations and token counts pooled', ok_stats)
pa = read_attrs(conn_file(ptmp, 'lodo_d1', 'halfA'))
check('pools: provenance names the members, their files and fingerprints',
      pa.get('pooled_from') == 'd2, d3' and pa.get('key') == 'halfA'
      and pa.get('sources') == 'connectivity_d2_halfA.h5, connectivity_d3_halfA.h5'
      and len(str(pa.get('source_fingerprints')).split(', ')) == 2)
mtime = os.stat(conn_file(ptmp, 'pool_all', 'avg')).st_mtime_ns
run_pool_domains(ptmp, pools=pools, verbose=False)
check('pools: an existing pool is skipped', os.stat(conn_file(ptmp, 'pool_all', 'avg')).st_mtime_ns == mtime)
check('pools: a pool name reused with different members is refused',
      raises(lambda: run_pool_domains(ptmp, pools={'lodo_d1': ['d1', 'd3']}, verbose=False)))
check('pools: a pool may not take the name of a domain',
      raises(lambda: run_pool_domains(ptmp, pools={'d1': ['d2', 'd3']}, overwrite=True, verbose=False)))
check('pools: pooling a pool is refused', raises(lambda: run_pool_domains(ptmp, pools={'x': ['lodo_d1', 'd1']}, verbose=False)))
check('pools: a missing member is refused', raises(lambda: run_pool_domains(ptmp, pools={'x': ['d1', 'zz']}, verbose=False)))
other_coords = pcoords.copy()
other_coords[0, 1] = 999
write_domain(ptmp, 'd4', 7, other_coords)
check('pools: members with different units are refused',
      raises(lambda: run_pool_domains(ptmp, pools={'x': ['d1', 'd4']}, verbose=False)))
check('pools: a single-member pool is refused', raises(lambda: run_pool_domains(ptmp, pools={'x': ['d1']}, verbose=False)))
run_parcellation(output_dir=ptmp, n_networks=3, n_samples=4, clustering='kmeans', binarize_connectivity=False,
                 fisher_transform=True, standardize_profiles=True, connectivity_pca_components=None, seed=0,
                 variant='km', verbose=False)
check('pools are parcellated like domains',
      all(os.path.exists(os.path.join(ptmp, 'km', 'parcellation', 'parcellation_%s_%s.h5' % (p, key)))
          for p in pools for key in ('avg',) + HALF_NAMES))

mtmp = tempfile.mkdtemp()
for tree in (mtmp, mtmp + '_null'):
    os.makedirs(os.path.join(tree, CONNECTIVITY_NAME))
    for i, d in enumerate(('d1', 'd2')):
        write_domain(tree, d, i, pcoords)
cfg_path = os.path.join(mtmp, 'pool.yml')
with open(cfg_path, 'w') as f:
    yaml.safe_dump(dict(output_dir=mtmp, connectivity=dict(domains=['d1', 'd2'], null_model='circshift'),
                        pool_domains=dict(pools={'pool_d': ['d1', 'd2']})), f)
r = subprocess.run([sys.executable, '-m', 'parcelmate.bin.main', cfg_path, '-s', 'pool_domains'],
                   capture_output=True, text=True, env=ENV)
check('main.py -s pool_domains pools both trees',
      r.returncode == 0 and all(os.path.exists(conn_file(t, 'pool_d', key))
                                for t in (mtmp, mtmp + '_null') for key in ('avg',) + HALF_NAMES))

# ---------------------------------------------------------------- scorer: domains, pairs, coassoc
stmp = tempfile.mkdtemp()
ns, ks = 150, 5
scoords = np.stack([np.repeat(np.arange(3), ns // 3), np.tile(np.arange(ns // 3), 3)], 1).astype(np.int32)
for t_i, tree in enumerate((stmp, stmp + '_null')):
    os.makedirs(os.path.join(tree, CONNECTIVITY_NAME))
    for d_i, d in enumerate(('a', 'b', 'c')):
        for key_i, key in enumerate(('avg',) + HALF_NAMES):
            if t_i == 0:
                M, _ = planted_shared(ns, ks, 0.08, structure_seed=10 + d_i, noise_seed=100 * d_i + key_i)
            else:
                rr = np.random.RandomState(100 + 10 * d_i + key_i)
                M = rr.uniform(0, 0.3, (ns, ns))
                M = (M + M.T) / 2
                np.fill_diagonal(M, 1.0)
            save_h5_data(dict(connectivity=M, coordinates=scoords), conn_file(tree, d, key), verbose=False)
    for variant, over in (('km', {}), ('bin', dict(binarize_connectivity=True, fisher_transform=False,
                                                   standardize_profiles=False))):
        run_parcellation(**dict(dict(output_dir=tree, n_networks=ks, n_samples=6, clustering='kmeans',
                                     binarize_connectivity=False, fisher_transform=True, standardize_profiles=True,
                                     connectivity_pca_components=None, seed=0, variant=variant, verbose=False), **over))
ACROSS = ('fidelity_across', 'fidelity_across_halves', 'reliability_across', 'reliability_across_halves')
base_cfg = dict(output_dir=stmp, seed=1, connectivity=dict(domains=['a', 'b', 'c']),
                parcellation_variants=dict(km={}))
cfg = dict(base_cfg, score=dict(domains=['a', 'b', 'c'], across_pairs=[['a', 'b'], ['c', 'a']]))
rows, _ = score_config(cfg, verbose=False)
by_tree = {}
for row in rows:
    if row['metric'] in ACROSS:
        by_tree.setdefault(row['tree'], set()).add((row['fit'], row['eval']))
check('across_pairs: only the listed pairs are scored, in every tree (%s)' % sorted(by_tree.get('real', [])),
      all(v <= {('a', 'b'), ('c', 'a')} for v in by_tree.values())
      and all(by_tree.get(t) == {('a', 'b'), ('c', 'a')} for t in ('real', 'null', 'pnull')))
check('across_pairs: within-domain rows still cover every domain',
      {r_['fit'] for r_ in rows if r_['metric'] == 'reliability_within' and r_['tree'] == 'real'} == {'a', 'b', 'c'})


def coassoc_value(tree, domain):
    vals = [r_['value'] for r_ in rows if r_['metric'] == 'reliability_within_coassoc'
            and r_['tree'] == tree and r_['fit'] == domain and r_['variant'] == 'km']
    return vals[0] if len(vals) == 1 else None


def samples_of(root, domain, key):
    return load_h5_data(os.path.join(root, 'km', 'parcellation', 'parcellation_%s_%s.h5' % (domain, key)),
                        verbose=False)['samples']


sa, sb = samples_of(stmp, 'a', 'halfA'), samples_of(stmp, 'a', 'halfB')
na, nb = samples_of(stmp + '_null', 'a', 'halfA'), samples_of(stmp + '_null', 'a', 'halfB')
real_v, pnull_v = coassoc_value('real', 'a'), coassoc_value('pnull', 'a')
check('score: reliability_within_coassoc is the co-association reliability of the two halves (%.3f)' % real_v,
      real_v is not None and abs(real_v - coassociation_reliability(sa, sb)) < 1e-12)
check('score: its partition-null reference averages both orientations (%.3f)' % pnull_v,
      pnull_v is not None and abs(pnull_v - (coassociation_reliability(na, sb) + coassociation_reliability(sa, nb)) / 2) < 1e-12
      and coassoc_value('null', 'a') is not None)
check('score: a pair naming an unlisted domain, or one domain twice, is refused',
      raises(lambda: score_config(dict(base_cfg, score=dict(across_pairs=[['a', 'z']])), verbose=False))
      and raises(lambda: score_config(dict(base_cfg, score=dict(across_pairs=[['a', 'a']])), verbose=False)))
rows_all, _ = score_config(base_cfg, verbose=False)
check('score: without a score section every ordered pair is scored, as before',
      {(r_['fit'], r_['eval']) for r_ in rows_all if r_['metric'] == 'fidelity_across_halves' and r_['tree'] == 'real'}
      == {(x, y) for x in 'abc' for y in 'abc' if x != y})
rows_sub, _ = score_config(dict(base_cfg, score=dict(domains=['a', 'b'])), verbose=False)
check('score: score.domains restricts the domains scored',
      {r_['fit'] for r_ in rows_sub} == {'a', 'b'} and {r_['eval'] for r_ in rows_sub} == {'a', 'b'})

# ---------------------------------------------------------------- yardstick driver
ypath = os.path.join(stmp, 'y.csv')
r = subprocess.run([sys.executable, 'analysis/yardstick.py', '--root', stmp, '--domains', 'a', 'b', 'c',
                    '--variants', 'km', '--out', ypath], capture_output=True, text=True, env=ENV)
yrows = list(csv.DictReader(open(ypath))) if os.path.exists(ypath) else []
check('yardstick driver: runs, both trees, both directions, all metrics, finite values',
      r.returncode == 0 and {x['tree'] for x in yrows} == {'real', 'pnull'}
      and {x['direction'] for x in yrows} == {'halfA->halfB', 'halfB->halfA'}
      and {x['metric'] for x in yrows} == {'yardstick_ari', 'reliability_ari', 'inertia_ratio', 'n_iter', 'n_clusters'}
      and len(yrows) == 3 * 2 * 2 * 5 and all(np.isfinite(float(x['value'])) for x in yrows))
real_y = np.mean([float(x['value']) for x in yrows if x['tree'] == 'real' and x['metric'] == 'yardstick_ari'])
real_rel = np.mean([float(x['value']) for x in yrows if x['tree'] == 'real' and x['metric'] == 'reliability_ari'])
check('yardstick driver: on planted blocks the steered agreement is high (%.3f; split-half ARI %.3f)'
      % (real_y, real_rel), real_y > 0.8)
r = subprocess.run([sys.executable, 'analysis/yardstick.py', '--root', stmp, '--domains', 'a',
                    '--variants', 'bin', '--out', os.path.join(stmp, 'y2.csv')], capture_output=True, text=True, env=ENV)
check('yardstick driver: refuses an arm not clustered on standardized Fisher profiles',
      r.returncode != 0 and 'standardized Fisher' in r.stderr)

# ---------------------------------------------------------------- the run configs
sig = inspect.signature(run_parcellation)


def resolved(path):
    c = yaml.safe_load(open(path))
    out = {}
    for name, over in c['parcellation_variants'].items():
        full = dict(c['parcellation'])
        full.update(over or {})
        out[name] = full
    return c, out


def binds(arms_):
    try:
        for name, full in arms_.items():
            sig.bind(variant=name, **full)
    except TypeError as e:
        print('   ', e)
        return False
    return True


def diff(x, y):
    return {key for key in set(x) | set(y) if x.get(key) != y.get(key)}


last_cfg, last = resolved('configs/last_mlp.yml')
pool_cfg, pooled = resolved('configs/pooled_mlp.yml')
_, final = resolved('configs/final_mlp.yml')
yolo_mlp_cfg, _ = resolved('configs/yolo_mlp.yml')
check('last_mlp: 15 arms, all bind to run_parcellation', len(last) == 15 and binds(last))
check('pooled_mlp: 3 arms, all bind to run_parcellation', len(pooled) == 3 and binds(pooled))
check('both: connectivity identical to configs/yolo_mlp.yml, own trees with a null model',
      last_cfg['connectivity'] == yolo_mlp_cfg['connectivity'] == pool_cfg['connectivity']
      and last_cfg['output_dir'] == 'results/last_mlp' and pool_cfg['output_dir'] == 'results/pooled_mlp')
check('last_mlp: the base arm is final_mlp vmf_pca100_lloyd100 plus stored restart labels',
      diff(last['vmf_pca100_lloyd100'], final['vmf_pca100_lloyd100']) == {'store_samples'})
polish = {'blockmodel_refine_labels', 'blockmodel_refine_stage'}
intended = {
    'vmf_pca100_lloyd100_bm_raw': ('vmf_pca100_lloyd100', polish),
    'vmf_pca100_lloyd100_bm_double': ('vmf_pca100_lloyd100_bm_raw', {'blockmodel_center'}),
    'vmf_pca100_lloyd100_bm_degree': ('vmf_pca100_lloyd100_bm_raw', {'blockmodel_center'}),
    'vmf_sparse_pca100_lloyd100': ('vmf_pca100_lloyd100', {'sparsify_profiles'}),
    'vmf_sparse_pca100_lloyd100_bm_degree': ('vmf_pca100_lloyd100_bm_degree', {'sparsify_profiles'}),
    'vmf_pca100_lloyd50': ('vmf_pca100_lloyd100', {'n_networks'}),
    'vmf_pca100_lloyd50_bm_degree': ('vmf_pca100_lloyd100_bm_degree', {'n_networks'}),
    'vmf_pca100_lloyd200': ('vmf_pca100_lloyd100', {'n_networks'}),
    'vmf_pca100_lloyd200_bm_degree': ('vmf_pca100_lloyd100_bm_degree', {'n_networks'}),
    'vmf_pca100_lloyd100_coassoc': ('vmf_pca100_lloyd100', {'consensus'}),
    'vmf_pca100_lloyd100_coassoc_bm_degree': ('vmf_pca100_lloyd100_bm_degree', {'consensus'}),
    'vmf_pca100_lloyd100_n200': ('vmf_pca100_lloyd100', {'n_samples'}),
    'vmf_pca100_lloyd100_n200_coassoc': ('vmf_pca100_lloyd100_n200', {'consensus'}),
    # the confirmation arm (Iteration 21): 200 restarts plus sparse profiles
    'vmf_sparse_pca100_lloyd100_n200': ('vmf_pca100_lloyd100_n200', {'sparsify_profiles'}),
}
bad = [(a_, ref_, diff(last[a_], last[ref_])) for a_, (ref_, keys) in intended.items() if diff(last[a_], last[ref_]) != keys]
if bad:
    print('   ', bad)
check('last_mlp: every arm differs from its reference arm in exactly the intended settings',
      not bad and set(intended) | {'vmf_pca100_lloyd100'} == set(last))
check('last_mlp: values as designed',
      [last['vmf_pca100_lloyd%d' % kk]['n_networks'] for kk in (50, 100, 200)] == [50, 100, 200]
      and last['vmf_pca100_lloyd100_bm_double']['blockmodel_center'] == 'double'
      and last['vmf_pca100_lloyd100_bm_degree']['blockmodel_center'] == 'degree'
      and last['vmf_pca100_lloyd100_bm_raw'].get('blockmodel_center') is None
      and all(v['blockmodel_refine_stage'] == 'consensus' for v in last.values() if v.get('blockmodel_refine_labels'))
      and last['vmf_pca100_lloyd100_n200']['n_samples'] == 200
      and all(v['consensus'] == 'coassociation' for a_, v in last.items() if 'coassoc' in a_)
      and all(v['connectivity_pca_components'] == 100 and v['pca_whiten'] is False and v['store_samples'] is True
              and v['clustering'] == 'kmeans' and v['standardize_profiles'] and v['fisher_transform'] for v in last.values()))
check('pooled_mlp: every arm resolves to the same settings as its namesake in last_mlp',
      all(pooled[a_] == last[a_] for a_ in pooled))
prose = ['wikitext', 'bookcorpus', 'agnews', 'tldr17']
pools_cfg = pool_cfg['pool_domains']['pools']
lodo_ok = all(sorted(pools_cfg['lodo_%s' % d]) == sorted(set(prose) - {d}) for d in prose)
pair_names = [p for p in pools_cfg if p.startswith('pair_')]
complements = {}
for p in pair_names:
    comp = [q for q in pair_names if set(pools_cfg[q]) == set(prose) - set(pools_cfg[p])]
    complements[p] = comp
pairs_ok = (len(pair_names) == 6 and all(len(v) == 1 for v in complements.values())
            and all(p == 'pair_%s_%s' % tuple(pools_cfg[p]) for p in pair_names))
check('pooled_mlp: lodo pools hold the other three domains, 6 pairs in 3 complementary splits, pool_all holds all four',
      lodo_ok and pairs_ok and sorted(pools_cfg['pool_all']) == sorted(prose) and len(pools_cfg) == 11)
sdom = pool_cfg['score']['domains']
spairs = {tuple(p) for p in pool_cfg['score']['across_pairs']}
expected_pairs = ({(x, y) for x in prose for y in prose if x != y}
                  | {('lodo_%s' % d, d) for d in prose} | {(d, 'lodo_%s' % d) for d in prose}
                  | {(p, complements[p][0]) for p in pair_names})
check('pooled_mlp: scored domains are the prose domains and every pool; the 26 pairs are exactly the intended ones',
      sorted(sdom) == sorted(prose + list(pools_cfg)) and len(pool_cfg['score']['across_pairs']) == 26
      and spairs == expected_pairs)

# ---------------------------------------------------------------- the launcher
text = open('scripts/launch_yolo.sh').read()
lists = {name: value.split() for name, value in re.findall(r'^([A-Z0-9_]+)="([^"]*)"', text, re.M)}
check('launcher: the last_mlp lists name every arm exactly once',
      sorted(lists['LAST_MLP_FAST'] + lists['LAST_MLP_K200'] + lists['LAST_MLP_N200']
             + lists['LAST_MLP_CONFIRM']) == sorted(last))
check('confirmation arm: exactly the two changes kept in Iteration 20, sparse profiles and 200 restarts',
      lists['LAST_MLP_CONFIRM'] == ['vmf_sparse_pca100_lloyd100_n200']
      and diff(last['vmf_sparse_pca100_lloyd100_n200'], last['vmf_pca100_lloyd100']) == {'sparsify_profiles', 'n_samples'}
      and last['vmf_sparse_pca100_lloyd100_n200'] == dict(last['vmf_sparse_pca100_lloyd100'], n_samples=200))
check('launcher: the pooled list names every pooled arm', sorted(lists['POOLED_MLP']) == sorted(pooled))
check('launcher: the k = 200 and 200-restart lists hold exactly those arms',
      all(last[a_]['n_networks'] == 200 for a_ in lists['LAST_MLP_K200'])
      and all(last[a_]['n_samples'] == 200 for a_ in lists['LAST_MLP_N200']))
check('launcher: the yardstick reads final_mlp arms that exist and were clustered on standardized Fisher profiles',
      all(a_ in final and final[a_]['standardize_profiles'] and final[a_]['fisher_transform']
          and not final[a_].get('sparsify_profiles') for a_ in lists['YARDSTICK_VARIANTS']))
r = subprocess.run(['bash', '-n', 'scripts/launch_yolo.sh'], capture_output=True, text=True)
check('launcher: parses', r.returncode == 0)
r = subprocess.run(['bash', 'scripts/launch_yolo.sh', 'nope'], capture_output=True, text=True)
check('launcher: usage lists the last-round modes',
      r.returncode == 2 and 'generate_last' in r.stderr and 'submit_last' in r.stderr
      and 'generate_confirm' in r.stderr and 'submit_confirm' in r.stderr)

print('\n%d check(s), %d failure(s)' % (n_checks[0], len(failures)))
if failures:
    for f in failures:
        print('  - %s' % f)
    sys.exit(1)
