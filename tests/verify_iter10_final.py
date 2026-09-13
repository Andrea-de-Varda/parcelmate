"""Verification for the Iteration 17 additions (final tests T1, T2 and T5).

    PYTHONPATH=. python tests/verify_iter10_final.py

Covers: `sparsify_standardized` (kept set, re-standardization, and the invariance to a unit's
level that the sparsify-then-standardize order lacks) and its guards; `clustering:
ward_kmeans` (single sample, determinism, a Lloyd fixed point no worse than Ward, planted
recovery); `metrics.center_connectivity` (each centring exact on the effect it targets, the
rank-1 residue it leaves of the other, and refinement on the centred target keeping a planted
pattern partition that refinement on raw |r| trades for hubness); the centred refinement
inside `sample_parcellations` and its guards; provenance of the new settings through
`run_parcellation` and a restricted score; `variants_tag` for long `-V` lists; the two run
configs (every arm binds to `run_parcellation`, differs from its reference arm only where
intended, repeats the residual-stream settings it is compared with, and carries the
connectivity section of the tree it is linked from); and the launcher's arm lists.
"""

import inspect
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import yaml
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from parcelmate.bin.make_jobs import DEFAULTS, get_job
from parcelmate.bin.score import score_config
from parcelmate.constants import CONNECTIVITY_NAME, HALF_NAMES
from parcelmate.data import standardize_array
from parcelmate.metrics import block_sse, blockmodel_refine, center_connectivity
from parcelmate.model import run_parcellation, sample_parcellations, sparsify_standardized
from parcelmate.util import load_h5_data, read_attrs, save_h5_data, variants_tag

failures = []
n_checks = [0]


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


def planted_hub(kind, hub_scale=3.0, n=400, k=6, pattern=0.08, noise=0.03, seed=0):
    """A planted block pattern under a strong per-unit effect, additive or multiplicative.

    theta is lognormal with mean 1, so a few units are much stronger than the rest, as hub
    units are. Returns (R, pattern labels, theta).
    """
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, k, n)
    P = rng.uniform(0, pattern, (k, k))
    P = (P + P.T) / 2
    np.fill_diagonal(P, P.diagonal() + pattern)
    theta = rng.lognormal(0, 0.6, n)
    theta /= theta.mean()
    if kind == 'multiplicative':
        H = 0.05 * hub_scale * np.outer(theta, theta)
    else:
        H = 0.025 * hub_scale * (theta[:, None] + theta[None, :])
    E = noise * rng.randn(n, n)
    R = np.abs(H + P[labels][:, labels] + (E + E.T) / 2)
    np.fill_diagonal(R, 1.0)
    return R, labels, theta


def deciles(x, b=10):
    return np.argsort(np.argsort(x)) * b // len(x)


def corrupt(labels, frac, k, seed=1):
    rng = np.random.RandomState(seed)
    out = labels.copy()
    flip = rng.rand(len(out)) < frac
    out[flip] = rng.randint(0, k, flip.sum())
    return out


def inertia(X, L):
    return float(sum(((X[L == c] - X[L == c].mean(axis=0)) ** 2).sum() for c in np.unique(L)))


# ---------------------------------------------------------------- sparsify_standardized
rng = np.random.RandomState(0)
n_u, d = 60, 500
Z = standardize_array(rng.exponential(size=(n_u, d)), axis=-1)
S = sparsify_standardized(Z)
kept = Z > np.quantile(Z, 0.9, axis=1, keepdims=True)
check('sparsify_standardized: rows re-standardized (mean 0, sd 1)',
      np.allclose(S.mean(axis=1), 0, atol=1e-10) and np.allclose(S.std(axis=1), 1, atol=1e-10))
check("sparsify_standardized: the kept entries are each row's top 10%% (%.3f of entries)" % kept.mean(),
      np.array_equal(S > S.min(axis=1, keepdims=True) + 1e-9, kept) and abs(kept.mean() - 0.1) < 0.005)
check('sparsify_standardized: kept values are the z-scores up to a per-row affine map',
      all(np.corrcoef(S[i, kept[i]], Z[i, kept[i]])[0, 1] > 1 - 1e-10 for i in range(n_u)))
p = rng.exponential(size=d)
X2 = np.stack([0.001 + 0.1 * p, 1.0 + 0.1 * p])      # one pattern: a low and a high baseline
S2 = sparsify_standardized(standardize_array(X2, axis=-1))
F2 = standardize_array(np.where(X2 > np.quantile(X2, 0.9, axis=1, keepdims=True), X2, 0.0), axis=-1)
r_old = np.corrcoef(F2[0], F2[1])[0, 1]
check('sparsify_standardized: one pattern at two baselines gives one profile',
      np.allclose(S2[0], S2[1], atol=1e-10))
check('the sparsify-then-standardize order it replaces does not (r = %.3f between the profiles)' % r_old,
      r_old < 0.99 and not np.allclose(F2[0], F2[1], atol=1e-3))

R, truth = planted_abs_r()
k = truth.max() + 1
base = dict(n_networks=k, n_samples=1, binarize_connectivity=False, fisher_transform=True, verbose=False)
check('guard: sparsify_profiles needs standardize_profiles',
      raises(lambda: sample_parcellations(R, sparsify_profiles=True, **base)))
check('guard: sparsify_profiles and sparsify_fisher are alternatives',
      raises(lambda: sample_parcellations(R, standardize_profiles=True, sparsify_profiles=True,
                                          sparsify_fisher=True, **base)))
check('guard: sparsify_profiles and binarize_connectivity are alternatives',
      raises(lambda: sample_parcellations(R, **dict(base, binarize_connectivity=True, fisher_transform=False),
                                          standardize_profiles=True, sparsify_profiles=True)))
check('guard: sparsify_profiles is row-wise only',
      raises(lambda: sample_parcellations(R, standardize_profiles=True, sparsify_profiles=True,
                                          binarize_scope='global', **base)))
out_sp = sample_parcellations(R, n_networks=k, n_samples=2, binarize_connectivity=False, fisher_transform=True,
                              standardize_profiles=True, sparsify_profiles=True, clustering='kmeans', seed=0,
                              verbose=False)
ari = adjusted_rand_score(truth, out_sp['samples'][0])
check('sparse-profile path: valid labels, planted blocks recovered (ARI %.3f)' % ari,
      out_sp['samples'].shape == (2, R.shape[0]) and ari > 0.9)

# ---------------------------------------------------------------- ward_kmeans
rng = np.random.RandomState(3)
feat = dict(binarize_connectivity=False, fisher_transform=False, standardize_profiles=False, verbose=False)
for label, spread in (('separated', 2.0), ('overlapping', 0.6)):
    kg, ng, dg = 6, 600, 8
    lab_g = rng.randint(0, kg, ng)
    Xg = (rng.randn(kg, dg) * spread)[lab_g] + rng.randn(ng, dg)
    wk = sample_parcellations(Xg, n_networks=kg, n_samples=7, clustering='ward_kmeans', seed=0, **feat)
    wk_other = sample_parcellations(Xg, n_networks=kg, n_samples=1, clustering='ward_kmeans', seed=99, **feat)
    wd = sample_parcellations(Xg, n_networks=kg, n_samples=1, clustering='ward', seed=0, **feat)
    labs, wlabs = wk['samples'][0].astype(int), wd['samples'][0].astype(int)
    C = np.stack([Xg[labs == c].mean(axis=0) for c in range(kg)])
    nearest = ((Xg[:, None, :] - C[None]) ** 2).sum(axis=-1).argmin(axis=1)
    moved = int((adjusted_rand_score(wlabs, labs) < 1) and (labs != wlabs).sum())
    if label == 'separated':
        check('ward_kmeans forces a single sample', wk['samples'].shape == (1, ng))
        check('ward_kmeans is deterministic across seeds', np.array_equal(wk['samples'], wk_other['samples']))
        check('ward_kmeans recovers a separated planted mixture (ARI %.3f)' % adjusted_rand_score(lab_g, labs),
              adjusted_rand_score(lab_g, labs) > 0.95)
    check('ward_kmeans (%s): a Lloyd fixed point, every unit nearest its own centroid' % label,
          np.array_equal(nearest, labs))
    check('ward_kmeans (%s): score is the inertia of its labels' % label,
          abs(inertia(Xg, labs) - wk['scores'][0]) <= 1e-6 * inertia(Xg, labs))
    check("ward_kmeans (%s): inertia %.1f, no worse than Ward's %.1f (ARI with Ward %.3f)"
          % (label, inertia(Xg, labs), inertia(Xg, wlabs), adjusted_rand_score(wlabs, labs)),
          inertia(Xg, labs) <= inertia(Xg, wlabs) + 1e-9)
    if label == 'overlapping':
        check('ward_kmeans (overlapping): the Lloyd step changes the Ward partition and lowers the inertia',
              adjusted_rand_score(wlabs, labs) < 1 and inertia(Xg, labs) < inertia(Xg, wlabs))

# ---------------------------------------------------------------- center_connectivity
rng = np.random.RandomState(0)
n = 300
th = rng.uniform(0.2, 1.0, n)
Ra = th[:, None] + th[None, :]
Rm = np.outer(th, th)
np.fill_diagonal(Ra, 1.0)
np.fill_diagonal(Rm, 1.0)
iu = np.triu_indices(n, 1)
c = th - th.mean()
res = np.abs(center_connectivity(Ra, 'double')[iu]).max()
check('double centring is exact on an additive per-unit effect (max |residue| %.1e)' % res, res < 1e-12)
check('double centring: every row of the result sums to zero',
      np.allclose(center_connectivity(R, 'double').sum(axis=1), 0, atol=1e-9))
res = np.abs(center_connectivity(Rm, 'degree')[iu]).max() / Rm[iu].max()
check('degree centring removes a multiplicative effect to O(1/n) (max |residue| / max r = %.1e)' % res, res < 0.02)
r_res = np.corrcoef(center_connectivity(Rm, 'double')[iu], np.outer(c, c)[iu])[0, 1]
check('double centring on a multiplicative effect leaves (theta_i - mean)(theta_j - mean) (r = %.4f)' % r_res,
      r_res > 0.999)
r_res = np.corrcoef(center_connectivity(Ra, 'degree')[iu], -np.outer(c, c)[iu])[0, 1]
check('degree centring on an additive effect leaves -(a_i - mean)(a_j - mean) (r = %.4f)' % r_res, r_res > 0.99)
R_before = R.copy()
out_deg, out_dbl = center_connectivity(R, 'degree'), center_connectivity(R, 'double')
check('center_connectivity returns new arrays with a zero diagonal and leaves its input untouched',
      out_deg is not R and np.array_equal(R, R_before) and not np.any(np.diag(out_deg)) and not np.any(np.diag(out_dbl)))
check('guard: unknown centring mode rejected', raises(lambda: center_connectivity(R, 'triple')))

for kind, mode, other in (('multiplicative', 'degree', 'double'), ('additive', 'double', 'degree')):
    Rh, truth_h, theta = planted_hub(kind)
    kh = truth_h.max() + 1
    hub = deciles(theta)
    start = corrupt(truth_h, 0.2, kh)
    raw, _ = blockmodel_refine(Rh, start, kh)
    cen, _ = blockmodel_refine(center_connectivity(Rh, mode), start, kh)
    alt, _ = blockmodel_refine(center_connectivity(Rh, other), start, kh)
    a_raw, h_raw = adjusted_rand_score(truth_h, raw), adjusted_mutual_info_score(hub, raw)
    a_cen, h_cen = adjusted_rand_score(truth_h, cen), adjusted_mutual_info_score(hub, cen)
    a_alt = adjusted_rand_score(truth_h, alt)
    check('%s effect: refinement on raw |r| trades the planted partition for hubness (ARI %.2f, hub AMI %.2f)'
          % (kind, a_raw, h_raw), a_raw < 0.5 and h_raw > 0.3)
    check('%s effect: refinement on the %s-centred matrix keeps it (ARI %.3f, hub AMI %.3f)'
          % (kind, mode, a_cen, h_cen), a_cen > 0.99 and h_cen < 0.05)
    if kind == 'multiplicative':
        check('multiplicative effect: double centring leaks where degree does not (ARI %.3f against %.3f)'
              % (a_alt, a_cen), a_alt < a_cen)
    else:
        check('additive effect: degree centring keeps the partition too (ARI %.3f)' % a_alt, a_alt > 0.99)

# ------------------------------------------------- centred refinement in sample_parcellations
common = dict(n_networks=k, n_samples=1, binarize_connectivity=False, fisher_transform=True,
              standardize_profiles=True, clustering='ward', seed=0, verbose=False)
ward_labels = sample_parcellations(R, **common)['samples'][0].astype(int)
for mode in ('double', 'degree'):
    o = sample_parcellations(R, blockmodel_refine_labels=True, blockmodel_center=mode, **common)
    T = center_connectivity(R, mode)
    lab = o['samples'][0].astype(int)
    check('%s-centred refinement: its score is the block SSE of its labels on the centred target' % mode,
          abs(block_sse(T, lab, k) - o['scores'][0]) <= 1e-6 * max(1.0, abs(o['scores'][0])))
    check('%s-centred refinement: no worse than the Ward labels it starts from, on that target' % mode,
          o['scores'][0] <= block_sse(T, ward_labels, k) + 1e-9)
o_none = sample_parcellations(R, blockmodel_refine_labels=True, blockmodel_center='none', **common)
o_raw = sample_parcellations(R, blockmodel_refine_labels=True, **common)
check('blockmodel_center "none" is the uncentred refinement', np.array_equal(o_none['samples'], o_raw['samples']))
check('guard: blockmodel_center needs blockmodel_refine_labels',
      raises(lambda: sample_parcellations(R, blockmodel_center='double', **common)))
check('guard: unknown blockmodel_center rejected',
      raises(lambda: sample_parcellations(R, blockmodel_refine_labels=True, blockmodel_center='triple', **common)))

# ------------------------------------------------- run_parcellation provenance, restricted score
tmp = tempfile.mkdtemp()
conn_dir = os.path.join(tmp, CONNECTIVITY_NAME)
os.makedirs(conn_dir)
n = R.shape[0]
coords = np.stack([np.repeat(np.arange(3), n // 3), np.tile(np.arange(n // 3), 3)], 1).astype(np.int32)
for key in ('avg',) + HALF_NAMES:
    save_h5_data(dict(connectivity=R, coordinates=coords),
                 os.path.join(conn_dir, '%s_dom_%s.h5' % (CONNECTIVITY_NAME, key)), verbose=False)
arms = dict(
    sparse=dict(clustering='kmeans', n_samples=4, sparsify_profiles=True),
    wardlloyd=dict(clustering='ward_kmeans', n_samples=4),
    polish=dict(clustering='ward', blockmodel_refine_labels=True, blockmodel_center='degree'),
)
for name, over in arms.items():
    run_parcellation(output_dir=tmp, n_networks=k, binarize_connectivity=False, fisher_transform=True,
                     standardize_profiles=True, connectivity_pca_components=None, seed=0, variant=name,
                     verbose=False, **over)


def parc_file(name):
    return os.path.join(tmp, name, 'parcellation', 'parcellation_dom_halfA.h5')


a = read_attrs(parc_file('sparse'))
check('run: provenance records sparsify_profiles', a.get('sparsify_profiles') in (True, 'True', 1))
a = read_attrs(parc_file('polish'))
check('run: provenance records blockmodel_center', a.get('blockmodel_center') == 'degree')
a = read_attrs(parc_file('wardlloyd'))
dd = load_h5_data(parc_file('wardlloyd'), verbose=False)
check('run: ward_kmeans records one sample, and its restart-split halves are the parcellation',
      a.get('clustering') == 'ward_kmeans' and int(a.get('n_samples')) == 1
      and np.array_equal(dd['parcellation_split1'], dd['parcellation'])
      and np.array_equal(dd['parcellation_split2'], dd['parcellation']))
check('run: an arm without the new settings records their defaults',
      a.get('sparsify_profiles') in (False, 'False', 0) and a.get('blockmodel_center') == 'None')
cfg = dict(output_dir=tmp, seed=1, connectivity=dict(domains=['dom']),
           parcellation_variants={name: {} for name in arms})
rows, out_path = score_config(cfg, variants=['polish', 'wardlloyd'], cross_domain=False, verbose=False,
                              allow_partial=True)
scored = {r['variant'] for r in rows if r['metric'] == 'fidelity_within' and r['tree'] == 'real'
          and not r['variant'].startswith('(')}   # '(ceiling)' rows are the uncompressed reference
check('score: the new arms are scored, into scores_polish_wardlloyd.csv',
      scored == {'polish', 'wardlloyd'} and os.path.basename(out_path) == 'scores_polish_wardlloyd.csv')

# ---------------------------------------------------------------- variants_tag
check('variants_tag: short lists are joined, as before', variants_tag(['a', 'b']) == 'a_b')
yolo_partial = ['vmf_lloyd', 'nopca_lloyd', 'fisher_pca_lloyd', 'vmf_ward', 'vmf_ward100', 'vmf_lloyd100']
check('variants_tag: the longest list used before (the YOLO 1 partial score) keeps its name',
      variants_tag(yolo_partial) == '_'.join(yolo_partial))
mlp_cfg = yaml.safe_load(open('configs/final_mlp.yml'))
early = [v for v in mlp_cfg['parcellation_variants'] if v not in ('vmf_lloyd100', 'vmf_sparse_lloyd100')]
tag = variants_tag(early)
check('variants_tag: the 13-arm early score (%d characters joined) is named %s' % (len('_'.join(early)), tag),
      len(early) == 13 and len('_'.join(early)) == 215 and re.fullmatch(r'13arms_[0-9a-f]{8}', tag) is not None)
check('variants_tag: independent of order, distinct for a different subset',
      variants_tag(early[::-1]) == tag and variants_tag(early[:-1] + ['vmf_lloyd100']) != tag)
job = get_job('configs/final_mlp.yml', dict(DEFAULTS), steps=['score'], variants=early)
check('make_jobs: the early score job is named final_mlp.score.%s and still passes every arm' % tag,
      '#SBATCH --job-name=final_mlp.score.%s\n' % tag in job and ('-V ' + ' '.join(early)) in job)
jobs_dir = tempfile.mkdtemp()
r = subprocess.run([sys.executable, '-m', 'parcelmate.bin.make_jobs', 'configs/final_mlp.yml', '-s', 'score',
                    '-V'] + early + ['-o', jobs_dir], capture_output=True, text=True,
                   env=dict(os.environ, PYTHONPATH='.'))
check('make_jobs CLI: writes final_mlp.score.%s.pbs' % tag,
      r.returncode == 0 and os.listdir(jobs_dir) == ['final_mlp.score.%s.pbs' % tag])
cfg13 = dict(output_dir=tmp, seed=1, connectivity=dict(domains=['dom']),
             parcellation_variants={v: {} for v in early + ['vmf_lloyd100']})
_, out_path = score_config(cfg13, variants=early, cross_domain=False, verbose=False, allow_partial=True)
check('score: a long -V list writes scores_%s.csv' % tag, os.path.basename(out_path) == 'scores_%s.csv' % tag)

# ---------------------------------------------------------------- the run configs
sig = inspect.signature(run_parcellation)


def resolved(path):
    cfg = yaml.safe_load(open(path))
    out = {}
    for name, over in cfg['parcellation_variants'].items():
        full = dict(cfg['parcellation'])
        full.update(over or {})
        out[name] = full
    return cfg, out


def binds(arms):
    try:
        for name, full in arms.items():
            sig.bind(variant=name, **full)
    except TypeError as e:
        print('   ', e)
        return False
    return True


def diff(x, y):
    return {key for key in set(x) | set(y) if x.get(key) != y.get(key)}


mlp_cfg, mlp = resolved('configs/final_mlp.yml')
res_cfg, res = resolved('configs/final_resid.yml')
yolo_cfg, yolo = resolved('configs/yolo.yml')
yolo_mlp_cfg, yolo_mlp = resolved('configs/yolo_mlp.yml')
check('final_mlp: 15 arms, all bind to run_parcellation', len(mlp) == 15 and binds(mlp))
check('final_resid: 2 arms, both bind to run_parcellation', len(res) == 2 and binds(res))
check('final_mlp: connectivity section identical to configs/yolo_mlp.yml, the tree it is linked from',
      mlp_cfg['connectivity'] == yolo_mlp_cfg['connectivity'])
check('final_resid: connectivity section identical to configs/yolo.yml, the tree it is linked from',
      res_cfg['connectivity'] == yolo_cfg['connectivity'])
check('both: their own trees, and a null model, so every arm gets its matched null partition',
      mlp_cfg['output_dir'] == 'results/final_mlp' and res_cfg['output_dir'] == 'results/final_resid'
      and mlp_cfg['connectivity'].get('null_model') and res_cfg['connectivity'].get('null_model'))
check('final_mlp: MLP units, 832 per layer', mlp_cfg['connectivity'].get('unit_type') == 'mlp'
      and mlp_cfg['connectivity'].get('units_per_layer') == 832)
intended = {
    'vmf_ward': ('vmf_ward100', {'n_networks'}),
    'vmf_ward150': ('vmf_ward100', {'n_networks'}),
    'vmf_ward200': ('vmf_ward100', {'n_networks'}),
    'vmf_lloyd100': ('vmf_ward100', {'clustering'}),
    'vmf_pca20_ward100': ('vmf_ward100', {'connectivity_pca_components', 'pca_whiten'}),
    'vmf_pca100_ward100': ('vmf_ward100', {'connectivity_pca_components', 'pca_whiten'}),
    'vmf_pca20_lloyd100': ('vmf_lloyd100', {'connectivity_pca_components', 'pca_whiten'}),
    'vmf_pca100_lloyd100': ('vmf_lloyd100', {'connectivity_pca_components', 'pca_whiten'}),
    'vmf_sparse_ward100': ('vmf_ward100', {'sparsify_profiles'}),
    'vmf_sparse_lloyd100': ('vmf_lloyd100', {'sparsify_profiles'}),
    'vmf_wardlloyd100': ('vmf_ward100', {'clustering'}),
    'vmf_ward100_bm': ('vmf_ward100', {'blockmodel_refine_labels'}),
    'vmf_ward100_bm_double': ('vmf_ward100_bm', {'blockmodel_center'}),
    'vmf_ward100_bm_degree': ('vmf_ward100_bm', {'blockmodel_center'}),
}
bad = [(arm, ref, diff(mlp[arm], mlp[ref])) for arm, (ref, keys) in intended.items()
       if diff(mlp[arm], mlp[ref]) != keys]
if bad:
    print('   ', bad)
check('final_mlp: every arm differs from its reference arm in exactly the intended settings',
      not bad and set(intended) | {'vmf_ward100'} == set(mlp))
check('final_mlp: values as designed (k, PCA d unwhitened, centring modes, clustering)',
      [mlp[a]['n_networks'] for a in ('vmf_ward', 'vmf_ward100', 'vmf_ward150', 'vmf_ward200')] == [50, 100, 150, 200]
      and all(mlp[a]['n_networks'] == 100 for a in intended if a not in ('vmf_ward', 'vmf_ward150', 'vmf_ward200'))
      and [mlp[a]['connectivity_pca_components'] for a in ('vmf_pca20_ward100', 'vmf_pca100_lloyd100')] == [20, 100]
      and all(mlp[a]['pca_whiten'] is False for a in mlp if mlp[a].get('connectivity_pca_components'))
      and mlp['vmf_ward100_bm_double']['blockmodel_center'] == 'double'
      and mlp['vmf_ward100_bm_degree']['blockmodel_center'] == 'degree'
      and mlp['vmf_wardlloyd100']['clustering'] == 'ward_kmeans'
      and all(mlp[a]['clustering'] == 'kmeans' for a in mlp if 'lloyd100' in a and a != 'vmf_wardlloyd100')
      and all(mlp[a]['standardize_profiles'] and mlp[a]['fisher_transform'] and not mlp[a]['binarize_connectivity']
              for a in mlp))
check('T1: MLP vmf_ward, vmf_ward100, vmf_lloyd100 repeat the residual-stream settings of configs/yolo.yml',
      all(mlp[a] == yolo[a] for a in ('vmf_ward', 'vmf_ward100', 'vmf_lloyd100')))
check('T1: residual vmf_ward150/200 repeat configs/yolo.yml vmf_ward100 except for k (150, 200)',
      diff(res['vmf_ward150'], yolo['vmf_ward100']) == {'n_networks'}
      and diff(res['vmf_ward200'], yolo['vmf_ward100']) == {'n_networks'}
      and (res['vmf_ward150']['n_networks'], res['vmf_ward200']['n_networks']) == (150, 200))
check('T1: the scored MLP Lloyd arm of configs/yolo_mlp.yml is the k = 50 twin of vmf_lloyd100',
      diff(yolo_mlp['vmf_lloyd'], mlp['vmf_lloyd100']) == {'n_networks'})

# ---------------------------------------------------------------- the launcher
text = open('scripts/launch_yolo.sh').read()
lists = {name: value.split() for name, value in re.findall(r'^(FINAL_[A-Z_]+)="([^"]*)"', text, re.M)}
mlp_launch = lists['FINAL_MLP_WARD'] + lists['FINAL_MLP_WARD_PLUS'] + lists['FINAL_MLP_PCA'] + lists['FINAL_MLP_LLOYD']
check('launcher: the four MLP lists name every final_mlp arm exactly once', sorted(mlp_launch) == sorted(mlp))
check('launcher: the residual list names both final_resid arms', sorted(lists['FINAL_RESID']) == sorted(res))
check('launcher: the early score covers exactly the 13 arms that do not wait on the Lloyd arms',
      sorted(lists['FINAL_MLP_WARD'] + lists['FINAL_MLP_WARD_PLUS'] + lists['FINAL_MLP_PCA']) == sorted(early))
r = subprocess.run(['bash', '-n', 'scripts/launch_yolo.sh'], capture_output=True, text=True)
check('launcher: parses', r.returncode == 0)
r = subprocess.run(['bash', 'scripts/launch_yolo.sh', 'nope'], capture_output=True, text=True)
check('launcher: usage lists the final modes', r.returncode == 2 and 'generate_final' in r.stderr
      and 'submit_final' in r.stderr)

print('\n%d check(s), %d failure(s)' % (n_checks[0], len(failures)))
if failures:
    for f in failures:
        print('  - %s' % f)
    sys.exit(1)
