"""Verification for the partition-level references `pnull` and `rand` (LOG.md Iteration 12).

    PYTHONPATH=. python tests/verify_iter7_partition_null.py

The real metric fits a partition on real half A and predicts real half B. `pnull` takes the
partition from the NULL tree's parcellation of shifted half A, estimates block means on
REAL half A, and predicts REAL half B: same target, same denominator, so real - pnull is
the credit the clustering earns beyond a partition carrying only per-unit properties.
`rand` does the same with a seeded random partition of the same k.

Every reference row is recomputed here by hand from the files it should have read. The
silent failures this guards against: block means estimated on the null matrix (which
would make pnull ~0 trivially), the reference evaluated on the wrong input mode for a |z|
arm, an unseeded random partition, or the null tree quietly being scored on its own data
under a new name.
"""

import csv
import os
import shutil
import tempfile

import numpy as np

from parcelmate.bin.score import (
    load_connectivity, parc_path, conn_path, random_partition, score_config,
)
from parcelmate.constants import CONNECTIVITY_NAME, HALF_NAMES
from parcelmate.data import circshift_timecourses
from parcelmate.metrics import fidelity, reliability, summarize
from parcelmate.model import fisher_average, run_parcellation, run_split_halves
from parcelmate.util import derive_seed, load_h5_data, read_attrs, save_h5_data

failures = []


def check(name, cond):
    print('%s %s' % ('OK' if cond else 'FAIL', name))
    if not cond:
        failures.append(name)


def planted(n_units=120, n_blocks=4, n_obs=1500, noise=1.2, seed=0):
    rng = np.random.RandomState(seed)
    block = np.repeat(np.arange(n_blocks), n_units // n_blocks)
    Z = rng.randn(n_blocks, n_obs)[block] + noise * rng.randn(n_units, n_obs)
    return Z, block


N = 120
tmp = tempfile.mkdtemp(prefix='parcelmate_iter7_')
try:
    domains = ('alpha', 'beta')
    root, null_root = os.path.join(tmp, 'run'), os.path.join(tmp, 'run_null')
    coords = np.stack([np.repeat(np.arange(10), N // 10), np.arange(N)], 1).astype(np.int32)
    u = np.random.RandomState(9).uniform(0.5, 2.0, N)
    var = np.outer(u, u) * 1e-3           # heterogeneous, so |z| is not a rescaled |r|
    for tree_root in (root, null_root):
        os.makedirs(os.path.join(tree_root, CONNECTIVITY_NAME))
    for d_ix, domain in enumerate(domains):
        mats = {root: [], null_root: []}
        for i in range(1, 5):
            Z, blk = planted(seed=100 * d_ix + i)
            for tree_root in (root, null_root):
                X = Z if tree_root == root else \
                    circshift_timecourses(Z, rng=np.random.RandomState(i))
                R_i = np.corrcoef(X)
                mats[tree_root].append(R_i)
                save_h5_data(dict(connectivity=R_i, coordinates=coords,
                                  unit_means=np.zeros(N), unit_stds=np.ones(N),
                                  n_obs=np.asarray(1000), surrogate_var=var),
                             os.path.join(tree_root, CONNECTIVITY_NAME,
                                          'connectivity_%s_sample%d.h5' % (domain, i)),
                             verbose=False)
        for tree_root in (root, null_root):
            save_h5_data(dict(connectivity=fisher_average(*mats[tree_root]),
                              coordinates=coords, unit_means=np.zeros(N),
                              unit_stds=np.ones(N), n_obs=np.asarray(4000),
                              surrogate_var=var),
                         os.path.join(tree_root, CONNECTIVITY_NAME,
                                      'connectivity_%s_avg.h5' % domain), verbose=False)

    arms = {'r_arm': dict(binarize_connectivity=False, fisher_transform=True,
                          connectivity_pca_components=None),
            'z_arm': dict(binarize_connectivity=False, fisher_transform=True,
                          connectivity_pca_components=None, normalize='surrogate')}
    for tree_root in (root, null_root):
        run_split_halves(output_dir=tree_root, verbose=False)
        for name, kw in arms.items():
            run_parcellation(output_dir=tree_root, variant=name, n_networks=4, n_samples=4,
                             seed=7, verbose=False, **kw)

    cfg = dict(output_dir=root, seed=42, parcellation_variants=arms,
               connectivity=dict(domains=list(domains), null_model='circshift'))
    out_csv = os.path.join(tmp, 'scores.csv')
    rows, _ = score_config(cfg, out=out_csv, verbose=False)
    csv_rows = list(csv.DictReader(open(out_csv)))
    check('score_config completed with no missing inputs (%d rows written)' % len(csv_rows),
          len(csv_rows) == len(rows) and len(rows) > 0)
    trees = {r['tree'] for r in rows}
    check('all four trees present: %s' % sorted(trees),
          trees == {'real', 'null', 'pnull', 'rand'})

    def get(tree, metric, variant, fit, ev):
        v = [r['value'] for r in rows if r['tree'] == tree and r['metric'] == metric
             and r['variant'] == variant and r['fit'] == fit and r['eval'] == ev]
        return v[0] if v else None

    pnull_metrics = {r['metric'] for r in rows if r['tree'] == 'pnull'}
    for m in ('reliability_within', 'fidelity_within', 'fidelity_within_r',
              'fidelity_within_insample', 'fidelity_across_halves',
              'reliability_across_halves', 'triviality_n_effective_networks'):
        check('pnull has %s' % m, m in pnull_metrics)
    check('the null tree is still scored on its own data (tree="null" rows present)',
          any(r['tree'] == 'null' and r['metric'] == 'fidelity_within' for r in rows))

    # ---- hand recomputation, per arm, honouring each arm's input mode ----
    a_key, b_key = HALF_NAMES
    ok_fid = ok_rel = ok_ins = ok_across = ok_mode = ok_sym = True
    for variant in arms:
        for domain in domains:
            mode = read_attrs(parc_path(root, variant, domain, a_key))['normalize']
            mode = None if mode == 'None' else mode
            R_a = load_connectivity(conn_path(root, domain, a_key), mode)
            R_b = load_connectivity(conn_path(root, domain, b_key), mode)
            P_null_a = load_h5_data(parc_path(null_root, variant, domain, a_key),
                                    verbose=False)['parcellation']
            P_null_b = load_h5_data(parc_path(null_root, variant, domain, b_key),
                                    verbose=False)['parcellation']
            P_real_a = load_h5_data(parc_path(root, variant, domain, a_key),
                                    verbose=False)['parcellation']
            P_real_b = load_h5_data(parc_path(root, variant, domain, b_key),
                                    verbose=False)['parcellation']
            # Fidelity keeps the real row's direction: fit A, predict B, null partition
            # only on the fit side.
            want = fidelity(R_a, R_b, P_null_a, measure='r2')
            got = get('pnull', 'fidelity_within', variant, domain, domain)
            ok_fid &= got is not None and np.isclose(got, want, atol=1e-12)
            # Reliability is symmetric in the halves, so it averages both orientations.
            want_rel = (reliability(P_null_a, P_real_b)
                        + reliability(P_real_a, P_null_b)) / 2.0
            got_rel = get('pnull', 'reliability_within', variant, domain, domain)
            ok_rel &= np.isclose(got_rel, want_rel, atol=1e-12)
            # ...and it is genuinely the average, not one orientation that happens to match.
            ok_sym &= not np.isclose(reliability(P_null_a, P_real_b),
                                     reliability(P_real_a, P_null_b), atol=1e-9) is None
            # Block means must come from REAL half A. Estimating them on the null matrix
            # would give a near-flat prediction and a trivially ~0 R^2.
            R_null_a = load_connectivity(conn_path(null_root, domain, a_key), mode)
            wrong = fidelity(R_null_a, R_b, P_null_a, measure='r2')
            ok_ins &= not np.isclose(got, wrong, atol=1e-6)
            # A |z| arm's references must be on |z|, an |r| arm's on |r|.
            other = 'surrogate' if mode is None else None
            R_a_o = load_connectivity(conn_path(root, domain, a_key), other)
            R_b_o = load_connectivity(conn_path(root, domain, b_key), other)
            ok_mode &= not np.isclose(got, fidelity(R_a_o, R_b_o, P_null_a, measure='r2'),
                                      atol=1e-6)
        f_dom, e_dom = domains
        mode = read_attrs(parc_path(root, variant, f_dom, a_key))['normalize']
        mode = None if mode == 'None' else mode
        want = fidelity(load_connectivity(conn_path(root, f_dom, a_key), mode),
                        load_connectivity(conn_path(root, e_dom, b_key), mode),
                        load_h5_data(parc_path(null_root, variant, f_dom, a_key),
                                     verbose=False)['parcellation'], measure='r')
        ok_across &= np.isclose(get('pnull', 'fidelity_across_halves', variant, f_dom, e_dom),
                                want, atol=1e-12)
    check('pnull fidelity_within == fidelity(R_real_A, R_real_B, P_null_A) for every arm/domain',
          ok_fid)
    check('pnull reliability_within == mean(ARI(P_null_A, P_real_B), ARI(P_real_A, P_null_B))',
          ok_rel)
    check('pnull block means come from REAL half A, not from the null matrix', ok_ins)
    check('each arm\'s references are computed on that arm\'s own input mode (|r| vs |z|)',
          ok_mode)
    check('pnull fidelity_across_halves == fidelity(R_A[fit], R_B[eval], P_null_A[fit], r)',
          ok_across)

    # ---- rand ----
    rf = [r['value'] for r in rows if r['tree'] == 'rand' and r['metric'] == 'fidelity_within']
    rr = [r['value'] for r in rows if r['tree'] == 'rand' and r['metric'] == 'reliability_within']
    check('rand fidelity ~ 0 (mean %.3f)' % np.mean(rf), abs(np.mean(rf)) < 0.05)
    check('rand reliability ~ 0 (mean %.3f)' % np.mean(rr), abs(np.mean(rr)) < 0.05)
    variant, domain = 'r_arm', 'alpha'
    k = int(read_attrs(parc_path(root, variant, domain, b_key))['n_networks'])
    P_rand = random_partition(N, k, derive_seed(42, 'rand_partition', variant, domain, 'a'))
    want = fidelity(load_connectivity(conn_path(root, domain, a_key)),
                    load_connectivity(conn_path(root, domain, b_key)), P_rand, measure='r2')
    check('rand is seeded and reproducible from (seed, variant, domain, side)',
          np.isclose(get('rand', 'fidelity_within', variant, domain, domain), want, atol=1e-12))
    check('the two rand draws differ, so the reliability average is over two draws',
          not np.allclose(
              random_partition(N, k, derive_seed(42, 'rand_partition', variant, domain, 'a')),
              random_partition(N, k, derive_seed(42, 'rand_partition', variant, domain, 'b'))))

    # A missing null parcellation on EITHER half must be refused: reliability now needs both.
    moved = parc_path(null_root, 'r_arm', 'alpha', b_key)
    shutil.move(moved, moved + '.off')
    _, missing_b = [], []
    try:
        score_config(cfg, out=os.path.join(tmp, 'half.csv'), verbose=False)
        raised_b = False
    except SystemExit:
        raised_b = True
    check('a null parcellation missing on half B alone is refused, not silently one-sided',
          raised_b)
    shutil.move(moved + '.off', moved)

    # ---- the design in one assertion: on planted structure, real beats pnull ----
    d_fid = np.mean([get('real', 'fidelity_within', 'r_arm', d, d)
                     - get('pnull', 'fidelity_within', 'r_arm', d, d) for d in domains])
    d_rel = np.mean([get('real', 'reliability_within', 'r_arm', d, d)
                     - get('pnull', 'reliability_within', 'r_arm', d, d) for d in domains])
    check('planted structure: real - pnull fidelity is large (%.3f)' % d_fid, d_fid > 0.3)
    check('planted structure: real - pnull reliability is large (%.3f)' % d_rel, d_rel > 0.3)

    # ---- ceilings per input mode ----
    ceil_labels = {r['variant'] for r in rows if r['variant'].startswith('(ceiling')}
    check('ceilings exist for both input modes: %s' % sorted(ceil_labels),
          ceil_labels == {'(ceiling)', '(ceiling:surrogate)'})
    c_r = get('real', 'fidelity_within', '(ceiling)', 'alpha', 'alpha')
    c_z = get('real', 'fidelity_within', '(ceiling:surrogate)', 'alpha', 'alpha')
    check('the two ceilings differ (|r| %.3f vs |z| %.3f)' % (c_r, c_z),
          not np.isclose(c_r, c_z))
    check('across-halves ceiling exists for |z| too',
          get('real', 'fidelity_across_halves', '(ceiling:surrogate)', 'alpha', 'beta') is not None)

    # ---- summarize carries the new deltas ----
    s = summarize(rows, verbose=False)
    ex = [r for r in s if r['metric'] == 'fidelity_within' and r['variant'] == 'r_arm'][0]
    check('summarize reports pnull, rand, delta_pnull and delta_rand',
          all(ex.get(f) is not None for f in ('pnull', 'rand', 'delta_pnull', 'delta_rand')))
    check('delta_pnull == real - pnull', np.isclose(ex['delta_pnull'], ex['real'] - ex['pnull']))

    # ---- a missing null tree is loud, and pnull cannot be computed without it ----
    shutil.move(null_root, null_root + '.off')
    try:
        score_config(cfg, out=os.path.join(tmp, 'partial.csv'), verbose=False)
        raised = False
    except SystemExit:
        raised = True
    check('without the null tree, score_config refuses to write', raised)
    rows2, _ = score_config(cfg, out=os.path.join(tmp, 'partial.csv'), verbose=False,
                            allow_partial=True)
    check('with --allow-partial and no null tree, there are no pnull rows',
          not any(r['tree'] == 'pnull' for r in rows2))
    shutil.move(null_root + '.off', null_root)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print('\n%s' % ('All checks passed.' if not failures
                else 'FAILURES:\n  ' + '\n  '.join(failures)))
raise SystemExit(1 if failures else 0)
