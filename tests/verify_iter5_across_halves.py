"""Verification for the data-matched across-domain metrics (see LOG.md, Iteration 9).

    PYTHONPATH=. python tests/verify_iter5_across_halves.py

`*_across` compares two sample-averaged connectomes (4 samples a side); the within-domain
metrics compare two halves (2 samples a side). Quoting a within number against an across
number therefore compared measurements taken at different amounts of data. `*_across_halves`
removes that: fit on half A of one domain, evaluate on half B of another, which is the same
token budget and the same disjoint-set structure as within-domain.

These checks exist because the failure mode is silent. An across-halves metric that quietly
read the `avg` files, or that paired half A with half A, would still produce a full column of
plausible numbers. So each one is recomputed here by hand from the files it should have read.
"""

import os
import shutil
import tempfile

import numpy as np

from parcelmate.bin.score import load_connectivity, parc_path, conn_path, score_tree
from parcelmate.constants import CONNECTIVITY_NAME, HALF_NAMES
from parcelmate.data import circshift_timecourses
from parcelmate.metrics import fidelity, reliability
from parcelmate.model import fisher_average, run_parcellation, run_split_halves
from parcelmate.util import load_h5_data, save_h5_data

failures = []


def check(name, cond):
    print('%s %s' % ('OK' if cond else 'FAIL', name))
    if not cond:
        failures.append(name)


def planted(n_units=120, n_blocks=4, n_obs=1500, noise=1.2, seed=0):
    rng = np.random.RandomState(seed)
    block = np.repeat(np.arange(n_blocks), n_units // n_blocks)
    latent = rng.randn(n_blocks, n_obs)
    Z = latent[block] + noise * rng.randn(n_units, n_obs)
    return Z, np.abs(np.corrcoef(Z)), block


tmp = tempfile.mkdtemp(prefix='parcelmate_iter5_')
try:
    domains = ('alpha', 'beta', 'gamma')
    arms = {'binar': dict(binarize_connectivity=True, connectivity_pca_components=10),
            'fisher': dict(binarize_connectivity=False, fisher_transform=True,
                           connectivity_pca_components=None)}
    root = os.path.join(tmp, 'run')
    os.makedirs(os.path.join(root, CONNECTIVITY_NAME))
    for d_ix, domain in enumerate(domains):
        mats = []
        for i in range(1, 5):
            Z, R_i, blk = planted(seed=100 * d_ix + i)
            mats.append(R_i)
            save_h5_data(
                dict(connectivity=R_i,
                     coordinates=np.stack([np.repeat(np.arange(10), len(blk) // 10),
                                           np.arange(len(blk))], 1).astype(np.int32),
                     unit_means=np.zeros(len(blk)), unit_stds=np.ones(len(blk)),
                     n_obs=np.asarray(1000)),
                os.path.join(root, CONNECTIVITY_NAME,
                             'connectivity_%s_sample%d.h5' % (domain, i)), verbose=False)
        # A real `avg` file is the Fisher average of all four samples, which is what makes
        # it less noisy than a half. Writing a single sample here instead (as an earlier
        # fixture did) inverts that and makes the noise checks below meaningless.
        save_h5_data(
            dict(connectivity=fisher_average(*mats),
                 coordinates=np.stack([np.repeat(np.arange(10), len(blk) // 10),
                                       np.arange(len(blk))], 1).astype(np.int32),
                 unit_means=np.zeros(len(blk)), unit_stds=np.ones(len(blk)),
                 n_obs=np.asarray(4000)),
            os.path.join(root, CONNECTIVITY_NAME, 'connectivity_%s_avg.h5' % domain),
            verbose=False)
    run_split_halves(output_dir=root, verbose=False)
    for name, kw in arms.items():
        run_parcellation(output_dir=root, variant=name, n_networks=4, n_samples=4,
                         seed=7, verbose=False, **kw)

    rows, missing = [], []
    score_tree(root, 'real', sorted(arms), list(domains), rows, missing, verbose=False)
    check('no inputs reported missing', not missing)

    seen = {r['metric'] for r in rows}
    for m in ('reliability_across', 'fidelity_across',
              'reliability_across_halves', 'fidelity_across_halves'):
        check('%s is computed' % m, m in seen)

    def get(metric, variant, fit, ev):
        v = [r['value'] for r in rows if r['metric'] == metric and r['variant'] == variant
             and r['fit'] == fit and r['eval'] == ev]
        return v[0] if v else None

    # The two across variants must cover exactly the same ordered domain pairs.
    pairs = {m: {(r['fit'], r['eval']) for r in rows if r['metric'] == m}
             for m in ('fidelity_across', 'fidelity_across_halves')}
    check('across_halves covers the same %d ordered pairs as across'
          % len(pairs['fidelity_across']),
          pairs['fidelity_across'] == pairs['fidelity_across_halves']
          and len(pairs['fidelity_across']) == len(domains) * (len(domains) - 1))
    check('across pairs never compare a domain with itself',
          all(f != e for f, e in pairs['fidelity_across_halves']))

    # The heart of it: recompute from the half files the metric claims to read.
    a_key, b_key = HALF_NAMES
    ok_fid, ok_rel, ok_differs = True, True, False
    for variant in sorted(arms):
        for f_dom in domains:
            R_fit = load_connectivity(conn_path(root, f_dom, a_key))
            P_fit = load_h5_data(parc_path(root, variant, f_dom, a_key),
                                 verbose=False)['parcellation']
            for e_dom in domains:
                if e_dom == f_dom:
                    continue
                R_ev = load_connectivity(conn_path(root, e_dom, b_key))
                P_ev = load_h5_data(parc_path(root, variant, e_dom, b_key),
                                    verbose=False)['parcellation']
                want_f = fidelity(R_fit, R_ev, P_fit, measure='r')
                want_r = reliability(P_fit, P_ev)
                got_f = get('fidelity_across_halves', variant, f_dom, e_dom)
                got_r = get('reliability_across_halves', variant, f_dom, e_dom)
                ok_fid &= got_f is not None and np.isclose(got_f, want_f, atol=1e-12)
                ok_rel &= got_r is not None and np.isclose(got_r, want_r, atol=1e-12)
                # It must not silently be the avg measurement wearing a new name.
                got_avg = get('fidelity_across', variant, f_dom, e_dom)
                ok_differs |= got_avg is not None and not np.isclose(got_f, got_avg)
    check('fidelity_across_halves == fidelity(R[fit,A], R[eval,B], P[fit,A], r)', ok_fid)
    check('reliability_across_halves == ARI(P[fit,A], P[eval,B])', ok_rel)
    check('across_halves is a different measurement from across, not an alias', ok_differs)

    # Pairing half A with half A would make reliability_across_halves symmetric in the pair,
    # exactly as the avg version is. Asymmetry is the observable signature of A-vs-B.
    asym = [(get('reliability_across_halves', v, f, e),
             get('reliability_across_halves', v, e, f))
            for v in sorted(arms) for f in domains for e in domains if f < e]
    check('reliability_across_halves is asymmetric (fit half A vs eval half B)',
          any(not np.isclose(x, y) for x, y in asym if x is not None and y is not None))
    sym = [(get('reliability_across', v, f, e), get('reliability_across', v, e, f))
           for v in sorted(arms) for f in domains for e in domains if f < e]
    check('reliability_across (avg) stays symmetric, as before',
          all(np.isclose(x, y) for x, y in sym if x is not None and y is not None))

    # Halves hold half the samples, so their connectomes are noisier than the averages.
    # If the halves metric were reading avg files this would not hold.
    ceil_h = np.mean([r['value'] for r in rows if r['metric'] == 'fidelity_across_halves'
                      and r['variant'] == '(ceiling)'])
    ceil_a = np.mean([r['value'] for r in rows if r['metric'] == 'fidelity_across'
                      and r['variant'] == '(ceiling)'])
    check('the halves ceiling is below the avg ceiling (%.3f < %.3f), as less data implies'
          % (ceil_h, ceil_a), ceil_h < ceil_a)

    # cross_domain=False must suppress both, not just the original.
    rows2, missing2 = [], []
    score_tree(root, 'real', sorted(arms), list(domains), rows2, missing2,
               cross_domain=False, verbose=False)
    check('--no-cross-domain suppresses both across variants',
          not any('across' in r['metric'] for r in rows2))

    # A missing half must be reported against the half, not silently dropped.
    os.rename(conn_path(root, domains[0], a_key), conn_path(root, domains[0], a_key) + '.bak')
    rows3, missing3 = [], []
    score_tree(root, 'real', sorted(arms), list(domains), rows3, missing3, verbose=False)
    check('a missing half file is reported as missing, naming the half',
          any(a_key in m and domains[0] in m for m in missing3))
    os.rename(conn_path(root, domains[0], a_key) + '.bak', conn_path(root, domains[0], a_key))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print('\n%s' % ('All checks passed.' if not failures
                else 'FAILURES:\n  ' + '\n  '.join(failures)))
raise SystemExit(1 if failures else 0)
