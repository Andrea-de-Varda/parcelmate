"""Intrinsic timescales of MLP units (LOG.md Iteration 47, info/TIMESCALES_PLAN.md).

    python -m parcelmate.bin.timescales acf --model Qwen/Qwen3.5-2B --out results/timescales/qwen3.5-2b
    python -m parcelmate.bin.timescales analyze --out results/timescales/qwen3.5-2b \
        --networks results/qwen35/qwen3.5-2b-resid-pool4 \
        --patching results/patching_bank/Qwen_Qwen3-5-2B results/patching/Qwen_Qwen3-5-2B \
        --richclub results/richclub/qwen3.5-2b-resid-pool4-tiles/metrics/richclub

`acf` (GPU). Per dataset (wikitext, bookcorpus), 3 x --n-windows windows of --seq-len tokens
(`timescales.load_windows`): the first third is half A, the second half B, the last the
reference for the token-and-position fit. Per condition (intact; shuffled = whole words permuted
within each window, `data.shuffle_words_within`), one fit on the reference windows of both
datasets, then the ACF of every unit's residual over lags 0..--max-lag for each dataset and half.
Writes <out>/acf.h5: `<condition>/<dataset>/<half>/{acf, mean, var, positive_share, n_pairs}`, the
windows' token ids and sources, unit coordinates and the model's layer types. A finished
(condition, dataset, half) is skipped on a rerun. `--centring window` (third revision,
2026-10-29) computes the within-window ACF on the same windows (copied from acf.h5): each window
centred and scaled on its own, Fisher mean over windows; it writes <out>/acf_within.h5 with also
`var_within`, `var_between` and the counts `n_windows`, `n_undefined`, `n_clipped` per unit. Run
`analyze --acf acf_within.h5 --tag analysis_within` on it.

`analyze` (CPU). Per condition and half, the ACF is the mean over the two datasets, read at lags
1, 10 and 100 (`load_measures`): how similar the unit's residual is to itself the next token, about a
sentence later and about a paragraph later. Each measure is the mean of the halves. Lags 10 and 100
are tested against networks and circuits; lag 1 is descriptive. No curve is fitted: the first
definition (tau_int) was 80-99% plateau, and the exponential fit that replaced it decided arbitrarily
whether the slow drift of the shuffled top layers was decay or floor (LOG.md Iteration 47).
Writes to <out>/<--tag>/ (default analysis_lags; tau_int in analysis/, the fit in analysis_fit/):
  units.npz           the measures per unit, per half, the references (tau_int, tail level), covariates
  reliability.csv     A: split-half Spearman, relations between lags, to tau_int, to the tail level
                      and between conditions
  layers.csv          B: quantiles and mean of every measure by layer
  networks.csv        C: spread of layer-adjusted network values against within-layer permutations,
                      also as a share of the mean within-layer variance; every partition and measure
  network_values.csv  C: layer-adjusted value of every network
  strength.csv        C: Spearman with node strength (rich-club units), real and null
  circuits.csv        D: per task, mean and variance over the circuit against layer-matched random
                      sets, with the task's both-correct accuracy
  summary.json        aggregates (median z, sign tests, accuracy correlations)
"""

import argparse
import csv
import json
import os

import h5py
import numpy as np

from parcelmate.util import derive_seed, stderr

os.umask(0o002)

DATASETS = ('wikitext', 'bookcorpus')
CONDITIONS = ('intact', 'shuffled')
HALVES = ('halfA', 'halfB')
ACF_FILES = {'global': 'acf.h5', 'window': 'acf_within.h5'}   # --centring -> file under --out


def run_acf(args):
    import torch
    from parcelmate.data import shuffle_words_within, word_start_table
    from parcelmate.model import get_model_and_tokenizer
    from parcelmate.residual import fit_token_position_large
    from parcelmate.timescales import load_windows, unit_acf

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, ACF_FILES[args.centring])
    n = args.n_windows
    if args.centring == 'window' and not os.path.exists(path):
        # The same windows as the global ACF: copied from acf.h5, which must exist.
        src = os.path.join(args.out, ACF_FILES['global'])
        with h5py.File(src, 'r') as s, h5py.File(path, 'w') as f:
            for k in ('n_windows', 'seq_len', 'seed'):
                assert s.attrs[k] == getattr(args, k), '%s: %s differs from %s' % (src, k, getattr(args, k))
            s.copy(s['windows'], f, name='windows')
            if 'layer_types' in s:
                s.copy(s['layer_types'], f, name='layer_types')
    model, tok = get_model_and_tokenizer(args.model)
    windows = {}
    with h5py.File(path, 'a') as f:
        f.attrs.update(model=args.model, n_windows=n, seq_len=args.seq_len, max_lag=args.max_lag,
                       seed=args.seed, min_count=args.min_count, centring=args.centring)
        layer_types = getattr(model.config, 'layer_types', None)
        if layer_types is None and hasattr(model.config, 'text_config'):
            layer_types = getattr(model.config.text_config, 'layer_types', None)
        if layer_types is not None and 'layer_types' not in f:
            f['layer_types'] = np.array(layer_types, dtype='S')
        for d in DATASETS:
            key = 'windows/%s' % d
            if key in f:
                windows[d] = np.asarray(f[key])
                continue
            stderr('windows: %s\n' % d)
            w, src = load_windows(d, tok, 3 * n, args.seq_len, args.seed)
            f[key] = w.astype(np.int32)
            f['windows/%s_source' % d] = src
            windows[d] = w
    sets = {d: dict(halfA=windows[d][:n], halfB=windows[d][n:2 * n], ref=windows[d][2 * n:]) for d in DATASETS}
    starts = word_start_table(tok)

    for cond in CONDITIONS:
        with h5py.File(path, 'r') as f:
            todo = [(d, h) for d in DATASETS for h in HALVES if '%s/%s/%s/acf' % (cond, d, h) not in f]
        if not todo:
            stderr('%s: done\n' % cond)
            continue
        ids = {}
        for d in DATASETS:
            for s, w in sets[d].items():
                if cond == 'shuffled':
                    w = np.asarray(shuffle_words_within(w.tolist(), starts,
                                                        derive_seed(args.seed, 'timescales_shuffle', d, s)))
                ids[(d, s)] = torch.as_tensor(w, dtype=torch.long)
        stderr('%s: token-and-position fit on %d reference windows\n' % (cond, sum(len(ids[(d, 'ref')]) for d in DATASETS)))
        ref = torch.cat([ids[(d, 'ref')] for d in DATASETS])
        fit = fit_token_position_large(model, ref, torch.ones_like(ref), len(tok), batch_size=args.batch_size,
                                       min_count=args.min_count, indent=2)
        for d, h in todo:
            stderr('%s / %s / %s\n' % (cond, d, h))
            out = unit_acf(model, ids[(d, h)], fit, args.max_lag, batch_size=args.batch_size, indent=2,
                           centring=args.centring)
            with h5py.File(path, 'a') as f:
                g = f.require_group('%s/%s/%s' % (cond, d, h))
                for k in ('mean', 'var', 'positive_share', 'n_pairs', 'var_within', 'var_between', 'n_windows',
                          'n_undefined', 'n_clipped'):
                    if k in out:
                        g[k] = out[k]
                if args.centring == 'window':
                    stderr('  windows with zero variance (unit-windows): %d; clipped r: %d\n'
                           % (out['n_undefined'].sum(), out['n_clipped'].sum()))
                g.attrs.update(n_tokens=out['n_tokens'], **fit.provenance())
                g['acf'] = out['acf']                     # written last: its presence marks done
                if 'coordinates' not in f:
                    width = len(out['mean']) // _n_layers(model)
                    f['coordinates'] = np.stack(np.divmod(np.arange(len(out['mean'])), width), 1).astype(np.int32)
        del fit
    stderr('wrote %s\n' % path)


def _n_layers(model):
    from parcelmate.model import mlp_projections
    return len(list(mlp_projections(model)))


# ---------------------------------------------------------------------------- analyze

def _spearman(x, y):
    from scipy.stats import spearmanr
    ok = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[ok], y[ok])[0]) if ok.sum() > 2 else float('nan')


def _write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    stderr('wrote %s (%d rows)\n' % (path, len(rows)))


def load_measures(path, max_lag=None):
    """The per-unit measures (Iteration 47, revised again 2026-10-29). Per condition and half, the
    ACF is the mean over the two datasets, read at fixed lags (LAGS):

      <condition>_acf<k>   the ACF at lag k: the correlation of the unit's residual at token t with
                           its residual at token t + k, over every such pair inside a window; mean
                           of the halves

    No curve is fitted: the exponential fit labelled the slow drift of the shuffled top layers as
    floor, and a model-free width labelled it as decay (LOG.md Iteration 47, second revision).
    tau_int (sum of the ACF over lags 1..max_lag, the first definition) and the tail level (mean
    ACF over the last half of the lags) are kept as references."""
    from parcelmate.timescales import tau_int
    with h5py.File(path, 'r') as f:
        coords = np.asarray(f['coordinates'])
        layer_types = [x.decode() for x in f['layer_types']] if 'layer_types' in f else None
        L = int(f.attrs['max_lag']) if max_lag is None else max_lag
        assert L >= max(LAGS), 'the ACF stops at lag %d, before lag %d' % (L, max(LAGS))
        acf = {(c, h): np.mean([np.asarray(f['%s/%s/%s/acf' % (c, d, h)])[:, :L + 1] for d in DATASETS], 0)
               for c in CONDITIONS for h in HALVES}
        cov = {k: np.mean([np.asarray(f['intact/%s/%s/%s' % (d, h, k)]) for d in DATASETS for h in HALVES], 0)
               for k in ('positive_share', 'var')}
    half, ref = {}, {}
    for key, a in acf.items():
        for k in LAGS:
            half[('%s_acf%d' % (key[0], k), key[1])] = a[:, k].astype(np.float64)
        ref[(key[0] + '_tau_int', key[1])] = tau_int(a, L)
        ref[(key[0] + '_tail', key[1])] = a[:, L // 2 + 1:].mean(1)
    measures = {}
    for c in CONDITIONS:
        for k in LAGS:
            m = '%s_acf%d' % (c, k)
            measures[m] = (half[(m, 'halfA')] + half[(m, 'halfB')]) / 2
    return dict(coords=coords, layer_types=layer_types, half=half, ref=ref, measures=measures, cov=cov, max_lag=L)


LAGS = (1, 10, 100)
TESTED = ('intact_acf10', 'intact_acf100', 'shuffled_acf10', 'shuffled_acf100')
DESCRIBED = tuple('%s_acf%d' % (c, k) for c in CONDITIONS for k in LAGS)


def run_analyze(args):
    from scipy.stats import binomtest
    from parcelmate.bin.bank_concentration import load_tasks
    from parcelmate.bin.compare_circuits import load_partitions
    from parcelmate.circuits import circuit_indices, flat_labels
    from parcelmate.timescales import (circuit_tau_test, layer_adjusted_network_tau, network_test,
                                       residualize_within_layer)

    out = os.path.join(args.out, args.tag)
    os.makedirs(out, exist_ok=True)
    T = load_measures(os.path.join(args.out, args.acf))
    coords = T['coords']
    layers = coords[:, 0].astype(np.int64)
    n_layers, width = int(layers.max()) + 1, int(coords[:, 1].max()) + 1
    assert np.array_equal(coords[:, 0] * width + coords[:, 1], np.arange(len(coords))), 'units not in flat order'
    M = dict(T['measures'])
    cov = np.column_stack([T['cov']['positive_share'], np.log(np.clip(T['cov']['var'], 1e-30, None))])
    for m in ('intact_acf10', 'intact_acf100'):
        M[m + '_cov'] = residualize_within_layer(M[m], cov, layers)
    tested = list(TESTED) + ['intact_acf10_cov', 'intact_acf100_cov']
    np.savez(os.path.join(out, 'units.npz'), coordinates=coords, **M,
             **{'%s_%s' % k: v for k, v in T['half'].items()}, **{'%s_%s' % k: v for k, v in T['ref'].items()},
             positive_share=T['cov']['positive_share'], var=T['cov']['var'])
    summary = dict(max_lag=T['max_lag'], lags=list(LAGS), n_units=int(len(coords)),
                   n_finite={m: int(np.isfinite(M[m]).sum()) for m in M})

    # A. reliability, and how the lags relate to each other and to the old measures
    rows = []
    H = T['half']
    for c in CONDITIONS:
        tau_int = (T['ref'][(c + '_tau_int', 'halfA')] + T['ref'][(c + '_tau_int', 'halfB')]) / 2
        tail = (T['ref'][(c + '_tail', 'halfA')] + T['ref'][(c + '_tail', 'halfB')]) / 2
        for k in LAGS:
            m = '%s_acf%d' % (c, k)
            rows.append(dict(measure=m, comparison='halfA_vs_halfB', spearman=_spearman(H[(m, 'halfA')], H[(m, 'halfB')])))
            for ref_name, ref in (('tau_int', tau_int), ('tail', tail)):
                rows.append(dict(measure=m, comparison='vs_' + ref_name, spearman=_spearman(M[m], ref)))
        for i, k1 in enumerate(LAGS):
            for k2 in LAGS[i + 1:]:
                rows.append(dict(measure='%s_acf%d' % (c, k1), comparison='vs_%s_acf%d' % (c, k2),
                                 spearman=_spearman(M['%s_acf%d' % (c, k1)], M['%s_acf%d' % (c, k2)])))
    for k in LAGS:
        rows.append(dict(measure='intact_acf%d' % k, comparison='vs_shuffled_acf%d' % k,
                         spearman=_spearman(M['intact_acf%d' % k], M['shuffled_acf%d' % k])))
    _write_csv(os.path.join(out, 'reliability.csv'), rows)
    summary['split_half'] = {r['measure']: r['spearman'] for r in rows if r['comparison'] == 'halfA_vs_halfB'}
    summary['gate_passed'] = bool(min(summary['split_half'][m] for m in TESTED) >= 0.7)
    if not summary['gate_passed'] and not args.ignore_gate:
        stderr('reliability gate failed (%s): stopping before B-D\n' % summary['split_half'])
        json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
        return

    # B. depth
    rows = []
    for l in range(n_layers):
        r = dict(layer=l, layer_type=T['layer_types'][l] if T['layer_types'] else '')
        for m in DESCRIBED:
            v = M[m][(layers == l) & np.isfinite(M[m])]
            q = np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9])
            r.update({'%s_q10' % m: q[0], '%s_q25' % m: q[1], '%s_median' % m: q[2],
                      '%s_q75' % m: q[3], '%s_q90' % m: q[4], '%s_mean' % m: float(v.mean())})
        rows.append(r)
    _write_csv(os.path.join(out, 'layers.csv'), rows)
    summary['depth_spearman'] = {m: _spearman(layers.astype(float), M[m]) for m in DESCRIBED}

    # C. networks
    parts = load_partitions(args.networks, args.variant)
    assert parts, 'no partitions under %s' % args.networks
    within_var = {m: float(np.nanmean([np.nanvar(M[m][layers == l]) for l in range(n_layers)])) for m in tested}
    rows, net_rows = [], []
    for (domain, half, tree), (P, pc) in sorted(parts.items()):
        k = P.shape[1]
        labels = flat_labels(P, pc, n_layers, width)
        mean_layer = np.bincount(labels, weights=layers.astype(float), minlength=k) / np.maximum(np.bincount(labels, minlength=k), 1)
        for m in tested:
            rng = np.random.RandomState(derive_seed(args.seed, 'timescales_networks', domain, half, tree, m) % (2 ** 32))
            res = network_test(M[m], labels, layers, k, n_layers, args.perm, rng)
            res.update(within_layer_var=within_var[m], share=res['spread'] / within_var[m],
                       perm_share=res['perm_mean'] / within_var[m])
            rows.append(dict(text_domain=domain, half=half, tree=tree, measure=m, **res))
            adj, mean, c = layer_adjusted_network_tau(M[m], labels, layers, k, n_layers)
            for i in range(k):
                net_rows.append(dict(text_domain=domain, half=half, tree=tree, measure=m, network=i,
                                     n_units=int(c[i]), mean_layer=float(mean_layer[i]), value_mean=float(mean[i]),
                                     value_layer_adjusted=float(adj[i])))
            stderr('  networks %s %s %s %s: share of within-layer variance %.3f (permutation %.4f), z %.1f\n'
                   % (domain, half, tree, m, res['share'], res['perm_share'], res['z']))
    _write_csv(os.path.join(out, 'networks.csv'), rows)
    _write_csv(os.path.join(out, 'network_values.csv'), net_rows)
    summary['networks'] = {'%s_%s_%s' % (r['half'], r['tree'], r['measure']): dict(share=r['share'], z=r['z'], p=r['p'])
                           for r in rows}

    if args.richclub:
        rows = []
        for tree in ('real', 'null'):
            z = np.load(os.path.join(args.richclub, 'units_%s.npz' % tree))
            assert np.array_equal(z['coordinates'], coords), 'rich-club units not in the same order'
            s = (z['strength_halfA'] + z['strength_halfB']) / 2
            for m in TESTED:
                within = [_spearman(s[layers == l], M[m][layers == l]) for l in range(n_layers)]
                rows.append(dict(tree=tree, measure=m, spearman_all=_spearman(s, M[m]),
                                 spearman_within_layer_mean=float(np.nanmean(within))))
        _write_csv(os.path.join(out, 'strength.csv'), rows)

    # D. circuits
    acc = {}
    for root in args.patching:
        with open(os.path.join(root, 'accuracy.csv')) as f:
            for r in csv.DictReader(f):
                acc[(r['domain'], r['task'])] = r
    tasks = load_tasks(args.patching)
    rows = []
    for t in tasks:
        assert t['attribution'].shape == (n_layers, width), 'attribution shape differs from the units'
        a = acc[(t['group'], t['task'])]
        c = circuit_indices(t['attribution'], args.pct)
        for m in TESTED:
            # The same draws for every measure of a task.
            rng = np.random.RandomState(derive_seed(args.seed, 'timescales_circuit', t['root'], t['group'], t['task'],
                                                    args.pct) % (2 ** 32))
            res = circuit_tau_test(M[m], c, width, n_layers, args.draws, rng)
            rows.append(dict(root=t['root'], group=t['group'], task=t['task'], pct=args.pct, measure=m,
                             accuracy=float(a['both_correct_accuracy']), n_both_correct=int(a['n_both_correct']),
                             circuit_mean_layer=float((c // width).mean()), **res))
    _write_csv(os.path.join(out, 'circuits.csv'), rows)
    summary['circuits'] = {}
    for m in TESTED:
        R = [r for r in rows if r['measure'] == m]
        s = dict(mean_z_vs_var_z_spearman=_spearman(np.array([r['mean_z'] for r in R]), np.array([r['var_z'] for r in R])))
        for stat in ('mean', 'var'):
            z = np.array([r[stat + '_z'] for r in R])
            n_pos = int((z > 0).sum())
            s[stat] = dict(median_z=float(np.nanmedian(z)), n_above=n_pos, n_tasks=len(z),
                           sign_test_p=float(binomtest(n_pos, int(np.isfinite(z).sum()), 0.5).pvalue),
                           spearman_with_accuracy=_spearman(np.array([r['accuracy'] for r in R]), z))
        summary['circuits'][m] = s
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    stderr(json.dumps(summary, indent=1) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='mode', required=True)
    a = sub.add_parser('acf')
    a.add_argument('--model', required=True)
    a.add_argument('--out', required=True)
    a.add_argument('--n-windows', type=int, default=100)
    a.add_argument('--seq-len', type=int, default=1024)
    a.add_argument('--max-lag', type=int, default=256)
    a.add_argument('--batch-size', type=int, default=8)
    a.add_argument('--min-count', type=int, default=3)
    a.add_argument('--seed', type=int, default=42)
    a.add_argument('--centring', choices=('global', 'window'), default='global',
                   help='global: one mean and variance per sample (acf.h5); window: per window, Fisher '
                        'mean over windows (acf_within.h5, same windows as acf.h5)')
    b = sub.add_parser('analyze')
    b.add_argument('--out', required=True)
    b.add_argument('--acf', default='acf.h5', help='ACF file under --out (acf_within.h5 for the within-window ACF)')
    b.add_argument('--networks', required=True)
    b.add_argument('--patching', nargs='+', required=True)
    b.add_argument('--richclub', default=None)
    b.add_argument('--variant', default='final')
    b.add_argument('--pct', type=float, default=0.1)
    b.add_argument('--draws', type=int, default=1000)
    b.add_argument('--perm', type=int, default=1000)
    b.add_argument('--seed', type=int, default=42)
    b.add_argument('--ignore-gate', action='store_true')
    b.add_argument('--tag', default='analysis_lags', help='output subdirectory of --out')
    args = ap.parse_args()
    run_acf(args) if args.mode == 'acf' else run_analyze(args)


if __name__ == '__main__':
    main()
