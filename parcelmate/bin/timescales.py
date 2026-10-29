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
(condition, dataset, half) is skipped on a rerun.

`analyze` (CPU). Per condition, the ACF of each half is the mean over the two datasets, tau_int is
its sum over lags 1..max_lag, and the analysed tau is the mean of the two halves ('diff' =
intact - shuffled). Writes to <out>/analysis/:
  units.npz        tau per unit (condition x half x dataset), exponential fits, covariates
  reliability.csv  A: Spearman across units, half A vs half B and wikitext vs bookcorpus
  layers.csv       B: tau by layer
  networks.csv     C: spread of layer-adjusted network tau against within-layer permutations,
                   every partition (halves, real and null trees) and measure
  network_tau.csv  C: layer-adjusted tau of every network
  strength.csv     C: Spearman of tau with node strength (rich-club units), real and null
  circuits.csv     D: per task, mean and variance of tau over the circuit against layer-matched
                   random sets, with the task's both-correct accuracy
  summary.json     aggregates (median z, sign tests, accuracy correlations)
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


def run_acf(args):
    import torch
    from parcelmate.data import shuffle_words_within, word_start_table
    from parcelmate.model import get_model_and_tokenizer
    from parcelmate.residual import fit_token_position_large
    from parcelmate.timescales import load_windows, unit_acf

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, 'acf.h5')
    model, tok = get_model_and_tokenizer(args.model)
    n = args.n_windows
    windows = {}
    with h5py.File(path, 'a') as f:
        f.attrs.update(model=args.model, n_windows=n, seq_len=args.seq_len, max_lag=args.max_lag,
                       seed=args.seed, min_count=args.min_count)
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
            out = unit_acf(model, ids[(d, h)], fit, args.max_lag, batch_size=args.batch_size, indent=2)
            with h5py.File(path, 'a') as f:
                g = f.require_group('%s/%s/%s' % (cond, d, h))
                for k in ('mean', 'var', 'positive_share', 'n_pairs'):
                    g[k] = out[k]
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


def _fit_chunk(a):
    from parcelmate.timescales import exp_fit
    return exp_fit(a)


def load_taus(path, max_lag=None):
    """tau_int per condition, half and dataset, and the analysed measures."""
    from parcelmate.timescales import tau_int
    with h5py.File(path, 'r') as f:
        coords = np.asarray(f['coordinates'])
        layer_types = [x.decode() for x in f['layer_types']] if 'layer_types' in f else None
        L = int(f.attrs['max_lag']) if max_lag is None else max_lag
        acf = {(c, d, h): np.asarray(f['%s/%s/%s/acf' % (c, d, h)]) for c in CONDITIONS for d in DATASETS for h in HALVES}
        cov = {k: np.mean([np.asarray(f['intact/%s/%s/%s' % (d, h, k)]) for d in DATASETS for h in HALVES], 0)
               for k in ('positive_share', 'var')}
    per = {k: tau_int(v, L) for k, v in acf.items()}
    half = {(c, h): tau_int(np.mean([acf[(c, d, h)] for d in DATASETS], 0), L) for c in CONDITIONS for h in HALVES}
    for h in HALVES:
        half[('diff', h)] = half[('intact', h)] - half[('shuffled', h)]
    tau = {c: (half[(c, 'halfA')] + half[(c, 'halfB')]) / 2 for c in CONDITIONS + ('diff',)}
    mean_acf = {c: np.mean([acf[(c, d, h)] for d in DATASETS for h in HALVES], 0) for c in CONDITIONS}
    return dict(coords=coords, layer_types=layer_types, per=per, half=half, tau=tau, cov=cov,
                mean_acf=mean_acf, max_lag=L)


def run_analyze(args):
    from scipy.stats import binomtest
    from parcelmate.bin.bank_concentration import load_tasks
    from parcelmate.bin.compare_circuits import load_partitions
    from parcelmate.circuits import circuit_indices, flat_labels
    from parcelmate.timescales import (circuit_tau_test, layer_adjusted_network_tau, network_test,
                                       residualize_within_layer)

    out = os.path.join(args.out, 'analysis')
    os.makedirs(out, exist_ok=True)
    T = load_taus(os.path.join(args.out, 'acf.h5'))
    coords = T['coords']
    layers = coords[:, 0].astype(np.int64)
    n_layers, width = int(layers.max()) + 1, int(coords[:, 1].max()) + 1
    assert np.array_equal(coords[:, 0] * width + coords[:, 1], np.arange(len(coords))), 'units not in flat order'
    tau = T['tau']
    measures = ['intact', 'shuffled', 'diff']
    cov = np.column_stack([T['cov']['positive_share'], np.log(np.clip(T['cov']['var'], 1e-30, None))])
    tau['intact_cov'] = residualize_within_layer(tau['intact'], cov, layers)

    # exponential fits on the ACF averaged over datasets and halves
    from multiprocessing import Pool
    fits = {}
    for c in CONDITIONS:
        chunks = np.array_split(T['mean_acf'][c], max(1, args.cores * 4))
        with Pool(args.cores) as p:
            res = p.map(_fit_chunk, chunks)
        fits[c] = {k: np.concatenate([r[k] for r in res]) for k in ('tau', 'A', 'B', 'ok')}
        stderr('exponential fit, %s: %.1f%% succeeded\n' % (c, 100 * fits[c]['ok'].mean()))
    np.savez(os.path.join(out, 'units.npz'), coordinates=coords,
             **{'tau_' + k: v for k, v in tau.items()},
             **{'tau_%s_%s' % k: v for k, v in T['half'].items()},
             **{'tau_%s_%s_%s' % k: v for k, v in T['per'].items()},
             **{'exp_%s_%s' % (c, k): v for c in fits for k, v in fits[c].items()},
             positive_share=T['cov']['positive_share'], var=T['cov']['var'])

    summary = dict(max_lag=T['max_lag'], exp_fit_ok={c: float(fits[c]['ok'].mean()) for c in fits},
                   n_units=int(len(coords)), n_finite={m: int(np.isfinite(tau[m]).sum()) for m in tau})

    # A. reliability
    rows = []
    for m in measures:
        rows.append(dict(measure=m, comparison='halfA_vs_halfB',
                         spearman=_spearman(T['half'][(m, 'halfA')], T['half'][(m, 'halfB')])))
    for c in CONDITIONS:
        t = {d: (T['per'][(c, d, 'halfA')] + T['per'][(c, d, 'halfB')]) / 2 for d in DATASETS}
        rows.append(dict(measure=c, comparison='wikitext_vs_bookcorpus', spearman=_spearman(*[t[d] for d in DATASETS])))
        for d in DATASETS:
            rows.append(dict(measure=c, comparison='%s_halfA_vs_halfB' % d,
                             spearman=_spearman(T['per'][(c, d, 'halfA')], T['per'][(c, d, 'halfB')])))
    rows.append(dict(measure='intact_vs_shuffled', comparison='units', spearman=_spearman(tau['intact'], tau['shuffled'])))
    _write_csv(os.path.join(out, 'reliability.csv'), rows)
    summary['split_half'] = {r['measure']: r['spearman'] for r in rows if r['comparison'] == 'halfA_vs_halfB'}
    summary['gate_passed'] = bool(summary['split_half']['intact'] >= 0.7)
    if not summary['gate_passed'] and not args.ignore_gate:
        stderr('reliability gate failed (split-half Spearman %.3f < 0.7): stopping before B-D\n'
               % summary['split_half']['intact'])
        json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
        return

    # B. depth
    rows = []
    for l in range(n_layers):
        r = dict(layer=l, layer_type=T['layer_types'][l] if T['layer_types'] else '')
        for m in measures:
            v = tau[m][(layers == l) & np.isfinite(tau[m])]
            q = np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9])
            r.update({'%s_mean' % m: float(v.mean()), '%s_q10' % m: q[0], '%s_q25' % m: q[1],
                      '%s_median' % m: q[2], '%s_q75' % m: q[3], '%s_q90' % m: q[4]})
        for c in CONDITIONS:
            r['%s_exp_tau_median' % c] = float(np.nanmedian(np.where(fits[c]['ok'], fits[c]['tau'], np.nan)[layers == l]))
            r['%s_exp_ok' % c] = float(fits[c]['ok'][layers == l].mean())
        rows.append(r)
    _write_csv(os.path.join(out, 'layers.csv'), rows)
    summary['depth_spearman'] = {m: _spearman(layers.astype(float), tau[m]) for m in measures}

    # C. networks
    parts = load_partitions(args.networks, args.variant)
    assert parts, 'no partitions under %s' % args.networks
    rows, net_rows = [], []
    for (domain, half, tree), (P, pc) in sorted(parts.items()):
        k = P.shape[1]
        labels = flat_labels(P, pc, n_layers, width)
        for m in measures + ['intact_cov']:
            rng = np.random.RandomState(derive_seed(args.seed, 'timescales_networks', domain, half, tree, m) % (2 ** 32))
            res = network_test(tau[m], labels, layers, k, n_layers, args.perm, rng)
            rows.append(dict(text_domain=domain, half=half, tree=tree, measure=m, **res))
            adj, mean, c = layer_adjusted_network_tau(tau[m], labels, layers, k, n_layers)
            mean_layer = np.bincount(labels, weights=layers.astype(float), minlength=k) / np.maximum(np.bincount(labels, minlength=k), 1)
            for i in range(k):
                net_rows.append(dict(text_domain=domain, half=half, tree=tree, measure=m, network=i,
                                     n_units=int(c[i]), mean_layer=float(mean_layer[i]), tau_mean=float(mean[i]),
                                     tau_layer_adjusted=float(adj[i])))
            stderr('  networks %s %s %s %s: spread %.3g, permutation %.3g +- %.3g, z %.1f\n'
                   % (domain, half, tree, m, res['spread'], res['perm_mean'], res['perm_sd'], res['z']))
    _write_csv(os.path.join(out, 'networks.csv'), rows)
    _write_csv(os.path.join(out, 'network_tau.csv'), net_rows)
    summary['networks'] = {'%s_%s_%s' % (r['half'], r['tree'], r['measure']): dict(z=r['z'], p=r['p']) for r in rows}

    if args.richclub:
        rows = []
        for tree in ('real', 'null'):
            z = np.load(os.path.join(args.richclub, 'units_%s.npz' % tree))
            assert np.array_equal(z['coordinates'], coords), 'rich-club units not in the same order'
            s = (z['strength_halfA'] + z['strength_halfB']) / 2
            for m in measures:
                within = [_spearman(s[layers == l], tau[m][layers == l]) for l in range(n_layers)]
                rows.append(dict(tree=tree, measure=m, spearman_all=_spearman(s, tau[m]),
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
        for m in measures:
            # The same draws for every measure of a task.
            rng = np.random.RandomState(derive_seed(args.seed, 'timescales_circuit', t['root'], t['group'], t['task'],
                                                    args.pct) % (2 ** 32))
            res = circuit_tau_test(tau[m], c, width, n_layers, args.draws, rng)
            rows.append(dict(root=t['root'], group=t['group'], task=t['task'], pct=args.pct, measure=m,
                             accuracy=float(a['both_correct_accuracy']), n_both_correct=int(a['n_both_correct']),
                             circuit_mean_layer=float((c // width).mean()), **res))
    _write_csv(os.path.join(out, 'circuits.csv'), rows)
    summary['circuits'] = {}
    for m in measures:
        R = [r for r in rows if r['measure'] == m]
        s = {}
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
    b = sub.add_parser('analyze')
    b.add_argument('--out', required=True)
    b.add_argument('--networks', required=True)
    b.add_argument('--patching', nargs='+', required=True)
    b.add_argument('--richclub', default=None)
    b.add_argument('--variant', default='final')
    b.add_argument('--pct', type=float, default=0.1)
    b.add_argument('--draws', type=int, default=1000)
    b.add_argument('--perm', type=int, default=1000)
    b.add_argument('--cores', type=int, default=8)
    b.add_argument('--seed', type=int, default=42)
    b.add_argument('--ignore-gate', action='store_true')
    args = ap.parse_args()
    run_acf(args) if args.mode == 'acf' else run_analyze(args)


if __name__ == '__main__':
    main()
