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

`analyze` (CPU). Per condition and half, the ACF is the mean over the two datasets, fitted with
A (exp(-k / tau) + B) (`load_measures`). Two measures per unit and condition, each the mean of the
halves: decay = log tau (the decaying part's time constant) and persistence = A x B (the level kept
at long lags). They replace tau_int, which was 80-99% plateau (LOG.md Iteration 47, first results).
Writes to <out>/<--tag>/ (default analysis_fit; the first run's tau_int results are in analysis/):
  units.npz           the measures per unit, per half, the references (tau_int, tail level), covariates
  reliability.csv     A: split-half Spearman (decay also by tertile of the amplitude A), relations
                      to tau_int and to the model-free tail level
  layers.csv          B: tau (in tokens), persistence and amplitude by layer
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


def _fit_all(acf, cores):
    """`timescales.exp_fit` of every row of `acf`, in parallel over chunks."""
    from multiprocessing import Pool
    chunks = np.array_split(acf, max(1, cores * 4))
    with Pool(cores) as p:
        res = p.map(_fit_chunk, chunks)
    return {k: np.concatenate([r[k] for r in res]) for k in ('tau', 'A', 'B', 'ok')}


def load_measures(path, cores, max_lag=None):
    """The per-unit measures (Iteration 47, revised 2026-10-29). Per condition and half, the ACF is
    the mean over the two datasets and is fitted with A (exp(-k / tau) + B) over lags 1..max_lag:

      <condition>_decay        log tau, the time constant of the decaying part, mean of the halves
      <condition>_persistence  A x B, the level the ACF keeps at long lags (variance that is constant
                               within a window and differs between windows), mean of the halves

    A unit whose fit fails in either half gets NaN for both. tau_int (sum of the ACF over lags
    1..max_lag, the first definition) and the model-free tail level (mean ACF over the last half of
    the lags) are kept as references."""
    from parcelmate.timescales import tau_int
    with h5py.File(path, 'r') as f:
        coords = np.asarray(f['coordinates'])
        layer_types = [x.decode() for x in f['layer_types']] if 'layer_types' in f else None
        L = int(f.attrs['max_lag']) if max_lag is None else max_lag
        acf = {(c, h): np.mean([np.asarray(f['%s/%s/%s/acf' % (c, d, h)])[:, :L + 1] for d in DATASETS], 0)
               for c in CONDITIONS for h in HALVES}
        cov = {k: np.mean([np.asarray(f['intact/%s/%s/%s' % (d, h, k)]) for d in DATASETS for h in HALVES], 0)
               for k in ('positive_share', 'var')}
    fits, half, ref = {}, {}, {}
    for key, a in acf.items():
        fits[key] = _fit_all(a, cores)
        stderr('exponential fit, %s %s: %.2f%% succeeded\n' % (key + (100 * fits[key]['ok'].mean(),)))
        ok = fits[key]['ok']
        half[(key[0] + '_decay', key[1])] = np.where(ok, np.log(fits[key]['tau']), np.nan)
        half[(key[0] + '_persistence', key[1])] = np.where(ok, fits[key]['A'] * fits[key]['B'], np.nan)
        half[(key[0] + '_amplitude', key[1])] = np.where(ok, fits[key]['A'], np.nan)
        ref[(key[0] + '_tau_int', key[1])] = tau_int(a, L)
        ref[(key[0] + '_tail', key[1])] = a[:, L // 2 + 1:].mean(1)
    measures = {}
    for c in CONDITIONS:
        both = fits[(c, 'halfA')]['ok'] & fits[(c, 'halfB')]['ok']
        for m in ('decay', 'persistence', 'amplitude'):
            v = (half[(c + '_' + m, 'halfA')] + half[(c + '_' + m, 'halfB')]) / 2
            measures['%s_%s' % (c, m)] = np.where(both, v, np.nan)
    return dict(coords=coords, layer_types=layer_types, half=half, ref=ref, measures=measures, cov=cov,
                fits=fits, max_lag=L)


TESTED = ('intact_decay', 'intact_persistence', 'shuffled_decay', 'shuffled_persistence')


def run_analyze(args):
    from scipy.stats import binomtest
    from parcelmate.bin.bank_concentration import load_tasks
    from parcelmate.bin.compare_circuits import load_partitions
    from parcelmate.circuits import circuit_indices, flat_labels
    from parcelmate.timescales import (circuit_tau_test, layer_adjusted_network_tau, network_test,
                                       residualize_within_layer)

    out = os.path.join(args.out, args.tag)
    os.makedirs(out, exist_ok=True)
    T = load_measures(os.path.join(args.out, 'acf.h5'), args.cores)
    coords = T['coords']
    layers = coords[:, 0].astype(np.int64)
    n_layers, width = int(layers.max()) + 1, int(coords[:, 1].max()) + 1
    assert np.array_equal(coords[:, 0] * width + coords[:, 1], np.arange(len(coords))), 'units not in flat order'
    M = dict(T['measures'])
    cov = np.column_stack([T['cov']['positive_share'], np.log(np.clip(T['cov']['var'], 1e-30, None))])
    for m in ('intact_decay', 'intact_persistence'):
        M[m + '_cov'] = residualize_within_layer(M[m], cov, layers)
    tested = list(TESTED) + ['intact_decay_cov', 'intact_persistence_cov']
    np.savez(os.path.join(out, 'units.npz'), coordinates=coords, **M,
             **{'%s_%s' % k: v for k, v in T['half'].items()}, **{'%s_%s' % k: v for k, v in T['ref'].items()},
             positive_share=T['cov']['positive_share'], var=T['cov']['var'])
    summary = dict(max_lag=T['max_lag'], n_units=int(len(coords)),
                   fit_ok={'%s_%s' % k: float(v['ok'].mean()) for k, v in T['fits'].items()},
                   n_finite={m: int(np.isfinite(M[m]).sum()) for m in M})

    # A. reliability, and how the new measures relate to the old ones
    rows = []
    H = T['half']
    for c in CONDITIONS:
        for m in ('decay', 'persistence', 'amplitude'):
            k = c + '_' + m
            rows.append(dict(measure=k, comparison='halfA_vs_halfB', subset='all',
                             spearman=_spearman(H[(k, 'halfA')], H[(k, 'halfB')])))
        # decay reliability by tertile of the decaying amplitude: tau is only defined when A is not ~0
        amp = M[c + '_amplitude']
        q = np.nanquantile(amp, [1 / 3, 2 / 3])
        for name, sel in (('amplitude_low', amp < q[0]), ('amplitude_mid', (amp >= q[0]) & (amp < q[1])),
                          ('amplitude_high', amp >= q[1])):
            rows.append(dict(measure=c + '_decay', comparison='halfA_vs_halfB', subset=name,
                             spearman=_spearman(H[(c + '_decay', 'halfA')][sel], H[(c + '_decay', 'halfB')][sel])))
        tau_int = (T['ref'][(c + '_tau_int', 'halfA')] + T['ref'][(c + '_tau_int', 'halfB')]) / 2
        tail = (T['ref'][(c + '_tail', 'halfA')] + T['ref'][(c + '_tail', 'halfB')]) / 2
        for k, ref_name, ref in ((c + '_persistence', 'tail', tail), (c + '_persistence', 'tau_int', tau_int),
                                 (c + '_decay', 'tau_int', tau_int), (c + '_decay', c + '_persistence', M[c + '_persistence'])):
            rows.append(dict(measure=k, comparison='vs_' + ref_name, subset='all', spearman=_spearman(M[k], ref)))
    for m in ('decay', 'persistence'):
        rows.append(dict(measure='intact_vs_shuffled', comparison=m, subset='all',
                         spearman=_spearman(M['intact_' + m], M['shuffled_' + m])))
    _write_csv(os.path.join(out, 'reliability.csv'), rows)
    summary['split_half'] = {r['measure']: r['spearman'] for r in rows
                             if r['comparison'] == 'halfA_vs_halfB' and r['subset'] == 'all'}
    summary['gate_passed'] = bool(min(summary['split_half']['intact_decay'],
                                      summary['split_half']['intact_persistence']) >= 0.7)
    if not summary['gate_passed'] and not args.ignore_gate:
        stderr('reliability gate failed (%s): stopping before B-D\n' % summary['split_half'])
        json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
        return

    # B. depth (decay reported in tokens)
    rows = []
    for l in range(n_layers):
        r = dict(layer=l, layer_type=T['layer_types'][l] if T['layer_types'] else '')
        for m in TESTED + ('intact_amplitude', 'shuffled_amplitude'):
            v = M[m][(layers == l) & np.isfinite(M[m])]
            v = np.exp(v) if m.endswith('decay') else v
            q = np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9])
            name = m.replace('decay', 'tau_tokens')
            r.update({'%s_q10' % name: q[0], '%s_q25' % name: q[1], '%s_median' % name: q[2],
                      '%s_q75' % name: q[3], '%s_q90' % name: q[4]})
            r['%s_finite' % name] = float(np.isfinite(M[m][layers == l]).mean())
        rows.append(r)
    _write_csv(os.path.join(out, 'layers.csv'), rows)
    summary['depth_spearman'] = {m: _spearman(layers.astype(float), M[m]) for m in TESTED}

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
    b.add_argument('--tag', default='analysis_fit', help='output subdirectory of --out')
    args = ap.parse_args()
    run_acf(args) if args.mode == 'acf' else run_analyze(args)


if __name__ == '__main__':
    main()
