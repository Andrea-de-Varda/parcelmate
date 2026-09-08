"""Verification for the iteration-2 changes: S3 config plumbing, S1 per-network knockout,
mean-ablation, and the cached per-unit activation statistics (see LOG.md).

Run from the repo root with the `analysis` conda environment:
    PYTHONPATH=. python tests/verify_iter2_fixes.py

The GPT-2 sections need the model cached locally (it is).
"""

import os
import shutil
import tempfile

import numpy as np
import torch

from parcelmate.model import (
    get_model_and_tokenizer,
    get_timecourses,
    pool_unit_stats,
    resolve_networks,
    run_knockout,
    select_network_units,
)
from parcelmate.util import h5_keys, load_h5_array, save_h5_data

failures = []


def check(name, cond):
    print('%s %s' % ('OK' if cond else 'FAIL', name))
    if not cond:
        failures.append(name)


# ------------------------------------------------------- S1: network selection is per-network
# Rows sum to <= 1, so at 0.5 a unit clears threshold for at most one network. The old code
# OR-ed across all columns, which is why it could only ever build the union.
parc = np.array([
    [0.9, 0.1, 0.0],
    [0.2, 0.8, 0.0],
    [0.3, 0.3, 0.4],
    [0.0, 0.0, 1.0],
], dtype=float)
check('S1: selecting network 0 picks only its confident units',
      list(select_network_units(parc, 0)) == [True, False, False, False])
check('S1: selecting network 2 picks only its confident units',
      list(select_network_units(parc, 2)) == [False, False, False, True])
check('S1: an ambiguous unit (max 0.4) is in no network at thresh 0.5',
      not any(select_network_units(parc, k)[2] for k in range(3)))
check('S1: threshold is honoured', list(select_network_units(parc, 2, knockout_thresh=0.35))
      == [False, False, True, True])
try:
    select_network_units(parc, 7)
    check('S1: out-of-range network raises', False)
except AssertionError:
    check('S1: out-of-range network raises', True)

# ------------------------------------------------------- S1: knockout is opt-in
check('S1: networks=None selects nothing (knockout is opt-in)', resolve_networks(None, 20) == [])
check('S1: networks="all" selects every network', resolve_networks('all', 4) == [0, 1, 2, 3])
check('S1: an explicit list is preserved', resolve_networks([2, 0], 4) == [2, 0])
check('S1: a bare int is accepted', resolve_networks(3, 4) == [3])
for bad in ([0, 9], 'every', -1):
    try:
        resolve_networks(bad, 4)
        check('S1: rejects %r' % (bad,), False)
    except AssertionError:
        check('S1: rejects %r' % (bad,), True)

# ------------------------------------------------------- pooled statistics are exact
rng = np.random.RandomState(0)
a = rng.randn(3, 500)
b = rng.randn(3, 250) * 2 + 1  # different mean AND spread, so naive averaging would be wrong
pm, ps = pool_unit_stats([a.mean(1), b.mean(1)], [a.std(1), b.std(1)], [a.shape[1], b.shape[1]])
allx = np.concatenate([a, b], axis=1)
check('pooling: mean matches the concatenated data', np.allclose(pm, allx.mean(1), atol=1e-5))
check('pooling: std matches the concatenated data', np.allclose(ps, allx.std(1), atol=1e-5))
naive = (a.std(1) + b.std(1)) / 2
check('pooling: differs from naively averaging stds (the bug it avoids)',
      not np.allclose(naive, allx.std(1), atol=1e-3))

# ------------------------------------------------------- per-unit stats from the real model
tok_model, tok = get_model_and_tokenizer('gpt2')
# No padding: GPT-2 has no pad token, and the pipeline pads manually in data.pad().
enc = tok(['The quick brown fox jumps over the lazy dog.'] * 2, return_tensors='pt')
out = get_timecourses(tok_model, enc['input_ids'], enc['attention_mask'], batch_size=2, verbose=False)
tc = out['timecourses']
check('stats: unit_means has one entry per unit', out['unit_means'].shape == (tc.shape[0],))
check('stats: unit_stds has one entry per unit', out['unit_stds'].shape == (tc.shape[0],))
check('stats: n_obs equals the token count', out['n_obs'] == tc.shape[1])
check('stats: means match the timecourses', np.allclose(out['unit_means'], tc.mean(axis=1), atol=1e-5))
check('stats: stds match the timecourses', np.allclose(out['unit_stds'], tc.std(axis=1), atol=1e-5))

# ------------------------------------------------------- mean-ablation on the real model
d = 768
n_layers_plus_1 = 13
coords = np.stack(
    [np.repeat(np.arange(n_layers_plus_1), d), np.tile(np.arange(d), n_layers_plus_1)], axis=1
).astype(np.int32)
probs = np.zeros((n_layers_plus_1 * d, 2))
targets = [(0, 5), (3, 5), (12, 5)]
for layer, unit in targets:
    probs[layer * d + unit, 0] = 1.0     # network 0
probs[1 * d + 7, 1] = 1.0                # network 1, a different unit

m0, _ = get_model_and_tokenizer('gpt2', knockout_probs=probs, coordinates=coords, network=0)
with torch.no_grad():
    hs = m0(**enc, output_hidden_states=True).hidden_states
check('S1: network 0 lesion zeroes its own units',
      all(bool(torch.all(hs[l][..., u] == 0)) for l, u in targets))
check('S1: network 0 lesion leaves network 1 units alone',
      not bool(torch.all(hs[1][..., 7] == 0)))

m1, _ = get_model_and_tokenizer('gpt2', knockout_probs=probs, coordinates=coords, network=1)
with torch.no_grad():
    hs1 = m1(**enc, output_hidden_states=True).hidden_states
check('S1: network 1 lesion zeroes its own unit', bool(torch.all(hs1[1][..., 7] == 0)))
check('S1: network 1 lesion leaves network 0 units alone',
      not any(bool(torch.all(hs1[l][..., u] == 0)) for l, u in targets))

# set_perturbation_values must reach every wrapped layer, including the two special ones
# (embedding via model.drop and the final state via model.ln_f)
sel = select_network_units(probs, 0)
fill = np.array([3.0, -2.0, 7.0])[:sel.sum()]
m0.set_perturbation_values(fill)
with torch.no_grad():
    hs_m = m0(**enc, output_hidden_states=True).hidden_states
observed = [float(hs_m[l][..., u].flatten()[0]) for l, u in targets]
check('mean-ablation: updated values reach embedding, mid and final layers (%s)' %
      np.round(observed, 3).tolist(),
      np.allclose(sorted(observed), sorted(fill.tolist()), atol=1e-3))
check('mean-ablation: every position gets the value, not just the first',
      all(bool(torch.allclose(hs_m[l][..., u], torch.full_like(hs_m[l][..., u], v), atol=1e-3))
          for (l, u), v in zip(targets, observed)))
try:
    m0.set_perturbation_values(np.zeros(len(fill) + 1))
    check('mean-ablation: wrong-length values raise', False)
except AssertionError:
    check('mean-ablation: wrong-length values raise', True)

# ------------------------------------------------------- run_knockout is a no-op by default
tmp = tempfile.mkdtemp(prefix='parcelmate_iter2_')
try:
    sub = os.path.join(tmp, 'subnetwork')
    os.makedirs(sub)
    save_h5_data(
        dict(parcellation=parc, coordinates=np.zeros((4, 2), dtype=np.int32)),
        os.path.join(sub, 'parcellation_shared_avg.h5'), verbose=False
    )
    run_knockout(output_dir=tmp, networks=None, verbose=False)
    check('S1: networks=None writes nothing', not os.path.exists(os.path.join(tmp, 'knockout')))

    missing = os.path.join(tmp, 'no_such_run')
    run_knockout(output_dir=missing, networks='all', verbose=False)
    check('S1: a missing subnetwork dir is reported, not raised',
          not os.path.exists(os.path.join(missing, 'knockout')))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    raise SystemExit('%d check(s) failed: %s' % (len(failures), failures))
print('All checks passed.')
