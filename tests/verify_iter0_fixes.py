"""Verification script for the iteration-0 bug fixes (see LOG.md).

Run with the `analysis` conda environment:
    python tests/verify_iter0_fixes.py

Covers:
  S2 - connectivity binarization thresholds each unit's own row (keepdims fix)
  S4 - PerturbedModel maps the final hidden-state index onto ln_f, so the lesioned
       unit is exactly the measured unit (tested end-to-end on GPT-2)
  S5 - align_samples with weight_samples=True uses bounded [0, 1] weights and the
       sample pointer advances past zero-weight samples
"""

import numpy as np
import torch
from scipy import optimize

from parcelmate.model import get_model_and_tokenizer, align_samples

failures = []


def check(name, cond):
    status = 'OK' if cond else 'FAIL'
    print('%s %s' % (status, name))
    if not cond:
        failures.append(name)


# --- S4: knockout coordinates land on the recorded hidden states ---
n_layers_plus_1 = 13
d = 768
coords = np.stack(
    [np.repeat(np.arange(n_layers_plus_1), d), np.tile(np.arange(d), n_layers_plus_1)],
    axis=1
).astype(np.int32)
probs = np.zeros((n_layers_plus_1 * d, 1))
for layer in (0, 3, 12):
    probs[layer * d + 5, 0] = 1.0  # knock out unit (layer, 5)

model, tok = get_model_and_tokenizer('gpt2', knockout_probs=probs, knockout_thresh=0.5, coordinates=coords)
enc = tok('The quick brown fox jumps over the lazy dog', return_tensors='pt')
with torch.no_grad():
    out = model(**enc, output_hidden_states=True)
hs = out.hidden_states

check('S4: embedding-layer unit (0, 5) zeroed', bool(torch.all(hs[0][..., 5] == 0)))
check('S4: mid-layer unit (3, 5) zeroed', bool(torch.all(hs[3][..., 5] == 0)))
check('S4: untargeted layer (11, 5) untouched', not bool(torch.all(hs[11][..., 5] == 0)))
check('S4: final-layer unit (12, 5) zeroed (post-ln_f)', bool(torch.all(hs[12][..., 5] == 0)))
check('S4: untargeted unit (12, 6) untouched', not bool(torch.all(hs[12][..., 6] == 0)))

# --- S2: binarization density (mirrors the expression in sample_parcellations) ---
n = 200
X = np.random.rand(n, n)
X = (X + X.T) / 2
B = (X > np.quantile(X, 0.9, axis=1, keepdims=True)).astype(int)
check('S2: every unit has identical fingerprint density', len(set(B.sum(axis=1))) == 1)
check('S2: density is 10%', B.sum(axis=1)[0] == int(0.1 * n))

# --- S5: weighted sample alignment ---
rng = np.random.default_rng(0)
true = rng.integers(0, 5, size=300)
samples = []
for k in range(4):
    perm = rng.permutation(5)
    lab = perm[true].copy()
    flip = rng.random(300) < 0.1  # 10% label noise
    lab[flip] = rng.integers(0, 5, size=flip.sum())
    samples.append(lab)
samples = np.array(samples, dtype=float)
scores = np.array([1e6, 2e6, 3e6, 4e6])  # realistic k-means inertia scale

p = align_samples(samples, scores, weight_samples=True, verbose=False)
check('S5: weighted parcellation values in [0, 1]', bool((p >= 0).all() and (p <= 1).all()))
check('S5: weighted parcellation rows sum to 1', bool(np.allclose(p.sum(axis=1), 1)))

rec = p.argmax(axis=1)
conf = np.array([[np.sum((true == a) & (rec == b)) for b in range(5)] for a in range(5)])
r, c = optimize.linear_sum_assignment(-conf)
agreement = conf[r, c].sum() / 300
check('S5: recovers ground-truth clustering (>0.9 agreement, got %.3f)' % agreement, agreement > 0.9)

p2 = align_samples(samples, scores, weight_samples=False, verbose=False)
check('S5: unweighted path unaffected (rows sum to 1)', bool(np.allclose(p2.sum(axis=1), 1)))

print()
if failures:
    raise SystemExit('%d check(s) failed: %s' % (len(failures), failures))
print('All checks passed.')
