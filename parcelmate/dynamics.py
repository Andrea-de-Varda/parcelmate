"""Descriptive measures of the connectome and its networks across training (LOG.md Iteration 28).

Reliability and fidelity say whether a partition is real; they do not describe the network,
and a random and a trained model pass them equally well (Iteration 27). This module measures
what the network looks like, in three groups, agreed with Andrea on 2026-09-23:

  connectome   dimensionality (eigenvalue spectrum of the unit correlation matrix),
               coupling strength and tail (off-diagonal |r|), how concentrated each unit's
               connections are, how unequal unit strength is (hubs), and how segregated the
               stored networks are (within- against between-network |r|, weighted modularity);
               plus a fixed subsample of units whose |r| is kept, for connectome similarity
               between checkpoints.
  activations  how often each MLP neuron fires (post-GELU > 0), its mean activation on seven
               string-defined token classes, and the model's language-modelling loss on the
               same text.
  partitions   (no model needed) how crisp, how evenly sized and how deep the stored networks
               are, how domain-specific each is, when each final network first appears, and
               whether networks split, merge or are rebuilt between checkpoints.

The connectome is recomputed with the same data loader, seeds and timecourse code as the
connectivity step (`model.domain_data_kwargs`, `get_dataset`, `get_timecourses`,
`get_connectivity`, the Fisher mean of two samples per half), so it equals the connectome the
stored partitions were fit on up to GPU rounding, and those partitions apply to it directly.
Nothing N x N is written except the subsample.
"""

import os
import time
import unicodedata

import h5py
import numpy as np
import torch

from parcelmate.constants import HALF_NAMES, N_TOKENS
from parcelmate.data import fisher_average, get_dataset
from parcelmate.metrics import hard_labels
from parcelmate.util import derive_seed, stderr

# ---------------------------------------------------------------------------- token classes

TOKEN_CLASSES = ('whitespace', 'punctuation', 'digit', 'function_word', 'content_word',
                 'continuation', 'other')
_WS, _PUNCT, _DIGIT, _ALPHA, _OTHER = range(5)

# English closed-class words: articles and determiners, pronouns, prepositions,
# conjunctions, auxiliaries and modals, negation, and a few closed-class adverbs.
FUNCTION_WORDS = frozenset('''
a an the this that these those some any each every either neither no another such what which
whose whatever whichever all both half several many much more most few fewer little less
i me my mine myself you your yours yourself yourselves he him his himself she her hers herself
it its itself we us our ours ourselves they them their theirs themselves one ones who whom
someone anyone everyone nobody somebody anybody everybody something anything everything nothing
of in on at by for with about against between into through during before after above below to
from up down out off over under again further than per via upon within without among amongst
across along around behind beyond beside besides toward towards onto near like unlike despite
since until till
and or but nor so yet because although though while whereas if unless whether as than once
whenever wherever
be am is are was were been being have has had having do does did doing done
will would shall should can could may might must ought
not n't never
there here then now also very too just only even still already
'''.split())


def token_class_table(tokenizer):
    """Per-vocabulary-id attributes from the decoded string alone.

    Returns dict of int arrays of length vocab: `base` (whitespace, punctuation or symbol,
    digit, alphabetic, other), `lead` (the string starts with whitespace, i.e. the token
    opens a word in byte-level BPE), `func` (alphabetic and a function word).
    """
    V = len(tokenizer)
    base = np.full(V, _OTHER, dtype=np.int8)
    lead = np.zeros(V, dtype=bool)
    func = np.zeros(V, dtype=bool)
    for i in range(V):
        s = tokenizer.decode([i])
        core = s.strip()
        lead[i] = bool(s) and s[0].isspace()
        if core == '':
            base[i] = _WS
        elif all(unicodedata.category(ch)[0] in 'PS' for ch in core):
            base[i] = _PUNCT
        elif all(ch.isdigit() for ch in core):
            base[i] = _DIGIT
        elif core.isalpha():
            base[i] = _ALPHA
            func[i] = core.lower() in FUNCTION_WORDS
    return dict(base=base, lead=lead, func=func)


def token_classes(input_ids, table):
    """Class index (into TOKEN_CLASSES) of every position of `input_ids` (B x L numpy).

    An alphabetic token opens a word if its string starts with whitespace, if it is the first
    position of its row, or if the previous token is whitespace or punctuation (a word after
    a newline or a quote); otherwise it continues a word.
    """
    ids = np.asarray(input_ids)
    base = table['base'][ids]
    lead = table['lead'][ids]
    func = table['func'][ids]
    prev = np.full_like(base, _WS)
    prev[:, 1:] = base[:, :-1]
    initial = lead | (prev == _WS) | (prev == _PUNCT)
    out = np.full(ids.shape, TOKEN_CLASSES.index('other'), dtype=np.int8)
    out[base == _WS] = TOKEN_CLASSES.index('whitespace')
    out[base == _PUNCT] = TOKEN_CLASSES.index('punctuation')
    out[base == _DIGIT] = TOKEN_CLASSES.index('digit')
    alpha = base == _ALPHA
    out[alpha & initial & func] = TOKEN_CLASSES.index('function_word')
    out[alpha & initial & ~func] = TOKEN_CLASSES.index('content_word')
    out[alpha & ~initial] = TOKEN_CLASSES.index('continuation')
    return out


# ---------------------------------------------------------------------------- small measures

def gini(x):
    """Gini coefficient of a non-negative vector (0: all equal; toward 1: one holds all)."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float('nan')
    i = np.arange(1, n + 1)
    return float(2.0 * (i * x).sum() / (n * x.sum()) - (n + 1.0) / n)


def spectrum_measures(eig, prefix='dim_'):
    """Dimensionality summaries of a correlation matrix's eigenvalues (any order)."""
    lam = np.sort(np.clip(np.asarray(eig, dtype=np.float64), 0, None))[::-1]
    tot = lam.sum()
    cum = np.cumsum(lam) / tot
    out = {
        'participation_ratio': float(tot ** 2 / (lam ** 2).sum()),
        'top1_share': float(lam[0] / tot),
        'top10_share': float(lam[:10].sum() / tot),
        'top100_share': float(lam[:100].sum() / tot),
        'n_for_50pct': int(np.searchsorted(cum, 0.5) + 1),
        'n_for_90pct': int(np.searchsorted(cum, 0.9) + 1),
    }
    out['participation_ratio_frac'] = out['participation_ratio'] / len(lam)
    return {prefix + k: v for k, v in out.items()}


def eigenvalues(R, device=None):
    """All eigenvalues of a symmetric matrix (lower triangle read), robust to large n.

    Tried in order (LOG.md Iteration 29): cuSOLVER on the GPU; MAGMA on the GPU, since
    cuSOLVER's syevd rejects n = 36,864 (Pythia-160m) with CUSOLVER_STATUS_INVALID_VALUE, a
    size limit of its workspace query rather than bad input; LAPACK on the CPU in float32
    (values only, so the workspace is O(n); minutes at 36,864 units on 8 cores). The same
    eigenvalues each way, up to float32 rounding. Which one ran is logged.
    """
    R = np.asarray(R, dtype=np.float32)
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device != 'cpu':
        for backend in ('cusolver', 'magma'):
            try:
                torch.backends.cuda.preferred_linalg_library(backend)
                t = torch.as_tensor(R, device=device)
                ev = torch.linalg.eigvalsh(t).double().cpu().numpy()
                del t
                torch.cuda.empty_cache()
                if backend != 'cusolver':
                    stderr('eigenvalues: computed with %s\n' % backend)
                return ev
            except (RuntimeError, torch.OutOfMemoryError) as e:
                torch.cuda.empty_cache()
                stderr('eigenvalues: %s failed at n = %d (%s)\n' % (backend, R.shape[0], str(e).split('\n')[0][:120]))
            finally:
                torch.backends.cuda.preferred_linalg_library('default')
        stderr('eigenvalues: computing on the CPU\n')
    import scipy.linalg
    return scipy.linalg.eigh(R, eigvals_only=True, driver='evd', check_finite=False).astype(np.float64)


def coupling_measures(A, seed=0, n_pairs=5_000_000, top_frac=0.01, prefix='coupling_'):
    """Off-diagonal |r| distribution, per-unit concentration and hub inequality.

    `A` is |r| with a zero diagonal (N x N). Quantiles come from a seeded sample of unit
    pairs; the mean is exact. `concentration` is, per unit, the share of its total |r| held
    by its top 1% of partners (median over units): high means few strong partners, low many
    weak ones.
    """
    N = A.shape[0]
    rng = np.random.RandomState(seed % (2 ** 32))
    i = rng.randint(0, N, size=n_pairs)
    j = rng.randint(0, N, size=n_pairs)
    keep = i != j
    v = A[i[keep], j[keep]].astype(np.float64)
    strength = A.sum(1, dtype=np.float64)
    k = max(1, int(np.ceil(top_frac * (N - 1))))
    # Row blocks, so no full copy of A is made (5.4 GB at 36,864 units).
    top = np.concatenate([np.partition(A[s:s + 2048], N - k, axis=1)[:, N - k:].sum(1, dtype=np.float64)
                          for s in range(0, N, 2048)])
    with np.errstate(invalid='ignore', divide='ignore'):
        conc = np.where(strength > 0, top / strength, np.nan)
    out = {
        prefix + 'mean_absr': float(strength.sum() / (N * (N - 1))),
        prefix + 'median_absr': float(np.median(v)),
        prefix + 'p90_absr': float(np.quantile(v, 0.9)),
        prefix + 'p99_absr': float(np.quantile(v, 0.99)),
        prefix + 'frac_absr_gt_0.1': float((v > 0.1).mean()),
        prefix + 'frac_absr_gt_0.3': float((v > 0.3).mean()),
        prefix + 'concentration_top1pct': float(np.nanmedian(conc)),
        'hub_strength_gini': gini(strength),
        'hub_strength_cv': float(strength.std() / strength.mean()) if strength.mean() > 0 else float('nan'),
    }
    return out


def segregation(A, labels, n_networks):
    """How separated a partition's networks are on the connectome `A` (|r|, zero diagonal).

    within_mean  mean |r| over pairs in the same network
    between_mean mean |r| over pairs in different networks
    ratio        within / between
    modularity   weighted Newman modularity, sum_c [W_cc / 2m - (K_c / 2m)^2]: the share of
                 connection weight inside networks beyond what a network of the same unit
                 strengths wired at random would put there
    """
    labels = np.asarray(labels)
    N = len(labels)
    # float32 product, float64 block sums: no N x N float64 copy (11 GB at 36,864 units).
    onehot = np.zeros((N, n_networks), dtype=np.float32)
    onehot[np.arange(N), labels] = 1.0
    W = onehot.T.astype(np.float64) @ (np.asarray(A, dtype=np.float32) @ onehot).astype(np.float64)
    sizes = onehot.sum(0).astype(np.float64)
    within_pairs = (sizes * (sizes - 1)).sum()
    total_pairs = N * (N - 1)
    within = np.trace(W)
    total = W.sum()
    K = W.sum(1)
    two_m = total
    out = {
        'segregation_within_mean': float(within / within_pairs) if within_pairs > 0 else float('nan'),
        'segregation_between_mean': float((total - within) / (total_pairs - within_pairs)),
    }
    out['segregation_ratio'] = out['segregation_within_mean'] / out['segregation_between_mean']
    out['segregation_modularity'] = float((np.diag(W) / two_m - (K / two_m) ** 2).sum())
    return out


# ---------------------------------------------------------------------------- the checkpoint job

def _samples(input_ids, attention_mask, n_samples):
    n = int(np.ceil(len(input_ids) / n_samples))
    return [(input_ids[k * n:(k + 1) * n], attention_mask[k * n:(k + 1) * n]) for k in range(n_samples)]


def lm_loss(lm, input_ids, attention_mask, batch_size=8):
    """Sum and count of next-token cross-entropy over positions where both tokens are real."""
    device = next(lm.parameters()).device
    tot, cnt = 0.0, 0
    for s in range(0, input_ids.size(0), batch_size):
        ids = input_ids[s:s + batch_size].to(device)
        m = attention_mask[s:s + batch_size].to(device).bool()
        with torch.no_grad():
            logits = lm(input_ids=ids, attention_mask=m.long()).logits[:, :-1].float()
        target = ids[:, 1:]
        valid = m[:, :-1] & m[:, 1:]
        ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                               target.reshape(-1), reduction='none')
        ce = ce.view(target.shape)[valid]
        tot += float(ce.double().sum())
        cnt += int(valid.sum())
        del logits
    return tot, cnt


def checkpoint_measures(cfg, out_dir, step=None, variant='final', n_sub=4096, keep_halves=False,
                        verbose=True):
    """Every connectome and activation measure for one checkpoint, all domains of `cfg`.

    `cfg` is the checkpoint's pipeline config (configs/pythia/...). Reads the stored real and
    null partitions from its `output_dir`; writes into `out_dir`:
      connectome_<step>.csv            tidy rows (step, domain, key, partition, eval_key,
                                       measure, value)
      units_<step>_<domain>.h5         per half: firing rate, class means, class counts;
                                       per sample: loss sum and count
      subsample_<step>_<domain>_<key>.h5  |r| on a fixed seeded subsample of units (fp16)
    With `keep_halves` the two half matrices are also returned (for tests).
    """
    import csv
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from parcelmate.model import domain_data_kwargs, get_connectivity, get_timecourses
    from parcelmate.bin.score import parc_path

    conn = cfg['connectivity']
    seed = cfg.get('seed')
    model_name, revision = conn['model_name'], conn.get('revision')
    assert conn.get('unit_type') == 'mlp' and not conn.get('units_per_layer'), \
        'dynamics measures are for all MLP units'
    n_samples = int(conn['n_samples'])
    assert n_samples >= 2 and n_samples % 2 == 0
    seq_len, batch_size = int(conn['seq_len']), int(conn['batch_size'])
    n_tokens = conn.get('n_tokens') or (N_TOKENS // (seq_len * batch_size)) * seq_len * batch_size
    if step is None:
        step = int(str(revision).replace('step', '')) if revision else -1
    root = cfg['output_dir']
    null_root = root.rstrip('/') + '_null'
    os.makedirs(out_dir, exist_ok=True)

    kw = {} if revision is None else dict(revision=str(revision))
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kw)
    lm = AutoModelForCausalLM.from_pretrained(model_name, **kw).float().eval()
    base = lm.base_model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    table = token_class_table(tokenizer)
    C = len(TOKEN_CLASSES)
    rows, halves_out = [], {}

    def row(domain, key, partition, eval_key, measure, value):
        rows.append(dict(model=model_name, step=step, domain=domain, key=key, partition=partition,
                         eval_key=eval_key, measure=measure, value=float(value)))

    for domain in conn['domains']:
        t0 = time.time()
        if verbose:
            stderr('%s step %s: %s\n' % (model_name, step, domain))
        dkw = domain_data_kwargs(domain, conn.get('data_kwargs'))
        dkw['tokenizer'] = tokenizer
        input_ids, attention_mask = get_dataset(
            n_tokens=n_tokens * n_samples, split=conn.get('split', 'train'), take=conn.get('take', 100000),
            seq_len=seq_len, wrap=conn.get('wrap', True), shuffle=conn.get('shuffle', True),
            seed=derive_seed(seed, 'data', domain), verbose=False, **dkw)
        per_half = {h: dict(R=[], firing=[], cls_sum=None, cls_cnt=None, n=0) for h in HALF_NAMES}
        losses = []
        coordinates = None
        for k, (ids, mask) in enumerate(_samples(input_ids, attention_mask, n_samples)):
            h = HALF_NAMES[0] if k < n_samples // 2 else HALF_NAMES[1]
            out = get_timecourses(
                base, ids, mask, batch_size=batch_size, highpass=conn.get('highpass'),
                lowpass=conn.get('lowpass'), step=conn.get('step', 0.2), unit_type='mlp',
                units_per_layer=None, unit_seed=seed,
                seed=derive_seed(seed, 'timecourses', domain, k + 1), verbose=False)
            X = out['timecourses']
            coordinates = out['coordinates']
            del out   # else the sample's timecourses (14.5 GB at 160m) outlive the loop
            # Activation measures first: get_connectivity centres and normalises X in place.
            cls = token_classes(ids.numpy(), table)[mask.numpy().astype(bool)]
            onehot = np.zeros((len(cls), C), dtype=np.float32)
            onehot[np.arange(len(cls)), cls] = 1.0
            ph = per_half[h]
            ph['firing'].append((X > 0).sum(1).astype(np.float64))
            s_ = (X @ onehot).astype(np.float64)
            c_ = onehot.sum(0).astype(np.float64)
            ph['cls_sum'] = s_ if ph['cls_sum'] is None else ph['cls_sum'] + s_
            ph['cls_cnt'] = c_ if ph['cls_cnt'] is None else ph['cls_cnt'] + c_
            ph['n'] += X.shape[1]
            ph['R'].append(get_connectivity(X))
            del X
            lm.to(device)
            # Two sequences at a time: the logits are the largest GPU allocation of the job
            # (1.6 GB at a batch of 8 over a 50k vocabulary). Sum and count are unchanged.
            losses.append(lm_loss(lm, ids, mask, batch_size=min(batch_size, 2)))
            lm.to('cpu')
            torch.cuda.empty_cache()
        N = coordinates.shape[0]
        # Fixed subsample of units: the same units at every checkpoint (same seed, same N).
        sub = np.sort(np.random.RandomState(derive_seed(seed, 'dynamics_subsample') % (2 ** 32))
                      .choice(N, size=min(n_sub, N), replace=False))
        halves = {}
        for h in HALF_NAMES:
            R = fisher_average(*per_half[h]['R'], eps=conn.get('eps', 1e-3))
            per_half[h]['R'] = None
            halves[h] = R
        if keep_halves:
            halves_out[domain] = {h: halves[h].copy() for h in HALF_NAMES}
        # Per-unit activation file.
        with h5py.File(os.path.join(out_dir, 'units_step%d_%s.h5' % (step, domain)), 'w') as f:
            f.create_dataset('coordinates', data=coordinates)
            for h in HALF_NAMES:
                ph = per_half[h]
                f.create_dataset('firing_rate_%s' % h, data=np.sum(ph['firing'], 0) / ph['n'])
                with np.errstate(invalid='ignore', divide='ignore'):
                    f.create_dataset('class_mean_%s' % h, data=ph['cls_sum'] / ph['cls_cnt'][None, :])
                f.create_dataset('class_count_%s' % h, data=ph['cls_cnt'])
            f.create_dataset('loss_sum', data=np.array([l[0] for l in losses]))
            f.create_dataset('loss_count', data=np.array([l[1] for l in losses]))
            f.attrs['token_classes'] = ', '.join(TOKEN_CLASSES)
            f.attrs['model_name'] = model_name
            f.attrs['revision'] = '' if revision is None else str(revision)
        # Activation summaries as rows.
        for h in HALF_NAMES:
            fr = np.sum(per_half[h]['firing'], 0) / per_half[h]['n']
            row(domain, h, '-', h, 'act_firing_rate_median', np.median(fr))
            row(domain, h, '-', h, 'act_frac_dead', (fr < 1e-3).mean())
            row(domain, h, '-', h, 'act_frac_dense', (fr > 0.5).mean())
            row(domain, h, '-', h, 'act_firing_rate_gini', gini(fr))
        tot = sum(l[0] for l in losses)
        cnt = sum(l[1] for l in losses)
        row(domain, '-', '-', '-', 'lm_loss', tot / cnt)
        # Connectome measures, per half; segregation in sample and held out.
        for h in HALF_NAMES:
            R = halves[h]
            # In place where possible (Iteration 28, for 160m): with every unit alive the
            # eigenvalues are taken on R itself, with its diagonal set to 1; eigvalsh reads one
            # triangle only, so the ~1e-7 asymmetry of the tiled GPU product needs no
            # symmetrising copy. R is turned into |r| in place right after.
            live = np.isfinite(np.diag(R))
            Rl = R if live.all() else R[np.ix_(live, live)]
            np.nan_to_num(Rl, copy=False)
            np.fill_diagonal(Rl, 1.0)
            for name, v in spectrum_measures(eigenvalues(Rl)).items():
                row(domain, h, '-', h, name, v)
            row(domain, h, '-', h, 'n_live_units', int(live.sum()))
            del Rl
            A = np.nan_to_num(R, copy=False)
            np.abs(A, out=A)
            np.fill_diagonal(A, 0.0)
            for name, v in coupling_measures(A, seed=derive_seed(seed, 'dynamics_pairs', domain, h)).items():
                row(domain, h, '-', h, name, v)
            with h5py.File(os.path.join(out_dir, 'subsample_step%d_%s_%s.h5' % (step, domain, h)), 'w') as f:
                f.create_dataset('absr', data=A[np.ix_(sub, sub)].astype(np.float16))
                f.create_dataset('units', data=sub)
            halves[h] = A
        for h in HALF_NAMES:
            other = HALF_NAMES[1] if h == HALF_NAMES[0] else HALF_NAMES[0]
            for tree, r in (('real', root), ('null', null_root)):
                p = parc_path(r, variant, domain, h)
                if not os.path.exists(p):
                    continue
                with h5py.File(p, 'r') as f:
                    P = np.asarray(f['parcellation'])
                labels, k_ = hard_labels(P), P.shape[1]
                for eval_key in (h, other):
                    for name, v in segregation(halves[eval_key], labels, k_).items():
                        row(domain, h, tree, eval_key, name, v)
        del halves
        if verbose:
            stderr('  done in %.0f s\n' % (time.time() - t0))

    path = os.path.join(out_dir, 'connectome_step%d.csv' % step)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if verbose:
        stderr('wrote %s (%d rows)\n' % (path, len(rows)))
    return (rows, halves_out) if keep_halves else rows


# ---------------------------------------------------------------------------- partitions only

def jaccard_matrix(a, b, ka, kb):
    """Jaccard index between every network of labelling `a` and every network of `b`."""
    a, b = np.asarray(a), np.asarray(b)
    inter = np.zeros((ka, kb), dtype=np.float64)
    np.add.at(inter, (a, b), 1.0)
    sa, sb = np.bincount(a, minlength=ka).astype(np.float64), np.bincount(b, minlength=kb).astype(np.float64)
    union = sa[:, None] + sb[None, :] - inter
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(union > 0, inter / union, 0.0)


def partition_measures(P, coordinates, prefix=''):
    """Crispness, size distribution and depth profile of one soft partition."""
    P = np.asarray(P, dtype=np.float64)
    labels = hard_labels(P)
    k = P.shape[1]
    top = P.max(1)
    sizes = np.bincount(labels, minlength=k).astype(np.float64)
    nz = sizes[sizes > 0]
    p = nz / nz.sum()
    layers = np.asarray(coordinates)[:, 0].astype(int)
    L = int(layers.max()) + 1
    comp = np.zeros((k, L))
    np.add.at(comp, (labels, layers), 1.0)
    comp = comp[sizes > 0]
    frac = comp / comp.sum(1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        ent = -(np.where(frac > 0, frac * np.log(frac), 0.0)).sum(1)
    eff_layers = np.exp(ent)
    per_net_conf = np.array([top[labels == c].mean() for c in np.flatnonzero(sizes > 0)])
    out = {
        'crisp_median_max_membership': float(np.median(top)),
        'crisp_frac_confident': float((top > 0.5).mean()),
        'crisp_network_median': float(np.median(per_net_conf)),
        'size_n_networks': int((sizes > 0).sum()),
        'size_effective_n': float(np.exp(-(p * np.log(p)).sum())),
        'size_gini': gini(nz),
        'size_largest_share': float(nz.max() / nz.sum()),
        'size_frac_small': float((nz < 0.1 * nz.mean()).mean()),
        'depth_median_effective_layers': float(np.median(eff_layers)),
        'depth_frac_single_layer': float((frac.max(1) >= 0.8).mean()),
        'depth_mean_layer_span': float(((frac >= 0.1).sum(1)).mean()),
    }
    return {prefix + k_: v for k_, v in out.items()}


def cross_domain_generality(labels_by_domain, k, thr=0.3):
    """Per network, how well it is matched in the other domains (same half, same step).

    For each domain's networks, the best Jaccard with any network of each other domain,
    averaged over the other domains. Returns (per-domain mean generality, fraction of that
    domain's networks whose best match exceeds `thr` in every other domain).
    """
    domains = sorted(labels_by_domain)
    out = {}
    for d in domains:
        best = []
        for e in domains:
            if e == d:
                continue
            J = jaccard_matrix(labels_by_domain[d], labels_by_domain[e], k, k)
            best.append(J.max(1))
        best = np.stack(best)                      # (n_other, k)
        present = np.bincount(labels_by_domain[d], minlength=k) > 0
        out[d] = dict(generality_mean=float(best.mean(0)[present].mean()),
                      generality_frac_general=float((best.min(0) > thr)[present].mean()))
    return out


def transition_measures(a, b, k, share=0.2, thr=0.5):
    """How the networks of labelling `a` (step t) turn into those of `b` (step t + 1).

    split_degree  per network of a, how many networks of b receive at least `share` of its units
    merge_degree  per network of b, how many networks of a contribute at least `share` of its units
    continued     fraction of a's networks with a reciprocal best Jaccard match above `thr`
    """
    a, b = np.asarray(a), np.asarray(b)
    inter = np.zeros((k, k))
    np.add.at(inter, (a, b), 1.0)
    sa, sb = inter.sum(1), inter.sum(0)
    pa, pb = sa > 0, sb > 0
    with np.errstate(invalid='ignore', divide='ignore'):
        to_b = inter / sa[:, None]
        from_a = inter / sb[None, :]
    J = jaccard_matrix(a, b, k, k)
    best_ab, best_ba = J.argmax(1), J.argmax(0)
    recip = np.array([best_ba[best_ab[i]] == i and J[i, best_ab[i]] > thr for i in range(k)])
    return dict(
        trans_split_degree=float((to_b >= share).sum(1)[pa].mean()),
        trans_merge_degree=float((from_a >= share).sum(0)[pb].mean()),
        trans_continued=float(recip[pa].mean()),
        trans_mean_best_jaccard=float(J.max(1)[pa].mean()),
    )


def birth_steps(labels_by_step, k, thr=0.5):
    """For each network of the final step, its best Jaccard at every step and its birth step.

    Birth is the earliest step from which the network is matched above `thr` at every later
    step (the final step always matches itself). Returns (tracks: final network x step best
    Jaccard, births: per final network, the birth step).
    """
    steps = sorted(labels_by_step)
    final = labels_by_step[steps[-1]]
    tracks = np.zeros((k, len(steps)))
    for j, s in enumerate(steps):
        tracks[:, j] = jaccard_matrix(final, labels_by_step[s], k, k).max(1)
    births = []
    for c in range(k):
        above = tracks[c] > thr
        b = len(steps) - 1
        while b > 0 and above[b - 1]:
            b -= 1
        births.append(steps[b])
    present = np.bincount(final, minlength=k) > 0
    return tracks[present], np.asarray(births)[present], steps


def network_selectivity(class_means, labels, k, classes, thr=0.5):
    """How consistently the units of each network prefer the same token classes.

    Each unit's mean activation on the token classes is z-scored ACROSS the classes, which
    keeps the shape of its preference and drops its overall level and scale. A network's
    profile is the mean of its units' z-scored vectors; its coherence is the profile's norm
    over sqrt(C), 1 when every unit has the same preference shape and near 0 when their
    preferences are unrelated. Classes absent from the text are left out. Returns the median
    coherence over networks, the fraction above `thr`, and the fraction of networks whose
    profile peaks on each class.
    """
    M = np.asarray(class_means, dtype=np.float64)
    ok = np.isfinite(M).all(0)
    M = M[:, ok]
    names = [c for c, o in zip(classes, ok) if o]
    mu, sd = M.mean(1, keepdims=True), M.std(1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        Z = np.where(sd > 0, (M - mu) / sd, 0.0)
    labels = np.asarray(labels)
    present = np.flatnonzero(np.bincount(labels, minlength=k) > 0)
    prof = np.stack([Z[labels == c].mean(0) for c in present])
    coh = np.linalg.norm(prof, axis=1) / np.sqrt(Z.shape[1])
    out = {'selectivity_coherence_median': float(np.median(coh)),
           'selectivity_frac_coherent': float((coh > thr).mean())}
    pref = prof.argmax(1)
    for j, name in enumerate(names):
        out['selectivity_prefers_%s' % name] = float((pref == j).mean())
    return out
