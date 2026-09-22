"""Task evaluation and attribution patching on minimal-pair tasks (LOG.md Iteration 23).

A plain-PyTorch re-implementation of the evaluation and neuron-attribution code of
Pengrui Han's LLM_Modularity repository (data and task configs vendored verbatim under
external/llm_modularity/). Kept behaviourally identical where it matters, and verified
against that code in tests/verify_iter13_patching.py:

  prompts    prefix + problem + concatenation + " " + suffix(answers), through the chat
             template (system "" + user, generation prompt) unless the task sets
             `force_base_evaluation`, in which case the raw text is used;
  scoring    full-sequence teacher forcing: the sum of log P(answer token | context) over
             the answer tokens, with the CLEAN answers used under both prompts;
  correct    an item is both-correct when the clean prompt prefers the correct answer
             AND the corrupted prompt prefers the incorrect one;
  filter     items whose clean and corrupted prompts tokenize to different lengths are
             dropped before anything is scored (their token-alignment filter);
  metric     normalized teacher-forcing difference, (d - d_corr) / (d_clean - d_corr),
             with both baselines the means over the both-correct items;
  attribution  for each MLP neuron, grad of the metric at the last prompt token of the
             corrupted+correct run times (clean activation - corrupted activation) there,
             summed over items.

Two deliberate differences from the original. Models run in float32 (they used fp16 on
24B-123B models). And raw-text tasks may be given the model's BOS token (`bos=True`),
which the original never adds; GPT-2 has no BOS, but LFM2.5 expects one. The choice is
recorded in every output.

The MLP hook is the same one connectivity uses (`model.MLP_PROJECTIONS`): a forward
pre-hook on the MLP's output projection, whose input is the post-nonlinearity neuron
activation. For gated MLPs (Llama-style, LFM2) that is act(w1 x) * w3 x, exactly what the
original hooks as `mlp.down_proj.input`.
"""

import json
import os
import time

import numpy as np
import torch

from parcelmate.util import stderr

DOMAINS = ('Lan', 'MD', 'phys', 'ToM')
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'external', 'llm_modularity')


# ---------------------------------------------------------------------------- tasks

def load_domain_config(domain, root=DATA_ROOT):
    with open(os.path.join(root, 'config', domain, 'config.json')) as f:
        return json.load(f)


def load_task(domain, task_config, root=DATA_ROOT):
    with open(os.path.join(root, 'data', domain, task_config['path'])) as f:
        return json.load(f)


def chat_prompt(tokenizer, message):
    """The original's template: an empty system turn, the user turn, a generation prompt."""
    if getattr(tokenizer, 'chat_template', None) is None:
        return message
    return tokenizer.apply_chat_template(
        [{'role': 'system', 'content': ''}, {'role': 'user', 'content': message}],
        tokenize=False, add_generation_prompt=True)


def format_prompts(data, task_config, tokenizer, bos=False):
    """Clean and corrupted prompts and the CLEAN answer pair, as the original builds them.

    Returns dict(clean_prompts, corrupted_prompts, clean_correct, clean_incorrect,
    force_base). `bos` prepends the tokenizer's BOS token to raw-text prompts only; chat
    templates supply their own.
    """
    assert not task_config.get('single_answer'), 'single-answer tasks are not supported'
    prefix = task_config.get('prefix') or ''
    suffix_t = task_config.get('suffix') or ''
    concat = task_config.get('concatenation') or ''
    force_base = bool(task_config.get('force_base_evaluation', False))
    clean_prompts, corrupted_prompts = [], []
    for item in data:
        clean = prefix + item['clean_problem'] + concat
        if suffix_t:
            clean += ' ' + suffix_t.format(correct_answer=str(item['clean_correct']),
                                           incorrect_answer=str(item['clean_incorrect']))
        corrupted = prefix + item['corrupted_problem'] + concat
        if suffix_t:
            corrupted += ' ' + suffix_t.format(correct_answer=str(item['corrupted_correct']),
                                               incorrect_answer=str(item['corrupted_incorrect']))
        if force_base:
            if bos and tokenizer.bos_token:
                clean, corrupted = tokenizer.bos_token + clean, tokenizer.bos_token + corrupted
        else:
            clean, corrupted = chat_prompt(tokenizer, clean), chat_prompt(tokenizer, corrupted)
        clean_prompts.append(clean)
        corrupted_prompts.append(corrupted)
    return dict(clean_prompts=clean_prompts, corrupted_prompts=corrupted_prompts,
                clean_correct=[item['clean_correct'] for item in data],
                clean_incorrect=[item['clean_incorrect'] for item in data],
                force_base=force_base)


def encode(tokenizer, prompts, answers):
    """Prompt + answer token ids per item (no special tokens added), and the prompt lengths."""
    ids, plens = [], []
    for p, a in zip(prompts, answers):
        pi = tokenizer(p, add_special_tokens=False)['input_ids']
        ai = tokenizer(str(a), add_special_tokens=False)['input_ids']
        ids.append(torch.tensor(pi + ai, dtype=torch.long))
        plens.append(len(pi))
    return ids, plens


def prompt_lengths(tokenizer, prompts):
    return [len(tokenizer(p, add_special_tokens=False)['input_ids']) for p in prompts]


# ---------------------------------------------------------------------------- scoring

def pad(ids, value=0):
    out = torch.full((len(ids), max(x.numel() for x in ids)), value, dtype=torch.long)
    for i, x in enumerate(ids):
        out[i, :x.numel()] = x
    return out


def answer_log_prob(logits, ids, plen):
    """Sum of log P(token_i | tokens_<i) over the answer tokens of one item."""
    lp = torch.log_softmax(logits[plen - 1:ids.numel() - 1].float(), dim=-1)
    return lp.gather(1, ids[plen:].unsqueeze(1)).sum()


def sequence_log_probs(model, ids, plens, batch_size=16, device=None, pad_id=0):
    """Per-item answer log probabilities under `model`, no gradient."""
    device = device or next(model.parameters()).device
    out = []
    for s in range(0, len(ids), batch_size):
        batch = ids[s:s + batch_size]
        x = pad(batch, pad_id).to(device)
        with torch.no_grad():
            logits = model(input_ids=x).logits
        for i, item in enumerate(batch):
            out.append(answer_log_prob(logits[i], item.to(device), plens[s + i]).item())
    return np.asarray(out, dtype=np.float64)


def evaluate_task(model, tokenizer, data, task_config, bos=False, batch_size=16, verbose=True):
    """Per-item correctness and the metric baselines for one task.

    Mirrors run_attribution.py's baseline block: the alignment filter first, then the four
    teacher-forced scores on the aligned items, both-correct as defined above, and the
    clean/corrupted baselines as means over the both-correct items. Returns a dict in the
    original baselines.json schema plus the per-item scores.
    """
    fmt = format_prompts(data, task_config, tokenizer, bos=bos)
    n_total = len(data)
    cl, co = prompt_lengths(tokenizer, fmt['clean_prompts']), prompt_lengths(tokenizer, fmt['corrupted_prompts'])
    misaligned = [i for i in range(n_total) if cl[i] != co[i]]
    aligned = [i for i in range(n_total) if cl[i] == co[i]]
    pad_id = tokenizer.pad_token_id or 0
    scores = {}
    for name, prompts_key, answers_key in (('clean_correct', 'clean_prompts', 'clean_correct'),
                                           ('clean_incorrect', 'clean_prompts', 'clean_incorrect'),
                                           ('corrupted_correct', 'corrupted_prompts', 'clean_correct'),
                                           ('corrupted_incorrect', 'corrupted_prompts', 'clean_incorrect')):
        ids, plens = encode(tokenizer, [fmt[prompts_key][i] for i in aligned],
                            [fmt[answers_key][i] for i in aligned])
        t0 = time.time()
        scores[name] = sequence_log_probs(model, ids, plens, batch_size=batch_size, pad_id=pad_id)
        if verbose:
            stderr('    %-20s %d items, %.0f s\n' % (name, len(ids), time.time() - t0))
    cc, ci, rc, ri = (scores[k] for k in ('clean_correct', 'clean_incorrect',
                                          'corrupted_correct', 'corrupted_incorrect'))
    clean_ok = cc > ci
    corrupted_ok = ri > rc
    both = clean_ok & corrupted_ok
    bc = np.flatnonzero(both)
    out = dict(
        n_total=n_total,
        n_token_misaligned=len(misaligned),
        token_misaligned_indices=misaligned,
        n_after_alignment_filter=len(aligned),
        clean_accuracy=float(clean_ok.mean()) if len(aligned) else float('nan'),
        corrupted_accuracy=float(corrupted_ok.mean()) if len(aligned) else float('nan'),
        both_correct_accuracy=float(both.mean()) if len(aligned) else float('nan'),
        n_both_correct=int(both.sum()),
        both_correct_indices=[aligned[i] for i in bc],
        not_both_correct_indices=[aligned[i] for i in np.flatnonzero(~both)],
        clean_baseline=float((cc[bc] - ci[bc]).mean()) if len(bc) else 0.0,
        corrupted_baseline=float((rc[bc] - ri[bc]).mean()) if len(bc) else 0.0,
        # Unfiltered accuracies as run_eval.py reports them, for comparison with tables
        # computed that way; identical to the above when nothing is misaligned.
        clean_tf_diff=float((cc - ci).mean()) if len(aligned) else float('nan'),
        corrupted_tf_diff=float((rc - ri).mean()) if len(aligned) else float('nan'),
        force_base=fmt['force_base'],
        bos=bool(bos and fmt['force_base']),
        example=dict(clean=fmt['clean_prompts'][0] + str(fmt['clean_correct'][0]),
                     corrupted=fmt['corrupted_prompts'][0] + str(fmt['clean_correct'][0])),
        per_item=dict(aligned_indices=aligned, clean_correct_lp=cc.tolist(),
                      clean_incorrect_lp=ci.tolist(), corrupted_correct_lp=rc.tolist(),
                      corrupted_incorrect_lp=ri.tolist()),
    )
    return out


# ---------------------------------------------------------------------------- attribution

def neuron_attribution(model, tokenizer, data, task_config, baselines, bos=False,
                       batch_size=8, verbose=True):
    """Attribution of the normalized metric to every MLP neuron (n_layers x width).

    For each batch of both-correct items: corrupted+incorrect (no grad) for the incorrect
    log-probs; clean+correct with the neuron activations captured; corrupted+correct with
    the activations captured and requiring grad, the metric backpropagated to them. The
    attribution at the last prompt token is grad * (clean - corrupted), summed over items.
    """
    from parcelmate.model import mlp_projections
    fmt = format_prompts(data, task_config, tokenizer, bos=bos)
    keep = baselines['both_correct_indices']
    assert keep, 'no both-correct items'
    pad_id = tokenizer.pad_token_id or 0
    device = next(model.parameters()).device
    cln_ids, cln_pl = encode(tokenizer, [fmt['clean_prompts'][i] for i in keep],
                             [fmt['clean_correct'][i] for i in keep])
    rc_ids, rc_pl = encode(tokenizer, [fmt['corrupted_prompts'][i] for i in keep],
                           [fmt['clean_correct'][i] for i in keep])
    ri_ids, ri_pl = encode(tokenizer, [fmt['corrupted_prompts'][i] for i in keep],
                           [fmt['clean_incorrect'][i] for i in keep])
    scale = baselines['clean_baseline'] - baselines['corrupted_baseline']
    assert scale != 0, 'clean and corrupted baselines coincide; the metric is undefined'
    projections = mlp_projections(getattr(model, 'base_model', model))
    captured = {}

    def make_hook(layer, grad):
        def hook(module, inputs):
            x = inputs[0]
            if grad:
                # The parameters are frozen and the input is token ids, so nothing in
                # the forward pass requires grad until an activation is flagged. The
                # first hooked activation becomes a leaf; every later one depends on it
                # and is a non-leaf that can retain its grad (the original does the same
                # through nnsight's `requires_grad_(True)`).
                if not x.requires_grad:
                    x.requires_grad_(True)
                x.retain_grad()
            captured[layer] = x
        return hook

    attribution = None
    for s in range(0, len(keep), batch_size):
        b = slice(s, s + batch_size)
        # 1. corrupted + incorrect, no hooks, no grad
        x = pad(ri_ids[b], pad_id).to(device)
        with torch.no_grad():
            logits = model(input_ids=x).logits
            inc = torch.stack([answer_log_prob(logits[i], ri_ids[s + i].to(device), ri_pl[s + i])
                               for i in range(x.shape[0])])
        del logits
        # 2. clean + correct, capture activations
        hooks = [p.register_forward_pre_hook(make_hook(l, False)) for l, p in projections]
        x = pad(cln_ids[b], pad_id).to(device)
        with torch.no_grad():
            model(input_ids=x)
        for h in hooks:
            h.remove()
        clean_acts = {l: captured[l][torch.arange(x.shape[0]), torch.tensor(cln_pl[b]) - 1].clone()
                      for l, _ in projections}
        captured.clear()
        # 3. corrupted + correct, capture activations with grad, backprop the metric
        hooks = [p.register_forward_pre_hook(make_hook(l, True)) for l, p in projections]
        x = pad(rc_ids[b], pad_id).to(device)
        with torch.enable_grad():
            logits = model(input_ids=x).logits
            cor = torch.stack([answer_log_prob(logits[i], rc_ids[s + i].to(device), rc_pl[s + i])
                               for i in range(x.shape[0])])
            metric = ((cor - inc).mean() - baselines['corrupted_baseline']) / scale
            metric.backward()
        for h in hooks:
            h.remove()
        rows = torch.arange(x.shape[0])
        pos = torch.tensor(rc_pl[b]) - 1
        if attribution is None:
            attribution = np.zeros((len(projections), clean_acts[projections[0][0]].shape[-1]))
        for l, _ in projections:
            g = captured[l].grad[rows, pos]
            r = captured[l][rows, pos].detach()
            attribution[l] += (g * (clean_acts[l] - r)).sum(0).float().cpu().numpy()
        captured.clear()
        del logits, clean_acts
        model.zero_grad(set_to_none=True)
        if verbose:
            stderr('\r    attribution %d/%d' % (min(s + batch_size, len(keep)), len(keep)))
    if verbose:
        stderr('\n')
    return attribution
