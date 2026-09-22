"""Verification for the Iteration 23 additions (minimal-pair evaluation and attribution).

    PYTHONPATH=. python tests/verify_iter13_patching.py [path/to/LLM_Modularity]

Covers: prompt formatting and tokenization equal to the original repository's
`format_prompts` / `prepare_sequence_inputs` (imported from a clone if one is given or found
at ../LLM_Modularity, otherwise skipped); answer log-probs equal to its
`compute_sequence_log_prob`; both-correct, alignment filter and baselines by hand on a tiny
GPT-2 with the real GPT-2 tokenizer; the attribution against an explicit autograd
computation on the same model; the LFM2 hook; and the BOS switch.
"""

import json
import os
import sys
import tempfile

import numpy as np
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, Lfm2Config, Lfm2ForCausalLM

from parcelmate.model import mlp_projections
from parcelmate.patching import (
    DATA_ROOT, answer_log_prob, encode, evaluate_task, format_prompts, load_domain_config,
    load_task, neuron_attribution, sequence_log_probs,
)

failures = []
n_checks = [0]
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')


def check(name, cond):
    n_checks[0] += 1
    print('%s %s' % ('OK' if cond else 'FAIL', name))
    if not cond:
        failures.append(name)


torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained('gpt2')
gpt2 = GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=len(tok),
                                  n_positions=256)).eval()
for p in gpt2.parameters():
    p.requires_grad_(False)

# ---------------------------------------------------------------- against the original code
orig = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, '..', 'LLM_Modularity')
if os.path.isdir(os.path.join(orig, 'src')):
    sys.path.insert(0, orig)
    from src import data_utils as odu, metrics as om   # noqa: E402
    ok = True
    for domain in ('Lan', 'MD', 'phys', 'ToM'):
        cfg = load_domain_config(domain)
        for task, tc in list(cfg.items())[:3]:
            data = load_task(domain, tc)[:5]
            mine = format_prompts(data, tc, tok, bos=False)
            theirs = odu.format_prompts(data, tc, 'gpt2', tok)
            ok &= mine['clean_prompts'] == theirs['clean_prompts']
            ok &= mine['corrupted_prompts'] == theirs['corrupted_prompts']
            ok &= mine['clean_correct'] == theirs['clean_correct']
            ok &= mine['clean_incorrect'] == theirs['clean_incorrect']
            ids, pl = encode(tok, mine['clean_prompts'], mine['clean_correct'])
            tids, tpl = odu.prepare_sequence_inputs(theirs['clean_prompts'], theirs['clean_correct'], tok)
            ok &= pl == tpl and all(torch.equal(a, b) for a, b in zip(ids, tids))
    check('formatting and tokenization equal the original on 12 tasks x 5 items', ok)
    ids, pl = encode(tok, mine['clean_prompts'], mine['clean_correct'])
    with torch.no_grad():
        logits = gpt2(input_ids=ids[0][None]).logits[0]
    check('answer log-prob equals the original compute_sequence_log_prob',
          abs(answer_log_prob(logits, ids[0], pl[0]).item()
              - om.compute_sequence_log_prob(logits, ids[0], pl[0]).item()) < 1e-5)
else:
    print('SKIP original-code comparison (no clone at %s)' % orig)

# ---------------------------------------------------------------- evaluation by hand
cfg = load_domain_config('Lan')
tc = cfg['subject_verb_agreement']
data = load_task('Lan', tc)[:24]
b = evaluate_task(gpt2, tok, data, tc, bos=False, batch_size=7, verbose=False)
fmt = format_prompts(data, tc, tok)
by_hand = {}
for name, pk, ak in (('cc', 'clean_prompts', 'clean_correct'), ('ci', 'clean_prompts', 'clean_incorrect'),
                     ('rc', 'corrupted_prompts', 'clean_correct'), ('ri', 'corrupted_prompts', 'clean_incorrect')):
    vals = []
    for i in b['per_item']['aligned_indices']:
        ids, pl = encode(tok, [fmt[pk][i]], [fmt[ak][i]])
        with torch.no_grad():
            lg = gpt2(input_ids=ids[0][None]).logits[0]
        vals.append(answer_log_prob(lg, ids[0], pl[0]).item())
    by_hand[name] = np.array(vals)
check('per-item scores equal unbatched single-item forward passes (batched, padded)',
      all(np.allclose(by_hand[k], b['per_item'][v], atol=1e-4) for k, v in
          (('cc', 'clean_correct_lp'), ('ci', 'clean_incorrect_lp'),
           ('rc', 'corrupted_correct_lp'), ('ri', 'corrupted_incorrect_lp'))))
cc, ci, rc, ri = (by_hand[k] for k in ('cc', 'ci', 'rc', 'ri'))
both = (cc > ci) & (ri > rc)
check('both-correct = clean prefers correct AND corrupted prefers incorrect',
      b['n_both_correct'] == int(both.sum())
      and b['both_correct_indices'] == [b['per_item']['aligned_indices'][i] for i in np.flatnonzero(both)])
check('accuracies over the aligned items',
      abs(b['clean_accuracy'] - (cc > ci).mean()) < 1e-12
      and abs(b['corrupted_accuracy'] - (ri > rc).mean()) < 1e-12)
if both.any():
    check('baselines are means over the both-correct items only',
          abs(b['clean_baseline'] - (cc - ci)[both].mean()) < 1e-9
          and abs(b['corrupted_baseline'] - (rc - ri)[both].mean()) < 1e-9)
cl = [len(tok(p, add_special_tokens=False)['input_ids']) for p in fmt['clean_prompts']]
co = [len(tok(p, add_special_tokens=False)['input_ids']) for p in fmt['corrupted_prompts']]
check('alignment filter drops items whose prompts differ in token count',
      b['token_misaligned_indices'] == [i for i in range(len(data)) if cl[i] != co[i]]
      and b['n_after_alignment_filter'] + b['n_token_misaligned'] == len(data))

# BOS switch: raw-text task gets the BOS token; a chat task does not (gpt2 has no template).
f_bos = format_prompts(data, tc, tok, bos=True)
check('bos=True prepends the BOS token to raw-text prompts',
      all(p.startswith(tok.bos_token) for p in f_bos['clean_prompts'])
      and not any(p.startswith(tok.bos_token) for p in fmt['clean_prompts']))
md = load_domain_config('MD')['add_sub_2op_symbolic']
f_md = format_prompts(load_task('MD', md)[:2], md, tok, bos=True)
check('a chat-template task ignores bos (GPT-2 has no template: raw text, no BOS)',
      f_md['clean_prompts'][0] == 'What is the answer of 862 - 158 =? Please directly give the final answer written in digits.'
      and not f_md['force_base'])

# ---------------------------------------------------------------- attribution by hand
if both.sum() >= 2:
    attr = neuron_attribution(gpt2, tok, data, tc, b, bos=False, batch_size=3, verbose=False)
    check('attribution shape is (n_layers, width)', attr.shape == (2, 128))
    # Explicit computation: one item at a time, activations via hooks on c_proj input.
    keep = b['both_correct_indices']
    ref = np.zeros((2, 128))
    inc_all, cor_all = [], []
    scale = b['clean_baseline'] - b['corrupted_baseline']
    projections = mlp_projections(gpt2)
    acts = {}

    def hook(l):
        def f(m, inp):
            if not inp[0].requires_grad:
                inp[0].requires_grad_(True)
            inp[0].retain_grad()
            acts[l] = inp[0]
        return f
    # The metric is a batch mean; with batch_size=3 the gradient of the mean over a batch
    # equals 1/B times the per-item gradient, so recompute with the same batching.
    for s in range(0, len(keep), 3):
        batch = keep[s:s + 3]
        B = len(batch)
        per_item = []
        for i in batch:
            ids_ri, pl_ri = encode(tok, [fmt['corrupted_prompts'][i]], [fmt['clean_incorrect'][i]])
            with torch.no_grad():
                inc = answer_log_prob(gpt2(input_ids=ids_ri[0][None]).logits[0], ids_ri[0], pl_ri[0])
            ids_c, pl_c = encode(tok, [fmt['clean_prompts'][i]], [fmt['clean_correct'][i]])
            hs = [p.register_forward_pre_hook(hook(l)) for l, p in projections]
            with torch.no_grad():
                gpt2(input_ids=ids_c[0][None])
            for h in hs:
                h.remove()
            clean = {l: acts[l][0, pl_c[0] - 1].detach().clone() for l, _ in projections}
            acts.clear()
            ids_r, pl_r = encode(tok, [fmt['corrupted_prompts'][i]], [fmt['clean_correct'][i]])
            hs = [p.register_forward_pre_hook(hook(l)) for l, p in projections]
            with torch.enable_grad():
                cor = answer_log_prob(gpt2(input_ids=ids_r[0][None]).logits[0], ids_r[0], pl_r[0])
                ((cor - inc) / B / scale).backward()
            for h in hs:
                h.remove()
            for l, _ in projections:
                g = acts[l].grad[0, pl_r[0] - 1]
                r = acts[l][0, pl_r[0] - 1].detach()
                ref[l] += (g * (clean[l] - r)).numpy()
            acts.clear()
    check('attribution equals an explicit per-item autograd computation',
          np.allclose(attr, ref, atol=1e-5, rtol=1e-4))
else:
    print('SKIP attribution check (random model produced < 2 both-correct items)')

# ---------------------------------------------------------------- LFM2 hook
c = Lfm2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
               num_key_value_heads=1, vocab_size=len(tok), layer_types=['conv', 'full_attention'])
lfm = Lfm2ForCausalLM(c).eval()
proj = mlp_projections(lfm)
check('LFM2: every block (conv and attention) has an MLP projection; causal-LM wrapper unwrapped',
      [l for l, _ in proj] == [0, 1] and all(p is lfm.model.layers[l].feed_forward.w2 for l, p in proj))
seen, ff_in = {}, {}
hs = [p.register_forward_pre_hook(lambda m, inp, l=l: seen.__setitem__(l, inp[0].detach())) for l, p in proj]
hs += [lfm.model.layers[l].feed_forward.register_forward_pre_hook(
    lambda m, inp, l=l: ff_in.__setitem__(l, inp[0].detach())) for l, _ in proj]
x = torch.randint(0, 100, (1, 6))
with torch.no_grad():
    lfm(input_ids=x)
for hh in hs:
    hh.remove()
ok = True
for l, _ in proj:
    ff = lfm.model.layers[l].feed_forward
    with torch.no_grad():
        gated = torch.nn.functional.silu(ff.w1(ff_in[l])) * ff.w3(ff_in[l])
    ok &= seen[l].shape[-1] == ff.w2.in_features and torch.allclose(seen[l], gated, atol=1e-6)
check('LFM2: the hooked quantity is the gated activation silu(w1 x) * w3 x in every block', ok)

print('\n%d checks, %d failure(s)' % (n_checks[0], len(failures)))
for f in failures:
    print('  FAIL', f)
sys.exit(1 if failures else 0)
