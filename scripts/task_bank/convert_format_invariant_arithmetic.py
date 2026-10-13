"""format_invariant_arithmetic (arXiv 2607.16693; SharathHN/format-invariant-arithmetic): registry entry only, NO conversion.

Reason: the repository has no LICENSE file and its README states none, so redistribution of converted items is unclear and the plan keeps such sources out of git. The repo's datasets are model-conditioned (Llama-3 correct prompts, .pkl files, not unpickled here) rather than a fixed release; the prompt templates (src/prompts.py) are read and described in REPORT_arithmetic.md.
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _arith_common as C

SID = 'format_invariant_arithmetic'
man = json.load(open(os.path.join(C.TB, 'raw', SID, 'MANIFEST.json')))
key = {p: s for p, s in man['files'].items() if p.split('/')[-1] in ('README.md', 'prompts.py', 'config.py', '01_datasets.py', '.gitignore')}
C.write_yaml(os.path.join(C.TB, 'registry', f'{SID}.yaml'), dict(
    source_id=SID, title='Are Arithmetic Heuristic Neurons Form-Invariant? A Mechanistic Analysis of Symbols, Text, and Code in LLMs',
    bibkey='naganna2026formInvariant', bibkey_data=None, url=man['url'], revision=man['commit'], download_date=man['download_date'],
    raw_files=[dict(path=f'external/task_bank/raw/{SID}/{p}', sha256=s) for p, s in key.items()] + [dict(path=f'external/task_bank/raw/{SID}/MANIFEST.json', sha256=C.sha256(os.path.join(C.TB, 'raw', SID, 'MANIFEST.json')), note='sha256 of all 145 tracked files')],
    licence='unclear: no LICENSE file and no licence statement in the README (the arXiv page lists CC BY 4.0 for the paper only)', licence_source='absence of LICENSE at the pinned commit; arXiv abs page for the paper', redistribute='no (unclear, treated as no)',
    tier=1, tasks=[], status='unverified',
    notes=dict(authors='arXiv: Sharath Naganna, Tanvir Ahmed Sijan, Uddipta Kalita (2026, "Under Review"); repo commits by SharathHN and sijantanvir, consistent.',
               fit='three formats of the Nikankin prompts (plain "x+y=", Python script ending "Ans = ", word problem "Tom has X apples and Alice gives him Y apples. So he now has "), operators + - * /, operands up to 999 (Llama-3 single-token range), 200 prompts per operator and format, model-conditioned (Llama-3 8B/3B/1B correct prompts); no corruption pairs in the data (patching pairs are random other prompts). A flipping variant would be easy to build from the templates (random same-operator partner with a different answer, as for nikankin_arithmetic) and the code and word formats are not covered by the other sources, so it is worth asking the authors for a licence or re-implementing the three templates with permission.',
               neuron_lists='data/<model>/importance/*.pt and heuristics JSON are committed (Llama-3 8B/3B/1B, per format and operator): candidate neuron lists for the later neuron-list pass, licence permitting; Llama tokenizer, not Qwen.',
               pickles='datasets/*.pkl were not unpickled (untrusted); JSON versions exist only for datasets_wrong of llama1b.')))
