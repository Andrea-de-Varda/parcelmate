"""Every task in external/task_bank/eval_tasks.txt goes through the patching prompt builder.

    PYTHONPATH=. python tests/verify_task_bank_eval.py

Checks, with the GPT-2 tokenizer (no model): the config loads, format_prompts accepts it,
the item count matches the file, and prompt/answer fields are strings.
"""
from transformers import AutoTokenizer

from parcelmate.patching import bank_eval_list, format_prompts, load_bank_config, load_bank_task

tok = AutoTokenizer.from_pretrained('gpt2')
pairs = bank_eval_list()
fails = 0
for source, task in pairs:
    try:
        cfg = load_bank_config(source)[task]
        data = load_bank_task(source, cfg)
        fmt = format_prompts(data, cfg, tok)
        assert len(fmt['clean_prompts']) == len(fmt['corrupted_prompts']) == len(data) > 0
        assert all(isinstance(a, str) for a in fmt['clean_correct'] + fmt['clean_incorrect'])
        assert fmt['force_base'], 'bank tasks are raw text'
    except Exception as e:
        fails += 1
        print('FAIL %s/%s: %r' % (source, task, e))
print('%d tasks, %d failures' % (len(pairs), fails))
assert not fails
