"""Fetch 'Do Language Models Track Entities Across State Changes?' (Tang et al., ICML 2026; arXiv 2605.30233) at a pinned commit of PootieT/entity-tracking-mi
and run its box-data generator (itself derived from Kim & Schuster 2023's generator; the Kim & Schuster password-protected data is NOT used) with seed 42.

The generator is untrusted third-party code: it was read in full before running (only local file I/O; no network, subprocess or eval). It is run from a patched copy
because the original fails on Python >= 3.11 (random.sample on a set); the two patched call sites sort the set first (which also makes the output independent of
set iteration order). PYTHONHASHSEED=0 is fixed because list(set_of_strings) is still used elsewhere. The three parameter sets are the paper's own shell scripts
(dataset_generation/boxes_AltForm_*.sh), with --seed 42 added (their scripts use the default 2255). Run under `python -s -P` (not -I, which would ignore PYTHONHASHSEED)."""
import os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knowledge_common as kc

SID, URL, SHA = 'entity_tracking_mi', 'https://github.com/PootieT/entity-tracking-mi', '21b1304872708fb85b21cf86730bd7df3d73da73'
# name -> (paper script, extra generator args)
RUNS = {
    'boxes_AltForm_1remove': ('boxes_AltForm_1remove.sh', ['--num_samples', '7000', '--allowed_operations', 'remove', '--num_operations', '1']),
    'boxes_AltForm_1move': ('boxes_AltForm_1move.sh', ['--num_samples', '5000', '--allowed_operations', 'move', '--num_operations', '1']),
    'boxes_AltForm_1put_1fixObj': ('boxes_AltForm_1put_1fixObj.sh', ['--num_samples', '5000', '--allowed_operations', 'put', '--num_operations', '1', '--fix_object_count_per_phrase', '1']),
}
COMMON = ['--alternative_forms', 'always', '--seed', '42', '--cft_formats', '']

def main():
    d = kc.git_checkout(SID, 'repo', URL, SHA)
    gen_dir = os.path.join(kc.RAW, SID, 'generated')
    os.makedirs(gen_dir, exist_ok=True)
    orig = os.path.join(d, 'dataset_generation', 'generate_boxes_data_modified.py')
    patched = os.path.join(gen_dir, 'generate_boxes_data_patched.py')
    src = open(orig).read()
    p1 = 'random.sample(self.boxes[box], len(self.boxes[box]))'
    p2 = 'random.sample(contents, len(contents))'
    assert src.count(p1) == 1 and src.count(p2) == 1
    src = src.replace(p1, 'random.sample(sorted(self.boxes[box]), len(self.boxes[box]))').replace(p2, 'random.sample(sorted(contents), len(contents))')
    with open(patched, 'w') as f:
        f.write(src)
    vocab = os.path.join(d, 'data', 'objects', 'llama_friendly_objects.csv')
    files = [orig, patched, vocab, os.path.join(d, 'LICENSE')]
    cmds = {}
    for name, (script, args) in RUNS.items():
        out = os.path.join(gen_dir, name)
        outf = os.path.join(out, 'train-t5.jsonl')
        cmd = [sys.executable, '-s', '-P', patched, '--object_vocabulary_file', vocab, '--output_dir', out] + args + COMMON
        cmds[name] = dict(paper_script='dataset_generation/' + script, command=' '.join(cmd[2:]).replace(kc.REPO, '<repo>'), env='PYTHONHASHSEED=0')
        if not os.path.exists(outf):
            env = dict(os.environ, PYTHONHASHSEED='0')
            subprocess.run(cmd, check=True, cwd=gen_dir, env=env, stdout=subprocess.DEVNULL)
        files.append(outf)
    kc.write_manifest(SID, dict(source_id=SID, url=URL, revision=SHA, generator_commit=SHA, generator_seed=42, generator_runs=cmds,
                                note='raw_files includes the frozen generator outputs (generated/*/train-t5.jsonl)'), files)
    print('ok', SID)

if __name__ == '__main__':
    main()
