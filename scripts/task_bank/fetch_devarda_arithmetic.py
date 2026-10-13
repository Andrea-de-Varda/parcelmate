"""Fetch de Varda et al. arithmetic dataset repo at a pinned commit, re-run its generator (seed 42, n=10000) into raw/devarda_arithmetic/generated/, check it reproduces the committed data, and write a sha256 manifest.

Third-party code is run with `python -I`, cwd = the output directory (the generator writes data/json/ relative to cwd).
"""
import hashlib, json, os, subprocess, sys, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, 'external', 'task_bank', 'raw', 'devarda_arithmetic')
REPO = os.path.join(RAW, 'repo')
GEN = os.path.join(RAW, 'generated')
URL = 'https://github.com/Andrea-de-Varda/arithmetic-dataset'
SHA = '19d5bcf5b31aa1d8c37ff3161eaba782d1a7a8fb'


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def run(*a, **k):
    return subprocess.run(a, check=True, capture_output=True, text=True, **k).stdout.strip()


def main():
    os.makedirs(RAW, exist_ok=True)
    if not os.path.isdir(REPO):
        run('git', 'clone', '-q', URL, REPO)
    run('git', '-C', REPO, 'checkout', '-q', SHA)
    assert run('git', '-C', REPO, 'rev-parse', 'HEAD') == SHA
    os.makedirs(GEN, exist_ok=True)
    if not os.path.exists(os.path.join(GEN, 'data', 'json', 'arith_dataset_numeric.json')):
        subprocess.run([sys.executable, '-I', os.path.join(REPO, 'generate_arithmetic_data.py'), '-n', '10000', '--seed', '42'],
                       cwd=GEN, check=True, capture_output=True)
    manifest = {'url': URL, 'commit': SHA, 'download_date': datetime.date.today().isoformat(), 'files': {}, 'generator_reproduces_committed': {}}
    for fm in ['numeric', 'english', 'spanish', 'italian', 'embedded', 'embedded_verbal']:
        rel = f'data/json/arith_dataset_{fm}.json'
        manifest['files'][f'repo/{rel}'] = sha256(os.path.join(REPO, rel))
        manifest['files'][f'generated/{rel}'] = sha256(os.path.join(GEN, rel))
        manifest['generator_reproduces_committed'][fm] = (json.load(open(os.path.join(REPO, rel))) == json.load(open(os.path.join(GEN, rel))))
    for rel in ['generate_arithmetic_data.py', 'generate_arithmetic_counterfactual.py', 'eval_models.py', 'README.md']:
        manifest['files'][f'repo/{rel}'] = sha256(os.path.join(REPO, rel))
    json.dump(manifest, open(os.path.join(RAW, 'MANIFEST.json'), 'w'), indent=1)
    print(json.dumps(manifest['generator_reproduces_committed']))


if __name__ == '__main__':
    main()
