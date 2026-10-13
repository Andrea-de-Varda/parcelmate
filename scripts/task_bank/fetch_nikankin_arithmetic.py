"""Clone https://github.com/technion-cs-nlp/llm-arithmetic-heuristics at the pinned commit into raw/nikankin_arithmetic/repo and write a sha256 manifest of every tracked file (raw is never edited)."""
import datetime, hashlib, json, os, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, 'external', 'task_bank', 'raw', 'nikankin_arithmetic')
REPO = os.path.join(RAW, 'repo')
URL, SHA = 'https://github.com/technion-cs-nlp/llm-arithmetic-heuristics', 'accf6b2bca7e1145520e96d0ceb60beb6b769c05'


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def run(*a):
    return subprocess.run(a, check=True, capture_output=True, text=True).stdout.strip()


os.makedirs(RAW, exist_ok=True)
if not os.path.isdir(REPO):
    run('git', 'clone', '-q', URL, REPO)
run('git', '-C', REPO, 'checkout', '-q', SHA)
assert run('git', '-C', REPO, 'rev-parse', 'HEAD') == SHA
files = run('git', '-C', REPO, 'ls-files').split('\n')
man = dict(url=URL, commit=SHA, download_date=datetime.date.today().isoformat(),
           files={f'repo/{f}': sha256(os.path.join(REPO, f)) for f in files})
json.dump(man, open(os.path.join(RAW, 'MANIFEST.json'), 'w'), indent=1)
print(len(files), 'files hashed')
