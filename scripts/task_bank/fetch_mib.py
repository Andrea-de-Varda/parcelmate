"""Fetch the MIB (Mechanistic Interpretability Benchmark) datasets from Hugging Face at pinned revisions.
Writes raw files under external/task_bank/raw/mib/<dataset>/ and external/task_bank/raw/mib/MANIFEST.json (revision, sha256, download date)."""
import hashlib, json, os, datetime
from huggingface_hub import snapshot_download

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'external', 'task_bank', 'raw', 'mib')
REVISIONS = {
    'ioi': 'e5f3468f3af4c0883be35cd3bced8c711c95d286',
    'arithmetic_addition': 'd56d68b7b1d8f9c7e9a262ec1f89331ed52c7516',
    'arithmetic_subtraction': '9eeb7f0f69c2adc131d81200f501a7f83c06fa8b',
    'copycolors_mcqa': '682676b0e80d3a80e847a21810157710c1f23e27',
    'arc_easy': 'be6999b975e8348387cc9a21f3c8b6be8021e7d6',
    'arc_challenge': 'ab946b6cf3e2543e590adbeebd531c52b8eb4ff8',
    'ravel': '53dc7a4dbe84276567b895d8fb608b3f169b9276',
}

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def main():
    manifest = {}
    for name, rev in REVISIONS.items():
        d = os.path.join(ROOT, name)
        snapshot_download(repo_id=f'mib-bench/{name}', repo_type='dataset', revision=rev, local_dir=d,
                          ignore_patterns=['*.DS_Store', '.cache/*'])
        files = {}
        for dp, dn, fn in os.walk(d):
            dn[:] = [x for x in dn if x != '.cache']
            for f in sorted(fn):
                p = os.path.join(dp, f)
                files[os.path.relpath(p, ROOT)] = sha256(p)
        manifest[name] = dict(revision=rev, files=files)
    manifest['_download_date'] = datetime.date.today().isoformat()
    with open(os.path.join(ROOT, 'MANIFEST.json'), 'w') as f:
        json.dump(manifest, f, indent=1)

if __name__ == '__main__':
    main()
