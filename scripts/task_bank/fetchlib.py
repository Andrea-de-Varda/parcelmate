"""Clone git repos at pinned commits and write a sha256 manifest. Cloned code is never executed from the raw directory."""
import datetime, json, os, subprocess
from common import TB, sha256_file

def clone_pinned(source_id, repos):
    """repos: {dirname: (url, sha)}. Clones into raw/<source_id>/<dirname>, checks out the sha, writes MANIFEST.json."""
    root = os.path.join(TB, 'raw', source_id)
    os.makedirs(root, exist_ok=True)
    man = dict(download_date=datetime.date.today().isoformat(), repos={})
    for name, (url, sha) in repos.items():
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            subprocess.run(['git', 'clone', '-q', url, d], check=True)
        subprocess.run(['git', '-C', d, 'checkout', '-q', sha], check=True)
        head = subprocess.run(['git', '-C', d, 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
        assert head == sha, (head, sha)
        files = {}
        for dp, dn, fn in os.walk(d):
            dn[:] = [x for x in dn if x not in ('.git', '__pycache__')]
            for f in sorted(fn):
                p = os.path.join(dp, f)
                files[os.path.relpath(p, os.path.join(TB, '..', '..')).replace('\\', '/')] = sha256_file(p)
        man['repos'][name] = dict(url=url, commit=sha, files=files)
    with open(os.path.join(root, 'MANIFEST.json'), 'w') as f:
        json.dump(man, f, indent=1)
    return man
