"""Fetch aieng-lab/gradiend-mib-ioi at a pinned Hugging Face revision."""
import datetime, json, os
from huggingface_hub import snapshot_download
from common import TB, sha256_file
REV = '070f111e5782148619b11917024e82807f8d1f8d'
d = os.path.join(TB, 'raw', 'gradiend_mib_ioi')
snapshot_download('aieng-lab/gradiend-mib-ioi', repo_type='dataset', revision=REV, local_dir=d, ignore_patterns=['.cache/*'])
files = {}
for dp, dn, fn in os.walk(d):
    dn[:] = [x for x in dn if x != '.cache']
    for f in sorted(fn):
        files[os.path.relpath(os.path.join(dp, f), os.path.join(TB, '..', '..'))] = sha256_file(os.path.join(dp, f))
json.dump(dict(revision=REV, download_date=datetime.date.today().isoformat(), files=files), open(os.path.join(d, 'MANIFEST.json'), 'w'), indent=1)
