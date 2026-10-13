"""Tigges et al. 2024 (arXiv 2407.10827) adds no dataset; download the PDF (arXiv v-latest at download date) to check the data sections."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
import datetime, json, os, urllib.request
from common import TB, sha256_file
d = os.path.join(TB, 'raw', 'tigges'); os.makedirs(d, exist_ok=True)
p = os.path.join(d, '2407.10827.pdf')
urllib.request.urlretrieve('https://arxiv.org/pdf/2407.10827', p)
json.dump(dict(download_date=datetime.date.today().isoformat(), files={os.path.relpath(p, os.path.join(TB, '..', '..')): sha256_file(p)}), open(os.path.join(d, 'MANIFEST.json'), 'w'), indent=1)
