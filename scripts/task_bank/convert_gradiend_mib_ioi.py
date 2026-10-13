"""aieng-lab/gradiend-mib-ioi: check that it is a re-packaging of mib-bench/ioi, record the result in the registry; no tasks are written (duplicate)."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
import json, os, sys
import pandas as pd
from common import TB, write_registry
RAW = os.path.join(TB, 'raw', 'gradiend_mib_ioi'); man = json.load(open(os.path.join(RAW, 'MANIFEST.json')))
g = pd.concat([pd.read_parquet(os.path.join(RAW, 'default', s + '.parquet')) for s in ('train', 'validation', 'test')])
mib = {s: json.load(open(os.path.join(TB, 'raw', 'mib', 'ioi', s + '.json'))) for s in ('train', 'dev', 'test')}
mp = {x['prompt']: x for v in mib.values() for x in v}
hit = flip = 0
for _, r in g.iterrows():
    p = r.masked.replace(' [MASK]', '')
    if p in mp:
        hit += 1
        flip += r.cf_masked.replace(' [MASK]', '') == mp[p]['s2_io_flip_counterfactual']['prompt']
print(len(g), hit, flip)
write_registry('gradiend_mib_ioi', dict(source_id='gradiend_mib_ioi', title='GRADIEND MIB-IOI cloze data (aieng-lab/gradiend-mib-ioi)', bibkey='mueller2025mibmechanisticinterpretabilitybenchmark',
    bibkey_data=['GRADIEND dataset card has "Citation: TODO"; no bib available'], url='https://huggingface.co/datasets/aieng-lab/gradiend-mib-ioi', revision=man['revision'], download_date=man['download_date'],
    raw_files=[dict(path=p, sha256=s) for p, s in man['files'].items()], licence='MIT', licence_source='README.md front matter at the pinned revision (raw/gradiend_mib_ioi/README.md)', redistribute='yes', tier='2 (masked cloze) / 1 via cf_masked',
    status='duplicate_of_mib', notes='All %d rows (10,000 train / 10,000 validation / 1,000 test) have a masked prompt identical to a mib-bench/ioi prompt (%d matched); in all %d the `cf_masked` text equals MIB s2_io_flip_counterfactual. The data therefore adds nothing beyond mib/ioi_s2_flip and was not converted separately (tier-1 equivalent exists).' % (len(g), hit, flip)),
    [dict(name='ioi_cloze', family='ioi', n_items_raw=len(g), n_items_converted=0, status='duplicate_of_mib:ioi_s2_flip', adaptations=['masked cloze in encoder-style form; for causal LMs identical to MIB prompt + IO name'])])
