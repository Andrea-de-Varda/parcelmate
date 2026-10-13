"""Registry entry for Tigges et al. 2024: no new data (tasks reuse Wang IOI generator, Hanna greater-than/gendered pronoun data and Newman et al. SVA templates)."""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
import json, os
from common import TB, write_registry
man = json.load(open(os.path.join(TB, 'raw', 'tigges', 'MANIFEST.json')))
tasks = [dict(name=n, family=f, n_items_converted=0, status='no_new_data', adaptations=[a]) for n, f, a in [
    ('ioi', 'ioi', 'Paper Sec. 2.4: 70 examples made with the Wang et al. generator; covered by wang_ioi (our frozen generator output).'),
    ('greater_than', 'greater_than', 'Hanna et al. data; not fitting tier 1, see eap_ig.'),
    ('gendered_pronoun', 'gender_pronoun', 'Hanna et al. data; covered by eap_ig gender_bias.'),
    ('sva', 'sva', 'Newman et al. 2021 templates (MIT per appendix); group 2 (newman_sva) handles the templates.')]]
write_registry('tigges', dict(source_id='tigges', title='LLM Circuit Analyses Are Consistent Across Training and Scale', bibkey='tigges2024llmcircuitanalysesconsistent', bibkey_data=[],
    url='https://arxiv.org/abs/2407.10827', revision='arXiv PDF as downloaded on the download date (no version pin available)', download_date=man['download_date'],
    raw_files=[dict(path=p, sha256=s) for p, s in man['files'].items()], licence='paper; its appendix states Wang IOI and Newman SVA data are MIT and greater-than / gendered-pronoun data have no licence',
    licence_source='raw/tigges/2407.10827.pdf, "Licenses of Artifacts Used"', redistribute='not applicable (no data)', tier=1, status='no_new_data',
    notes='The paper introduces no dataset. The four tasks reuse existing generators/data. Pythia-tokenizer-filtered versions of greater-than, gender and SVA data are the pythia.csv files of eap-ig-faithfulness (not verified that these are exactly the files the paper used).'), tasks)
