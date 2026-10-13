"""Export the BibTeX of every group-4 source from its canonical page into external/task_bank/references/<source_id>.bib.

Canonical exports: ACL Anthology (https://aclanthology.org/<id>.bib), arXiv (https://arxiv.org/bibtex/<id>) and the DOI resolver (Accept: application/x-bibtex).
DBLP and OpenReview refused automated access (bot challenge) on 2026-10-12, so the ICLR / NeurIPS / ICML papers use the arXiv export plus a `note` line carrying the venue,
which was checked against (a) the OpenReview search API record (venue string and forum id, queried before the challenge appeared) for Meng et al. (NeurIPS 2022, 2QNG8btAqA),
Hernandez et al. (ICLR 2024 spotlight, w7LU2s14kE), Todd et al. (ICLR 2024 poster, AwyxtyMwaG), Feng & Steinhardt (ICLR 2024 poster, zb3b6oKO77), Prakash et al. (ICLR 2024 poster, 8sKcAWOf2D)
and (b) the ICML 2026 poster page https://icml.cc/virtual/2026/poster/64207 for Tang et al. Titles and author lists were compared with the arXiv abstract pages (citation_* meta tags)."""
import os, re, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knowledge_common as kc

def get(url, accept=None):
    h = {'User-Agent': 'parcelmate-task-bank'}
    if accept:
        h['Accept'] = accept
    for k in range(3):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=30).read().decode('utf-8')
        except Exception:
            time.sleep(3 * (k + 1))
    raise SystemExit('failed ' + url)

def acl(i):
    return get('https://aclanthology.org/%s.bib' % i).strip()

def arxiv(i, note=None):
    b = get('https://arxiv.org/bibtex/%s' % i).strip()
    if note:
        body = re.sub(r'[,\s]*\}\s*$', '', b)
        b = body + ',\n      note={%s}\n}' % note
    return b

def main():
    time.sleep(1)
    out = {
        'ravel': [acl('2024.acl-long.470')],
        'counterfact': [arxiv('2202.05262', 'Published at NeurIPS 2022 (OpenReview 2QNG8btAqA); the arXiv export carries the year of its latest version')],
        'lre_relations': [arxiv('2308.09124', 'Published at ICLR 2024, spotlight (OpenReview w7LU2s14kE)')],
        'linear_relational_concepts': [acl('2024.naacl-long.85'), arxiv('2308.09124', 'Published at ICLR 2024, spotlight (OpenReview w7LU2s14kE); source of the relation data')],
        'function_vectors': [arxiv('2310.15213', 'Published at ICLR 2024 (OpenReview AwyxtyMwaG)')],
        'binding_iclr': [arxiv('2310.17191', 'Published at ICLR 2024 (OpenReview zb3b6oKO77)')],
        'oi_subspace': [acl('2024.emnlp-main.967'), arxiv('2310.17191', 'Published at ICLR 2024 (OpenReview zb3b6oKO77); origin of the Binding ID mechanism')],
        'entity_tracking_prakash': [arxiv('2402.14811', 'Published at ICLR 2024 (OpenReview 8sKcAWOf2D)'), acl('2023.acl-long.213')],
        'entity_tracking_mi': [arxiv('2605.30233', 'ICML 2026 (https://icml.cc/virtual/2026/poster/64207)'), acl('2023.acl-long.213')],
        'pragworld': [get('https://doi.org/10.1609/aaai.v40i39.40618', 'application/x-bibtex').strip(), arxiv('2511.13021')],
    }
    d = os.path.join(kc.TB, 'references')
    os.makedirs(d, exist_ok=True)
    for sid, entries in out.items():
        with open(os.path.join(d, sid + '.bib'), 'w') as f:
            f.write('% exported 2026-10-12 by scripts/task_bank/fetch_references_knowledge.py (ACL Anthology / arXiv / doi.org exports; see that script for the venue checks)\n')
            f.write('\n\n'.join(entries) + '\n')
        print(sid, len(entries))

if __name__ == '__main__':
    main()
