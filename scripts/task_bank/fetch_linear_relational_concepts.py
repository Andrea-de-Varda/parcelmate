"""Fetch the code/data repo of Chanin et al. (NAACL 2024) at a pinned commit of chanind/linear-relational-concepts. Its data/lre/ folder is a copy of the LRE relation data."""
import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knowledge_common as kc

SID, URL, SHA = 'linear_relational_concepts', 'https://github.com/chanind/linear-relational-concepts', 'ec5db9ad942b84b998536a97a621dab6c75733e7'

def main():
    d = kc.git_checkout(SID, 'repo', URL, SHA)
    files = sorted(glob.glob(os.path.join(d, 'data', 'lre', '*', '*.json'))) + [os.path.join(d, 'LICENSE'), os.path.join(d, 'README.md'), os.path.join(d, 'CITATION.cff')]
    kc.write_manifest(SID, dict(source_id=SID, url=URL, revision=SHA), files)
    print('ok', SID, len(files))

if __name__ == '__main__':
    main()
