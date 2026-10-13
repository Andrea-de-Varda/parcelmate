"""Fetch the LRE relation data (Hernandez et al., ICLR 2024) at a pinned commit of evandez/relations (data/ holds the 48 relation JSONs)."""
import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knowledge_common as kc

SID, URL, SHA = 'lre_relations', 'https://github.com/evandez/relations', '1b9ec3cf2b8368e42bde7e80fcef384312a6ec07'

def main():
    d = kc.git_checkout(SID, 'repo', URL, SHA)
    files = sorted(glob.glob(os.path.join(d, 'data', '*', '*.json'))) + [os.path.join(d, 'LICENSE'), os.path.join(d, 'README.md')]
    kc.write_manifest(SID, dict(source_id=SID, url=URL, revision=SHA), files)
    print('ok', SID, len(files))

if __name__ == '__main__':
    main()
