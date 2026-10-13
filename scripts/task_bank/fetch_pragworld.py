"""Fetch PragWorld (Vashistha et al., AAAI 2026; arXiv 2511.13021) at a pinned commit of SachinVashisth/PRAGWORLD (Manual and Synthetic CSVs, MIT)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knowledge_common as kc

SID, URL, SHA = 'pragworld', 'https://github.com/SachinVashisth/PRAGWORLD', 'c20b079a2882b8a933247c85e78df3e47ab95cec'

def main():
    d = kc.git_checkout(SID, 'repo', URL, SHA)
    files = [os.path.join(d, f) for f in ['Manual_Dataset_500_Final.csv', 'synthetic_final.csv', 'LICENSE', 'README.md']]
    kc.write_manifest(SID, dict(source_id=SID, url=URL, revision=SHA), files)
    print('ok', SID)

if __name__ == '__main__':
    main()
