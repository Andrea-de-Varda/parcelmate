"""Fetch the Feng & Steinhardt (ICLR 2024) binding-ID repository at a pinned commit of jiahai-feng/binding-iclr. The repo contains the dataset generators
and their word lists (coref/datasets/raw/*.csv); the repo has no top-level LICENSE file (only the vendored transformer_lens one)."""
import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knowledge_common as kc

SID, URL, SHA = 'binding_iclr', 'https://github.com/jiahai-feng/binding-iclr', 'fb7608dd4c255ac37d7e95c5c8bf45b5512901eb'

def main():
    d = kc.git_checkout(SID, 'repo', URL, SHA)
    ds = os.path.join(d, 'coref', 'datasets')
    files = sorted(glob.glob(os.path.join(ds, 'raw', '*.csv')) + glob.glob(os.path.join(ds, '*.py')) + [os.path.join(d, 'README.md'), os.path.join(d, 'transformer_lens', 'LICENSE')])
    kc.write_manifest(SID, dict(source_id=SID, url=URL, revision=SHA, note='biosbias.json (56 MB, Bias in Bios) is present in the clone but not used'), files)
    print('ok', SID, len(files))

if __name__ == '__main__':
    main()
