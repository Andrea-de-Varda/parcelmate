"""Copy the arithmetic data used in de Varda et al. (arXiv 2609.04463) into raw/devarda_paper_arithmetic.

Source: Andrea's local clone of the paper repository (github.com/Andrea-de-Varda/arithmetic-inconsistencies,
private), at the commit recorded in MANIFEST.json. The four counterfactual files (5,000 problems per format,
ids aligned) and the paper's 2,000-id attribution fold are copied verbatim with their sha256.

    python scripts/task_bank/fetch_devarda_paper_arithmetic.py [--src /path/to/arithmetic-inconsistencies]
"""
import argparse, json, os, shutil, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _arith_common as C

FILES = ['arithmetic-dataset/arith_dataset_%s_counterfactual.json' % f for f in ('numeric', 'english', 'spanish', 'italian')] \
    + ['attribution/data/attribution_fold_ids.json', 'attribution/src/data.py']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/home/andrea/Documents/postdoc/side-projects/arithmetic-inconsistencies')
    a = ap.parse_args()
    commit = subprocess.check_output(['git', '-C', a.src, 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', a.src, 'status', '--porcelain', '--'] + FILES, text=True).strip()
    assert not dirty, 'source files differ from the commit: ' + dirty
    out = os.path.join(C.TB, 'raw', 'devarda_paper_arithmetic')
    files = {}
    for rel in FILES:
        dst = os.path.join(out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(a.src, rel), dst)
        files[rel] = C.sha256(dst)
    C.write_json(os.path.join(out, 'MANIFEST.json'), dict(
        url='https://github.com/Andrea-de-Varda/arithmetic-inconsistencies (private; copied from a local clone)',
        commit=commit, download_date=C.today(), local_source=a.src, files=files))
    print('copied %d files at %s' % (len(files), commit))


if __name__ == '__main__':
    main()
