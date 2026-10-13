"""sha256 of every converted task file, so the gitignored tasks/ tree is checkable against git.

    python scripts/task_bank/manifest.py          # write external/task_bank/tasks_manifest.tsv
    python scripts/task_bank/manifest.py --check  # verify tasks/ against it (e.g. after rsync to the cluster)

tasks/ is rebuilt by the convert_<source>.py scripts (deterministic) or copied by rsync; it is not
committed (241 MB, and several sources have no redistribution licence).
"""
import argparse, glob, hashlib, os, sys

TB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'external', 'task_bank')
MANIFEST = os.path.join(TB, 'tasks_manifest.tsv')


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    files = sorted(os.path.relpath(p, TB) for p in glob.glob(os.path.join(TB, 'tasks', '*', '*')))
    if not a.check:
        with open(MANIFEST, 'w') as f:
            for rel in files:
                f.write('%s\t%s\n' % (rel, sha256(os.path.join(TB, rel))))
        print('%d files in %s' % (len(files), MANIFEST))
        return
    want = dict(line.rstrip('\n').split('\t') for line in open(MANIFEST))
    bad = [rel for rel, h in want.items() if not os.path.exists(os.path.join(TB, rel)) or sha256(os.path.join(TB, rel)) != h]
    print('%d files checked, %d missing or different' % (len(want), len(bad)))
    for rel in bad[:20]:
        print('  ' + rel)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
