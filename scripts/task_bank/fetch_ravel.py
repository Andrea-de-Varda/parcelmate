"""Fetch RAVEL (Huang et al., ACL 2024) at a pinned commit of explanare/ravel and unpack data.tgz (the five entity types)."""
import os, sys, tarfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knowledge_common as kc

SID, URL, SHA = 'ravel', 'https://github.com/explanare/ravel', 'e421fc9e132b3613b1d0020f856d539217fb30ca'

def main():
    d = kc.git_checkout(SID, 'repo', URL, SHA)
    out = os.path.join(kc.RAW, SID, 'data_unpacked')
    tgz = os.path.join(d, 'data.tgz')
    if not os.path.isdir(out):
        os.makedirs(out)
        with tarfile.open(tgz) as t:
            for m in t.getmembers():  # only plain files, no path escapes
                assert m.isfile() or m.isdir(), m.name
                assert not m.name.startswith('/') and '..' not in m.name.split('/'), m.name
            t.extractall(out)
    files = [tgz, os.path.join(d, 'LICENSE'), os.path.join(d, 'README.md')]
    for root, _, fs in os.walk(out):
        files += [os.path.join(root, f) for f in sorted(fs)]
    kc.write_manifest(SID, dict(source_id=SID, url=URL, revision=SHA, note='data/*.json unpacked from data.tgz into data_unpacked/'), sorted(files))
    print('ok', SID)

if __name__ == '__main__':
    main()
