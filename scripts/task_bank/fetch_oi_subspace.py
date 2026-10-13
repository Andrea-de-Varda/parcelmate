"""Fetch the ordering-ID binding data (Dai, Heinzerling, Inui, EMNLP 2024). The GitHub repo cl-tohoku/OI-Subspace (pinned commit) holds only a README and the MIT
LICENSE; the data ships as the ACL Anthology software attachment (unversioned; identified by sha256)."""
import os, sys, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knowledge_common as kc

SID, URL, SHA = 'oi_subspace', 'https://github.com/cl-tohoku/OI-Subspace', 'e70f4717d40d399974167f386235051014b0c783'
ZIP_URL = 'https://aclanthology.org/attachments/2024.emnlp-main.967.software.zip'

def main():
    d = kc.git_checkout(SID, 'repo', URL, SHA)
    z = kc.download(ZIP_URL, os.path.join(kc.RAW, SID, 'acl_software', '2024.emnlp-main.967.software.zip'))
    out = os.path.join(kc.RAW, SID, 'acl_software')
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            assert not n.startswith('/') and '..' not in n.split('/'), n
        if not os.path.isdir(os.path.join(out, 'model')):
            zf.extractall(out)
    files = [z, os.path.join(d, 'LICENSE'), os.path.join(d, 'README.md')]
    for root, _, fs in os.walk(os.path.join(out, 'model')):
        files += [os.path.join(root, f) for f in sorted(fs)]
    kc.write_manifest(SID, dict(source_id=SID, url=URL, revision=SHA, data_url=ZIP_URL,
                                data_revision='none (unversioned ACL Anthology attachment; identified by sha256 of the zip)'), sorted(files))
    print('ok', SID)

if __name__ == '__main__':
    main()
