"""Fetch CounterFact (Meng et al., NeurIPS 2022) from the official ROME host. The file has no git revision, so the version is the
sha256 of the downloaded file plus the download date; the code repo kmeng01/rome is cloned at a pinned commit for reference (licence, loader)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knowledge_common as kc

SID = 'counterfact'
URL = 'https://rome.baulab.info/data/dsets/counterfact.json'
ROME_URL, ROME_SHA = 'https://github.com/kmeng01/rome', '0874014cd9837e4365f3e6f3c71400ef11509e04'
EXPECT_SHA256 = 'd017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f'

def main():
    r = kc.git_checkout(SID, 'rome_repo', ROME_URL, ROME_SHA)
    f = kc.download(URL, os.path.join(kc.RAW, SID, 'official', 'counterfact.json'))
    assert kc.sha256_file(f) == EXPECT_SHA256, 'official counterfact.json changed upstream'
    kc.write_manifest(SID, dict(source_id=SID, url=URL, revision='none (unversioned file; identified by sha256 %s)' % EXPECT_SHA256,
                                code_repo=ROME_URL, code_repo_revision=ROME_SHA), [f, os.path.join(r, 'LICENSE'), os.path.join(r, 'dsets', 'counterfact.py')])
    print('ok', SID)

if __name__ == '__main__':
    main()
