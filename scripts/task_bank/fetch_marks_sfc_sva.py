"""Fetch Marks et al. (Sparse Feature Circuits) repo at a pinned commit; the SVA data are in data/."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _syntax_common import *  # noqa

URL, SHA = 'https://github.com/saprmarks/feature-circuits', '7fbd82b895ae16294f4e6fc7bfc675d1d680d659'

if __name__ == '__main__':
    git_clone_pinned(URL, os.path.join(raw_dir('marks_sfc_sva'), 'repo'), SHA)
    write_manifest('marks_sfc_sva', dict(github=URL, github_commit=SHA, download_date=today()))
    print('ok')
