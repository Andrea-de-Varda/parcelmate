"""Fetch the Refining-TSE repository of Newman et al. (2021) at a pinned commit."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _syntax_common import *  # noqa

URL, SHA = 'https://github.com/bnewm0609/refining-tse', 'db423f80246ba2d7c2fc602fc514943949cfaf71'

if __name__ == '__main__':
    git_clone_pinned(URL, os.path.join(raw_dir('newman_sva'), 'repo'), SHA)
    write_manifest('newman_sva', dict(github=URL, github_commit=SHA, download_date=today()))
    print('ok')
