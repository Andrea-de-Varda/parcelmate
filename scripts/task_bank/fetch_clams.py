"""Fetch CLAMS (Mueller et al. 2020) at a pinned commit."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _syntax_common import *  # noqa

URL, SHA = 'https://github.com/aaronmueller/clams', '0881d24f8d66a705cda186d5bf267f939392aba1'

if __name__ == '__main__':
    git_clone_pinned(URL, os.path.join(raw_dir('clams'), 'repo'), SHA)
    write_manifest('clams', dict(github=URL, github_commit=SHA, download_date=today()))
    print('ok')
