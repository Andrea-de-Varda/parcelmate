"""Fetch CausalGym at pinned revisions (HF dataset + GitHub repo with the templates). Rerunnable."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _syntax_common import *  # noqa

HF_ID, HF_REV = 'aryaman/causalgym', '95349c3a5e53e2506e8b212482ea6dd784978156'
GH_URL, GH_SHA = 'https://github.com/aryamanarora/causalgym', '0f3129ff3b6c5c8264892f30a25be25150ae9179'

if __name__ == '__main__':
    from huggingface_hub import snapshot_download
    d = raw_dir('causalgym')
    snapshot_download(HF_ID, repo_type='dataset', revision=HF_REV, local_dir=os.path.join(d, 'hf'))
    git_clone_pinned(GH_URL, os.path.join(d, 'repo'), GH_SHA)
    write_manifest('causalgym', dict(hf_dataset=HF_ID, hf_revision=HF_REV, github=GH_URL, github_commit=GH_SHA, download_date=today()))
    print('ok')
