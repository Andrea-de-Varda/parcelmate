"""Fetch the entity-tracking (boxes) data of Prakash et al. (ICLR 2024) at a pinned commit of Nix07/finetuning. The repo has no LICENSE file."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knowledge_common as kc

SID, URL, SHA = 'entity_tracking_prakash', 'https://github.com/Nix07/finetuning', '6d6c0980bf073d305b915669c33533576a70aa77'

def main():
    d = kc.git_checkout(SID, 'repo', URL, SHA)
    files = [os.path.join(d, 'data', 'dataset.jsonl'), os.path.join(d, 'data', 'objects.csv'), os.path.join(d, 'data', 'data_utils.py'), os.path.join(d, 'README.md')]
    kc.write_manifest(SID, dict(source_id=SID, url=URL, revision=SHA), files)
    print('ok', SID)

if __name__ == '__main__':
    main()
