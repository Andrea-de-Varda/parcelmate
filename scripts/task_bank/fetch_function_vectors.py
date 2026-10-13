"""Fetch the function-vector task datasets (Todd et al., ICLR 2024) at a pinned commit of ericwtodd/function_vectors."""
import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knowledge_common as kc

SID, URL, SHA = 'function_vectors', 'https://github.com/ericwtodd/function_vectors', 'fb9eac7b6dc707ea1475a717379916007fe448d5'

def main():
    d = kc.git_checkout(SID, 'repo', URL, SHA)
    files = sorted(glob.glob(os.path.join(d, 'dataset_files', '*', '*.json'))) + [os.path.join(d, 'LICENSE'), os.path.join(d, 'dataset_files', 'README.md'),
             os.path.join(d, 'src', 'utils', 'prompt_utils.py')]
    kc.write_manifest(SID, dict(source_id=SID, url=URL, revision=SHA), files)
    print('ok', SID, len(files))

if __name__ == '__main__':
    main()
