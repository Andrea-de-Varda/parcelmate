"""Linear relational concepts (Chanin et al., NAACL 2024; chanind/linear-relational-concepts): no separate conversion.

The repository's data/lre/{bias,commonsense,factual,linguistic}/*.json are copies of the LRE relation data (Hernandez et al.). This script verifies that claim file by file
(parsed JSON equality, matching by file stem; person_native_language sits under bias/ in the Chanin copy and factual/ in evandez/relations) and writes the registry entry that records the relation.
The LRC paper's own contribution (concept directions, classification/causality benchmarks) needs model activations and is not a prompt dataset: tier 2 at most (prompt + object, no corrupted prompt)."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knowledge_common as kc

SID = 'linear_relational_concepts'

def main():
    man = kc.load_manifest(SID)
    lre = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(kc.RAW, 'lre_relations', 'repo', 'data', '*', '*.json'))}
    mine = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(kc.RAW, SID, 'repo', 'data', 'lre', '*', '*.json'))}
    same = [k for k in mine if k in lre and mine[k] == lre[k]]
    diff = [k for k in mine if k not in lre or mine[k] != lre[k]]
    only_lre = [k for k in lre if k not in mine]
    assert not diff, diff
    reg = kc.base_registry(SID, man, title='Linear relational concepts (Chanin et al.): code and a copy of the LRE relation data',
                           bibkey='chanin-etal-2024-identifying', bibkey_data='hernandez2024linearityrelationdecodingtransformer (the data)', url=man['url'], revision=man['revision'], licence='MIT (Copyright 2023 David Chanin)', licence_source='external/task_bank/raw/linear_relational_concepts/repo/LICENSE',
                           redistribute='yes (MIT)', tier='1 via lre_relations (same data); the LRC benchmark itself is tier 2 at most', status='not_converted_duplicate_of:lre_relations',
                           notes='data/lre/*: %d of %d files equal (as parsed JSON) to the same-named files in evandez/relations (revision %s); %d relation files exist only in evandez/relations: %s. Items are converted once, in lre_relations.' % (
                               len(same), len(mine), kc.load_manifest('lre_relations')['revision'], len(only_lre), only_lre))
    kc.write_registry(SID, reg, [dict(name='(none)', status='not_converted_duplicate_of:lre_relations', notes='see source notes')])
    print(len(same), len(mine), only_lre)

if __name__ == '__main__':
    main()
