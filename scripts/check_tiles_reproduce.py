"""Reproduction check of recomputed tiles (LOG.md Iteration 45, info/RICH_CLUB_PLAN.md section 7).

    python scripts/check_tiles_reproduce.py results/qwen35/qwen3.5-2b-resid-pool4/metrics/scores.csv \\
        results/richclub/qwen3.5-2b-resid-pool4-tiles/metrics/scores.csv

The stored partitions were copied into the new tree and re-scored on the recomputed tiles.
Every row of the original score table must be present in the new one with the same value
up to `--tol` (absolute; the metrics that read no matrix, such as the ARI reliabilities,
must agree exactly up to float formatting). Exits 1 otherwise, which stops a dependent job.
"""

import argparse
import csv
import sys


def load(path):
    with open(path) as f:
        return {(r['tree'], r['variant'], r['metric'], r['fit'], r['eval']): float(r['value'])
                for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('old')
    ap.add_argument('new')
    ap.add_argument('--tol', type=float, default=2e-3)
    args = ap.parse_args()
    old, new = load(args.old), load(args.new)
    missing = sorted(set(old) - set(new))
    worst, bad = 0.0, []
    for key, v in sorted(old.items()):
        if key not in new:
            continue
        d = abs(new[key] - v)
        worst = max(worst, d)
        print('%-60s %12.6f %12.6f %10.2e' % ('/'.join(key), v, new[key], d))
        if d > args.tol:
            bad.append(key)
    print('\n%d rows compared, max |difference| %.2e, tolerance %.0e' % (len(old) - len(missing), worst, args.tol))
    if missing:
        print('MISSING in the new table: %s' % ', '.join('/'.join(k) for k in missing))
    if bad:
        print('OUT OF TOLERANCE: %s' % ', '.join('/'.join(k) for k in bad))
    if missing or bad:
        sys.exit(1)
    print('REPRODUCED')


if __name__ == '__main__':
    main()
