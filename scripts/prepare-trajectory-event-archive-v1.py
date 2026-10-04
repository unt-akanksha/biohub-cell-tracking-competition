"""Prepare one image-only batch's private archive ranges; never print URLs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=int, required=True)
    args = p.parse_args()
    report = json.loads((ROOT / 'reports/experiments/trajectory-event-source-v1-plan.json').read_text())
    row = next(b for b in report['batches'] if b['index'] == args.batch)
    folder = ROOT / '.biohub/cache/trajectory-event-source-v1-plan' / ('batch-' + str(args.batch))
    assert hashlib.sha256((folder / 'MOVIES.json').read_bytes()).hexdigest() == row['plan_sha256']
    plan = json.loads((folder / 'MOVIES.json').read_text())
    assert all(m['role'] == 'optimization' for m in plan['movies'])
    assert not plan['ground_truth_included'] and not plan['selection_or_validation_opened']
    base = ROOT / 'scripts/prepare-native-archive-plan-v2.py'
    assert hashlib.sha256(base.read_bytes()).hexdigest() == 'dcb511901ba8ecae21f10f55fe8ccc2e24c316553ba0734905fc74bff02e2fb2'
    source = base.read_text(encoding='utf-8')
    assert source.count("movies['sealed_audit_opened']") == 1
    source = source.replace("movies['sealed_audit_opened']", "movies['selection_or_validation_opened']")
    sys.argv = [str(base), '--plan-dir', str(folder)]
    exec(compile(source, str(base), 'exec'), dict(__name__='__main__', __file__=str(base)))


if __name__ == '__main__':
    main()
