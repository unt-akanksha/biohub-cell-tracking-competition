"""Reuse pinned counts-only physical source audit on a frozen expanded batch."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=int, required=True)
    args = p.parse_args()
    name = 'trajectory-event-source-v1-b' + str(args.batch)
    report = json.loads((ROOT / 'reports/experiments/trajectory-event-source-v1-plan.json').read_text())
    record = next(b for b in report['batches'] if b['index'] == args.batch)
    scope_path = '.biohub/cache/trajectory-event-source-v1-plan/batch-' + str(args.batch) + '/MOVIES.json'
    assert hashlib.sha256((ROOT / scope_path).read_bytes()).hexdigest() == record['plan_sha256']
    features_path = ROOT / '.biohub/cache' / (name + '-features/RESULT.json')
    features = json.loads(features_path.read_text())
    assert features['status'] == 'current_candidate_source_features_frozen'
    assert not features['ground_truth_opened'] and not features['selection_or_validation_opened']
    assert features['source_scope_sha256'] == record['plan_sha256']
    for stem, item in features['per_movie'].items():
        path = features_path.parent / (stem + '-prediction.json')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['prediction_sha256']
    base = ROOT / 'scripts/audit-trajectory-source-division-capacity-v1.py'
    assert hashlib.sha256(base.read_bytes()).hexdigest() == 'abce8ea351bde94206d38a8d920c061c5b8e7fa6eb5815c22b140f7316e89143'
    source = base.read_text(encoding='utf-8')
    changes = [
        ('trajectory-source-division-capacity-v1.json', name + '-capacity.json'),
        ('.biohub/cache/trajectory-disagreement-source-v1-plan/MOVIES.json', scope_path),
        ('5c4b64793068565598537db6bb3af439513bde3d46e357ffeab59164bda020bf', record['plan_sha256']),
        ('trajectory-disagreement-source-v1-full-output', name + '-full-output'),
        ('trajectory-disagreement-source-v1-full-harvest.json', name + '-full-harvest.json'),
        ("    reference = json.loads((ROOT / 'reports/experiments/trajectory-joint-source-v1-result.json').read_text())\n    expected = {r['stem']: r for r in reference['rows']['original']}\n", ''),
        ('reports/experiments/trajectory-candidate-supervision-v1.json', '.biohub/cache/' + name + '-features/RESULT.json'),
        ("(base / (stem + '-original') / 'repaired-prediction.json')", "(ROOT / '.biohub/cache/" + name + "-features' / (stem + '-prediction.json'))"),
        ("        for key in scorer.COUNT_COLUMNS:\n            assert getattr(evaluation, key) == expected[stem][key]\n", ''),
        ("ROOT / '.biohub/cache/trajectory-candidate-supervision-v1' / (stem + '-candidates.npz')",
         "ROOT / '.biohub/cache/" + name + "-features' / (stem + '-groups.npz')"),
        ("inventory['candidate_sha256'][stem]", "inventory['per_movie'][stem]['groups_sha256']"),
        ("source_sha256=sha(Path(__file__)), source_only=True", "source_sha256=sha(Path(__file__)), effective_source_sha256=EFFECTIVE_SHA, paired_baseline_score_reproduction_required=False, source_only=True"),
    ]
    for old, new in changes:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    exec(compile(source, str(base), 'exec'), dict(__name__='__main__', __file__=__file__,
         EFFECTIVE_SHA=hashlib.sha256(source.encode()).hexdigest()))


if __name__ == '__main__':
    main()
