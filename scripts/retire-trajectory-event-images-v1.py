"""Retire completed event-source RAM images only after exact output backups."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=int, required=True)
    args = p.parse_args()
    name = 'trajectory-event-source-v1-b' + str(args.batch)
    build = json.loads((ROOT / 'reports/experiments' / (name + '-build.json')).read_text())
    launch = json.loads((ROOT / 'reports/experiments' / (name + '-full-launch.json')).read_text())
    remote = launch['remote_root']
    assert re.fullmatch(r'/dev/shm/biohub-event-source-v1-b' + str(args.batch) + r'\.[A-Za-z0-9]{6}', remote)
    assert not (ROOT / 'reports/experiments' / (name + '-images-cleanup.json')).exists()
    feature_root = ROOT / '.biohub/cache' / (name + '-features')
    features = json.loads((feature_root / 'RESULT.json').read_text())
    assert features['status'] == 'current_candidate_source_features_frozen'
    assert set(features['per_movie']) == set(build['movies'])
    for stem, row in features['per_movie'].items():
        for suffix, key in (('-prediction.json', 'prediction_sha256'), ('-groups.npz', 'groups_sha256'), ('-features.npz', 'features_sha256')):
            assert hashlib.sha256((feature_root / (stem + suffix)).read_bytes()).hexdigest() == row[key]
    base = ROOT / 'scripts/retire-trajectory-selection-images-v1.py'
    assert hashlib.sha256(base.read_bytes()).hexdigest() == 'e7b3c2e04f2afc3f357091dfc120365f9f53cd70f5b10c8ac44fca02e2ff10b8'
    source = base.read_text(encoding='utf-8')
    changes = [
        ('trajectory-ranker-selection-v1-full-output', name + '-full-output'),
        ('trajectory-ranker-selection-v1-full-harvest.json', name + '-full-harvest.json'),
        ('trajectory-ranker-selection-v1-plan', 'trajectory-event-source-v1-plan/batch-' + str(args.batch)),
        ('18a13bbe5c582fd58cbc48687339a1ee33cef88a94ecd2299f99f5c6310fce2f', build['images_sha256']),
        ('1020', str(len(build['movies']) * 102)),
        ('1d9cbb244bbdc23d631c45811ccd222bb2e29d5b70a2cd6f05fa87aeb5ff297e', build['source_scope_sha256']),
        ('/dev/shm/biohub-ranker-selection-v1.LJ0JSl', remote),
        ("len(terminal['movies']) == 10", "len(terminal['movies']) == " + str(len(build['movies']))),
        ('trajectory-selection-images-v1-cleanup.json', name + '-images-cleanup.json'),
    ]
    for old, new in changes:
        assert old in source
        source = source.replace(old, new)
    exec(compile(source, str(base), 'exec'), dict(__name__='__main__', __file__=__file__))


if __name__ == '__main__':
    main()
