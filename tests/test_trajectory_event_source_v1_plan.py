import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_source_expansion_excludes_all_selection_and_reuses_previous_predictions():
    report = json.loads((ROOT / 'reports/experiments/trajectory-event-source-v1-plan.json').read_text())
    roles = json.loads((ROOT / '.biohub/cache/native-correspondence-v2-plan/MOVIES.json').read_text())
    allowed = {m['stem'] for m in roles['movies'] if m['role'] == 'optimization'}
    forbidden = {m['stem'] for m in roles['movies'] if m['role'] != 'optimization'}
    selected = set(report['selected_movies'])
    reused, new = map(set, (report['reused_prediction_movies'], report['new_prediction_movies']))
    assert selected <= allowed and not selected & forbidden
    assert reused | new == selected and not reused & new
    assert len(reused) == 8 and len(new) == 60 and len(selected) == 68
    assert report['counts']['44b6_events'] == 19
    assert report['counts']['6bba_events'] == 93
    assert not report['gpu_launched'] and not report['training_or_submission_performed']


def test_image_only_batches_are_hash_pinned_small_and_complete_movies():
    report = json.loads((ROOT / 'reports/experiments/trajectory-event-source-v1-plan.json').read_text())
    covered = []
    for record in report['batches']:
        path = ROOT / '.biohub/cache/trajectory-event-source-v1-plan' / ('batch-' + str(record['index'])) / 'MOVIES.json'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['plan_sha256']
        plan = json.loads(path.read_text())
        assert 0 < len(plan['movies']) <= 8
        assert [m['stem'] for m in plan['movies']] == record['movies']
        assert all(set(m) == {'stem', 'embryo', 'role', 'image_frames'} for m in plan['movies'])
        assert all(m['image_frames'] == list(range(100)) for m in plan['movies'])
        assert not plan['ground_truth_included'] and not plan['authorized_for_submission']
        assert record['full_wall_cap_seconds'] == 2700 and record['smoke_required']
        covered.extend(record['movies'])
    assert len(covered) == len(set(covered)) == 60
    assert covered == report['new_prediction_movies']
