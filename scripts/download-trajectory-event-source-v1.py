"""Image-only bounded batch downloader specialization of the verified reader."""
import hashlib
import json
from pathlib import Path


def configured_source(base, scope):
    assert hashlib.sha256(base.read_bytes()).hexdigest() == '65bc09ef9734a41173623fd46bfa0731eaa86e1cde5b6336607be0cb3313e09c'
    plan = json.loads(scope.read_text())
    stems = tuple(m['stem'] for m in plan['movies'])
    assert 1 <= len(stems) <= 8 and len(set(stems)) == len(stems)
    assert all(m['role'] == 'optimization' and m['image_frames'] == list(range(100)) for m in plan['movies'])
    assert not plan['ground_truth_included'] and not plan['competition_test_data_read'] and not plan['selection_or_validation_opened']
    digest = hashlib.sha256(scope.read_bytes()).hexdigest()
    text = base.read_text(encoding='utf-8')
    replacements = [
        ("STEMS = ('44b6_12dfb391', '44b6_267148e4', '6bba_062c8d37', '6bba_07e24132')", 'STEMS = ' + repr(stems)),
        ('    plan = json.loads(payload)', "    plan = json.loads(payload)\n    if plan['movie_plan_sha256'] != " + repr(digest) + ": raise ValueError('Source scope changed')\n    plan.update(stems=list(STEMS),ground_truth_included=False,total_bytes=plan['total_image_bytes'])"),
        ('len(records) != 408', 'len(records) != ' + str(102 * len(stems))),
        ('total=408', 'total=' + str(102 * len(stems))),
        ("plan['total_bytes'] > 3 * 1024 ** 3", "plan['total_bytes'] > 5_500_000_000"),
        ("run_id='trajectory-division-archive-v1'", 'run_id=' + repr(plan['run_id'])),
    ]
    for old, new in replacements:
        assert text.count(old) == 1
        text = text.replace(old, new)
    compile(text, str(base), 'exec')
    return text


if __name__ == '__main__':
    folder = Path(__file__).parent
    base = folder / 'download-trajectory-division-archive-v1.py'
    exec(compile(configured_source(base, folder / 'MOVIES.json'), str(base), 'exec'),
         dict(__name__='__main__', __file__=str(base)))
