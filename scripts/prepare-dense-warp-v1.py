"""Freeze two optimization-only images before a dense-correspondence smoke."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / '.biohub/cache/native-correspondence-v2-plan/MOVIES.json'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == '60300fbc7251f842e80b3ae1687e78d097b7dcda876908f3832b34174c0a0ef6'
    old = json.loads(source.read_text())
    movies = []
    for embryo in ('44b6', '6bba'):
        m = next(m for m in old['movies'] if m['embryo']==embryo and m['role']=='optimization')
        t = m['selected_transitions'][len(m['selected_transitions'])//2]
        movies.append(dict(stem=m['stem'], embryo=embryo, role='optimization', image_frames=[t]))
    target = ROOT / '.biohub/cache/dense-warp-v1-plan'
    target.mkdir(exist_ok=False)
    plan = dict(run_id='dense-warp-v1', movies=movies, competition_test_data_read=False,
                sealed_audit_opened=False, ground_truth_used=False, authorized_for_submission=False)
    (target/'MOVIES.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps(plan))


if __name__ == '__main__': main()
