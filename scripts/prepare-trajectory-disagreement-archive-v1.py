"""Reuse the pinned authorized ZIP index reader with an image-only source plan."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
SCOPE=ROOT/'.biohub/cache/trajectory-disagreement-source-v1-plan'


def main():
    plan=SCOPE/'MOVIES.json'
    assert hashlib.sha256(plan.read_bytes()).hexdigest()=='5c4b64793068565598537db6bb3af439513bde3d46e357ffeab59164bda020bf'
    payload=json.loads(plan.read_text())
    assert not payload['ground_truth_included'] and not payload['selection_or_test_movies_included']
    old=ROOT/'.biohub/cache/native-correspondence-v2-plan/MOVIES.json'
    assert hashlib.sha256(old.read_bytes()).hexdigest()==payload['source_plan_sha256']
    roles={m['stem']:m['role'] for m in json.loads(old.read_text())['movies']}
    assert len(payload['movies'])==8 and len({m['stem'] for m in payload['movies']})==8
    assert all(roles[m['stem']]=='optimization' and m['image_frames']==list(range(100)) for m in payload['movies'])
    path=ROOT/'scripts/prepare-native-archive-plan-v2.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='dcb511901ba8ecae21f10f55fe8ccc2e24c316553ba0734905fc74bff02e2fb2'
    source=path.read_text(encoding='utf-8')
    old_guard="if movies['competition_test_data_read'] or movies['sealed_audit_opened']:"
    assert source.count(old_guard)==1
    source=source.replace(old_guard,"if movies['selection_or_test_movies_included'] or movies['ground_truth_included']:")
    sys.argv=[str(path),'--plan-dir',str(SCOPE)]
    exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})


if __name__=='__main__':main()
