"""Freeze all ten existing selection movies, unchanged model, and image ranges."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    source=ROOT/'.biohub/cache/native-correspondence-v2-plan/MOVIES.json'
    assert sha(source)=='60300fbc7251f842e80b3ae1687e78d097b7dcda876908f3832b34174c0a0ef6'
    audit=ROOT/'reports/experiments/trajectory-candidate-ranker-v2-audit.json'
    proof=json.loads(audit.read_text())
    assert proof['qualified_for_independent_selection']==['6bba']
    old=json.loads(source.read_text())
    movies=[dict(stem=m['stem'],embryo=m['embryo'],role='selection',image_frames=list(range(100))) for m in old['movies'] if m['role']=='selection']
    assert len(movies)==10 and len({m['stem'] for m in movies})==10
    target=ROOT/'.biohub/cache/trajectory-ranker-selection-v1-plan';target.mkdir(exist_ok=False)
    plan=dict(run_id='trajectory-ranker-selection-v1',movies=movies,source_plan_sha256=sha(source),
              source_ranker_audit_sha256=sha(audit),selected_source_model='6bba',
              selected_weights_sha256=proof['weights_sha256'],ground_truth_included=False,
              competition_test_data_read=False,sealed_audit_opened=False,
              purpose='Unchanged baseline predictions for fixed ranker selection and graph integration',
              training_or_hyperparameter_search_allowed=False,authorized_for_submission=False)
    (target/'MOVIES.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    base=ROOT/'scripts/prepare-native-archive-plan-v2.py'
    assert sha(base)=='dcb511901ba8ecae21f10f55fe8ccc2e24c316553ba0734905fc74bff02e2fb2'
    print(json.dumps(dict(status='selection_scope_frozen',movies=[m['stem'] for m in movies],scope_sha256=sha(target/'MOVIES.json'))),flush=True)
    sys.argv=[str(base),'--plan-dir',str(target)]
    runpy.run_path(str(base),run_name='__main__')


if __name__=='__main__':main()
