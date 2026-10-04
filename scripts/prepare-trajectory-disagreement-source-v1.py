"""Freeze a small source-only whole-movie corpus without inspecting its errors."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'.biohub/cache/native-correspondence-v2-plan/MOVIES.json'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest=='60300fbc7251f842e80b3ae1687e78d097b7dcda876908f3832b34174c0a0ef6'
    old=json.loads(source.read_text());selected=[]
    for embryo in ('44b6','6bba'):
        candidates=[m for m in old['movies'] if m['embryo']==embryo and m['role']=='optimization']
        candidates.sort(key=lambda m:hashlib.sha256(('trajectory-disagreement-source-v1/'+m['stem']).encode()).hexdigest())
        assert len(candidates)>=4
        selected.extend(dict(stem=m['stem'],embryo=embryo,role='optimization',image_frames=list(range(100))) for m in candidates[:4])
    assert len({r['stem'] for r in selected})==8
    target=ROOT/'.biohub/cache/trajectory-disagreement-source-v1-plan'
    target.mkdir(exist_ok=False)
    result=dict(run_id='trajectory-disagreement-source-v1',status='source_scope_frozen_not_launched',
                source_plan_sha256=digest,selection_rule='first four optimization movies per embryo by SHA256(salt/stem)',
                movies=selected,ground_truth_included=False,selection_or_test_movies_included=False,
                purpose='Inventory actual neural/ILP versus final-motion disagreements before choosing or fitting a model',
                public_backbone_training_overlap=True,independently_held_out=False,
                authorized_for_submission=False,training_authorized_by_this_plan=False,
                gpu_launch_requires_successful_smoke=True,requires_sequential_gpu_gate=True)
    path=target/'MOVIES.json';path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],stems=[m['stem'] for m in selected],frames=800,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest())))


if __name__=='__main__':main()
