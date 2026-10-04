"""Frozen training and screening scope; no new held-out movies."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'.biohub/cache/native-correspondence-v2-plan/MOVIES.json'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='60300fbc7251f842e80b3ae1687e78d097b7dcda876908f3832b34174c0a0ef6'
    old=json.loads(source.read_text());movies=[]
    for m in old['movies']:
        row={k:m[k] for k in ('stem','embryo','role')};times=m['selected_transitions']
        if m['role']=='optimization':row['image_frames']=[times[len(times)//2]]
        else:
            selected=[times[i] for i in np.linspace(0,len(times)-1,4).round().astype(int)]
            row.update(transitions=selected,image_frames=sorted({f for t in selected for f in (t,t+1)}),nodes=m['nodes'],edges=m['edges'])
        movies.append(row)
    target=ROOT/'.biohub/cache/dense-warp-v1-full-plan';target.mkdir(exist_ok=False)
    plan=dict(run_id='dense-warp-v1-full',movies=movies,competition_test_data_read=False,
              sealed_audit_opened=False,authorized_for_submission=False,steps_per_source=2000,views_per_movie=4,
              lr=1e-5,weight_decay=.01,anchor_coefficient=1e-4,seed=9731,
              source_gate='correct >= parent; CE strictly lower; at least one more correct; per-movie correct no worse',
              opposite_gate='same as source; final checkpoint only; no gate tuning',
              public_backbone_training_overlap=True,independently_held_out=False)
    (target/'MOVIES.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(movies=len(movies),frames=sum(len(m['image_frames']) for m in movies),sha256=hashlib.sha256((target/'MOVIES.json').read_bytes()).hexdigest())))


if __name__=='__main__':main()
