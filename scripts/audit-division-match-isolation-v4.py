"""Read-only source geometry diagnostic: wider proximity is not a training label."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.native_correspondence_data_v2 import VOXEL


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_frame(nodes,points):
    truth=np.array([n[2:] for n in nodes],np.float32).reshape(-1,3)*VOXEL
    distances=np.linalg.norm(truth[:,None]-points[None],axis=-1)
    cost=np.concatenate((np.where(distances<=7,distances,1e6),np.full((len(nodes),len(nodes)),7.0001)),axis=1)
    rows,cols=linear_sum_assignment(cost)
    matched={int(row):int(col) for row,col in zip(rows,cols) if col<len(points) and distances[row,col]<=7}
    result={}
    for i,node in enumerate(nodes):
        order=np.argsort(distances[i],kind='stable')
        d1=float(distances[i,order[0]]) if len(order) else None
        d2=float(distances[i,order[1]]) if len(order)>1 else None
        nearest=int(order[0]) if len(order) else None
        assigned=matched.get(i)
        # Isolation refers to other IMAGE proposals, not unannotated negative cells.
        isolated=assigned is not None and assigned==nearest and (d2 is None or d2-d1>=3.25)
        result[int(node[0])]=dict(nearest_um=d1,second_um=d2,assigned_index=assigned,nearest_index=nearest,
                                 nearest_is_assigned=assigned is not None and assigned==nearest,
                                 isolated_margin_3_25=isolated)
    return result


def main():
    folder=ROOT/'.biohub/cache/native-division-proposal-audit-v4-full-output'
    evidence_path=folder/'RESULT.json'
    if sha(evidence_path)!='0f12f8a011c0575f63f9160e97db2dcc2dc8ccab6cb8c898e3963cc18e6a2d4e':raise ValueError('Proposal evidence changed')
    result=json.loads(evidence_path.read_text())
    plan_path=ROOT/'.biohub/cache/native-division-proposal-audit-v4-plan/MOVIES.json'
    if sha(plan_path)!=result['movie_plan_sha256']:raise ValueError('Source plan changed')
    movie_lookup={m['stem']:m for m in json.loads(plan_path.read_text())['movies']}
    cache={};totals={};events=[]
    for artifact in result['point_artifacts']:
        path=folder/artifact['path']
        if sha(path)!=artifact['sha256']:raise ValueError('Image points changed')
        stem=path.stem;movie=movie_lookup[stem]
        if movie['role']!='optimization':raise ValueError('Source optimization only')
        with np.load(path,allow_pickle=False) as packet:
            for name in packet.files:
                variant,t=name.rsplit('_',1);t=int(t)
                nodes=[n for n in movie['nodes'] if int(n[1])==t]
                cache[(stem,variant,t)]=inspect_frame(nodes,packet[name])
    for row in result['records']:
        key=row['embryo']+'-'+row['variant'];count=totals.setdefault(key,Counter())
        for event in row['events']:
            nodes=[cache[(row['stem'],row['variant'],row['transition'])][event['parent']]]
            nodes += [cache[(row['stem'],row['variant'],row['transition']+1)][c] for c in event['children']]
            all_assigned=all(n['assigned_index'] is not None for n in nodes)
            all_nearest=all(n['nearest_is_assigned'] for n in nodes)
            isolated=all(n['isolated_margin_3_25'] for n in nodes)
            shared_nearest=nodes[1]['nearest_index']==nodes[2]['nearest_index']
            count.update(events=1,all_assigned_within_7um=int(all_assigned),all_nearest_assigned=int(all_nearest),
                         all_isolated_margin_3_25=int(isolated),daughters_share_nearest_peak=int(shared_nearest),
                         newly_isolated_beyond_old_radius=int(isolated and not event['eligible']))
            events.append(dict(stem=row['stem'],transition=row['transition'],variant=row['variant'],parent=event['parent'],
                               children=event['children'],old_eligible=event['eligible'],nodes=nodes,
                               all_assigned_within_7um=all_assigned,all_isolated=isolated))
    report=dict(status='source_match_isolation_diagnostic',totals={k:dict(v) for k,v in totals.items()},events=events,
                labels_changed=False,selection_opened=False,target_pilot_labels_used=False,authorized_for_submission=False,
                source_result_sha256=sha(evidence_path))
    destination=ROOT/'reports/experiments/native-division-match-isolation-v4.json'
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='events'}))


if __name__=='__main__':main()
