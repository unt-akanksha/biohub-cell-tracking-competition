"""Image-only alternative sampling and label-safe source coverage diagnostics."""
from collections import Counter
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from research.native_correspondence_data_v2 import VOXEL, match_queries


def proposals_xy2(pooled):
    if pooled.shape != (64,128,128) or not np.isfinite(pooled).all():
        raise ValueError('Expected finite half-XY native volume')
    response=gaussian_filter(pooled,(.7,1.4,1.4))-gaussian_filter(pooled,(1.5,3.,3.))
    locations=np.argwhere((response==maximum_filter(response,(3,5,5))) & (response>.025)).astype(np.float32)
    if len(locations)>4096:raise ValueError('Candidate cap; do not truncate')
    locations[:,1:]=locations[:,1:]*2+.5
    return locations*VOXEL


def coverage(movie,t,parent_points,child_points):
    nodes={int(n[0]):n for n in movie['nodes']}
    parents=[n for n in movie['nodes'] if int(n[1])==t]
    children=[n for n in movie['nodes'] if int(n[1])==t+1]
    p_truth=np.array([n[2:] for n in parents],np.float32).reshape(-1,3)*VOXEL
    c_truth=np.array([n[2:] for n in children],np.float32).reshape(-1,3)*VOXEL
    pm=match_queries(p_truth,parent_points);cm=match_queries(c_truth,child_points)
    pd={int(n[0]):i for i,n in enumerate(parents)};cd={int(n[0]):i for i,n in enumerate(children)}
    outgoing={}
    for p,c in movie['edges']:
        if int(nodes[p][1])==t:outgoing.setdefault(int(p),[]).append(int(c))
    events=[]
    for parent,daughters in sorted(outgoing.items()):
        if len(daughters)!=2:continue
        ids=[parent,*sorted(daughters)];distances=[];matched=[];reasons=[];points=[]
        for index,node_id in enumerate(ids):
            truth=np.array(nodes[node_id][2:],np.float32)*VOXEL
            proposals=parent_points if index==0 else child_points
            mapping=pm if index==0 else cm;lookup=pd if index==0 else cd
            distance=float(np.linalg.norm(proposals-truth,axis=1).min()) if len(proposals) else None
            distances.append(distance)
            found=lookup[node_id] in mapping;matched.append(found)
            points.append(proposals[mapping[lookup[node_id]]] if found else None)
            reasons.append('matched' if found else ('no_proposal_within_3.25um' if distance is None or distance>3.25 else 'one_to_one_conflict'))
        displacement=[float(np.linalg.norm(points[j]-points[0])) if matched[0] and matched[j] else None for j in (1,2)]
        eligible=all(matched) and all(d<=20 for d in displacement)
        category='eligible' if eligible else ('missing_parent_and_daughter' if not matched[0] and not all(matched[1:]) else
                  'missing_parent_only' if not matched[0] else 'missing_daughter_only' if not all(matched[1:]) else 'displacement_over_20um')
        events.append(dict(parent=parent,children=ids[1:],nearest_um=distances,matched=matched,reasons=reasons,
                           displacement_um=displacement,eligible=eligible,category=category,
                           nearest_all_within_7um=all(d is not None and d<=7 for d in distances)))
    counts=dict(Counter(e['category'] for e in events))
    counts.update(events=len(events),parent_matched=sum(e['matched'][0] for e in events),
                  daughters_matched=sum(sum(e['matched'][1:]) for e in events),
                  nearest_all_within_7um=sum(e['nearest_all_within_7um'] for e in events),
                  one_to_one_conflicts=sum(e['reasons'].count('one_to_one_conflict') for e in events),
                  parent_proposals=len(parent_points),child_proposals=len(child_points))
    return dict(events=events,counts=counts)
