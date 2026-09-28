"""Full-movie event refinement with exact geometry and fork dominance reduction."""
from collections import Counter
import time
import numpy as np

from research.trajectory_event_candidates_v1 import frames
from research.trajectory_event_features_v1 import FEATURES
from research.trajectory_event_fast_features_v1 import FeatureContext
from research.trajectory_event_dominance_v1 import infer


def refine(initial,graph,groups,edge_matrix,weights,*,per_frame_seconds=2.,max_seconds=120.):
    weights=np.asarray(weights,np.float64)
    if weights.shape!=(len(FEATURES),) or not np.isfinite(weights).all():
        raise ValueError('Invalid event model weights')
    if not all(np.isfinite(x) and x>0 for x in (per_frame_seconds,max_seconds)):
        raise ValueError('Positive finite inference deadlines required')
    start=time.monotonic()
    context=FeatureContext(graph,edge_matrix)
    original={(int(e['source_id']),int(e['target_id'])):e for e in graph['edges']}
    if len(original)!=len(graph['edges']):raise ValueError('Duplicate baseline edges')
    current=dict(original);records=[];budget_exhausted=False
    for case in frames(initial,graph,groups):
        remaining=max_seconds-(time.monotonic()-start)
        if remaining<=0:
            budget_exhausted=True;break
        x=context.features(case)
        remaining=max_seconds-(time.monotonic()-start)
        if remaining<=0:
            budget_exhausted=True;break
        chosen,evidence=infer(case,x.astype(np.float64)@weights,case['incumbent'],
                             time_limit=min(per_frame_seconds,remaining))
        row=dict(t=case['t'],options=len(chosen),**evidence)
        if evidence['changed']:
            parents=set(map(int,case['parents']));children=set(map(int,case['children']))
            old={(a,b) for a,b in original if a in parents and b in children}
            new=set()
            for p,a,b in case['options'][chosen.astype(bool)]:
                if p<0:continue
                for c in (a,b):
                    if c>=0:new.add((int(case['parents'][p]),int(case['children'][c])))
            for pair in old-new:del current[pair]
            for a,b in new-old:current[(a,b)]=dict(source_id=a,target_id=b)
            row.update(removed_edges=len(old-new),added_edges=len(new-old))
        records.append(row)
    added=set(current)-set(original);removed=set(original)-set(current)
    if added or removed:
        incoming,outgoing=Counter(),Counter()
        for a,b in current:
            assert a!=b and graph['nodes'][str(a)]['t']<graph['nodes'][str(b)]['t']
            incoming[b]+=1;outgoing[a]+=1
        if max(incoming.values(),default=0)>1 or max(outgoing.values(),default=0)>2:
            raise ValueError('Event composition violated lineage degrees')
        result=dict(graph,edges=[current[pair] for pair in sorted(current)])
    else:
        result=dict(graph)
    assert result['nodes']==graph['nodes']
    return result,dict(frames=records,processed_frames=len(records),budget_exhausted=budget_exhausted,
        solver_fallbacks=sum(r['fallback'] for r in records),added_edges=len(added),removed_edges=len(removed),
        nodes_and_coordinates_unchanged=True,seconds=time.monotonic()-start)
