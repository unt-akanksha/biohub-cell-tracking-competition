"""Diagnostic counts only; never generate oracle-selected prediction edges."""
from collections import defaultdict
import numpy as np
from research.independent_motion_prior import SCALE,VARIANCE,NULL_LOGIT


def classify_unlinked(coords,flow,index_to_gt,predicted_edges,truth_edges):
    points=np.asarray(coords,dtype=float)
    flow=np.asarray(flow,dtype=float)
    if points.ndim!=2 or points.shape[1]!=4 or flow.shape!=(len(points),3):
        raise ValueError('Aligned raw nodes and motion required')
    if not np.isfinite(points).all() or not np.isfinite(flow).all():
        raise ValueError('Finite geometry required')
    mapping={int(i):int(g) for i,g in index_to_gt.items() if g is not None and int(g)>=0}
    represented=defaultdict(list)
    for i,g in mapping.items():
        if not 0<=i<len(points): raise ValueError('Mapping references absent prediction')
        represented[g].append(i)
    truth=set(map(tuple,truth_edges))
    recovered={(mapping[s],mapping[d]) for s,d in predicted_edges if s in mapping and d in mapping}&truth
    grouped=defaultdict(list)
    for s,d in sorted(truth-recovered):
        if s not in represented or d not in represented: continue
        si,di=represented[s],represented[d]
        times=set(points[si,0])
        if len(times)!=1 or any(points[j,0]!=next(iter(times))+1 for j in di):
            raise ValueError('Official matches must preserve exact consecutive times')
        grouped[int(next(iter(times)))].append((si,di))
    result=dict(both_detected_unlinked=0,posterior_above_half_but_topology_blocked=0,
                distance_beats_null_but_parent_competition=0,all_pairs_at_or_beyond_null_distance=0)
    for t,missing in grouped.items():
        src,tgt=np.flatnonzero(points[:,0]==t),np.flatnonzero(points[:,0]==t+1)
        if max(len(src),len(tgt))>2048: raise ValueError('Same memory guard required')
        delta=points[src,None,1:]*SCALE-(points[None,tgt,1:]*SCALE+flow[None,tgt])
        costs=.5*np.sum(delta**2/VARIANCE,axis=-1)
        weights=np.exp(-costs)
        probability=weights/(weights.sum(axis=0,keepdims=True)+np.exp(NULL_LOGIT))
        sr,dr={int(i):j for j,i in enumerate(src)},{int(i):j for j,i in enumerate(tgt)}
        for si,di in missing:
            ix=np.ix_([sr[i] for i in si],[dr[i] for i in di])
            if float(probability[ix].max())>.5:
                key='posterior_above_half_but_topology_blocked'
            elif float(costs[ix].min()) < -NULL_LOGIT:
                key='distance_beats_null_but_parent_competition'
            else:
                key='all_pairs_at_or_beyond_null_distance'
            result[key]+=1
            result['both_detected_unlinked']+=1
    return result
