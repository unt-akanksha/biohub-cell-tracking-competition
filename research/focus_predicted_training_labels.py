"""Sparse parent supervision on raw FOCUS detections, never unknown negatives."""
from collections import defaultdict
import numpy as np

SCALE=np.asarray([1.625,.40625,.40625])


def role(source_frame):
    """Disjoint frame blocks, with an unused temporal gap for adaptation audit."""
    if int(source_frame)!=source_frame or not 0<=source_frame<99:
        raise ValueError('Adjacent transitions in a complete100-frame movie required')
    if source_frame<=68:return 'fitting'
    if source_frame>=80:return 'diagnostic'
    return 'embargo'


def targets(coords,index_to_gt,truth_nodes,truth_edges,source_frame):
    points=np.asarray(coords,dtype=float)
    if (points.ndim!=2 or points.shape[1]!=4 or not np.isfinite(points).all()
        or (points<0).any() or (points[:,0]!=np.floor(points[:,0])).any()
        or (points[:,0]>=100).any()):
        raise ValueError('Finite complete-movie raw coordinates required')
    role(source_frame)
    if set(index_to_gt)!=set(range(len(points))):raise ValueError('Complete detection match map required')
    truth={int(k):np.asarray(v,dtype=float) for k,v in truth_nodes.items()}
    if (len(truth)!=len(truth_nodes) or any(v.shape!=(4,) or not np.isfinite(v).all() for v in truth.values())):
        raise ValueError('Unique finite ground-truth node geometry required')
    edges=set((int(s),int(d)) for s,d in truth_edges)
    if any(s not in truth or d not in truth for s,d in edges):raise ValueError('GT edge outside node inventory')
    incoming=defaultdict(list)
    for s,d in edges:incoming[d].append(s)
    by_gt=defaultdict(list)
    for i,g in index_to_gt.items():
        if g is None or g==-1:continue
        if int(g)!=g or int(g) not in truth:raise ValueError('Invalid matched GT identity')
        if points[i,0]!=truth[int(g)][0]:raise ValueError('Cross-frame match is invalid')
        by_gt[int(g)].append(i)
    src=np.flatnonzero(points[:,0]==source_frame);tgt=np.flatnonzero(points[:,0]==source_frame+1)
    if max(len(src),len(tgt))>2048:raise ValueError('No truncation beyond attention guard')
    local={int(i):j for j,i in enumerate(src)}
    labels=np.full(len(tgt),-1,dtype=np.int64);status=[]
    for column,i in enumerate(tgt):
        child=index_to_gt[int(i)]
        if child is None or child==-1:
            status.append('unknown_target');continue
        child=int(child)
        if len(by_gt[child])!=1:
            status.append('ambiguous_target_match');continue
        parents=incoming[child]
        if len(parents)!=1 or truth[parents[0]][0]!=source_frame:
            status.append('no_unique_adjacent_annotated_parent');continue
        parent=parents[0];matches=by_gt[parent]
        if len(matches)==1 and matches[0] in local:
            labels[column]=local[matches[0]];status.append('known_parent');continue
        distances=np.linalg.norm((points[src,1:]-truth[parent][1:])*SCALE,axis=1)
        if not len(distances) or np.all(distances>7.):
            labels[column]=len(src);status.append('known_parent_absent');continue
        status.append('ambiguous_or_unmatched_near_parent')
    return dict(source_indices=src,target_indices=tgt,labels=labels,status=status,
        source_frame=int(source_frame),role=role(source_frame),null_index=len(src))
