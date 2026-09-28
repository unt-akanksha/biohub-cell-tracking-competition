"""Prediction-only atomic corrections for a future calibrated abstention gate.

No learned gate or thresholds are supplied here. Every subset of complete
bipartite change components preserves lineage degrees. IDs locate graph
entities but never enter the feature matrix.
"""
from collections import Counter, defaultdict
import numpy as np
from research.trajectory_candidate_ranker_v1 import FEATURES as EDGE_FEATURES

FEATURES=('log_removed_edges','log_added_edges')+tuple(
    side+'_'+name for side in ('removed_mean','added_mean') for name in EDGE_FEATURES)


def _edges(graph):
    nodes=graph['nodes'];pairs={};incoming=Counter();outgoing=Counter()
    for edge in graph['edges']:
        a,b=int(edge['source_id']),int(edge['target_id'])
        if ((a,b) in pairs or str(a) not in nodes or str(b) not in nodes
                or nodes[str(a)]['t']>=nodes[str(b)]['t']):
            raise ValueError('Invalid or duplicate temporal edge')
        pairs[a,b]=edge;incoming[b]+=1;outgoing[a]+=1
    if max(incoming.values(),default=0)>1 or max(outgoing.values(),default=0)>2:
        raise ValueError('Invalid lineage degree')
    return pairs,outgoing


def components(initial,baseline,candidate):
    if baseline['nodes']!=candidate['nodes']:
        raise ValueError('Candidate changed cells or coordinates')
    old,outgoing=_edges(baseline);new,_=_edges(candidate)
    nodes=baseline['nodes'];observed=set(map(int,initial['nodes']))
    protected=set()
    for a,b in old:
        if (a not in observed or b not in observed or outgoing[a]==2
                or nodes[str(b)]['t']-nodes[str(a)]['t']!=1):
            protected.update((a,b))
    old_pairs,new_pairs=set(old),set(new)
    changed=old_pairs^new_pairs;adj=defaultdict(set)
    for a,b in changed:
        if (a in protected or b in protected or a not in observed or b not in observed
                or nodes[str(b)]['t']-nodes[str(a)]['t']!=1):
            raise ValueError('Correction touches protected or nonconsecutive incidence')
        t=int(nodes[str(b)]['t']);left=(t,0,a);right=(t,1,b)
        adj[left].add(right);adj[right].add(left)
    result=[];unseen=set(adj)
    while unseen:
        seed=min(unseen);pending=[seed];vertices=set()
        while pending:
            vertex=pending.pop()
            if vertex in vertices:continue
            vertices.add(vertex);pending.extend(adj[vertex]-vertices)
        unseen.difference_update(vertices)
        pairs={(v[2],w[2]) for v in vertices if v[1]==0 for w in adj[v]}
        result.append(dict(t=seed[0],removed=tuple(sorted(pairs&old_pairs)),
                           added=tuple(sorted(pairs&new_pairs))))
    return sorted(result,key=lambda r:(r['t'],r['removed'],r['added']))


def apply_mask(initial,baseline,candidate,accept):
    parts=components(initial,baseline,candidate);mask=np.asarray(accept)
    if mask.dtype.kind!='b' or mask.shape!=(len(parts),):
        raise ValueError('One boolean decision per complete correction required')
    current,_=_edges(baseline);proposed,_=_edges(candidate)
    for part,selected in zip(parts,mask):
        if not selected:continue
        for pair in part['removed']:del current[pair]
        for pair in part['added']:current[pair]=proposed[pair]
    result=dict(baseline,edges=[current[p] for p in sorted(current)])
    _edges(result)
    return result


def features(parts,groups,edge_matrix):
    matrix=np.asarray(edge_matrix)
    if matrix.shape!=(len(groups['parents']),len(EDGE_FEATURES)) or not np.isfinite(matrix).all():
        raise ValueError('Invalid prediction edge feature schema')
    children=np.repeat(groups['children'],np.diff(groups['offsets']))
    pairs=list(zip(map(int,groups['parents']),map(int,children)))
    index={pair:i for i,pair in enumerate(pairs)}
    if len(index)!=len(pairs):raise ValueError('Duplicate candidate edge')
    rows=[]
    for part in parts:
        row=[np.log1p(len(part['removed'])),np.log1p(len(part['added']))]
        for side in ('removed','added'):
            if any(pair not in index for pair in part[side]):
                raise ValueError('Changed edge lacks frozen prediction features')
            values=matrix[[index[p] for p in part[side]]]
            row.extend(values.mean(axis=0) if len(values) else np.zeros(len(EDGE_FEATURES)))
        rows.append(row)
    return np.asarray(rows,dtype=np.float64).reshape(len(parts),len(FEATURES))
