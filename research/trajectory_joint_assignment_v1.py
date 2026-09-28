"""Fixed learned-cost assignment preserving node positions and lineage degrees."""
from collections import Counter, defaultdict
from copy import deepcopy
import numpy as np
from scipy.optimize import linear_sum_assignment
from research.trajectory_disagreement_data_v1 import adjacency


def apply(final, groups, matrix, weights):
    nodes = {int(k):v for k,v in final['nodes'].items()}
    incoming, outgoing = adjacency(final['edges'])
    if any(len(v)>1 for v in incoming.values()) or any(len(v)>2 for v in outgoing.values()):
        raise ValueError('Invalid input degrees')
    protected = set()
    for edge in final['edges']:
        a,b = int(edge['source_id']),int(edge['target_id'])
        dt = int(nodes[b]['t'])-int(nodes[a]['t'])
        if dt<=0:
            raise ValueError('Noncausal input edge')
        if dt!=1 or len(outgoing[a])==2:
            protected.update((a,b))
    scores = matrix.astype(np.float64) @ weights
    if not np.isfinite(scores).all() or len(scores)!=len(groups['parents']):
        raise ValueError('Invalid scores')
    index = {int(c):i for i,c in enumerate(groups['children'])}
    if len(index)!=len(groups['children']):
        raise ValueError('Duplicate query')
    by_t = defaultdict(list)
    for child,i in index.items():
        parent = int(groups['current'][i])
        if parent<0 or child in protected or parent in protected:
            continue
        if incoming[child]!=[parent] or outgoing[parent]!=[child]:
            raise ValueError('Group current parent does not reproduce graph')
        by_t[int(nodes[child]['t'])].append(child)
    changed = {}; per_frame = []
    for t, children in sorted(by_t.items()):
        children.sort()
        parents = sorted(incoming[c][0] for c in children)
        if len(set(parents))!=len(parents):
            raise ValueError('Assignment capacity drift')
        parent_index = {p:i for i,p in enumerate(parents)}
        cost = np.full((len(children),len(parents)),np.inf)
        for row,child in enumerate(children):
            i = index[child];a,b = groups['offsets'][i:i+2]
            for p,score in zip(groups['parents'][a:b],scores[a:b]):
                if int(p) in parent_index:
                    cost[row,parent_index[int(p)]] = -score
        current_cols = np.array([parent_index[incoming[c][0]] for c in children])
        if not np.isfinite(cost[np.arange(len(children)),current_cols]).all():
            raise ValueError('Current assignment must remain feasible')
        rr,cc = linear_sum_assignment(cost)
        improvement = float(cost[np.arange(len(children)),current_cols].sum()-cost[rr,cc].sum())
        edits = {}
        if improvement>1e-9:
            for row,col in zip(rr,cc):
                child,parent = children[int(row)],parents[int(col)]
                if incoming[child]!=[parent]:
                    edits[child]=parent
        changed.update(edits)
        per_frame.append(dict(t=t,queries=len(children),changed=len(edits),cost_improvement=improvement))
    result = deepcopy(final)
    for edge in result['edges']:
        if int(edge['target_id']) in changed:
            edge['source_id'] = changed[int(edge['target_id'])]
    old_pairs = [(int(e['source_id']),int(e['target_id'])) for e in final['edges']]
    new_pairs = [(int(e['source_id']),int(e['target_id'])) for e in result['edges']]
    assert len(set(new_pairs))==len(new_pairs) and len(new_pairs)==len(old_pairs)
    assert Counter(a for a,b in new_pairs)==Counter(a for a,b in old_pairs)
    assert Counter(b for a,b in new_pairs)==Counter(b for a,b in old_pairs)
    assert {(a,b) for a,b in new_pairs if a in protected or b in protected}=={(a,b) for a,b in old_pairs if a in protected or b in protected}
    assert result['nodes']==final['nodes']
    return result,dict(changed_edges=len(changed),protected_nodes=len(protected),
                       per_frame=per_frame,node_positions_unchanged=True,degrees_unchanged=True,
                       division_and_gap_incident_edges_unchanged=True)
