"""Image-only feasible assignment problems and counts-only source upper bounds."""
from collections import Counter,defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from research.trajectory_disagreement_data_v1 import adjacency


def problems(final, groups):
    """Same protected-node/capacity policy as frozen joint-assignment v1."""
    nodes={int(k):v for k,v in final['nodes'].items()}
    incoming,outgoing=adjacency(final['edges'])
    if any(len(v)>1 for v in incoming.values()) or any(len(v)>2 for v in outgoing.values()):
        raise ValueError('Invalid graph degrees')
    protected=set()
    for e in final['edges']:
        a,b=int(e['source_id']),int(e['target_id'])
        dt=int(nodes[b]['t'])-int(nodes[a]['t'])
        if dt<=0:raise ValueError('Noncausal edge')
        if dt!=1 or len(outgoing[a])==2:protected.update((a,b))
    index={int(c):i for i,c in enumerate(groups['children'])}
    if len(index)!=len(groups['children']):raise ValueError('Duplicate query')
    by_t=defaultdict(list)
    for child,i in index.items():
        p=int(groups['current'][i])
        if p<0 or child in protected or p in protected:continue
        if incoming[child]!=[p] or outgoing[p]!=[child]:raise ValueError('Current graph mismatch')
        by_t[int(nodes[child]['t'])].append(child)
    for t,children in sorted(by_t.items()):
        children=sorted(children);parents=sorted(incoming[c][0] for c in children)
        if len(set(parents))!=len(parents):raise ValueError('Duplicate capacity')
        pindex={p:i for i,p in enumerate(parents)}
        edge_index=np.full((len(children),len(parents)),-1,np.int64)
        for row,child in enumerate(children):
            i=index[child];a,b=groups['offsets'][i:i+2]
            for j in range(a,b):
                p=int(groups['parents'][j])
                if p in pindex:edge_index[row,pindex[p]]=j
        current=np.array([pindex[incoming[c][0]] for c in children])
        assert np.all(edge_index[np.arange(len(children)),current]>=0)
        yield dict(t=t,children=np.asarray(children),parents=np.asarray(parents),edge_index=edge_index,
                   current_cols=current,group_indices=np.array([index[c] for c in children]))


def counts_only(final,groups,targets,safe):
    """Uses source labels for feasibility only; returns no chosen edges/graphs."""
    known=targets>=0
    current_correct=int((groups['current'][known]==targets[known]).sum())
    upper=current_correct;counts=Counter();known_eligible=0
    for problem in problems(final,groups):
        ix=problem['edge_index'];valid=ix>=0;indices=problem['group_indices']
        expected=targets[indices];is_known=expected>=0
        wanted=(problem['parents'][None,:]==expected[:,None]) & is_known[:,None]
        reward=wanted.astype(np.float64)
        # Current assignment only breaks ties; the full tie budget stays below one label.
        rr=np.arange(len(indices));cc=problem['current_cols']
        reward[rr,cc]+=1e-6/max(1,len(indices))
        row,col=linear_sum_assignment(np.where(valid,-reward,np.inf))
        before=int(wanted[rr,cc].sum());after=int(wanted[row,col].sum())
        assert after>=before
        upper+=after-before;known_eligible+=int(is_known.sum())
        counts['eligible_frames']+=1;counts['eligible_queries']+=len(indices)
        counts['eligible_known_errors']+=int((is_known & ~wanted[rr,cc]).sum())
        counts['reachable_known_errors']+=int((is_known & ~wanted[rr,cc] & (wanted & valid).any(axis=1)).sum())
        counts['jointly_recoverable_additional_choices']+=after-before
        # Labels from close alternatives remain ambiguity, not definite negatives.
        for q,i in enumerate(indices):
            if not is_known[q] or wanted[q,cc[q]]:continue
            current_edge=ix[q,cc[q]]
            if safe[current_edge]:counts['eligible_definite_errors']+=1
    return dict(**counts,positive_queries=int(known.sum()),known_eligible_queries=known_eligible,
                current_correct=current_correct,constrained_upper_bound_correct=upper,
                upper_bound_is_not_a_model=True,oracle_predictions_exported=False)
