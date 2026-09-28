"""Exact event dominance reduction, not a heuristic probability threshold.

A fork consuming parent p and children a,b is unnecessary when an allowed
continuation p->a plus an allowed birth b has strictly greater total score (or
vice versa). The replacement consumes exactly the same cells, so preserves all
constraints regardless of other selected events. Partial-label masks matter:
a forbidden birth cannot justify pruning an annotated two-daughter fork.
"""
import numpy as np

from research.trajectory_event_assignment_v1 import solve,validate_choice,EventSolveError


def allowed_options(case,scores,allowed=None):
    scores=np.asarray(scores,np.float64)
    if scores.shape!=(len(case['options']),) or not np.isfinite(scores).all():
        raise ValueError('Invalid event scores')
    if allowed is None:allowed=np.ones(len(scores),bool)
    allowed=np.asarray(allowed)
    if allowed.shape!=scores.shape or allowed.dtype.kind!='b':raise ValueError('Invalid allowed-option mask')
    retained=allowed.copy();births={};links={}
    for i,(p,a,b) in enumerate(case['options']):
        if not allowed[i]:continue
        if p<0:births[int(a)]=i
        elif a>=0 and b<0:links[(int(p),int(a))]=i
    pruned=0
    for i,(p,a,b) in enumerate(case['options']):
        if b<0 or not allowed[i]:continue
        for linked,born in ((a,b),(b,a)):
            first=links.get((int(p),int(linked)));second=births.get(int(born))
            if first is not None and second is not None:
                replacement=scores[first]+scores[second]
                # Keep numerical ties and near-ties. No tuned cutoff controls
                # biological event confidence; this is strict objective dominance.
                tolerance=1e-10*max(1.,abs(float(replacement)),abs(float(scores[i])))
                if replacement>scores[i]+tolerance:
                    retained[i]=False;pruned+=1;break
    return retained,dict(allowed_before=int(allowed.sum()),dominated_forks=pruned,allowed_after=int(retained.sum()))


def infer(case,scores,incumbent,*,time_limit=5.):
    validate_choice(case,incumbent)
    retained,reduction=allowed_options(case,scores)
    try:
        chosen=solve(case,scores,allowed=retained,time_limit=time_limit)
    except EventSolveError as error:
        return np.asarray(incumbent).copy(),dict(changed=False,fallback=True,reason=str(error),**reduction)
    improvement=float(np.asarray(scores)@(chosen.astype(float)-incumbent))
    if improvement<=1e-9:
        return np.asarray(incumbent).copy(),dict(changed=False,fallback=False,gain=improvement,**reduction)
    return chosen,dict(changed=not np.array_equal(chosen,incumbent),fallback=False,gain=improvement,**reduction)
