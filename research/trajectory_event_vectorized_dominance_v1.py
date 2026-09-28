"""Experimental exact vectorization of fork dominance; not deployed in live fits.

Valid event problems have unique options. Replacement arithmetic and the
reference's strict relative tie tolerance are unchanged. No score is fitted.
"""
import numpy as np

from research.trajectory_event_dominance_v1 import allowed_options as reference


def allowed_options(case,scores,allowed=None):
    options=np.asarray(case['options'])
    scores=np.asarray(scores,np.float64)
    if scores.shape!=(len(options),) or not np.isfinite(scores).all():
        raise ValueError('Invalid event scores')
    if allowed is None:allowed=np.ones(len(scores),bool)
    allowed=np.asarray(allowed)
    if allowed.shape!=scores.shape or allowed.dtype.kind!='b':
        raise ValueError('Invalid allowed-option mask')
    retained=allowed.copy()
    nchildren=int(case['nchildren'])
    if int(case['nparents'])>np.iinfo(np.int64).max//nchildren:
        return reference(case,scores,allowed)
    births=np.full(nchildren,-1,np.int64)
    birth_rows=np.flatnonzero(allowed & (options[:,0]<0))
    births[options[birth_rows,1]]=birth_rows
    link_rows=np.flatnonzero(allowed & (options[:,0]>=0) & (options[:,1]>=0) & (options[:,2]<0))
    fork_rows=np.flatnonzero(allowed & (options[:,2]>=0))
    if len(link_rows) and len(birth_rows) and len(fork_rows):
        keys=options[link_rows,0]*nchildren+options[link_rows,1]
        order=np.argsort(keys);keys=keys[order];link_rows=link_rows[order]
        p,a,b=options[fork_rows].T
        dominated=np.zeros(len(fork_rows),bool)
        for linked,born in ((a,b),(b,a)):
            wanted=p*nchildren+linked
            indices=np.searchsorted(keys,wanted)
            bounded=np.minimum(indices,len(keys)-1)
            second=births[born]
            valid=(indices<len(keys)) & (keys[bounded]==wanted) & (second>=0)
            rows=np.flatnonzero(valid)
            replacement=scores[link_rows[bounded[rows]]]+scores[second[rows]]
            current=scores[fork_rows[rows]]
            tolerance=1e-10*np.maximum(1.,np.maximum(np.abs(replacement),np.abs(current)))
            dominated[rows]|=replacement>current+tolerance
        retained[fork_rows[dominated]]=False
    return retained,dict(allowed_before=int(allowed.sum()),
        dominated_forks=int((allowed & ~retained).sum()),allowed_after=int(retained.sum()))
