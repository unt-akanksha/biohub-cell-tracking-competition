"""Compare nearest-only training labels with radius-constrained one-to-one labels.

CPU diagnostic only: not a scorer change, graph rewrite, or training patch.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def compare_matches(detections_um, truth_um, radius_um=5.):
    detections=np.asarray(detections_um,dtype=float)
    truth=np.asarray(truth_um,dtype=float)
    if (detections.ndim!=2 or truth.ndim!=2 or detections.shape[1]!=3 or truth.shape[1]!=3
        or not np.isfinite(detections).all() or not np.isfinite(truth).all()
        or not np.isfinite(radius_um) or radius_um<=0):
        raise ValueError('Finite physical coordinates and positive radius required')
    greedy={}; exact={}
    if len(detections) and len(truth):
        distance=cdist(detections,truth)
        nearest=distance.argmin(1); minimum=distance.min(1)
        taken=set()
        # Same nearest-only rule as the official training code. Equal-distance
        # ordering is explicitly stable here, not a claimed bitwise Torch replay.
        for index in np.argsort(minimum,kind='stable'):
            if minimum[index]>radius_um: break
            target=int(nearest[index])
            if target not in taken:
                greedy[int(index)]=target; taken.add(target)
        n=len(truth)
        dummy=(n+1)*(radius_um+1)
        cost=np.full((n,len(detections)+n),dummy)
        cost[:,:len(detections)]=np.where(distance.T<=radius_um,distance.T,dummy*2)
        rows,cols=linear_sum_assignment(cost)
        exact={int(c):int(r) for r,c in zip(rows,cols) if c<len(detections) and distance[c,r]<=radius_um}
        ambiguous=int(((distance<=radius_um).sum(1)>1).sum())
    else:
        ambiguous=0
    if len(exact)<len(greedy): raise ValueError('Maximum-cardinality matching lost available supervision')
    return dict(annotations=len(truth),detections=len(detections),greedy_matched=len(greedy),
        exact_matched=len(exact),additional_matched=len(exact)-len(greedy),
        changed_assignments=sum(greedy.get(k)!=v for k,v in exact.items()),ambiguous_detections=ambiguous)
