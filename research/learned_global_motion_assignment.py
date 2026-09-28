"""Fixed ordinary-link assignment using existing cached image-derived flow."""
import numpy as np
from global_motion_assignment import assign,contract as static_contract
from independent_motion_prior import SCALE,VARIANCE,NULL_LOGIT


def contract():
    return dict(static_contract(),motion='Frozen image-derived backward displacement at each child',
        flow_checkpoint_sha256='3006ee0f904640b16dd988d6404a1b3ac4f933ca68ca912d69fe4598f96d4788')


def link(coords,backward_um):
    points=np.asarray(coords,dtype=float); flow=np.asarray(backward_um,dtype=float)
    if (points.ndim!=2 or points.shape[1]!=4 or flow.shape!=(len(points),3)
        or not np.isfinite(points).all() or not np.isfinite(flow).all() or (points<0).any()
        or np.any(points[:,0]!=np.floor(points[:,0])) or np.any(flow[points[:,0]==0]!=0)):
        raise ValueError('Finite unchanged TZYX nodes and aligned child-to-parent micron flow required')
    edges=[]
    for t in sorted(set(points[:,0].astype(int))):
        source=np.flatnonzero(points[:,0]==t); target=np.flatnonzero(points[:,0]==t+1)
        if max(len(source),len(target))>2048: raise ValueError('Assignment guard exceeded; no truncation')
        delta=points[source,None,1:]*SCALE-(points[None,target,1:]*SCALE+flow[None,target])
        costs=.5*np.sum(delta**2/VARIANCE,axis=-1)
        edges.extend((int(source[s]),int(target[d])) for s,d in assign(costs,-NULL_LOGIT))
    return edges
