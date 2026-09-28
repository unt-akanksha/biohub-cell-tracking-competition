"""Fixed raw nodes, owned neural residual plus unchanged physical flow prior."""
import numpy as np
try:
    from independent_motion_prior import SCALE,VARIANCE,NULL_LOGIT
except ModuleNotFoundError:
    from research.independent_motion_prior import SCALE,VARIANCE,NULL_LOGIT


def link(coords,flow,scores):
    coords=np.asarray(coords,dtype=float); flow=np.asarray(flow,dtype=float)
    if coords.ndim!=2 or coords.shape[1]!=4 or flow.shape!=(len(coords),3):
        raise ValueError('Aligned raw points and physical flow required')
    if not np.isfinite(coords).all() or not np.isfinite(flow).all() or (coords<0).any() or (coords[:,0]!=np.floor(coords[:,0])).any():
        raise ValueError('Finite physical inputs required')
    edges=[]; seen=set()
    for t in sorted(set(coords[:,0].astype(int))):
        src,tgt=np.flatnonzero(coords[:,0]==t),np.flatnonzero(coords[:,0]==t+1)
        if max(len(src),len(tgt))>2048: raise ValueError('No truncation above node guard')
        if not len(src) or not len(tgt): continue
        raw=np.asarray(scores[t],dtype=float); seen.add(t)
        if raw.shape!=(len(src),len(tgt)) or not np.isfinite(raw).all(): raise ValueError('Exact finite neural matrix required')
        delta=coords[src,None,1:]*SCALE-(coords[None,tgt,1:]*SCALE+flow[None,tgt])
        combined=raw-.5*np.sum(delta**2/VARIANCE,axis=-1)
        maximum=np.maximum(combined.max(axis=0),NULL_LOGIT)
        weights=np.exp(combined-maximum)
        posterior=weights/(weights.sum(axis=0)+np.exp(NULL_LOGIT-maximum))
        proposed=[(float(posterior[i,j]),int(src[i]),int(tgt[j])) for i,j in zip(*np.where(posterior>.5))]
        degree,assigned={},set()
        for p,s,d in sorted(proposed,key=lambda r:(-r[0],r[1],r[2])):
            if d not in assigned and degree.get(s,0)<2:
                edges.append((s,d,p));assigned.add(d);degree[s]=degree.get(s,0)+1
    if set(scores)!=seen: raise ValueError('Unexpected/missing nonempty transition matrices')
    return edges
