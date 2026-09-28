"""Fixed-node physical backward-flow links; same variance/null/topology controls."""
import numpy as np

from research.independent_motion_prior import SCALE,VARIANCE,NULL_LOGIT


def link_backward_flow(coords,backward_um):
    points = np.asarray(coords,dtype=float).reshape(-1,4)
    flow = np.asarray(backward_um,dtype=float)
    if (flow.shape != (len(points),3) or not np.isfinite(flow).all()
        or not np.isfinite(points).all() or (points<0).any()
        or (points[:,0] != np.floor(points[:,0])).any()):
        raise ValueError('Finite N,4 coordinates and aligned N,3 physical backward flows required')
    edges = []
    for t in sorted(set(points[:,0].astype(int))):
        src,tgt = np.flatnonzero(points[:,0] == t),np.flatnonzero(points[:,0] == t+1)
        if not len(src) or not len(tgt):
            continue
        if max(len(src),len(tgt)) > 2048:
            raise ValueError('Attention guard exceeded; no truncation')
        predicted_parent = points[tgt,1:]*SCALE+flow[tgt]
        delta = points[src,None,1:]*SCALE-predicted_parent[None]
        weights = np.exp(-.5*np.sum(delta**2/VARIANCE,axis=-1))
        probs = weights/(weights.sum(axis=0,keepdims=True)+np.exp(NULL_LOGIT))
        proposals = [(float(probs[i,j]),int(src[i]),int(tgt[j]))
                     for i,j in zip(*np.where(probs > .5))]
        degree,assigned = {},set()
        for probability,s,d in sorted(proposals,key=lambda row:(-row[0],row[1],row[2])):
            if d not in assigned and degree.get(s,0) < 2:
                edges.append((s,d,probability))
                assigned.add(d); degree[s] = degree.get(s,0)+1
    return edges
