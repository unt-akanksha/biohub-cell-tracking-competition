"""Frozen physical-motion diagnostic; returns edges without changing detections."""
import numpy as np

SCALE = np.asarray([1.625,.40625,.40625])
TRAIN_STD = np.asarray([2.1445976973620673,1.1764392031579605,1.46154607742509])
# Difference of two uniform quantization errors, step 1.625um after downsampling.
VARIANCE = TRAIN_STD**2 + 1.625**2/6.
NULL_LOGIT = -4.5  # unassigned option at a 3-standard-deviation displacement


def link_motion(coords):
    points = np.asarray(coords,dtype=float).reshape(-1,4)
    if not np.isfinite(points).all() or np.any(points < 0) or np.any(points[:,0] != np.floor(points[:,0])):
        raise ValueError('Finite nonnegative coordinates with integer times required')
    edges = []
    for t in sorted(set(points[:,0].astype(int))):
        src,tgt = np.flatnonzero(points[:,0] == t),np.flatnonzero(points[:,0] == t+1)
        if not len(src) or not len(tgt):
            continue
        if max(len(src),len(tgt)) > 2048:
            raise ValueError('Diagnostic memory guard exceeded; no truncation')
        delta = (points[src,None,1:]-points[None,tgt,1:])*SCALE
        logits = -.5*np.sum(delta**2/VARIANCE,axis=-1)
        weights = np.exp(logits)
        probs = weights/(weights.sum(axis=0,keepdims=True)+np.exp(NULL_LOGIT))
        proposals = [(float(probs[i,j]),int(src[i]),int(tgt[j]))
                     for i,j in zip(*np.where(probs > .5))]
        degree,assigned = {},set()
        for probability,s,d in sorted(proposals,key=lambda row:(-row[0],row[1],row[2])):
            if d not in assigned and degree.get(s,0) < 2:
                edges.append((s,d,probability))
                assigned.add(d); degree[s] = degree.get(s,0)+1
    return edges
