"""Training-only Gaussian motion residual calibration; fixed topology/null."""
from collections import defaultdict,Counter
import numpy as np
from research.independent_motion_prior import SCALE,NULL_LOGIT

TRAIN_STEMS=('6bba_f1fde7e0','6bba_23af9eeb')
VARIANCE_FLOOR=SCALE**2/6.


def matched_residuals(coords,flow,index_to_gt,truth_edges):
    points=np.asarray(coords,dtype=float)
    flow=np.asarray(flow,dtype=float)
    if points.ndim!=2 or points.shape[1]!=4 or flow.shape!=(len(points),3):
        raise ValueError('Aligned raw geometry required')
    by_gt=defaultdict(list)
    for i,g in index_to_gt.items():
        if g is not None and int(g)>=0: by_gt[int(g)].append(int(i))
    degree=Counter(s for s,d in truth_edges)
    values=[]
    for s,d in truth_edges:
        if degree[s]!=1 or len(by_gt[s])!=1 or len(by_gt[d])!=1: continue
        a,b=by_gt[s][0],by_gt[d][0]
        if points[b,0]!=points[a,0]+1: continue
        values.append(points[a,1:]*SCALE-(points[b,1:]*SCALE+flow[b]))
    array=np.asarray(values,dtype=float).reshape(-1,3)
    if not np.isfinite(array).all(): raise ValueError('Finite residuals required')
    return array


def fit(residuals):
    values=np.asarray(residuals,dtype=float)
    if values.ndim!=2 or values.shape[1]!=3 or len(values)<100 or not np.isfinite(values).all():
        raise ValueError('At least100 finite training residuals required')
    return dict(mean_um=values.mean(axis=0).tolist(),variance_um2=np.maximum(values.var(axis=0),VARIANCE_FLOOR).tolist(),
                unconstrained_variance_um2=values.var(axis=0).tolist(),variance_floor_um2=VARIANCE_FLOOR.tolist(),
                observations=len(values),method='Pooled diagonal Gaussian maximum likelihood with native-voxel variance floor')


def link(coords,flow,parameters):
    points=np.asarray(coords,dtype=float)
    flow=np.asarray(flow,dtype=float)
    mean=np.asarray(parameters['mean_um'],dtype=float)
    variance=np.asarray(parameters['variance_um2'],dtype=float)
    if (points.ndim!=2 or points.shape[1]!=4 or flow.shape!=(len(points),3)
            or mean.shape!=(3,) or variance.shape!=(3,) or (variance<VARIANCE_FLOOR).any()
            or not all(np.isfinite(a).all() for a in (points,flow,mean,variance))
            or (points<0).any() or np.any(points[:,0]!=np.floor(points[:,0]))):
        raise ValueError('Finite unchanged coordinates, motion and constrained Gaussian required')
    edges=[]
    for t in sorted(set(points[:,0].astype(int))):
        src,tgt=np.flatnonzero(points[:,0]==t),np.flatnonzero(points[:,0]==t+1)
        if max(len(src),len(tgt))>2048: raise ValueError('Guard exceeded; no truncation')
        if not len(src) or not len(tgt): continue
        delta=points[src,None,1:]*SCALE-(points[None,tgt,1:]*SCALE+flow[None,tgt]+mean)
        weights=np.exp(-.5*np.sum(delta**2/variance,axis=-1))
        probabilities=weights/(weights.sum(axis=0,keepdims=True)+np.exp(NULL_LOGIT))
        proposals=[(float(probabilities[i,j]),int(src[i]),int(tgt[j])) for i,j in zip(*np.where(probabilities>.5))]
        degree,assigned={},set()
        for p,s,d in sorted(proposals,key=lambda r:(-r[0],r[1],r[2])):
            if d not in assigned and degree.get(s,0)<2:
                edges.append((s,d,p)); assigned.add(d); degree[s]=degree.get(s,0)+1
    return edges
