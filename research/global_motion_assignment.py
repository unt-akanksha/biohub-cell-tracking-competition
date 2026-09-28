"""Ordinary-link LAP ablation; image-derived nodes unchanged, no division repair.

Uses the existing source-trained physical variance and null cost. No threshold
sweep, graph-count objective, gap edges, node deletion, or fabricated detections.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from independent_motion_prior import SCALE,VARIANCE,NULL_LOGIT


def contract():
    return dict(version=1,method='Rectangular linear assignment with one private null column per child',
        scale=SCALE.tolist(),variance_um2=VARIANCE.tolist(),null_cost=-NULL_LOGIT,
        max_parents=1,max_children=1,divisions='Not modeled in this ordinary-link ablation',
        node_policy='All coordinates and node IDs preserved',maximum_nodes_per_frame=2048,
        threshold_sweep=False,graph_count_objective=False)


def assign(costs, null_cost):
    """Source x target costs; each child may decline every real parent."""
    costs=np.asarray(costs,dtype=np.float64)
    if costs.ndim!=2 or not np.isfinite(costs).all() or (costs<0).any() or not np.isfinite(null_cost) or null_cost<=0:
        raise ValueError('Finite nonnegative matrix and positive null cost required')
    ns,nt=costs.shape
    if max(ns,nt)>2048: raise ValueError('Assignment guard exceeded; no truncation')
    if not ns or not nt: return []
    matrix=np.full((nt,ns+nt),np.inf)
    # A cost tied with null is rejected rather than arbitrarily forced to a cell.
    matrix[:,:ns]=np.where(costs.T<null_cost,costs.T,np.inf)
    matrix[np.arange(nt),ns+np.arange(nt)]=null_cost
    rows,cols=linear_sum_assignment(matrix)
    return sorted((int(source),int(target)) for target,source in zip(rows,cols) if source<ns)


def link(coords):
    points=np.asarray(coords,dtype=np.float64)
    if (points.ndim!=2 or points.shape[1]!=4 or not np.isfinite(points).all() or (points<0).any()
        or np.any(points[:,0]!=np.floor(points[:,0]))):
        raise ValueError('Finite nonnegative TZYX nodes and integral time required')
    edges=[]
    for t in sorted(set(points[:,0].astype(int))):
        source=np.flatnonzero(points[:,0]==t); target=np.flatnonzero(points[:,0]==t+1)
        if max(len(source),len(target))>2048: raise ValueError('Assignment guard exceeded; no truncation')
        delta=(points[source,None,1:]-points[None,target,1:])*SCALE
        costs=.5*np.sum(delta**2/VARIANCE,axis=-1)
        edges.extend((int(source[s]),int(target[d])) for s,d in assign(costs,-NULL_LOGIT))
    return edges
