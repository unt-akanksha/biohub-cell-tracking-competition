"""Recover complete sampled flow from immutable adjacent feature packets."""
import numpy as np


def assemble_flow(coords,packets):
    points=np.asarray(coords)
    if (points.ndim!=2 or points.shape[1]!=4 or not np.isfinite(points).all()
        or (points<0).any() or set(points[:,0])!=set(range(100))):
        raise ValueError('Complete finite100-frame raw movie required')
    flow=np.zeros((len(points),3),np.float32);covered=np.zeros(len(points),bool);count=0
    for t,p in enumerate(packets):
        if t>=99 or np.asarray(p['source_frame']).shape!=() or int(p['source_frame'])!=t:raise ValueError('Exact99 ordered transitions required')
        src=np.flatnonzero(points[:,0]==t);tgt=np.flatnonzero(points[:,0]==t+1)
        if (not np.array_equal(p['source_indices'],src) or not np.array_equal(p['target_indices'],tgt)
            or not np.array_equal(p['source_coords'],points[src,1:].astype(np.float32))
            or not np.array_equal(p['target_coords'],points[tgt,1:].astype(np.float32))):raise ValueError('Packet changed raw identities or coordinates')
        values=np.asarray(p['backward_um'])
        if values.dtype!=np.float32 or values.shape!=(len(tgt),3) or not np.isfinite(values).all() or covered[tgt].any():raise ValueError('Exact finite unique sampled flow required')
        flow[tgt]=values;covered[tgt]=True;count+=1
    if count!=99 or not np.array_equal(covered,points[:,0]>0):raise ValueError('Incomplete movie flow coverage')
    return flow
