"""Candidate geometry in the parent's locally moving frame, using past edges only."""
import numpy as np
from research.learned_division_recovery import biological_geometry_score,VOXEL_SIZE_ZYX_UM

WIDTH=14


def geometry(first_step,second_step):
    a=np.asarray(first_step,dtype=float);b=np.asarray(second_step,dtype=float)
    score,_,midpoint,cosine,ratio=biological_geometry_score(a,b)
    distances=sorted((float(np.linalg.norm(a)),float(np.linalg.norm(b))))
    return np.asarray((*distances,float(np.linalg.norm(a-b)),midpoint,cosine,ratio,score))


def features(nodes,incoming,parent_id,first_id,second_id):
    ids=tuple(map(int,(parent_id,first_id,second_id)))
    if len(set(ids))!=3 or not set(ids)<=nodes.keys():raise ValueError('Three valid anchors required')
    rows=[nodes[i] for i in ids];t=int(rows[0][0])
    if any(int(r[0])!=t+1 for r in rows[1:]):raise ValueError('Daughters must follow parent')
    pos=np.asarray([r[1:] for r in rows],dtype=float)*VOXEL_SIZE_ZYX_UM
    if pos.shape!=(3,3) or not np.isfinite(pos).all():raise ValueError('Finite positions required')
    a,b=pos[1:]-pos[0];raw=geometry(a,b)
    previous=sorted(set(int(i) for i in incoming.get(ids[0],()) if i in nodes and int(nodes[i][0])==t-1))
    if len(previous)!=1:return np.zeros(WIDTH,dtype=np.float64),dict(available=False,raw_geometry=raw.tolist(),comoving_geometry=None)
    prev=previous[0];previous_position=np.asarray(nodes[prev][1:],dtype=float)*VOXEL_SIZE_ZYX_UM
    velocity=pos[0]-previous_position
    if not np.isfinite(velocity).all():raise ValueError('Nonfinite previous position')
    comoving=geometry(a-velocity,b-velocity)
    previous_previous=sorted(set(int(i) for i in incoming.get(prev,()) if i in nodes and int(nodes[i][0])==t-2))
    acceleration=0.
    if len(previous_previous)==1:
        older=np.asarray(nodes[previous_previous[0]][1:],dtype=float)*VOXEL_SIZE_ZYX_UM
        if not np.isfinite(older).all():raise ValueError('Nonfinite older position')
        acceleration=float(np.linalg.norm(velocity-(previous_position-older)))
    scales=np.asarray((12.,12.,15.,8.,1.,1.,8.))
    result=np.r_[1.,comoving/scales,float(np.linalg.norm(velocity))/8.,
                 (raw[6]-comoving[6])/8.,comoving[3]/(1.+raw[3]),
                 float(len(previous_previous)==1),acceleration/8.,
                 float(len(incoming.get(ids[0],()))==1)]
    if result.shape!=(WIDTH,) or not np.isfinite(result).all():raise ValueError('Invalid motion feature vector')
    return np.clip(result,-8.,8.),dict(available=True,raw_geometry=raw.tolist(),comoving_geometry=comoving.tolist())
