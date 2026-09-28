"""Aligned two-frame external division examples, no cross-domain time shortcut."""
import math
import numpy as np
from research.learned_division_recovery import biological_geometry_score


def aligned_biohub(patches):
    p=np.asarray(patches,dtype=np.float16)
    if p.ndim!=6 or p.shape[1:]!=(3,3,17,17,17):raise ValueError('Invalid Biohub patch layout')
    return p[:,:,[1,1,2]].copy()


def aligned_external(source_patches,target_patches,indices):
    source=np.asarray(source_patches,dtype=np.float16)
    target=np.asarray(target_patches,dtype=np.float16)
    ids=np.asarray(indices,dtype=np.int64)
    if source.shape[1:]!=(3,17,17,17) or target.shape[1:]!=(3,17,17,17) or ids.ndim!=2 or ids.shape[1]!=3:
        raise ValueError('Invalid external layout')
    if (ids<0).any() or (ids[:,0]>=len(source)).any() or (ids[:,1:]>=len(target)).any():
        raise ValueError('Invalid external node row')
    return np.stack((source[ids[:,0]][:,[1,1,2]],target[ids[:,1]][:,[0,0,1]],
                     target[ids[:,2]][:,[0,0,1]]),axis=1)


def shared_geometry(raw):
    g=np.asarray(raw,dtype=np.float32).copy()
    if g.ndim!=2 or g.shape[1]!=9 or not np.isfinite(g[:,:7]).all():raise ValueError('Invalid geometry')
    g[:,7:]=np.nan
    return g


def aligned_context(context,mask):
    c=np.asarray(context,dtype=np.float16).copy();m=np.asarray(mask,dtype=bool).copy()
    if c.shape[1:]!=(43,8) or m.shape!=c.shape[:2]:raise ValueError('Invalid context')
    c[:,3:11]=c[:,11:19];m[:,3:11]=m[:,11:19]
    c[:,3:11,0]=np.where(m[:,3:11],-.5,0.)
    return c,m


def geometry_and_anchors(parent,first,second):
    p=np.asarray(parent,dtype=np.float32);a=np.asarray(first,dtype=np.float32);b=np.asarray(second,dtype=np.float32)
    if p.shape!=(3,) or a.shape!=(3,) or b.shape!=(3,) or not np.isfinite([p,a,b]).all():raise ValueError('Invalid physical coordinates')
    x=a.astype(float)-p;y=b.astype(float)-p
    score,existing,midpoint,opposition,ratio=biological_geometry_score(x,y)
    geometry=np.asarray((np.linalg.norm(y),np.linalg.norm(x-y),existing,midpoint,opposition,ratio,score,np.nan,np.nan),dtype=np.float32)
    anchors=np.zeros((3,8),dtype=np.float32);anchors[0,5]=1.
    for i,delta in enumerate(sorted((a-p,b-p),key=lambda q:tuple(q)),1):
        anchors[i]=(.5,*np.clip(delta/20.,-2.,2.),math.log1p(float(np.linalg.norm(delta)))/math.log1p(30.),0.,1.,0.)
    eligible=bool(geometry[0]<=12. and geometry[1]<=15. and score>=3.)
    return geometry,anchors,eligible


def external_examples(data):
    positives=np.asarray(data['positive_mask'],dtype=bool)
    if positives.shape!=(len(data['source_ids']),len(data['target_ids'])):
        raise ValueError('External lineage mask misaligned')
    count=positives.sum(axis=1)
    if not np.array_equal(count==2,np.asarray(data['division_target'])>.5):raise ValueError('Division labels inconsistent')
    result=[]
    for row in range(len(positives)):
        children=np.flatnonzero(positives[row])
        children=sorted(children,key=lambda j:int(data['target_ids'][j]))
        if len(children) not in (1,2):continue
        first=children[0];p=data['source_coords_um'][row];a=data['target_coords_um'][first]
        def example(second,label):
            g,anchors,eligible=geometry_and_anchors(p,a,data['target_coords_um'][second])
            return dict(indices=(row,first,second),geometry=g,anchors=anchors,eligible=eligible,target=label)
        if len(children)==2:result.append(example(children[1],True))
        negatives=[]
        for proposed in range(positives.shape[1]):
            if positives[row,proposed]:continue
            value=example(proposed,False);g=value['geometry']
            if g[0]<=12. and g[1]<=15. and g[6]>=0.:negatives.append(value)
        negatives.sort(key=lambda r:(-float(r['geometry'][6]),float(r['geometry'][0]),float(r['geometry'][1]),int(data['target_ids'][r['indices'][2]])))
        result.extend(negatives[:2])
    return result
