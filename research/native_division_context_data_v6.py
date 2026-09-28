"""Recover image-patch roles and request genuinely new temporal observations."""
import numpy as np


def context_layout(packet,transition):
    count=len(packet['patches']);triples=packet['triples'];coords=packet['coords']
    if not 0<=transition<=98 or triples.shape!=(len(packet['labels']),3):raise ValueError('Invalid event timing/shape')
    parent=set(map(int,triples[:,0]));daughter=set(map(int,triples[:,1:].ravel()))
    if parent & daughter or parent | daughter != set(range(count)):raise ValueError('Patch frame identity is ambiguous')
    positions=np.full((count,3),np.nan,np.float32)
    for ids,xyz in zip(triples,coords):
        for index,point in zip(ids,xyz):
            if np.isfinite(positions[index]).all() and not np.allclose(positions[index],point,rtol=0,atol=1e-5):raise ValueError('Image point identity changed')
            positions[index]=point
    if not np.isfinite(positions).all():raise ValueError('Unbound image patch')
    is_parent=np.array([i in parent for i in range(count)],bool)
    requested=np.where(is_parent,transition-1,transition+2)
    valid=(requested>=0)&(requested<100)
    # No fabricated temporal image at the edge. Downstream sees explicit invalid.
    times=np.where(valid,requested,-1).astype(np.int64)
    return dict(positions=positions,is_parent=is_parent,context_times=times,context_valid=valid)
