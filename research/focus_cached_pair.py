"""Array contract for replayable owned-head training on fixed FOCUS proposals."""
import numpy as np

REQUIRED={'source_indices','target_indices','source_coords','target_coords',
          'source_features','target_features','source_pos','target_pos','backward_um','labels','source_frame'}


def validate_pair(packet,all_coords):
    if set(packet)!=REQUIRED:raise ValueError('Exact cached feature-pair schema required')
    coords=np.asarray(all_coords)
    if coords.ndim!=2 or coords.shape[1]!=4 or not np.isfinite(coords).all():raise ValueError('Finite global raw coordinate inventory required')
    frame=np.asarray(packet['source_frame'])
    if frame.shape!=() or not np.issubdtype(frame.dtype,np.integer) or not 0<=int(frame)<99:
        raise ValueError('One adjacent source frame in0..98 required')
    sizes=[]
    for side,t in [('source',int(frame)),('target',int(frame)+1)]:
        ids=np.asarray(packet[side+'_indices'])
        expected=np.flatnonzero(coords[:,0]==t)
        if ids.dtype!=np.int64 or ids.shape!=expected.shape or not np.array_equal(ids,expected):
            raise ValueError('Every raw node must retain exact global identity and frame ordering')
        n=len(ids);sizes.append(n)
        if n>2048:raise ValueError('No cached attention truncation above2048 nodes')
        geometry=np.asarray(packet[side+'_coords'])
        if geometry.shape!=(n,3) or not np.array_equal(geometry,coords[ids,1:]):
            raise ValueError('Native subvoxel geometry changed')
        for field in ('features','pos'):
            value=np.asarray(packet[side+'_'+field])
            if value.shape!=(n,32) or value.dtype!=np.float32 or not np.isfinite(value).all():
                raise ValueError('Finite32-channel float32 feature/position arrays required')
    ns,nt=sizes
    flow=np.asarray(packet['backward_um']);labels=np.asarray(packet['labels'])
    if flow.shape!=(nt,3) or flow.dtype!=np.float32 or not np.isfinite(flow).all():raise ValueError('Finite target-aligned physical backward flow required')
    if labels.shape!=(nt,) or labels.dtype!=np.int64 or (labels< -1).any() or (labels>ns).any():
        raise ValueError('Audited integer parent/null/unknown labels required')
    return dict(source_frame=int(frame),source_nodes=ns,target_nodes=nt,
        known_parent=int(((labels>=0)&(labels<ns)).sum()),known_absent=int((labels==ns).sum()),unknown=int((labels==-1).sum()))
