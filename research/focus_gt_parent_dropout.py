"""Exact annotated-parent candidate dropout, strictly fitting-only."""
import hashlib
import numpy as np
try:
    from focus_parent_dropout import SCALE,SEED
except ModuleNotFoundError:
    from research.focus_parent_dropout import SCALE,SEED


def augment(sample,parent_gt_coords):
    if sample.get('role')!='fitting':raise ValueError('Only fitting samples may use GT augmentation')
    p=sample['packet'];coords=np.asarray(p['source_coords'],float);labels=np.asarray(p['labels']);ns=len(coords)
    if coords.shape!=(ns,3) or not np.isfinite(coords).all() or labels.ndim!=1 or labels.dtype!=np.int64 or (labels< -1).any() or (labels>ns).any():raise ValueError('Finite source geometry and valid labels required')
    parents=np.unique(labels[(labels>=0)&(labels<ns)])
    if not len(parents):return None
    centers={int(k):np.asarray(v,float) for k,v in parent_gt_coords.items()}
    if set(centers)!=set(parents) or any(c.shape!=(3,) or not np.isfinite(c).all() or np.linalg.norm((coords[k]-c)*SCALE)>7.+1e-12 for k,c in centers.items()):raise ValueError('Exact GT centers for every unique matched supervised parent required')
    key=f"{SEED}:{sample['stem']}:{int(p['source_frame'])}".encode()
    chosen=int(parents[int.from_bytes(hashlib.sha256(key).digest()[:8],'big')%len(parents)])
    keep=np.linalg.norm((coords-centers[chosen])*SCALE,axis=1)>7.
    if keep[chosen]:raise ValueError('Selected matched parent must be removed')
    retained=np.flatnonzero(keep);new_ns=len(retained);remap=np.full(ns,-1,np.int64);remap[retained]=np.arange(new_ns)
    changed=np.full(labels.shape,-1,np.int64);changed[labels==ns]=new_ns;synthetic=[]
    for parent in parents:
        mask=labels==parent
        if keep[parent]:changed[mask]=remap[parent]
        elif not new_ns or np.all(np.linalg.norm((coords[retained]-centers[int(parent)])*SCALE,axis=1)>7.):
            changed[mask]=new_ns;synthetic.extend(np.flatnonzero(mask).tolist())
    out={k:np.asarray(v).copy() for k,v in p.items()}
    for name in ('source_indices','source_coords','source_features','source_pos'):
        if len(out[name])!=ns:raise ValueError('Aligned source arrays required')
        out[name]=out[name][retained].copy()
    out['labels']=changed
    if not np.all(changed[labels==chosen]==new_ns):raise ValueError('Selected true parent must become absent')
    return dict(packet=out,stem=sample['stem'],role='fitting',provenance=dict(selected_parent_row=chosen,
        selected_gt_parent_coords=centers[chosen].tolist(),removed_source_rows=np.flatnonzero(~keep).tolist(),
        synthetic_null_columns=sorted(synthetic),source_nodes_before=ns,source_nodes_after=new_ns,
        eligible_nonempty_source=new_ns>0,ambiguous_removed_parent_targets=int(((labels>=0)&(labels<ns)&(changed==-1)).sum())))
