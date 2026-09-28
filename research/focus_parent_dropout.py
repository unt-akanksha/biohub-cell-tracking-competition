"""Conservative fitting-only candidate dropout with provable absent labels."""
import hashlib
import numpy as np

SCALE = np.array([1.625, .40625, .40625])
RADIUS_UM = 14.
SEED = 244691


def augment(sample):
    if sample.get('role') != 'fitting':
        raise ValueError('Only fitting samples may be augmented')
    p = sample['packet']; labels = np.asarray(p['labels']); coords = np.asarray(p['source_coords'])
    ns = len(coords)
    if (coords.shape != (ns, 3) or not np.isfinite(coords).all() or labels.dtype != np.int64
            or labels.ndim != 1 or (labels < -1).any() or (labels > ns).any()
            or np.asarray(p['source_indices']).shape != (ns,)):
        raise ValueError('Aligned finite source geometry and parent labels required')
    parents = np.unique(labels[(labels >= 0) & (labels < ns)])
    if not len(parents): return None
    key = f"{SEED}:{sample['stem']}:{int(p['source_frame'])}".encode()
    chosen = int(parents[int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % len(parents)])
    keep = np.linalg.norm((coords.astype(float) - coords[chosen]) * SCALE, axis=1) > RADIUS_UM
    retained = np.flatnonzero(keep); new_ns = len(retained)
    remap = np.full(ns, -1, np.int64); remap[retained] = np.arange(new_ns)
    new_labels = np.full(labels.shape, -1, np.int64); new_labels[labels == ns] = new_ns
    synthetic = np.zeros(labels.shape, dtype=bool)
    for parent in parents:
        mask = labels == parent
        if keep[parent]: new_labels[mask] = remap[parent]
        elif not new_ns or np.all(np.linalg.norm((coords[retained].astype(float) - coords[parent]) * SCALE, axis=1) > RADIUS_UM):
            new_labels[mask] = new_ns; synthetic[mask] = True
    out = {k: np.asarray(v).copy() for k, v in p.items()}
    for key in ('source_indices', 'source_coords', 'source_features', 'source_pos'):
        if len(out[key]) != ns: raise ValueError('All source tensors must be aligned')
        out[key] = out[key][retained].copy()
    out['labels'] = new_labels
    if not synthetic[labels == chosen].all(): raise ValueError('Selected known parent must become provably absent')
    return dict(packet=out, stem=sample['stem'], role='fitting', provenance=dict(
        selected_parent_row=chosen, removed_source_rows=np.flatnonzero(~keep).tolist(),
        synthetic_null_columns=np.flatnonzero(synthetic).tolist(), source_nodes_before=ns,
        source_nodes_after=new_ns, eligible_nonempty_source=new_ns > 0,
        ambiguous_removed_parent_targets=int(((labels >= 0) & (labels < ns) & (new_labels == -1)).sum())))
