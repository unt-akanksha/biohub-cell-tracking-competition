"""Training-only frame selection and official one-to-one annotation records."""
import hashlib
import numpy as np


def calibration_scope(split, probe=True):
    fold=split['folds'][0]; train=fold['train']
    if len(train)!=120 or len(set(train))!=120 or set(train)&set(fold['selection']+fold['audit_order']):
        raise ValueError('Exact disjoint source-training pool required')
    diagnostic=train[::5]; diagnostic_set=set(diagnostic)
    fitting=[s for s in train if s not in diagnostic_set]
    if any(not s.startswith('6bba_') for s in train): raise ValueError('Source embryo only')
    return dict(fitting=fitting[:4] if probe else fitting,diagnostic=diagnostic[:2] if probe else diagnostic)


def select_frames(stem, annotated_times, total_frames):
    times=sorted(set(int(t) for t in annotated_times))
    if len(times)<3 or total_frames<2 or any(t<0 or t>=total_frames for t in times):
        raise ValueError('Three distinct in-bounds annotated frames required')
    ranked=sorted(times,key=lambda t:hashlib.sha256(f'{stem}:{t}:recall-calibration-v1'.encode()).digest())
    return sorted(ranked[:3])


def temporal_context(t, total_frames):
    if not isinstance(t,int) or not 0<=t<total_frames or total_frames<2:
        raise ValueError('Frame outside two-frame inference context')
    return ((0,1),0) if t==0 else ((t-1,t),1)


def annotated_confidences(pred_coords, probabilities, truth_coords):
    """All arrays in original t,z,y,x space; unmatched annotations remain zero."""
    import tracksdata as td
    import polars as pl
    from tracksdata.metrics import DistanceMatching
    coords=np.asarray(pred_coords,dtype=float); truth=np.asarray(truth_coords,dtype=float)
    probs=np.asarray(probabilities,dtype=np.float32)
    if (coords.ndim!=2 or coords.shape[1]!=4 or truth.ndim!=2 or truth.shape[1]!=4
        or not len(truth) or probs.shape!=(len(coords),)
        or not all(np.isfinite(a).all() for a in (coords,truth,probs))
        or (probs<=0).any() or (probs>1).any()
        or any((a<0).any() or np.any(a[:,0]!=np.floor(a[:,0])) for a in (coords,truth))):
        raise ValueError('Finite original-grid peaks and all frame annotations required')
    graphs=[]
    for values in (coords,truth):
        graph=td.graph.InMemoryGraph()
        for axis in ('z','y','x'): graph.add_node_attr_key(axis,pl.Float64,0.)
        if len(values): graph.bulk_add_nodes([dict(t=int(row[0]),z=float(row[1]),y=float(row[2]),x=float(row[3])) for row in values])
        graphs.append(graph)
    prediction,ground_truth=graphs
    result=np.zeros(len(truth),dtype=np.float32)
    if not len(coords): return result
    pred_ids=list(prediction.node_attrs()['node_id']); gt_ids=list(ground_truth.node_attrs()['node_id'])
    confidences=dict(zip(pred_ids,probs)); annotation_index={node:i for i,node in enumerate(gt_ids)}
    prediction.match(ground_truth,matching=DistanceMatching(max_distance=7.,scale=(1.625,.40625,.40625)))
    seen=set()
    for row in prediction.node_attrs().iter_rows(named=True):
        match=row[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
        if match is None or match<0: continue
        if match not in annotation_index or match in seen: raise ValueError('Non one-to-one official match')
        seen.add(match); result[annotation_index[match]]=confidences[row['node_id']]
    return result
