"""Isolated motion-field averaging on three training frames; fixed parent D4."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

CHECKPOINT_SHA='76f7da6e32c901e3e3f9d2ab870b8a22235b0b91086cfcf909aa41674896a144'
SPLIT_SHA='12eca8b1f77b549cebb241bd81ced8f3b4b38bef16d18dce2a551c40d31e9d13'


def main(args):
    for path,sha in ((args.checkpoint,CHECKPOINT_SHA),(args.manifest,SPLIT_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=sha: raise ValueError('Frozen provenance changed')
    sys.path[:0]=[str(args.repo/'scripts'),str(args.repo/'src'),str(args.runtime)]
    import numpy as np
    import torch
    import tracksdata as td
    from real_checkpoint_gpu_smoke import run_smoke
    from detector_spatial_tta import install_detector_spatial_tta
    from backward_flow_spatial_tta import install_backward_flow_spatial_tta
    from image_motion_residual import flow_hash
    if not torch.cuda.is_available(): raise RuntimeError('CUDA smoke required')
    torch.set_num_threads(2)
    fold=json.loads(args.manifest.read_text())['folds'][0]
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if state['step']!=1000 or state['identity']['training_stems']!=fold['train']:
        raise ValueError('Completed full-source checkpoint required')
    stem=fold['train'][0]
    if stem in fold['selection']+fold['audit_order']: raise ValueError('Training-only probe required')
    frozen=[]
    def control_flow(flow):
        frozen.append((flow,flow_hash(flow)))
        return dict(mode='native')
    def candidate_flow(flow):
        frozen.append((flow,flow_hash(flow)))
        return install_backward_flow_spatial_tta(flow)
    def guard(model):
        original=model.predict_edges
        receipt=dict(limit=2048,maximum_nodes=0,truncation=False)
        def predict(*values):
            count=max(values[0].shape[1],values[1].shape[1])
            receipt['maximum_nodes']=max(receipt['maximum_nodes'],count)
            if count>2048: raise RuntimeError('Node limit exceeded; no truncation')
            return original(*values)
        model.predict_edges=predict
        return receipt
    args.output.mkdir(parents=True,exist_ok=True)
    rows={}
    for arm,patch in (('control',control_flow),('optimized',control_flow),('candidate',candidate_flow)):
        rows[arm]=run_smoke(args.checkpoint,args.data/stem,args.output/arm,
            standalone_image_flow=True,pre_motion_patch=install_detector_spatial_tta,
            encode_patch=guard,flow_patch=patch,skip_zero_neural=arm!='control')
    coords=[]; edges=[]
    for arm in ('control','optimized','candidate'):
        graph=td.graph.IndexedRXGraph.from_geff(str(args.output/arm/'gpu_smoke.geff'))[0]
        values=graph.node_attrs(attr_keys=['node_id','t','z','y','x']).sort('node_id').select('t','z','y','x').to_numpy()
        coords.append(values)
        edge_keys=[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE,td.DEFAULT_ATTR_KEYS.EDGE_TARGET]
        edges.append(graph.edge_attrs(attr_keys=edge_keys).select(edge_keys).sort(edge_keys).to_numpy())
    if not len(coords[0]) or any(not np.array_equal(coords[0],value) for value in coords[1:]):
        raise ValueError('Flow-only probe changed detections')
    if not np.array_equal(edges[0],edges[1]) or rows['control']['scorer_counts']!=rows['optimized']['scorer_counts']:
        raise ValueError('Zero-weight shortcut changed real graph or official counts')
    for arm in ('optimized','candidate'):
        execution=rows[arm]['motion_execution_receipt']
        if execution['neural_forward_calls']!=0 or execution['zero_weight_skips']!=2:
            raise ValueError('Real zero-weight neural skip did not execute on both pairs')
    if rows['control']['motion_execution_receipt']['neural_forward_calls']!=2:
        raise ValueError('Native control did not execute both neural forward passes')
    hashes=[dict(before=before,after=flow_hash(flow)) for flow,before in frozen]
    receipt=rows['candidate']['flow_patch_receipt']
    if (any(row['before']!=row['after'] for row in hashes) or receipt['views']!=8
        or receipt['forward_calls']!=2 or receipt['maximum_mean_absolute_flow_delta_um']<=0):
        raise ValueError('Executed nontrivial frozen flow average required')
    result=dict(status='passed_flow_spatial_tta_functionality',checkpoint_sha256=CHECKPOINT_SHA,
        split_sha256=SPLIT_SHA,movie=stem,frames=3,**rows,frozen_flow_hashes=hashes,
        detections_identical=True,zero_weight_graph_identical=True,
        coordinate_sha256=hashlib.sha256(coords[0].tobytes()).hexdigest(),
        selection_opened=False,target_audit_opened=False,authorized_for_submission=False)
    (args.output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('repo','runtime','checkpoint','manifest','data','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
