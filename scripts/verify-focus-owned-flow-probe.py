"""Replay geometry/linking/serialization of the small GPU probe without labels."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RUN='focus-owned-flow-probe-v1'
M=runpy.run_path(str(ROOT/'scripts/run-focus-owned-flow-probe.py'))


def verify_outputs(result,output,raw_root,split_path):
    import numpy as np
    import tracksdata as td
    from research.independent_motion_prior import link_motion
    from research.backward_flow_linking import link_backward_flow
    coords,shape,raw_record,fold=M['validate_probe_input'](raw_root,split_path)
    if (result['status']!='passed_cached_centroid_flow_functionality'
        or result['movie']!=M['STEM'] or result['frames']!=3 or result['processed_pairs']!=2
        or result['checkpoint_sha256']!=M['FLOW_SHA'] or result['frozen_flow_tensor_sha256']!=M['FLOW_TENSOR_SHA']
        or result['split_sha256']!=M['SPLIT_SHA'] or result['raw_terminal_sha256']!=M['RAW_SHA']
        or result['raw_checkpoint_sha256']!=raw_record['sha256']
        or result['coordinate_sha256']!=hashlib.sha256(coords.tobytes()).hexdigest()
        or result['static_zero_flow_topology_identical'] is not True
        or not 0<=result['maximum_interior_parity_error_um']<=2e-5
        or not 0<result['elapsed_seconds']<=3600
        or any(result[k] is not False for k in ('ground_truth_opened','selection_opened','target_audit_opened','authorized_for_submission'))):
        raise ValueError('Exact completed training-only raw-centroid flow probe required')
    sample=output/'sampled_flow.npz'
    if hashlib.sha256(sample.read_bytes()).hexdigest()!=result['sample_sha256']:
        raise ValueError('Sampled motion artifact checksum mismatch')
    with np.load(sample,allow_pickle=False) as data:
        if set(data.files)!={'coords','backward_um'} or not np.array_equal(data['coords'],coords):
            raise ValueError('Sampled motion changed raw centroid identity/order')
        flows=data['backward_um'].copy()
    if (flows.shape!=(len(coords),3) or not np.isfinite(flows).all()
        or (flows[coords[:,0]==0]!=0).any() or not np.any(flows!=0)):
        raise ValueError('Finite nontrivial aligned backward motion required')
    if len(result['sampler_receipts'])!=2:
        raise ValueError('Exactly two sampled frame receipts required')
    for row,t in zip(result['sampler_receipts'],(1,2)):
        points=coords[coords[:,0]==t,1:]
        count=int(np.any(points/[1,4,4]>[63,63,63],axis=1).sum())
        if (row['frame']!=t or row['sampled_nodes']!=len(points)
            or row['trailing_border_extended_nodes']!=count
            or row['policy']!='constant flow extension through trailing unsampled voxel centers'
            or row['coordinates_modified'] is not False or row['nodes_deleted'] is not False):
            raise ValueError('Exact declared boundary sampling and node preservation required')
    computed={'control':link_motion(coords),'candidate':link_backward_flow(coords,flows)}
    zero=link_backward_flow(coords,np.zeros_like(flows))
    if [(s,t) for s,t,p in computed['control']]!=[(s,t) for s,t,p in zero]:
        raise ValueError('Zero-flow static control mismatch')
    for arm,edges in computed.items():
        graph=td.graph.IndexedRXGraph.from_geff(str(output/(arm+'.geff')))[0]
        nodes=graph.node_attrs().sort('node_id')
        if not np.array_equal(nodes.select('t','z','y','x').to_numpy(),coords):
            raise ValueError('Persisted graph changed raw centroids')
        mapping={node:i for i,node in enumerate(nodes['node_id'].to_list())}
        columns=[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE,td.DEFAULT_ATTR_KEYS.EDGE_TARGET]
        actual=[] if graph.num_edges()==0 else sorted((mapping[s],mapping[t])
            for s,t in graph.edge_attrs(attr_keys=columns).select(columns).iter_rows())
        if actual!=sorted((s,t) for s,t,p in edges):
            raise ValueError('Persisted graph differs from frozen CPU linking replay')
        if result['graphs'][arm]!=dict(nodes=len(coords),edges=len(edges),exact_coordinate_round_trip=True):
            raise ValueError('Graph receipt differs from actual serialization')
    return dict(raw_coordinates_exact=True,static_control_replayed=True,flow_links_replayed=True,
        complete_training_frames=3,labels_opened=False)


def main():
    notebook=ROOT/f'kaggle/biohub-{RUN}/biohub-{RUN}.ipynb'
    folder=ROOT/f'.biohub/cache/kernel-outputs/{RUN}/focus_owned_flow_probe'
    nb=json.loads(notebook.read_text()); source=''.join(nb['cells'][1]['source'])
    for name,file in (('sources','source_hashes.json'),('runtime_sources','runtime_hashes.json')):
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
            and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name)
        bundle=ast.literal_eval(node.value)
        expected={k:hashlib.sha256(v.encode()).hexdigest() for k,v in bundle.items()}
        if json.loads((folder/file).read_text())!=expected:
            raise ValueError('Executed source receipt differs from frozen notebook')
    terminal=json.loads((folder/'launcher_terminal.json').read_text())
    if (terminal['status']!='completed' or terminal['run_id']!=RUN
        or terminal['declared_budget_seconds']!=3600 or not 0<terminal['elapsed_seconds']<=3600
        or terminal['submission_performed'] is not False):
        raise ValueError('Completed bounded GPU terminal required')
    result_path=folder/'outputs/result.json'; result=json.loads(result_path.read_text())
    replay=verify_outputs(result,folder/'outputs',ROOT/'.biohub/cache/kernel-outputs/focus3d-raw-detections-v1',
        ROOT/'research/independent_real_baseline_v1_split.json')
    report=dict(status='verified_cached_focus_owned_flow_probe_not_accuracy',result=result,
        terminal=terminal,cpu_replay=replay,source_sha256=dict(
            notebook=hashlib.sha256(notebook.read_bytes()).hexdigest(),result=hashlib.sha256(result_path.read_bytes()).hexdigest()),
        authorized_for_submission=False,caveat='Three training frames verify functionality only; no detector or tracking accuracy gain established.')
    target=ROOT/f'reports/experiments/{RUN}-result.json'
    if target.exists(): raise ValueError('Refuse to overwrite verified probe')
    target.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
