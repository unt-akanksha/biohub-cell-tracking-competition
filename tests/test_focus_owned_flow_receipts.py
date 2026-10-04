import hashlib
from pathlib import Path
import runpy
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
M=runpy.run_path(str(ROOT/'scripts/run-focus-owned-flow-probe.py'))
V=runpy.run_path(str(ROOT/'scripts/verify-focus-owned-flow-probe.py'))
RAW=ROOT/'.biohub/cache/kernel-outputs/focus3d-raw-detections-v1'
SPLIT=ROOT/'research/independent_real_baseline_v1_split.json'


@pytest.fixture
def artifact(tmp_path):
    from research.independent_motion_prior import link_motion
    from research.backward_flow_linking import link_backward_flow
    from research.raw_centroid_flow_sampling import sample_raw_centroid_flow
    coords,shape,row,_=M['validate_probe_input'](RAW,SPLIT)
    field=np.full((3,64,64,64),.1,dtype=np.float32)
    flows=np.zeros((len(coords),3),dtype=np.float32); receipts=[]
    for t in (1,2):
        selected=np.flatnonzero(coords[:,0]==t)
        flows[selected],receipt=sample_raw_centroid_flow(field,coords[selected,1:],shape[1:])
        receipts.append(dict(frame=t,**receipt))
    sample=tmp_path/'sampled_flow.npz'; np.savez_compressed(sample,coords=coords,backward_um=flows)
    graphs={arm:M['persist_graph'](coords,edges,tmp_path/(arm+'.geff')) for arm,edges in (
        ('control',link_motion(coords)),('candidate',link_backward_flow(coords,flows)))}
    result=dict(status='passed_cached_centroid_flow_functionality',movie=M['STEM'],frames=3,processed_pairs=2,
        checkpoint_sha256=M['FLOW_SHA'],frozen_flow_tensor_sha256=M['FLOW_TENSOR_SHA'],
        split_sha256=M['SPLIT_SHA'],raw_terminal_sha256=M['RAW_SHA'],raw_checkpoint_sha256=row['sha256'],
        coordinate_sha256=hashlib.sha256(coords.tobytes()).hexdigest(),static_zero_flow_topology_identical=True,
        maximum_interior_parity_error_um=0.,elapsed_seconds=1.,ground_truth_opened=False,
        selection_opened=False,target_audit_opened=False,authorized_for_submission=False,
        sample_sha256=hashlib.sha256(sample.read_bytes()).hexdigest(),sampler_receipts=receipts,graphs=graphs)
    return result,tmp_path


def test_real_graph_cpu_replay_of_synthetic_motion_fixture(artifact):
    result,path=artifact
    replay=V['verify_outputs'](result,path,RAW,SPLIT)
    assert replay['flow_links_replayed'] and replay['raw_coordinates_exact']
    assert not replay['labels_opened']


@pytest.mark.parametrize('fault',['target','nan','boundary','nodes','sample_hash'])
def test_reject_bad_receipts(artifact,fault):
    result,path=artifact
    if fault=='target': result['target_audit_opened']=True
    if fault=='nan': result['maximum_interior_parity_error_um']=float('nan')
    if fault=='boundary': result['sampler_receipts'][0]['trailing_border_extended_nodes']+=1
    if fault=='nodes': result['graphs']['candidate']['nodes']-=1
    if fault=='sample_hash': result['sample_sha256']='0'*64
    with pytest.raises(ValueError): V['verify_outputs'](result,path,RAW,SPLIT)
