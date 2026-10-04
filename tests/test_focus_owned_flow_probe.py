import ast
import json
from pathlib import Path
import runpy
import subprocess
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
M=runpy.run_path(str(ROOT/'scripts/run-focus-owned-flow-probe.py'))
RAW=ROOT/'.biohub/cache/kernel-outputs/focus3d-raw-detections-v1'
SPLIT=ROOT/'research/independent_real_baseline_v1_split.json'


def test_real_cached_training_input_and_graph_round_trip(tmp_path):
    from research.independent_motion_prior import link_motion
    from research.backward_flow_linking import link_backward_flow
    coords,shape,row,fold=M['validate_probe_input'](RAW,SPLIT)
    assert row['stem']=='6bba_23af9eeb' and row['stem'] in fold['train']
    assert coords.shape==(285,4) and set(coords[:,0])=={0,1,2}
    before=coords.copy()
    edges=link_motion(coords)
    zero=link_backward_flow(coords,np.zeros((len(coords),3)))
    assert [(s,t) for s,t,p in edges]==[(s,t) for s,t,p in zero]
    result=M['persist_graph'](coords,edges,tmp_path/'real.geff')
    assert result['nodes']==285 and result['edges']>0
    np.testing.assert_array_equal(coords,before)


def test_real_raw_border_sampling_does_not_remove_or_move_nodes():
    from research.raw_centroid_flow_sampling import sample_raw_centroid_flow
    coords,shape,_,_=M['validate_probe_input'](RAW,SPLIT)
    before=coords.copy()
    field=np.zeros((3,64,64,64),dtype=np.float32)
    field[0]=1.; field[1]=-2.; field[2]=.5
    values,receipt=sample_raw_centroid_flow(field,coords[:,1:],shape[1:])
    np.testing.assert_array_equal(values,np.tile([1.,-2.,.5],(285,1)))
    assert receipt['trailing_border_extended_nodes']>0
    assert receipt['sampled_nodes']==285
    np.testing.assert_array_equal(coords,before)


@pytest.mark.parametrize('target',['split','terminal','npz'])
def test_real_input_hash_drift_rejected(monkeypatch,target):
    read=Path.read_bytes
    paths={'split':SPLIT,'terminal':RAW/'focus3d_raw_detections_terminal.json',
        'npz':RAW/'raw_detections/6bba_23af9eeb.npz'}
    monkeypatch.setattr(Path,'read_bytes',lambda p: read(p)+b' ' if p==paths[target] else read(p))
    with pytest.raises(ValueError): M['validate_probe_input'](RAW,SPLIT)


@pytest.mark.parametrize('edges',[
    [(0,1,.9),(0,1,.9)],[(0,2,.9)],[(0,1,.1)],[(0,1,float('nan'))],[(0,9,.9)],
    [(0,1,.9),(0,3,.8),(0,4,.7)],[(0,1,.9),(5,1,.8)]])
def test_graph_rejects_bad_edges(tmp_path,edges):
    coords=np.array([[0,1,1,1],[1,1,1,1],[2,1,1,1],[1,2,1,1],[1,3,1,1],[0,4,1,1]],dtype=float)
    with pytest.raises(ValueError): M['persist_graph'](coords,edges,tmp_path/'bad.geff')
    assert not (tmp_path/'bad.geff').exists()


def test_builder_real_cli_and_frozen_dependencies():
    proc=subprocess.run([sys.executable,'-I',str(ROOT/'scripts/build-focus-owned-flow-probe.py'),'--check'],
        cwd=ROOT.parent,capture_output=True,text=True,timeout=45)
    assert proc.returncode==0,proc.stderr
    assert json.loads(proc.stdout)['run_id']=='focus-owned-flow-probe-v1'
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-focus-owned-flow-probe.py'))['build']()
    assert meta['kernel_sources']==['indarkarhana/biohub-backward-flow-fit-v1/1',
        'indarkarhana/biohub-focus3d-raw-detections-v1/1']
    assert meta['enable_gpu'] and not meta['enable_internet']
    assert nb['metadata']['codex']['declared_budget_seconds']==3600
    source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(node.value)
    assert runtime['run_pilot.py']==(ROOT/'scripts/run-focus-owned-flow-probe.py').read_text()
    assert 'from research.independent_motion_prior' not in runtime['backward_flow_linking.py']
    assert "require_tracks=False" in runtime['run_pilot.py']
    assert 'predict_video' not in runtime['run_pilot.py']
    assert "'--steps'" not in ''.join(nb['cells'][-1]['source'])
