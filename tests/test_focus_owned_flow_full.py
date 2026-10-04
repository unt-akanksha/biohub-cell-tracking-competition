import ast
import json
from pathlib import Path
import runpy
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
G=runpy.run_path(str(ROOT/'research/focus_owned_flow_full_contract.py'))


def inputs():
    return [(ROOT/path).read_bytes() for path in (
        'reports/experiments/focus-owned-flow-probe-v1-result.json',
        'research/independent_real_baseline_v1_split.json')]


def test_actual_gpu_probe_authorizes_only_exposed_diagnostic():
    policy=G['receipt'](*inputs())
    assert policy['stems']==G['STEMS'] and policy['new_target_movies_opened']==0
    assert policy['flow_views']==1 and not policy['detector_inference']
    assert not policy['authorized_for_submission'] and not policy['independent_confirmation']
    for index in (0,1):
        data=inputs(); data[index]+=b' '
        with pytest.raises(ValueError): G['receipt'](*data)


def test_full_builder_retains_probe_bytes_and_adds_only_bounded_lane():
    path=ROOT/'kaggle/biohub-focus-owned-flow-probe-v1/biohub-focus-owned-flow-probe-v1.ipynb'
    before=path.read_bytes()
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-focus-owned-flow-full.py'))['build']()
    assert path.read_bytes()==before
    assert meta['enable_gpu'] and not meta['enable_internet'] and not meta['enable_tpu']
    assert meta['kernel_sources'][-1]=='indarkarhana/biohub-focus-owned-flow-probe-v1/1'
    assert nb['metadata']['codex']['contract']==G['receipt'](*inputs())
    source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    bundle=ast.literal_eval(node.value)
    original=json.loads(before); old_source=''.join(original['cells'][1]['source'])
    old_node=next(n for n in ast.parse(old_source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    old_bundle=ast.literal_eval(old_node.value)
    assert bundle['probe_runtime.py']==old_bundle['run_pilot.py']
    assert bundle['raw_centroid_flow_sampling.py']==old_bundle['raw_centroid_flow_sampling.py']
    assert bundle['backward_flow_linking.py']==old_bundle['backward_flow_linking.py']
    assert "range(0,99,2)" in bundle['run_pilot.py']
    assert "np.array_equal(flows[selected],probe_flow)" in bundle['run_pilot.py']
    assert 'predict_video' not in bundle['run_pilot.py']
    assert "require_tracks=False" in bundle['run_pilot.py']
    assert "'--probe',str(probe)" in ''.join(nb['cells'][-1]['source'])


def test_real_full_builder_cli_without_repository_import_context():
    proc=subprocess.run([sys.executable,'-I',str(ROOT/'scripts/build-focus-owned-flow-full.py'),'--check'],
        cwd=ROOT.parent,capture_output=True,text=True,timeout=45)
    assert proc.returncode==0,proc.stderr
    assert json.loads(proc.stdout)==dict(run_id='focus-owned-flow-full-v1',gpu=True)
