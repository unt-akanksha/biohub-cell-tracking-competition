import ast
import copy
import json
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_calibration_scope_rejects_nonjoint_and_evaluation_movies():
    scope = runpy.run_path(str(ROOT/'scripts/run-independent-bn-calibration.py'))['calibration_scope']
    split = json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text())
    identity = dict(joint_training=True,training_stems=list(split['folds'][0]['train']))
    assert len(scope(identity,split)) == 120
    bad = copy.deepcopy(identity); bad['joint_training'] = False
    with pytest.raises(ValueError):
        scope(bad,split)
    split['folds'][0]['selection'][0] = identity['training_stems'][0]
    with pytest.raises(ValueError,match='evaluation'):
        scope(identity,split)


def test_builder_binds_source_and_numerical_gate_without_training():
    builder = runpy.run_path(str(ROOT/'scripts/build-independent-bn-calibration.py'))
    nb,meta = builder['build']()
    launch = ''.join(nb['cells'][-1]['source'])
    assert "runtime/'run_recalibration.py'" in launch
    assert 'tests/test_bn_recalibration.py' in launch
    assert 'independent_joint_broad/outputs/last.pt' in launch
    assert builder['CHECKPOINT_SHA'] in launch
    assert launch.index('gate.returncode') < launch.index('process = subprocess.Popen')
    assert meta['kernel_sources'] == ['indarkarhana/biohub-independent-joint-broad-v1/1']
    assert not meta['enable_internet'] and nb['metadata']['codex']['optimizer_steps'] == 0
    source = (ROOT/'scripts/run-independent-bn-calibration.py').read_text()
    assert 'require_tracks=False' in source
    assert "probe_smoke['status'] != 'passed'" in source
    assert "recalibrate_batchnorm(model,batches(stems,records)" in source
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))


def test_partial_calibration_cannot_enter_full_selection():
    check = runpy.run_path(str(ROOT/'scripts/run-independent-selection-inference.py'))['check_completed_profile']
    state = dict(step=1000,identity=dict(max_steps=1000,motion_residual={'version':1},
        bn_recalibration=dict(status='probe',movie_count=2,batches=2,parameters_unchanged=True)))
    with pytest.raises(ValueError,match='recalibration'):
        check(state)
    state['identity']['bn_recalibration'].update(status='completed',movie_count=120,batches=120)
    check(state)


def test_bn_selection_and_scoring_bind_frozen_notebook(monkeypatch):
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-independent-bn-selection.py'))['build']('a'*64,1)
    launch = ''.join(nb['cells'][-1]['source'])
    assert 'independent_bn_calibration/outputs/last.pt' in launch and 'a'*64 in launch
    assert meta['kernel_sources'] == ['indarkarhana/biohub-independent-bn-calibration-v1/1']
    frozen = json.dumps(nb)
    path = ROOT/'kaggle/biohub-independent-bn-selection-v1/biohub-independent-bn-selection-v1.ipynb'
    original = Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda self,*a,**k: frozen if self == path else original(self,*a,**k))
    scored,meta = runpy.run_path(str(ROOT/'scripts/build-independent-bn-scoring.py'))['build'](1)
    source = ''.join(scored['cells'][-1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'scoring_sources')
    assert ast.literal_eval(assignment.value)['selection_launch.ipynb'] == frozen
    assert meta['kernel_sources'] == ['indarkarhana/biohub-independent-bn-selection-v1/1']
    assert not meta['enable_gpu'] and not meta['enable_internet']
