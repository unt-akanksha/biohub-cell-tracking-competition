import ast
import copy
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD = runpy.run_path(str(ROOT/'scripts/build-independent-division-specialist.py'))
RUNNER = runpy.run_path(str(ROOT/'scripts/run-independent-association-pilot.py'))


def test_initialization_rejects_unfinished_or_unrelated_models():
    stems = ['train']
    identity = dict(max_steps=1000,training_profile='full',training_stems=stems,
        motion_residual=dict(version=1),known_null=dict(version=1,absence_radius_um=7.,unknown_columns_supervised=False),
        loss='sparse_parent_with_annotated_missing_parent_null_v1',frozen_modules=['unet','detect_head'])
    state = dict(step=1000,identity=identity)
    RUNNER['validate_division_initialization'](state,stems,dict(version=1))
    for key,value in [('loss','sparse_parent_classification_with_null_v1'),('max_steps',100),('frozen_modules',[])]:
        bad = copy.deepcopy(state)
        bad['identity'][key] = value
        with pytest.raises(ValueError):
            RUNNER['validate_division_initialization'](bad,stems,dict(version=1))


@pytest.mark.parametrize('steps',[100,1000])
def test_builder_preserves_detector_and_head_with_real_gates(steps):
    nb,meta = BUILD['build'](steps)
    assert meta['kernel_sources'] == ['indarkarhana/biohub-independent-known-null-v1/2']
    assert not meta['enable_internet'] and nb['metadata']['codex']['detector_frozen']
    assert nb['metadata']['codex']['declared_budget_seconds'] == 3600
    launch = ''.join(nb['cells'][-1]['source'])
    assert f"'--steps','{steps}','--training-scope','full','--division-specialist'" in launch
    assert "'--known-null'" not in launch and "'--joint'" not in launch
    assert BUILD['CHECKPOINT_SHA'] in launch and 'independent_known_null/outputs/last.pt' in launch
    assert launch.index('tests/test_division_missing_null_loss.py') < launch.index('process = subprocess.Popen')
    source = ''.join(nb['cells'][1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'runtime_sources')
    runtime = ast.literal_eval(assignment.value)
    runner = runtime['run_association.py']
    assert "run_id='independent-division-specialist-v1'" in runner
    assert 'WeightedRandomSampler' in runner and 'replacement=True,generator=generator' in runner
    assert "meta['targets'].numpy() for meta,_ in dataset._data" in runner
    assert "r['division_columns'] for r in history) == 0" in runner
    assert 'if not args.joint and not known_null:' in runner


def test_profile_rejects_old_checkpoint_before_torch_import():
    args = SimpleNamespace(division_specialist=True,known_null=False,joint=False,row_negatives=False,
        motion_residual=True,training_scope='full',steps=100,sha256='c'*64)
    with pytest.raises(ValueError):
        RUNNER['main'](args)


def test_full_selection_rejects_probe_and_changed_sampling():
    verify = runpy.run_path(str(ROOT/'scripts/run-independent-selection-inference.py'))['check_completed_profile']
    state = dict(step=1000,optimizer=dict(state={0:dict(step=1000)}),identity=dict(
        max_steps=1000,training_profile='full',motion_residual=dict(version=1),
        division_specialist=dict(version=1,division_window_mass=.5),
        known_null=dict(version=1,absence_radius_um=7.,unknown_columns_supervised=False),
        frozen_modules=['unet','detect_head'],initialization_sha256=BUILD['CHECKPOINT_SHA'],
        loss='division_balanced_with_annotated_missing_parent_null_v1',
        division_sampling=dict(division_windows=100,ordinary_windows=11000,division_mass=.5,replacement=True)))
    verify(state)
    for change in ('probe','sampling','initial','optimizer'):
        bad = copy.deepcopy(state)
        if change == 'probe':
            bad['step'] = bad['identity']['max_steps'] = 100
        elif change == 'sampling':
            bad['identity']['division_sampling']['division_mass'] = .7
        elif change == 'initial':
            bad['identity']['initialization_sha256'] = 'a'*64
        else:
            bad['optimizer']['state'][0]['step'] = 20
        with pytest.raises(ValueError):
            verify(bad)


def test_selection_and_scoring_bind_b642_reference(monkeypatch):
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-division-specialist-selection.py'))['build']('a'*64,2)
    assert meta['kernel_sources'] == ['indarkarhana/biohub-independent-division-specialist-v1/2',
        'indarkarhana/biohub-independent-known-null-selection-v1/1']
    launch = ''.join(nb['cells'][-1]['source'])
    assert "p/'independent_known_null_selection'" in launch
    assert 'independent_division_specialist/outputs/last.pt' in launch
    assert nb['metadata']['codex']['division_specialist']
    assert not nb['metadata']['codex']['edge_feature_tta']
    frozen = json.dumps(nb)
    path = ROOT/'kaggle/biohub-division-specialist-selection-v1/biohub-division-specialist-selection-v1.ipynb'
    original = Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda self,*a,**k:frozen if self == path else original(self,*a,**k))
    scored,metadata = runpy.run_path(str(ROOT/'scripts/build-division-specialist-scoring.py'))['build']()
    source = ''.join(scored['cells'][-1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'scoring_sources')
    assert ast.literal_eval(assignment.value)['selection_launch.ipynb'] == frozen
    assert not metadata['enable_gpu']
