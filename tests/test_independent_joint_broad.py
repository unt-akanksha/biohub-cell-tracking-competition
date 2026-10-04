import ast
import copy
import json
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_joint_initialization_requires_exact_full_source_parent_only_checkpoint():
    validate = runpy.run_path(str(ROOT/'scripts/run-independent-association-pilot.py'))['validate_joint_initialization']
    stems = json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text())['folds'][0]['train']
    state = dict(step=1000,identity=dict(max_steps=1000,training_profile='full',training_stems=stems,
        motion_residual={'version':1},loss='sparse_parent_classification_with_null_v1'))
    validate(state,stems,{'version':1})
    for key,value in [('training_stems',stems[:4]),('training_profile','pilot'),
                      ('motion_residual',{'version':2}),('loss','sparse_parent_and_row_hard_negative_v1')]:
        bad = copy.deepcopy(state)
        bad['identity'][key] = value
        with pytest.raises(ValueError):
            validate(bad,stems,{'version':1})
    state['step'] = 100
    with pytest.raises(ValueError):
        validate(state,stems,{'version':1})


@pytest.mark.parametrize('steps',[100,1000])
def test_joint_notebook_binds_checkpoint_and_real_gpu_extension_gate(steps):
    builder = runpy.run_path(str(ROOT/'scripts/build-independent-joint-broad.py'))
    nb,meta = builder['build'](steps)
    launch = ''.join(nb['cells'][-1]['source'])
    assert f"'--steps','{steps}','--training-scope','full','--joint'" in launch
    assert 'independent_motion_broad_pair/outputs/control/last.pt' in launch
    assert builder['CHECKPOINT_SHA'] in launch
    assert "command.append('--motion-residual')" in launch
    assert '--row-negatives' not in launch
    assert meta['kernel_sources'] == ['indarkarhana/biohub-independent-motion-broad-pair-v1/1']
    assert not meta['enable_internet'] and not nb['metadata']['codex']['detector_frozen']
    assert nb['metadata']['codex']['declared_budget_seconds'] == 3600
    source = ''.join(nb['cells'][1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'runtime_sources')
    runtime = ast.literal_eval(assignment.value)
    runner = runtime['run_association.py']
    assert 'model.encode = joint_encode' in runner
    assert 'if not args.joint:' in runner
    assert 'scaler.state_dict()' in runner
    assert 'epoch_source = patch_train_epoch(epoch_source)' in runner
    assert "row['optimizer_steps_min'] < 90" in runner
    assert 'initial_module_hashes.items()' in runner
    assert 'if step > 100 and (step100_smoke is None' in runner
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))


def test_joint_selection_and_scoring_keep_exact_checkpoint_and_notebook(monkeypatch):
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-independent-joint-selection.py'))['build']('a'*64,1)
    launch = ''.join(nb['cells'][-1]['source'])
    assert 'independent_joint_broad/outputs/last.pt' in launch and 'a'*64 in launch
    assert meta['kernel_sources'] == ['indarkarhana/biohub-independent-joint-broad-v1/1']
    frozen = json.dumps(nb)
    path = ROOT/'kaggle/biohub-independent-joint-selection-v1/biohub-independent-joint-selection-v1.ipynb'
    original = Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda self,*a,**k: frozen if self == path else original(self,*a,**k))
    scored,metadata = runpy.run_path(str(ROOT/'scripts/build-independent-joint-scoring.py'))['build'](1)
    source = ''.join(scored['cells'][-1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'scoring_sources')
    assert ast.literal_eval(assignment.value)['selection_launch.ipynb'] == frozen
    assert metadata['kernel_sources'] == ['indarkarhana/biohub-independent-joint-selection-v1/1']
    assert not metadata['enable_gpu'] and not metadata['enable_internet']
