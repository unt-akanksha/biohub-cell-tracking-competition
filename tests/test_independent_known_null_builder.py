import ast
import copy
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = runpy.run_path(str(ROOT/'scripts/build-independent-known-null.py'))


@pytest.mark.parametrize('steps', [100, 1000])
def test_frozen_profile_pins_initialization_and_does_not_reset_head(steps):
    nb, meta = BUILDER['build'](steps)
    assert meta['kernel_sources'] == ['indarkarhana/biohub-independent-joint-broad-v1/1']
    assert not meta['enable_internet'] and nb['metadata']['codex']['detector_frozen']
    assert not nb['metadata']['codex']['joint_training']
    assert nb['metadata']['codex']['declared_budget_seconds'] == 3600
    launch = ''.join(nb['cells'][-1]['source'])
    assert f"'--steps','{steps}','--training-scope','full','--known-null'" in launch
    assert "'--joint'" not in launch
    assert 'independent_joint_broad/outputs/last.pt' in launch
    assert BUILDER['CHECKPOINT_SHA'] in launch
    assert launch.index('tests/test_sparse_parent_missing_null.py') < launch.index('process = subprocess.Popen')
    source = ''.join(nb['cells'][1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'runtime_sources')
    runtime = ast.literal_eval(assignment.value)
    assert 'annotated_missing_parent.py' in runtime
    assert 'if not args.joint and not known_null:' in runtime['run_association.py']
    assert "run_id='independent-known-null-v1'" in runtime['run_association.py']
    assert "r['known_null_columns'] for r in history) == 0" in runtime['run_association.py']


@pytest.mark.parametrize('field,value', [('joint', True), ('row_negatives', True),
    ('training_scope', 'pilot'), ('motion_residual', False), ('sha256', '0'*64)])
def test_unsafe_training_profiles_rejected_before_torch_import(field, value):
    args = dict(known_null=True, joint=False, row_negatives=False, motion_residual=True,
                training_scope='full', steps=100, sha256=BUILDER['CHECKPOINT_SHA'])
    args[field] = value
    with pytest.raises(ValueError):
        runpy.run_path(str(ROOT/'scripts/run-independent-association-pilot.py'))['main'](SimpleNamespace(**args))


def test_selection_rejects_probe_and_incomplete_or_changed_null_profiles():
    verify = runpy.run_path(str(ROOT/'scripts/run-independent-selection-inference.py'))['check_completed_profile']
    state = dict(step=1000, optimizer=dict(state={0: dict(step=1000)}), identity=dict(
        max_steps=1000, training_profile='full', motion_residual=dict(version=1),
        known_null=dict(version=1,absence_radius_um=7.,unknown_columns_supervised=False),
        frozen_modules=['unet','detect_head'], initialization_sha256=BUILDER['CHECKPOINT_SHA'],
        loss='sparse_parent_with_annotated_missing_parent_null_v1'))
    verify(state)
    for mutate in ('probe', 'joint', 'unknown', 'missing_updates'):
        bad = copy.deepcopy(state)
        if mutate == 'probe':
            bad['step'] = bad['identity']['max_steps'] = 100
        elif mutate == 'joint':
            bad['identity']['joint_training'] = True
        elif mutate == 'unknown':
            bad['identity']['known_null']['unknown_columns_supervised'] = True
        else:
            bad['optimizer']['state'][0]['step'] = 80
        with pytest.raises(ValueError):
            verify(bad)


def test_selection_and_scorer_bind_frozen_detector_reference(monkeypatch):
    build = runpy.run_path(str(ROOT/'scripts/build-independent-known-null-selection.py'))['build']
    with pytest.raises(ValueError):
        build('a'*64, 1)
    nb, meta = build('a'*64, 2)
    assert meta['kernel_sources'] == ['indarkarhana/biohub-independent-known-null-v1/2',
                                     'indarkarhana/biohub-independent-joint-selection-v1/1']
    assert nb['metadata']['codex']['known_null']
    assert not nb['metadata']['codex']['edge_feature_tta']
    launch = ''.join(nb['cells'][-1]['source'])
    assert "command.extend(['--node-reference',str(reference)])" in launch
    assert 'independent_known_null/outputs/last.pt' in launch and 'a'*64 in launch
    path = ROOT/'kaggle/biohub-independent-known-null-selection-v1/biohub-independent-known-null-selection-v1.ipynb'
    frozen = json.dumps(nb)
    original = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda self,*a,**k: frozen if self == path else original(self,*a,**k))
    scored, metadata = runpy.run_path(str(ROOT/'scripts/build-independent-known-null-scoring.py'))['build']()
    source = ''.join(scored['cells'][-1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'scoring_sources')
    assert ast.literal_eval(assignment.value)['selection_launch.ipynb'] == frozen
    assert not metadata['enable_gpu']
