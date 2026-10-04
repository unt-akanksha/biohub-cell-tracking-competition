import json
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
M = runpy.run_path(str(ROOT/'scripts/run-independent-selection-inference.py'))


def test_completed_residual_smoke_is_supported_but_partial_checkpoints_are_rejected():
    M['check_completed_profile'](dict(step=100,identity=dict(max_steps=100,motion_residual={'version':1})))
    with pytest.raises(ValueError):
        M['check_completed_profile'](dict(step=90,identity=dict(max_steps=100,motion_residual={'version':1})))
    with pytest.raises(ValueError):
        M['check_completed_profile'](dict(step=100,identity=dict(max_steps=100)))


def test_source_selection_excludes_fit_and_target_embryo():
    manifest = json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text())
    identity = dict(training_stems=manifest['folds'][0]['train'][:4])
    assert M['selection_scope'](identity,manifest) == manifest['folds'][0]['selection']
    manifest['folds'][0]['selection'][0] = manifest['folds'][0]['audit_order'][0]
    with pytest.raises(ValueError,match='Target embryo'):
        M['selection_scope'](identity,manifest)


def test_selection_overlap_is_rejected():
    manifest = json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text())
    identity = dict(training_stems=manifest['folds'][0]['train'][:4])
    manifest['folds'][0]['selection'][0] = identity['training_stems'][0]
    with pytest.raises(ValueError,match='disjoint'):
        M['selection_scope'](identity,manifest)


def test_full_fit_requires_explicit_profile_and_exact_frozen_list():
    manifest = json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text())
    identity = dict(training_stems=list(manifest['folds'][0]['train']),training_profile='full')
    assert M['selection_scope'](identity,manifest) == manifest['folds'][0]['selection']
    identity.pop('training_profile')
    with pytest.raises(ValueError,match='training scope'):
        M['selection_scope'](identity,manifest)
    identity['training_profile'] = 'full'
    identity['training_stems'].pop()
    with pytest.raises(ValueError,match='training scope'):
        M['selection_scope'](identity,manifest)


def test_selection_builder_binds_checkpoint_and_does_not_launch_training():
    builder = runpy.run_path(str(ROOT/'scripts/build-independent-selection-inference.py'))
    nb, meta = builder['build']('1'*64,4)
    launch = ''.join(nb['cells'][-1]['source'])
    assert "runtime/'run_selection.py'" in launch and "runtime/'run_pilot.py'" not in launch
    assert '1'*64 in launch and not meta['enable_internet']
    assert meta['kernel_sources'] == ['indarkarhana/biohub-independent-real-pilot-v1/4']
    assert nb['metadata']['codex']['target_audit_opened'] is False
    assert "'notebooks/indarkarhana/biohub-independent-real-pilot-v1'" in launch
    assert "'kernels/indarkarhana/biohub-independent-real-pilot-v1'" in launch
    assert "run_id='independent-selection-inference-v1'" in launch
    assert "'notebooks/indarkarhana/biohub-independent-selection-inference-v1'" not in launch
    with pytest.raises(ValueError):
        builder['build']('bad',4)
