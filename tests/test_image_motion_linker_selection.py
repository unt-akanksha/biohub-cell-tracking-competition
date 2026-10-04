import ast
import copy
import json
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
INFER = runpy.run_path(str(ROOT/'scripts/run-independent-selection-inference.py'))


def test_completed_profile_rejects_probe_missing_flow_or_changed_supervision():
    state = dict(step=1000,optimizer=dict(state={0:dict(step=1000)}),frozen_flow_model={'head.bias':'fixture'},
        identity=dict(max_steps=1000,training_profile='full',motion_residual=dict(version=1),
            known_null=dict(version=1,absence_radius_um=7.,unknown_columns_supervised=False),
            frozen_modules=['unet','detect_head'],
            initialization_sha256='b64254aece75ef709e517621757a5313ac665e0803050ce4f13fa2a76602d6ef',
            loss='sparse_parent_with_annotated_missing_parent_null_v1',image_motion=INFER['image_motion_contract'](),
            frozen_flow_sha256='f'*64))
    INFER['check_completed_profile'](state)
    for damage in ('probe','weights','flow','nulls','updates'):
        bad = copy.deepcopy(state)
        if damage == 'probe':
            bad['step'] = bad['identity']['max_steps'] = 100
        elif damage == 'weights':
            bad.pop('frozen_flow_model')
        elif damage == 'flow':
            bad['identity']['image_motion']['flow_frozen'] = False
        elif damage == 'nulls':
            bad['identity']['known_null']['unknown_columns_supervised'] = True
        else:
            bad['optimizer']['state'][0]['step'] = 100
        with pytest.raises(ValueError):
            INFER['check_completed_profile'](bad)


def test_selection_and_cpu_scorer_bind_both_models_and_native_nodes(monkeypatch):
    report = dict(steps=1000,status='verified_image_motion_linker_training_not_selection',flow_unchanged=True,
        detector_unchanged=True,probe_inputs_replayed=True,checkpoint_sha256='a'*64,frozen_flow_sha256='f'*64)
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-image-motion-linker-selection.py'))['build'](report)
    assert meta['kernel_sources'] == ['indarkarhana/biohub-image-motion-linker-v1/2',
                                      'indarkarhana/biohub-independent-known-null-selection-v1/1']
    assert nb['metadata']['codex']['frozen_flow_sha256'] == 'f'*64
    runtime_assignment = next(n for n in ast.parse(''.join(nb['cells'][1]['source'])).body
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
        and n.targets[0].id == 'runtime_sources')
    runner = ast.literal_eval(runtime_assignment.value)['run_selection.py']
    scorer = runpy.run_path(str(ROOT/'scripts/score-independent-selection.py'))
    assert scorer['emitted_manifest_run_id'](runner) == nb['metadata']['codex']['run_id']
    launch = ''.join(nb['cells'][-1]['source'])
    assert 'image_motion_linker/outputs/last.pt' in launch
    assert "p/'independent_known_null_selection'" in launch
    assert '--edge-feature-tta' not in launch
    path = ROOT/'kaggle/biohub-image-motion-linker-selection-v1/biohub-image-motion-linker-selection-v1.ipynb'
    frozen = json.dumps(nb)
    read = Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda self,*a,**k:frozen if self==path else read(self,*a,**k))
    scored,metadata = runpy.run_path(str(ROOT/'scripts/build-image-motion-linker-scoring.py'))['build']()
    source = ''.join(scored['cells'][-1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'scoring_sources')
    assert ast.literal_eval(assignment.value)['selection_launch.ipynb'] == frozen
    assert not metadata['enable_gpu'] and not metadata['enable_internet']
