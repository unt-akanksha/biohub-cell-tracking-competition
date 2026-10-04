import ast
import copy
import json
from pathlib import Path
import runpy
import pytest
ROOT=Path(__file__).resolve().parents[1]


def test_changed_detector_requires_explicit_fixed_flow_and_every_pair_averaged():
    module=runpy.run_path(str(ROOT/'tests/test_independent_selection_scoring.py'))
    manifest,terminal,codex,split=module['fixture']()
    policy=dict(neural_weight=0.,spatial_weight=1.,null_logit=-4.5,
        flow_checkpoint_sha256='3006ee0f904640b16dd988d6404a1b3ac4f933ca68ca912d69fe4598f96d4788')
    manifest.update(standalone_image_flow=policy,detector_spatial_tta=dict(version=1,views=8,encode_calls=32,
        features='native unchanged',logits='inverse-aligned XY D4 arithmetic mean',maximum_mean_absolute_logit_delta=.1),
        known_null_training=dict(version=1,absence_radius_um=7.,unknown_columns_supervised=False))
    with pytest.raises(ValueError,match='Unregistered'):
        module['M']['verify_manifest'](manifest,terminal,codex,split)
    codex.update(detector_spatial_tta=True,standalone_image_flow=copy.deepcopy(policy),known_null=True)
    module['M']['verify_manifest'](manifest,terminal,codex,split)
    for fault in ('views','pairs','policy','null_provenance'):
        bad=copy.deepcopy(manifest)
        if fault=='views': bad['detector_spatial_tta']['views']=4
        if fault=='pairs': bad['detector_spatial_tta']['encode_calls']=31
        if fault=='policy': bad['standalone_image_flow']['null_logit']=-5.
        if fault=='null_provenance': bad['known_null_training']['unknown_columns_supervised']=True
        with pytest.raises(ValueError): module['M']['verify_manifest'](bad,terminal,codex,split)


def test_builders_bind_detector_policy_without_fixed_coordinate_reference(monkeypatch):
    report_path=ROOT/'reports/experiments/detector-spatial-tta-probe-v1.json'
    training_path=ROOT/'reports/experiments/image-motion-linker-v2-training.json'
    replacements={report_path:json.dumps(dict(status='verified_detector_tta_probe_not_selection',source_sha256=dict(result='d'*64))),
        training_path:json.dumps(dict(steps=1000,status='verified_image_motion_linker_training_not_selection',flow_unchanged=True,
        detector_unchanged=True,probe_inputs_replayed=True,checkpoint_sha256='a'*64,frozen_flow_sha256='f'*64))}
    read=Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda self,*a,**k:replacements[self] if self in replacements else read(self,*a,**k))
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-detector-spatial-tta-selection.py'))['build']()
    launch=''.join(nb['cells'][-1]['source'])
    assert '--node-reference' not in launch and 'reference_candidates' not in launch
    assert '--standalone-image-flow' in launch and '--detector-spatial-tta' in launch
    assert nb['metadata']['codex']['detector_spatial_tta'] is True
    assert len(meta['kernel_sources'])==1
    assignment=next(n for n in ast.parse(''.join(nb['cells'][1]['source'])).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(assignment.value)
    scorer=runpy.run_path(str(ROOT/'scripts/score-independent-selection.py'))
    assert scorer['emitted_manifest_run_id'](runtime['run_selection.py'])==nb['metadata']['codex']['run_id']
    replacements[ROOT/'kaggle/biohub-detector-spatial-tta-selection-v1/biohub-detector-spatial-tta-selection-v1.ipynb']=json.dumps(nb)
    scored,metadata=runpy.run_path(str(ROOT/'scripts/build-detector-spatial-tta-scoring.py'))['build']()
    assert not metadata['enable_gpu'] and not metadata['enable_internet']
    for cell in scored['cells']: ast.parse(''.join(cell['source']))
