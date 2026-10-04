import ast
import copy
import hashlib
import json
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
INFER = runpy.run_path(str(ROOT/'scripts/run-independent-selection-inference.py'))


def calibration():
    previous = runpy.run_path(str(ROOT/'tests/test_association_calibration_receipts.py'))
    result,_,split = previous['fixture']()
    fitting,diag = previous['M']['RUN']['scope'](split,'full')
    result.update(profile='full',fitting_stems=fitting,diagnostic_stems=diag,probe_inputs_replayed=True)
    result['fit']['fit_metrics']['known_null_columns']=1
    return result,split


@pytest.mark.parametrize('fault',['probe','overlap','nulls','weights','parameters'])
def test_inference_rejects_unverified_calibration(fault):
    result,split = calibration()
    INFER['calibration_receipt'](result,split,result['checkpoint_sha256'],'e'*64)
    if fault=='probe': result['profile']='probe'
    if fault=='overlap': result['fitting_stems'][0]=split['folds'][0]['selection'][0]
    if fault=='nulls': result['fit']['fit_metrics']['known_null_columns']=0
    if fault=='weights': result['frozen_after']['neural']='f'*64
    if fault=='parameters': result['fit']['parameters'][1]=10.
    with pytest.raises(ValueError):
        INFER['calibration_receipt'](result,split,result['checkpoint_sha256'],'e'*64)


def test_builders_bind_actual_runtime_identity_and_calibration_bytes(monkeypatch):
    result,split = calibration(); payload=json.dumps(result).encode()
    report=dict(status='verified_calibration_not_tracking_candidate',profile='full',
        checkpoint_sha256=result['checkpoint_sha256'],source_sha256=dict(result=hashlib.sha256(payload).hexdigest()))
    report_path=ROOT/'reports/experiments/association-calibration-fit-v1.json'
    raw_path=ROOT/'.biohub/cache/kernel-outputs/association-calibration-fit-v1/association_calibration_fit/outputs/result.json'
    training_path=ROOT/'reports/experiments/image-motion-linker-v2-training.json'
    training=dict(steps=1000,status='verified_image_motion_linker_training_not_selection',flow_unchanged=True,
        detector_unchanged=True,probe_inputs_replayed=True,checkpoint_sha256=result['checkpoint_sha256'],
        frozen_flow_sha256=result['frozen_after']['flow'])
    text,raw=Path.read_text,Path.read_bytes
    replacements={report_path:json.dumps(report),training_path:json.dumps(training)}
    monkeypatch.setattr(Path,'read_text',lambda self,*a,**k:replacements[self] if self in replacements else text(self,*a,**k))
    monkeypatch.setattr(Path,'read_bytes',lambda self,*a,**k:payload if self==raw_path else raw(self,*a,**k))
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-calibrated-motion-selection.py'))['build']()
    source=''.join(nb['cells'][1]['source'])
    assignment=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(assignment.value)
    scorer=runpy.run_path(str(ROOT/'scripts/score-independent-selection.py'))
    assert scorer['emitted_manifest_run_id'](runtime['run_selection.py'])==nb['metadata']['codex']['run_id']
    assert runtime['association_calibration_result.json'].encode()==payload
    assert '--calibration-json' in ''.join(nb['cells'][-1]['source'])
    assert meta['kernel_sources'][0]=='indarkarhana/biohub-image-motion-linker-v1/2'
    replacements[ROOT/'kaggle/biohub-calibrated-motion-selection-v1/biohub-calibrated-motion-selection-v1.ipynb']=json.dumps(nb)
    scored,metadata=runpy.run_path(str(ROOT/'scripts/build-calibrated-motion-scoring.py'))['build']()
    assert not metadata['enable_gpu'] and not metadata['enable_internet']
    for cell in scored['cells']:
        ast.parse(''.join(cell['source']))


def test_scorer_rejects_unregistered_or_changed_calibration():
    module=runpy.run_path(str(ROOT/'tests/test_independent_selection_scoring.py'))
    manifest,terminal,codex,split=module['fixture']()
    manifest['association_calibration']=dict(parameters=[0.,1.,-4.5],result_sha256='1'*64)
    with pytest.raises(ValueError,match='Unregistered'):
        module['M']['verify_manifest'](manifest,terminal,codex,split)
    codex['association_calibration']=copy.deepcopy(manifest['association_calibration'])
    module['M']['verify_manifest'](manifest,terminal,codex,split)
    manifest['association_calibration']['parameters'][0]=.1
    with pytest.raises(ValueError,match='Exact fitted'):
        module['M']['verify_manifest'](manifest,terminal,codex,split)
