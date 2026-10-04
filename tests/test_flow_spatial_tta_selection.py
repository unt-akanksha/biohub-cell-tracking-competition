import ast
import copy
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
G=runpy.run_path(str(ROOT/'research/flow_spatial_tta_contract.py'))


def contract():
    return G['receipt']((ROOT/'reports/experiments/backward-flow-spatial-tta-probe-v2-result.json').read_bytes(),
        (ROOT/'research/independent_real_baseline_v1_split.json').read_bytes())


@pytest.mark.parametrize('fault',['ensemble','target','calibration','reference','checkpoint','tta'])
def test_fixed_source_only_request_rejects_other_recipes(fault):
    args=SimpleNamespace(sha256=G['PARENT_SHA'],standalone_image_flow=True,detector_spatial_tta=True,
        flow_spatial_tta_reference=Path('reference'))
    G['validate_request'](args)
    if fault=='ensemble': args.detector_ensemble_secondary=Path('secondary')
    if fault=='target': args.embryo_audit=True
    if fault=='calibration': args.calibration_json=Path('calibration')
    if fault=='reference': args.node_reference=Path('different_reference')
    if fault=='checkpoint': args.sha256='a'*64
    if fault=='tta': args.detector_spatial_tta=False
    with pytest.raises(ValueError): G['validate_request'](args)


def test_exact_probe_and_split_binding():
    assert contract()['skip_zero_neural'] is True
    payload=(ROOT/'reports/experiments/backward-flow-spatial-tta-probe-v2-result.json').read_bytes()
    split=(ROOT/'research/independent_real_baseline_v1_split.json').read_bytes()
    with pytest.raises(ValueError): G['receipt'](payload+b' ',split)
    with pytest.raises(ValueError): G['receipt'](payload,split+b' ')


def fixture():
    module=runpy.run_path(str(ROOT/'tests/test_independent_selection_scoring.py'))
    manifest,terminal,codex,split=module['fixture']()
    policy=dict(neural_weight=0.,spatial_weight=1.,null_logit=-4.5,
        flow_checkpoint_sha256='3006ee0f904640b16dd988d6404a1b3ac4f933ca68ca912d69fe4598f96d4788')
    d4=dict(version=1,views=8,encode_calls=32,features='native unchanged',
        logits='inverse-aligned XY D4 arithmetic mean',maximum_mean_absolute_logit_delta=.1)
    value=contract()
    codex.update(checkpoint_sha256=G['PARENT_SHA'],detector_spatial_tta=True,
        standalone_image_flow=policy,flow_spatial_tta=value)
    manifest.update(checkpoint_sha256=G['PARENT_SHA'],detector_spatial_tta=d4,standalone_image_flow=policy,
        node_reference_manifest_sha256=G['REFERENCE_SHA'],flow_spatial_tta=dict(contract=value,
            execution=dict(views=8,forward_calls=32,output_precision='FP32 arithmetic mean',
                maximum_mean_absolute_flow_delta_um=.1),
            motion_execution=dict(skip_zero_neural=True,neural_forward_calls=0,zero_weight_skips=32),
            frozen_hashes=dict(flow_before=G['FLOW_TENSOR_SHA'],flow_after=G['FLOW_TENSOR_SHA'],
                neural_before='a'*64,neural_after='a'*64)))
    for row in manifest['records']: row['reference_nodes_identical']=True
    return module['M']['verify_manifest'],manifest,terminal,codex,split


@pytest.mark.parametrize('fault',['reference','nodes','partial','neural','mutation','unregistered','nan'])
def test_scorer_accepts_only_registered_unchanged_detection_motion(fault):
    verify,manifest,terminal,codex,split=fixture()
    verify(manifest,terminal,codex,split)
    if fault=='reference': manifest['node_reference_manifest_sha256']='c'*64
    if fault=='nodes': manifest['records'][0]['reference_nodes_identical']=False
    if fault=='partial': manifest['flow_spatial_tta']['execution']['forward_calls']=31
    if fault=='neural': manifest['flow_spatial_tta']['motion_execution']['neural_forward_calls']=1
    if fault=='mutation': manifest['flow_spatial_tta']['frozen_hashes']['neural_after']='b'*64
    if fault=='unregistered': del codex['flow_spatial_tta']
    if fault=='nan': manifest['flow_spatial_tta']['execution']['maximum_mean_absolute_flow_delta_um']=float('nan')
    with pytest.raises(ValueError): verify(manifest,terminal,codex,split)


def test_bounded_builders_keep_frozen_ensemble_untouched(monkeypatch):
    path=ROOT/'kaggle/biohub-owned-detector-ensemble-selection-v1/biohub-owned-detector-ensemble-selection-v1.ipynb'
    before=path.read_bytes()
    module=runpy.run_path(str(ROOT/'scripts/build-flow-spatial-tta-selection.py'))
    nb,meta,target=module['build']()
    assert path.read_bytes()==before
    assert nb['metadata']['codex']['flow_spatial_tta']==contract()
    assert nb['metadata']['codex']['declared_budget_seconds']==3600
    assert meta['enable_gpu'] and not meta['enable_internet']
    assert meta['kernel_sources'][-1]=='indarkarhana/biohub-detector-spatial-tta-selection-v1/1'
    source=''.join(nb['cells'][1]['source'])
    runtime=ast.literal_eval(module['assignment'](source,'runtime_sources').value)
    for name in ('image_motion_residual','backward_flow_spatial_tta','calibrated_motion_scores','flow_spatial_tta_contract'):
        assert runtime[name+'.py']==(ROOT/f'research/{name}.py').read_text()
    assert '--flow-spatial-tta-reference' in ''.join(nb['cells'][-1]['source'])
    scorer=runpy.run_path(str(ROOT/'scripts/score-independent-selection.py'))
    assert scorer['emitted_manifest_run_id'](runtime['run_selection.py'])==module['RUN']
    read=Path.read_text; frozen=target/meta['code_file']
    monkeypatch.setattr(Path,'read_text',lambda self,*a,**k: json.dumps(nb) if self==frozen else read(self,*a,**k))
    scored,scored_meta,_=module['build'](True)
    assert not scored_meta['enable_gpu'] and not scored_meta['enable_internet']
    assert len(scored_meta['title'])<=50 and len(scored_meta['id'].split('/')[1])<=50
    sources=ast.literal_eval(module['assignment'](''.join(scored['cells'][-1]['source']),'scoring_sources').value)
    assert sources['scripts/score-independent-selection.py']==(ROOT/'scripts/score-independent-selection.py').read_text()
    assert 'research/flow_spatial_tta_contract.py' in sources
