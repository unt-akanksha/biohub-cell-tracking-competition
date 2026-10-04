import ast
import copy
import json
from pathlib import Path
import runpy
import pytest

ROOT=Path(__file__).resolve().parents[1]
G=runpy.run_path(str(ROOT/'research/owned_detector_ensemble_contract.py'))
REPORT=ROOT/'reports/experiments/owned-detector-ensemble-probe-v1-result.json'
SPLIT=ROOT/'research/independent_real_baseline_v1_split.json'


@pytest.mark.parametrize('fault',['weight','mutation','scope','calls','nan'])
def test_probe_gate_rejects_invalid_evidence(fault):
    report=json.loads(REPORT.read_bytes())
    G['receipt'](REPORT.read_bytes(),SPLIT.read_bytes())
    row=report['result']; ensemble=row['candidate']['pre_motion_patch_receipt']
    if fault=='weight': ensemble['weights']=[.25,.75]
    if fault=='mutation': row['frozen_hashes'][0]['after']='a'*64
    if fault=='scope': row['target_audit_opened']=True
    if fault=='calls': ensemble['secondary_d4']['encode_calls']=1
    if fault=='nan': ensemble['maximum_mean_absolute_logit_delta']=float('nan')
    with pytest.raises(ValueError): G['receipt'](json.dumps(report).encode(),SPLIT.read_bytes())


def test_full_source_and_cpu_builders_are_bounded_and_registered(monkeypatch):
    module=runpy.run_path(str(ROOT/'scripts/build-owned-detector-ensemble-selection.py'))
    nb,meta,target=module['build']()
    assert nb['metadata']['codex']['declared_budget_seconds']==3600
    assert meta['enable_gpu'] and not meta['enable_internet']
    launch=''.join(nb['cells'][-1]['source'])
    assert '--detector-ensemble-secondary' in launch and '--embryo-audit' not in launch
    source=''.join(nb['cells'][1]['source'])
    runtime=ast.literal_eval(module['assignment'](source,'runtime_sources').value)
    scorer=runpy.run_path(str(ROOT/'scripts/score-independent-selection.py'))
    assert scorer['emitted_manifest_run_id'](runtime['run_selection.py'])==module['RUN']
    for name,code in runtime.items():
        if name.endswith('.py'): ast.parse(code)
    path=target/meta['code_file']; read=Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda self,*a,**k: json.dumps(nb) if self==path else read(self,*a,**k))
    scoring,scoring_meta,_=module['build'](True)
    assert len(scoring_meta['title'])<=50 and len(scoring_meta['id'].split('/')[1])<=50
    assert not scoring_meta['enable_gpu'] and not scoring_meta['enable_internet']
    sources=ast.literal_eval(module['assignment'](''.join(scoring['cells'][-1]['source']),'scoring_sources').value)
    assert sources['scripts/score-independent-selection.py']==(ROOT/'scripts/score-independent-selection.py').read_text()
    assert 'research/owned_detector_ensemble_contract.py' in sources


def test_scorer_rejects_unregistered_changed_weight_and_partial_execution():
    module=runpy.run_path(str(ROOT/'tests/test_independent_selection_scoring.py'))
    manifest,terminal,codex,split=module['fixture']()
    policy=dict(neural_weight=0.,spatial_weight=1.,null_logit=-4.5,
        flow_checkpoint_sha256='3006ee0f904640b16dd988d6404a1b3ac4f933ca68ca912d69fe4598f96d4788')
    d4=dict(version=1,views=8,encode_calls=32,features='native unchanged',
        logits='inverse-aligned XY D4 arithmetic mean',maximum_mean_absolute_logit_delta=.1)
    contract=G['receipt'](REPORT.read_bytes(),SPLIT.read_bytes())
    manifest.update(checkpoint_sha256=G['PARENT_SHA'],standalone_image_flow=policy,detector_spatial_tta=d4)
    codex.update(checkpoint_sha256=G['PARENT_SHA'],standalone_image_flow=policy,detector_spatial_tta=True)
    ensemble=dict(contract=contract,execution=dict(weights=[.5,.5],probability_mixture=True,
        features='parent native unchanged',encode_calls=32,parent_d4=d4,secondary_d4=copy.deepcopy(d4),
        maximum_mean_absolute_logit_delta=.1),frozen_hashes=[dict(before='a'*64,after='a'*64)]*2,
        secondary_provenance=dict(objective='pu'))
    manifest['owned_detector_ensemble']=ensemble
    with pytest.raises(ValueError,match='Unregistered'): module['M']['verify_manifest'](manifest,terminal,codex,split)
    codex['owned_detector_ensemble']=contract
    module['M']['verify_manifest'](manifest,terminal,codex,split)
    for fault in ('weight','calls','mutation','target'):
        bad=copy.deepcopy(manifest)
        if fault=='weight': bad['owned_detector_ensemble']['execution']['weights']=[.3,.7]
        if fault=='calls': bad['owned_detector_ensemble']['execution']['secondary_d4']['encode_calls']=31
        if fault=='mutation': bad['owned_detector_ensemble']['frozen_hashes'][0]['after']='b'*64
        if fault=='target': bad['target_audit_opened']=True
        with pytest.raises(ValueError): module['M']['verify_manifest'](bad,terminal,codex,split)
