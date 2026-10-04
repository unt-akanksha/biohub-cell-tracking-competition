import copy
import json
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
G = runpy.run_path(str(ROOT/'research/owned_detector_transfer_contract.py'))
B = runpy.run_path(str(ROOT/'scripts/build-owned-detector-transfer.py'))


def payloads():
    return ((ROOT/'reports/experiments/owned-detector-selection-v1-comparison.json').read_bytes(),
            (ROOT/'research/independent_real_baseline_v1_split.json').read_bytes())


def test_failed_gate_preserved_and_only_exposed_movies_used():
    report,split = payloads(); receipt = G['verify'](report,split)
    assert receipt['previously_exposed'] and receipt['new_target_movies_opened']==0
    assert not receipt['source_combined_gate_passed'] and not receipt['authorized_for_submission']
    for a,b in [(report+b' ',split),(report,split+b' ')]:
        with pytest.raises(ValueError): G['verify'](a,b)
    altered=json.loads(split); altered['folds'][0]['initial_complete_movie_audit'].reverse()
    with pytest.raises(ValueError): G['contract'](altered)


def test_gpu_builder_uses_frozen_trained_pu_and_existing_runtime():
    path=ROOT/'kaggle/biohub-owned-detector-pu-selection-v1/biohub-owned-detector-pu-selection-v1.ipynb'
    before=path.read_bytes(); nb,meta,_=B['build']()
    assert path.read_bytes()==before
    assert meta['kernel_sources']==['indarkarhana/biohub-owned-detector-fit-pair-v1/1']
    assert meta['enable_gpu'] and not meta['enable_internet']
    codex=nb['metadata']['codex']
    assert codex['declared_budget_seconds']==3600 and not codex['authorized_for_submission']
    assert codex['checkpoint_sha256']==G['CHECKPOINT_SHA']
    source=''.join(nb['cells'][1]['source']); runtime=__import__('ast').literal_eval(B['assignment'](source,'runtime_sources').value)
    assert G['verify'](runtime['frozen_owned_comparison.json'].encode(),runtime['split.json'].encode())==codex['owned_transfer_diagnostic']
    assert '--owned-transfer-diagnostic' in ''.join(nb['cells'][-1]['source'])


def fixture():
    original=runpy.run_path(str(ROOT/'tests/test_independent_selection_scoring.py'))
    manifest,terminal,codex,_=original['fixture']()
    nb,_,_=B['build'](); meta=nb['metadata']['codex']; _,split=payloads(); split=json.loads(split)
    terminal['run_id']=meta['run_id']; manifest['run_id']=meta['run_id']
    for key in ('checkpoint_sha256','owned_transfer_diagnostic','target_audit_opened','owned_detector_fit','standalone_image_flow'):
        manifest[key]=copy.deepcopy(meta[key]); codex[key]=copy.deepcopy(meta[key])
    codex.update(run_id=meta['run_id'],detector_spatial_tta=True)
    manifest['records']=[dict(stem=s,image_shape=[100,64,256,256],processed_frames=100) for s in G['STEMS']]
    manifest['detector_spatial_tta']=dict(version=1,views=8,encode_calls=396,features='native unchanged',
        logits='inverse-aligned XY D4 arithmetic mean',maximum_mean_absolute_logit_delta=.1)
    return original['M'],manifest,terminal,codex,split


@pytest.mark.parametrize('fault',[None,'undeclared','new_movie','promoted','checkpoint','audit','missing'])
def test_score_scope_cannot_be_relabelled_or_extended(fault):
    scorer,manifest,terminal,codex,split=fixture()
    if fault=='undeclared': codex.pop('owned_transfer_diagnostic')
    if fault=='new_movie': manifest['records'][-1]['stem']=split['folds'][0]['audit_order'][4]
    if fault=='promoted': manifest['owned_transfer_diagnostic']['source_combined_gate_passed']=True
    if fault=='checkpoint': manifest['checkpoint_sha256']='f'*64
    if fault=='audit': codex['embryo_audit']={}
    if fault=='missing': manifest['records'].pop()
    if fault:
        with pytest.raises(ValueError): scorer['verify_manifest'](manifest,terminal,codex,split)
    else: scorer['verify_manifest'](manifest,terminal,codex,split)


def test_cpu_builder_binds_exact_new_notebook(monkeypatch):
    nb,meta,target=B['build'](); content=json.dumps(nb)
    path=target/meta['code_file']; read=Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda self,*a,**k: content if self==path else read(self,*a,**k))
    score,meta,_=B['build'](True)
    source=''.join(score['cells'][-1]['source'])
    bundle=__import__('ast').literal_eval(B['assignment'](source,'scoring_sources').value)
    assert bundle['selection_launch.ipynb']==content
    assert 'research/owned_detector_transfer_contract.py' in bundle
    assert meta['kernel_sources']==['indarkarhana/biohub-owned-detector-pu-transfer-v1/1']
    assert not meta['enable_gpu'] and not meta['enable_internet']


def test_transfer_gain_never_becomes_source_gate_or_submission_authorization():
    _,split=payloads(); split=json.loads(split); receipt=G['contract'](split)
    baseline=json.loads((ROOT/'reports/experiments/detector-spatial-tta-audit-v1-score.json').read_text())
    score=copy.deepcopy(baseline['result'])
    score.update(status='scored_exposed_transfer_diagnostic',owned_transfer_diagnostic=receipt,checkpoint_sha256=G['CHECKPOINT_SHA'])
    for key in ('score','edge_jaccard'): score['summary'][key]+=.01
    for row in score['per_movie']: row['adj_edge_jaccard']+=.01
    manifest=dict(status='completed',owned_transfer_diagnostic=receipt,checkpoint_sha256=G['CHECKPOINT_SHA'],
        authorized_for_submission=False,target_audit_opened=True,ground_truth_opened=False,
        records=[dict(stem=s,processed_frames=100,image_shape=[100,64,256,256]) for s in G['STEMS']])
    compare=runpy.run_path(str(ROOT/'scripts/summarize-owned-detector-transfer.py'))['compare']
    result=compare(score,manifest,baseline,split)
    assert result['diagnostic_transfer_supported']
    assert not result['source_combined_gate_passed'] and not result['authorized_for_submission']
    assert result['new_target_movies_opened']==0
    score['summary']['node_recall']-=.006
    assert not compare(score,manifest,baseline,split)['diagnostic_transfer_supported']
    score['per_movie'].pop()
    with pytest.raises(ValueError): compare(score,manifest,baseline,split)
