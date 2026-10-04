import ast
import json
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
G=runpy.run_path(str(ROOT/'research/flow_transfer_contract.py'))


def test_builder_real_cli_has_its_own_import_path():
    result=subprocess.run([sys.executable,'-I',str(ROOT/'scripts/build-flow-tta-transfer.py'),'--check'],
        cwd=ROOT.parent,capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['run_id']=='flow-tta-transfer-v1'


def payloads():
    return [(ROOT/path).read_bytes() for path in (
        'reports/experiments/flow-spatial-tta-selection-v1-result.json',
        'reports/experiments/backward-flow-spatial-tta-probe-v2-result.json',
        'research/independent_real_baseline_v1_split.json')]


def test_actual_gate_and_reference():
    policy=G['verify'](*payloads())
    assert policy['exposed_transfer_only'] and not policy['source_validation_only']
    assert policy['new_target_movies_opened']==0
    reference=G['load_reference'](ROOT/'.biohub/cache/kernel-outputs/detector-spatial-tta-audit-v1/detector_spatial_tta_audit',
        policy,'12eca8b1f77b549cebb241bd81ced8f3b4b38bef16d18dce2a551c40d31e9d13')
    assert list(reference['records'])==G['STEMS']
    for index in range(3):
        changed=payloads(); changed[index]+=b' '
        with pytest.raises(ValueError): G['verify'](*changed)


@pytest.mark.parametrize('fault',['mode','reference','checkpoint','tta','ensemble','target','calibration'])
def test_transfer_request_rejects_other_recipes(fault):
    p=G['verify'](*payloads())
    args=SimpleNamespace(flow_transfer_diagnostic=True,flow_spatial_tta_reference=Path('ref'),
        sha256=p['checkpoint_sha256'],standalone_image_flow=True,detector_spatial_tta=True)
    G['validate_request'](args)
    if fault=='mode': args.flow_transfer_diagnostic=False
    if fault=='reference': args.flow_spatial_tta_reference=None
    if fault=='checkpoint': args.sha256='0'*64
    if fault=='tta': args.detector_spatial_tta=False
    if fault=='ensemble': args.detector_ensemble_secondary=Path('secondary')
    if fault=='target': args.embryo_audit=True
    if fault=='calibration': args.calibration_json=Path('calibration')
    with pytest.raises(ValueError): G['validate_request'](args)


def fixture():
    verify,m,t,c,s=runpy.run_path(str(ROOT/'tests/test_flow_spatial_tta_selection.py'))['fixture']()
    p=G['verify'](*payloads()); s=json.loads(payloads()[2])
    m['records']=m['records'][:4]
    for row,stem in zip(m['records'],G['STEMS']): row['stem']=stem
    calls=sum(r['processed_frames']-1 for r in m['records'])
    c.update(flow_spatial_tta=p,target_audit_opened=True)
    m.update(target_audit_opened=True,node_reference_manifest_sha256=G['REFERENCE_SHA'])
    m['detector_spatial_tta']['encode_calls']=calls
    m['flow_spatial_tta']['contract']=p
    m['flow_spatial_tta']['execution']['forward_calls']=calls
    m['flow_spatial_tta']['motion_execution']['zero_weight_skips']=calls
    return verify,m,t,c,s


@pytest.mark.parametrize('fault',['scope','source','target','coordinates','views','reference'])
def test_transfer_scorer_is_fail_closed(fault):
    verify,m,t,c,s=fixture(); verify(m,t,c,s)
    if fault=='scope': m['records'][0]['stem']='44b6_unopened'
    if fault=='source': c['flow_spatial_tta']['source_comparison_sha256']='b'*64
    if fault=='target': m['target_audit_opened']=False
    if fault=='coordinates': m['records'][0]['reference_nodes_identical']=False
    if fault=='views': m['flow_spatial_tta']['execution']['views']=1
    if fault=='reference': m['node_reference_manifest_sha256']='c'*64
    with pytest.raises(ValueError): verify(m,t,c,s)


def test_builders_bind_exact_new_transfer_not_old_source(monkeypatch):
    b=runpy.run_path(str(ROOT/'scripts/build-flow-tta-transfer.py'))
    nb,meta,target=b['build']()
    assert meta['kernel_sources'][-1]=='indarkarhana/biohub-detector-spatial-tta-embryo-audit-v1/1'
    assert nb['metadata']['codex']['flow_spatial_tta']==G['verify'](*payloads())
    assert '--flow-transfer-diagnostic' in ''.join(nb['cells'][-1]['source'])
    runtime=ast.literal_eval(b['assignment'](''.join(nb['cells'][1]['source']),'runtime_sources').value)
    assert "run_id='flow-tta-transfer-v1'" in runtime['run_selection.py']
    assert runtime['flow_transfer_contract.py']==(ROOT/'research/flow_transfer_contract.py').read_text()
    read=Path.read_text; path=target/meta['code_file']
    monkeypatch.setattr(Path,'read_text',lambda self,*a,**k: json.dumps(nb) if self==path else read(self,*a,**k))
    score,sm,_=b['build'](True)
    bundle=ast.literal_eval(b['assignment'](''.join(score['cells'][-1]['source']),'scoring_sources').value)
    assert json.loads(bundle['selection_launch.ipynb'])==nb
    assert 'research/flow_transfer_contract.py' in bundle
    assert not sm['enable_gpu'] and not sm['enable_internet']
    assert len(sm['title'])<=50 and len(sm['id'].split('/')[1])<=50
