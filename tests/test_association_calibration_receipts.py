import copy
import hashlib
import json
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
M = runpy.run_path(str(ROOT/'scripts/verify-association-calibration.py'))


def fixture():
    split = json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text())
    fitting,diagnostic = M['RUN']['scope'](split,'probe')
    records = [dict(role=role,stem=stem,t_start=t,source_nodes=5,target_nodes=5,supervised_columns=1,
        known_null_columns=0,cache_sha256='a'*64,input_sha256='b'*64)
        for role,stems in [('fitting',fitting),('diagnostic',diagnostic)] for stem in stems for t in range(3)]
    hashes = dict(neural='c'*64,flow='e82b7255fb2cded608a800fb6627ec4972043491e636e22a0dcbd66fc5c93779')
    result = dict(status='completed_calibration_not_tracking_validation',profile='probe',functionality_passed=True,
        checkpoint_sha256=M['RUN']['CHECKPOINT_SHA'],split_sha256=M['RUN']['SPLIT_SHA'],
        fitting_stems=fitting,diagnostic_stems=diagnostic,frozen_before=hashes,frozen_after=dict(hashes),records=records,
        fit=dict(bounds=[[0.,1.],[.25,4.],[-12.,4.]],parameters=[.2,1.,-4.],converged=True,
            fit_metrics=dict(nll=.3,columns=12,known_null_columns=0),flow_control=dict(nll=.4)),
        diagnostic=dict(calibrated=dict(nll=.3,columns=6,known_null_columns=0)),
        selection_opened=False,target_audit_opened=False,authorized_for_submission=False)
    terminal = dict(status='completed',run_id='association-calibration-probe-v1',declared_budget_seconds=3600,
        elapsed_seconds=100,submission_performed=False)
    return result,terminal,split


def test_probe_receipt_is_functionality_only():
    report = M['verify'](*fixture(),'probe')
    assert report['windows']==18 and not report['authorized_for_submission']


@pytest.mark.parametrize('fault',['weights','scope','missing','duplicate','counts','bounds','nonfinite','timeout'])
def test_bad_probe_receipts_reject(fault):
    result,terminal,split = copy.deepcopy(fixture())
    if fault=='weights': result['frozen_after']['neural']='f'*64
    if fault=='scope': result['target_audit_opened']=True
    if fault=='missing': result['records'].pop()
    if fault=='duplicate': result['records'][1]=result['records'][0]
    if fault=='counts': result['fit']['fit_metrics']['columns']=13
    if fault=='bounds': result['fit']['parameters'][0]=2.
    if fault=='nonfinite': result['fit']['parameters'][0]=float('nan')
    if fault=='timeout': terminal['elapsed_seconds']=3601
    with pytest.raises(ValueError):
        M['verify'](result,terminal,split,'probe')


def test_full_builder_embeds_verified_probe_and_requires_full_scope(monkeypatch):
    result,terminal,split = fixture()
    report = M['verify'](result,terminal,split,'probe')
    payload = json.dumps(result).encode()
    report['source_sha256'] = dict(result=hashlib.sha256(payload).hexdigest())
    report_path = ROOT/'reports/experiments/association-calibration-probe-v1.json'
    probe_path = ROOT/'.biohub/cache/kernel-outputs/association-calibration-probe-v1/association_calibration_probe/outputs/result.json'
    read_text,read_bytes = Path.read_text,Path.read_bytes
    monkeypatch.setattr(Path,'read_text',lambda self,*a,**k:json.dumps(report) if self==report_path
        else payload.decode() if self==probe_path else read_text(self,*a,**k))
    monkeypatch.setattr(Path,'read_bytes',lambda self,*a,**k:payload if self==probe_path else read_bytes(self,*a,**k))
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-association-calibration-fit.py'))['build']()
    assert "'--scope', 'full'" in ''.join(nb['cells'][-1]['source'])
    assert 'probe_result.json' in ''.join(nb['cells'][1]['source'])
    assert nb['metadata']['codex']['run_id']=='association-calibration-fit-v1'
    assert meta['kernel_sources']==['indarkarhana/biohub-image-motion-linker-v1/2']
