import copy
import json
from pathlib import Path
import runpy
import pytest

ROOT=Path(__file__).resolve().parents[1]


def example():
    helpers=runpy.run_path(str(ROOT/'tests/test_flow_tta_transfer.py'))
    policy=helpers['G']['verify'](*helpers['payloads']())
    baseline=json.loads((ROOT/'reports/experiments/detector-spatial-tta-audit-v1-score.json').read_text())
    score=copy.deepcopy(baseline['result'])
    score.update(status='scored_exposed_flow_transfer_diagnostic',source_manifest_run_id='flow-tta-transfer-v1',
        checkpoint_sha256=policy['checkpoint_sha256'],flow_transfer_diagnostic=policy,
        target_audit_opened=True,authorized_for_submission=False)
    manifest=dict(flow_spatial_tta=dict(contract=policy),checkpoint_sha256=policy['checkpoint_sha256'],
        target_audit_opened=True,authorized_for_submission=False,
        records=[dict(stem=stem,processed_frames=100,image_shape=[100,64,256,256]) for stem in policy['stems']])
    return score,manifest,baseline,policy


def test_unchanged_result_does_not_pass_transfer():
    compare=runpy.run_path(str(ROOT/'scripts/summarize-flow-tta-transfer.py'))['compare']
    result=compare(*example())
    assert not result['diagnostic_transfer_supported']
    assert not result['authorized_for_submission']
    assert result['new_target_movies_opened']==0


@pytest.mark.parametrize('fault',['node_count','recall','movie','partial','nan','policy'])
def test_comparator_rejects_inconsistent_transfer_evidence(fault):
    compare=runpy.run_path(str(ROOT/'scripts/summarize-flow-tta-transfer.py'))['compare']
    score,manifest,baseline,policy=example()
    if fault=='node_count': score['per_movie'][0]['num_pred_nodes']+=1
    if fault=='recall': score['per_movie'][0]['node_recall']-=.01
    if fault=='movie': score['per_movie'][0]['stem']='44b6_unopened'
    if fault=='partial': manifest['records'][0]['processed_frames']=99
    if fault=='nan': score['summary']['score']=float('nan')
    if fault=='policy': score['flow_transfer_diagnostic']={}
    with pytest.raises(ValueError): compare(score,manifest,baseline,policy)


def test_frozen_queue_cannot_launch_gpu_or_submit_and_rejects_drift(monkeypatch):
    path=ROOT/'scripts/run-flow-tta-transfer-scoring-queue.py'
    module=runpy.run_path(str(path)); module['validate_staging']()
    text=path.read_text()
    assert "'--accelerator'" not in text and "'competitions','submit'" not in text
    assert "open('x'" in text and "helper.INSPECT(cpu)['present']" in text
    assert 'helper.verify_push(receipt,cpu)' in text
    read=Path.read_bytes
    monkeypatch.setattr(Path,'read_bytes',lambda self: b'changed' if self.name=='biohub-flow-tta-transfer-scoring-v1.ipynb' else read(self))
    with pytest.raises(ValueError,match='Frozen transfer notebook changed'):
        module['validate_staging']()
