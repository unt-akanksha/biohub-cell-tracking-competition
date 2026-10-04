import copy
import hashlib
import json
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
COMPARE = runpy.run_path(str(ROOT/'scripts/summarize-backward-flow-selection.py'))['compare']


def fixture():
    def read(path):
        return json.loads((ROOT/path).read_text())
    native = read('reports/experiments/independent-joint-selection-v1-score.json')['result']
    causal = read('reports/experiments/causal-motion-selection-v1-score.json')['result']
    learned = read('reports/experiments/independent-known-null-selection-v1-score.json')['result']
    static = read('.biohub/cache/kernel-outputs/joint-static-motion-selection-v1/joint_static_motion_score/static_motion_score.json')
    result = copy.deepcopy(static)
    manifest = json.dumps(dict(checkpoint_sha256='a'*64)).encode()
    result.update(flow_checkpoint_sha256='a'*64,flow_manifest_sha256=hashlib.sha256(manifest).hexdigest())
    fit = dict(checkpoint_sha256='a'*64,small_fit_gate_passed=True,steps=1000)
    return result,manifest,fit,native,causal,static,learned


def test_weaker_control_cannot_be_mistaken_for_strongest_candidate():
    report = COMPARE(*fixture())
    assert report['decision'] == 'not_strongest_standalone'
    assert report['deltas']['static'] == 0
    assert report['deltas']['causal'] < 0 and report['authorized_for_submission'] is False


def test_changed_detection_metrics_or_motion_policy_rejected():
    values = fixture()
    values[0]['per_movie']['motion'][0]['num_pred_nodes'] -= 1
    with pytest.raises(ValueError,match='detections'):
        COMPARE(*values)
    values = fixture()
    values[0]['null_logit'] += 1
    with pytest.raises(ValueError,match='policy'):
        COMPARE(*values)
