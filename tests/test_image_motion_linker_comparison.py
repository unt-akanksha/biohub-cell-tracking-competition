import copy
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT/'scripts/summarize-image-motion-linker-selection.py'))


def fixture():
    candidate,manifest,training,native,causal = runpy.run_path(str(ROOT/'tests/test_known_null_selection_comparison.py'))['inputs']()
    manifest.update(image_motion_training=dict(MODULE['TRAIN']['EXPECTED']),frozen_flow_sha256='f'*64)
    training.update(image_motion=dict(MODULE['TRAIN']['EXPECTED']),frozen_flow_sha256='f'*64,flow_unchanged=True)
    parent = copy.deepcopy(native)
    parent.update(checkpoint_sha256=MODULE['TRAIN']['INITIAL'],summary=dict(score=.85))
    flow = copy.deepcopy(causal)
    flow['flow_checkpoint_sha256'] = MODULE['TRAIN']['EXPECTED']['flow_checkpoint_sha256']
    flow['summaries']['motion']['score'] = .95
    return candidate,manifest,training,native,causal,parent,flow


def test_older_control_gain_is_not_enough():
    report = MODULE['compare'](*fixture())
    assert report['delta_vs_native'] > 0 and report['delta_vs_flow'] < 0
    assert report['delta_vs_parent'] < 0 and report['decision'] == 'not_strongest_standalone'
    assert report['authorized_for_submission'] is False


def test_flow_provenance_or_reference_nodes_cannot_change():
    values = fixture()
    values[6]['flow_checkpoint_sha256'] = 'e'*64
    with pytest.raises(ValueError,match='references'):
        MODULE['compare'](*values)
    values = fixture()
    values[6]['per_movie']['motion'][0]['num_pred_nodes'] += 1
    with pytest.raises(ValueError,match='detections'):
        MODULE['compare'](*values)
