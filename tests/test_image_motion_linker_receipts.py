from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT/'scripts/summarize-image-motion-linker.py'))
FIXTURE = runpy.run_path(str(ROOT/'tests/test_known_null_training_receipts.py'))['receipt']


def fixture():
    result,terminal,split = FIXTURE()
    result['identity'].update(image_motion=dict(MODULE['EXPECTED']),frozen_flow_sha256='c'*64,
                              initialization_sha256=MODULE['INITIAL'])
    result['flow_unchanged'] = True
    terminal.update(run_id='image-motion-linker-v1',declared_budget_seconds=3600)
    return result,terminal,split


def test_combined_receipts_require_both_models_frozen():
    report = MODULE['verify'](*fixture(),100)
    assert report['flow_unchanged'] and report['detector_unchanged']
    assert report['authorized_for_submission'] is False


@pytest.mark.parametrize('damage',['flow_changed','flow_identity','detector','initialization','nulls'])
def test_invalid_components_or_missing_supervision_rejected(damage):
    result,terminal,split = fixture()
    if damage == 'flow_changed':
        result['flow_unchanged'] = False
    elif damage == 'flow_identity':
        result['identity']['image_motion']['flow_checkpoint_sha256'] = 'd'*64
    elif damage == 'detector':
        result['detector_unchanged'] = False
    elif damage == 'initialization':
        result['identity']['initialization_sha256'] = 'e'*64
    else:
        result['known_null_supervised_total'] = 0
    with pytest.raises(ValueError):
        MODULE['verify'](result,terminal,split,100)
