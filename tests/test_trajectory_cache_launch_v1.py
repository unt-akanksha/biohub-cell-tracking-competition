from pathlib import Path
import runpy
import pytest

CHECK = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/launch-trajectory-overlap-cache-v1.py'))['verify_submission_terminals']


def test_terminal_inventory():
    assert len(CHECK('ref,status\n1,SubmissionStatus.COMPLETE\n2,SubmissionStatus.ERROR\n')) == 2


@pytest.mark.parametrize('data', ['ref,status\n', 'ref,state\n1,complete\n',
    'ref,status\n1,SubmissionStatus.PENDING\n', 'ref,status\n1,unknown\n'])
def test_unknown_or_pending_inventory_rejected(data):
    with pytest.raises(ValueError):
        CHECK(data)
