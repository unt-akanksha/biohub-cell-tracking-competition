import pytest
from research.trajectory_event_final_budget_v1 import final_budget


def history(status='SubmissionStatus.COMPLETE'):
    return [dict(ref='56231458', status=status)]


def test_final_budget_reserves_full_platform_cap():
    result = final_budget(27.52, history(), {})
    assert result['new_run_reserved_hours'] == 12
    assert result['remaining_after_all_reservations'] == pytest.approx(15.52)
    assert not result['hidden_evaluation_assumed_free']


def test_exact_reserve_boundary():
    assert final_budget(20, history(), {})['remaining_after_all_reservations'] == 8
    with pytest.raises(ValueError):
        final_budget(19.99, history(), {})


def test_pending_job_not_ignored():
    with pytest.raises(ValueError):
        final_budget(27.52, history('SubmissionStatus.PENDING'), {56231458: 12})


@pytest.mark.parametrize('remaining', [float('nan'), float('inf'), -1])
def test_invalid_quota(remaining):
    with pytest.raises(ValueError):
        final_budget(remaining, history(), {})


def test_unknown_pending_rejected():
    with pytest.raises(ValueError):
        final_budget(30, history('SubmissionStatus.RUNNING'), {})
