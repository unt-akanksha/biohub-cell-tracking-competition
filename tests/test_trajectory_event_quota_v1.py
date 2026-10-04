import pytest
from research.trajectory_event_quota_v1 import acceptance_budget,saved_kernel_complete


@pytest.mark.parametrize('status',['complete','KernelWorkerStatus.COMPLETE'])
def test_saved_kernel_complete_formats(status):
    assert saved_kernel_complete('owner/kernel has status "'+status+'"\n','owner/kernel')


@pytest.mark.parametrize('output',[
    'owner/kernel has status "KernelWorkerStatus.RUNNING"',
    'owner/kernel has status "KernelWorkerStatus.ERROR"',
    'other/kernel has status "KernelWorkerStatus.COMPLETE"',
    'owner/kernel has status "NOT_COMPLETE"',
    'warning complete',
])
def test_saved_kernel_incomplete_or_wrong_identity_rejected(output):
    assert not saved_kernel_complete(output,'owner/kernel')


def rows(status='SubmissionStatus.PENDING',ref=56231458):return [dict(ref=str(ref),status=status)]


def test_pending_full_platform_cap_is_reserved():
    r=acceptance_budget(28.12,rows(),{56231458:12})
    assert r['remaining_after_all_reservations']==pytest.approx(14.12)
    assert not r['hidden_evaluation_assumed_free']


def test_reserve_boundary():
    assert acceptance_budget(22,rows(),{56231458:12})['remaining_after_all_reservations']==8
    with pytest.raises(ValueError):acceptance_budget(21.999,rows(),{56231458:12})


@pytest.mark.parametrize('remaining',[float('nan'),float('inf'),-1])
def test_bad_quota(remaining):
    with pytest.raises(ValueError):acceptance_budget(remaining,rows(),{56231458:12})


def test_unknown_pending_and_status_rejected():
    with pytest.raises(ValueError):acceptance_budget(30,rows(ref=999),{56231458:12})
    with pytest.raises(ValueError):acceptance_budget(30,rows('unknown'),{56231458:12})


def test_terminal_releases_only_its_reservation():
    assert acceptance_budget(10,rows('SubmissionStatus.COMPLETE'),{})['remaining_after_all_reservations']==8


def test_empty_and_duplicate_inventory_rejected():
    with pytest.raises(ValueError):acceptance_budget(30,[],{})
    with pytest.raises(ValueError):acceptance_budget(30,rows()+rows(),{56231458:12})
