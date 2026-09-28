"""T4 notebook-hour accounting with conservative outstanding-run reservations."""
import math
import re


def saved_kernel_complete(output, kernel):
    """Accept only a complete status for the exact requested saved kernel."""
    return re.fullmatch(
        re.escape(kernel) + r' has status ["\'](?:KernelWorkerStatus\.)?COMPLETE["\']',
        output.strip(), flags=re.IGNORECASE,
    ) is not None


def acceptance_budget(remaining_hours, submissions, known_pending_caps):
    remaining=float(remaining_hours)
    if not math.isfinite(remaining) or remaining<0 or not submissions:
        raise ValueError('Fresh finite quota and submission inventory required')
    pending={};seen=set()
    for row in submissions:
        ref=int(row['ref']);status=row['status']
        if ref in seen:raise ValueError('Duplicate submission inventory entry')
        seen.add(ref)
        if status in ('SubmissionStatus.COMPLETE','SubmissionStatus.ERROR'):continue
        if status not in ('SubmissionStatus.PENDING','SubmissionStatus.RUNNING'):
            raise ValueError('Unknown submission status')
        if ref not in known_pending_caps:raise ValueError('Unknown pending submission needs its own reservation')
        cap=float(known_pending_caps[ref])
        if not math.isfinite(cap) or cap<=0:raise ValueError('Invalid pending-run cap')
        pending[str(ref)]=cap
    # One-hour two-T4 notebook, plus an extra hour of conservative headroom.
    acceptance_reservation=2.
    unreserved=remaining-sum(pending.values())-acceptance_reservation
    if unreserved<8.:raise ValueError('Pending plus new run would breach eight-hour reserve')
    return dict(remaining_hours=remaining,pending_reservations_hours=pending,
        new_acceptance_reserved_hours=acceptance_reservation,
        remaining_after_all_reservations=unreserved,reserve_hours=8.,
        t4x2_quota_hours_per_notebook_hour=1.,hidden_evaluation_assumed_free=False)
