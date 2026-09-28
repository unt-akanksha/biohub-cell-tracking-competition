"""Conservative final saved-run or submission reservation in T4 notebook hours."""
from research.trajectory_event_quota_v1 import acceptance_budget


def final_budget(remaining_hours, submissions, known_pending_caps):
    checked = acceptance_budget(remaining_hours, submissions, known_pending_caps)
    after = checked['remaining_after_all_reservations'] - 10.0
    if after < 8.0:
        raise ValueError('Twelve-hour final run would breach eight-hour reserve')
    return dict(
        remaining_hours=checked['remaining_hours'],
        pending_reservations_hours=checked['pending_reservations_hours'],
        new_run_reserved_hours=12.0, remaining_after_all_reservations=after,
        reserve_hours=8.0, t4x2_quota_hours_per_notebook_hour=1.0,
        hidden_evaluation_assumed_free=False,
    )
