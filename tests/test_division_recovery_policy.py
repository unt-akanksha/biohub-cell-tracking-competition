from __future__ import annotations

from research.temporal_contrastive.calibrate_division_recovery_policy import (
    recovery_metrics,
    select_threshold,
)


def row(score: float, *, division: bool, second: bool, top_two: bool) -> dict:
    return {
        "division_logit": score,
        "is_division": division,
        "second_is_positive": second,
        "top_two_are_daughters": top_two,
    }


def test_policy_selection_prefers_precise_external_division_gate() -> None:
    rows = [
        row(4.0, division=True, second=True, top_two=True),
        row(3.0, division=True, second=True, top_two=True),
        row(2.0, division=False, second=False, top_two=False),
        row(-1.0, division=False, second=False, top_two=False),
    ]

    threshold, metrics = select_threshold(rows)

    assert 2.0 < threshold <= 3.0
    assert metrics["division_tp"] == 2
    assert metrics["division_fp"] == 0
    assert metrics["edge_precision"] == 1.0


def test_recovery_metrics_penalize_a_wrong_second_daughter() -> None:
    rows = [
        row(2.0, division=True, second=True, top_two=True),
        row(1.0, division=False, second=False, top_two=False),
    ]

    metrics = recovery_metrics(rows, threshold=0.0)

    assert metrics["edge_tp"] == 1
    assert metrics["edge_fp"] == 1
    assert metrics["event_tp"] == 1
    assert metrics["event_fp"] == 1
    assert metrics["division_tp"] == 1
    assert metrics["division_fp"] == 1
    assert metrics["recovery_composite"] < 1.1
