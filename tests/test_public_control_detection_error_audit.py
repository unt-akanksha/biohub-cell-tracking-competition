from __future__ import annotations

import numpy as np

from research.audit_public_control_detection_errors import classify_frame, summarize_rows


def test_error_audit_separates_conflict_localization_and_absence() -> None:
    predicted = np.asarray([[0, 0, 0], [0, 20, 0]], dtype=np.float32)
    truth = np.asarray(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 40, 0],
            [0, 80, 0],
        ],
        dtype=np.float32,
    )
    rows = classify_frame(predicted, truth)
    summary = summarize_rows(rows)

    assert summary["failure_counts"] == {
        "matched": 1,
        "crowding_conflict": 1,
        "localization_miss": 1,
        "absent_detection": 1,
    }
    assert summary["annotated_node_recall"] == 0.25


def test_error_audit_is_diagnostic_and_never_submits() -> None:
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "research/audit_public_control_detection_errors.py"
    ).read_text(encoding="utf-8")

    assert '"model_or_threshold_selected": False' in source
    assert '"public_leaderboard_used_for_selection": False' in source
    assert '"authorized_for_submission": False' in source
    assert "kaggle competitions submit" not in source
