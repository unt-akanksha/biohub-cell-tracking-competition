from __future__ import annotations

from research.temporal_contrastive.diagnose_v3_division_recovery import (
    RUN_ID,
    diagnostic_result,
)


def test_v3_diagnostic_cannot_authorize_competition_use() -> None:
    result = diagnostic_result(
        model_hashes={"target_44b6": "a" * 64, "target_6bba": "b" * 64},
        threshold=1.25,
        selection={"division_tp": 3},
        audit={"division_tp": 2},
    )

    assert result["status"] == "diagnostic_only"
    assert result["run_id"] == RUN_ID
    assert result["authorized_for_competition_graph_evaluation"] is False
    assert result["authorized_for_submission"] is False
    assert result["competition_data_read"] is False
    assert "already opened" in result["audit_reuse_disclosure"]
