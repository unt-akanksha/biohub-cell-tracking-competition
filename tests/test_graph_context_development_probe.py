from pathlib import Path

from research.temporal_contrastive.graph_context_division_inference import (
    EXPECTED_PARAMETER_COUNT,
    calibration_free_scores,
    public_contract,
)
import torch


ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / "research/temporal_contrastive/score_graph_context_division_development_probe.py"


def test_calibration_free_score_is_equal_rank_mean() -> None:
    result = calibration_free_scores(
        [torch.tensor((3.0, 1.0, 2.0)), torch.tensor((0.0, 2.0, 1.0))],
        [10, 20, 30],
    )

    assert result == {10: 0.5, 20: 0.5, 30: 0.5}
    assert EXPECTED_PARAMETER_COUNT == 74_732_308
    assert public_contract()["absolute_threshold_used"] is False


def test_probe_requires_a_precommitted_audited_policy_unit() -> None:
    source = SCORER.read_text(encoding="utf-8")

    assert "graph_context_division_sweep_terminal.json" in source
    assert 'terminal.get("policy_audit_passed") is True' in source
    assert 'members == terminal.get("precommitted_members")' in source
    assert "v1_constituents_passed" in source
    assert "v2_audit_unit_passed" in source
    assert 'terminal.get("policy_unit_audited") is True' in source
    assert 'terminal.get("constituent_audit_gate_required") is False' in source
    assert 'audit.get("audit_gate_passed") is True' in source
    assert 'worker.get("parameter_count") == EXPECTED_PARAMETER_COUNT' in source
    assert 'inventory.get("graph_context_edges_read") is False' in source
    assert 'inventory.get("graph_context_labels_used") is False' in source
    assert '"absolute_threshold_used": False' in source
    assert '"weights_searched_on_probe": False' in source
    assert '"model_subset_searched_on_probe": False' in source
    assert '"authorized_for_submission": False' in source
    assert "kaggle competitions submit" not in source
