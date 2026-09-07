from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build-graph-context-consensus-division-dataset.py"
SPEC = importlib.util.spec_from_file_location("graph_context_runtime", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_runtime_inventory_contains_context_and_independent_voters() -> None:
    assert "graph_context_division_model.py" in MODULE.RUNTIME_FILES
    assert "graph_context_division_inference.py" in MODULE.RUNTIME_FILES
    assert "graph_context_features.py" in MODULE.RUNTIME_FILES
    assert "learned_division_recovery.py" in MODULE.RUNTIME_FILES
    assert "handcrafted_division.py" in MODULE.RUNTIME_FILES
    assert MODULE.EXPECTED_PARAMETER_COUNT == 74_732_308


def test_runtime_policy_is_additive_two_gpu_and_submission_ineligible() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert '"exact_two_t4_required": True' in source
    assert '"maximum_added_edges_per_movie": 1' in source
    assert '"external_policy_additive_only": True' in source
    assert '"model_subset_searched_on_audit": False' in source
    assert '"authorized_for_submission": False' in source
    assert "kaggle competitions submit" not in source


def test_runtime_accepts_audited_v2_ensemble_without_relabeling_members() -> None:
    members = [
        {"audit_gate_passed": False},
        {"audit_gate_passed": True},
    ]
    policy = {
        "graph_context_policy": "equal_rank_selection_admitted_ensemble",
        "policy_contract": MODULE.V2_POLICY_CONTRACT,
        "policy_unit_audited": True,
        "constituent_audit_gate_required": False,
    }

    assert MODULE.audited_member_contract(policy, members)
    assert not MODULE.audited_member_contract(
        {**policy, "policy_unit_audited": False}, members
    )
