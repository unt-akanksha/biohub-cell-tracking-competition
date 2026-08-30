from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify-graph-context-consensus-submission-candidate.py"
SPEC = importlib.util.spec_from_file_location("graph_context_candidate_verifier", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_verifier_preserves_strict_external_promotion_gate() -> None:
    assert module.MINIMUM_PROXY_GAIN == 0.005
    assert module.MAXIMUM_ADJUSTED_EDGE_REGRESSION == 0.001
    assert module.EXPECTED_PARAMETER_COUNT == 74_732_308
    assert module.verify_candidate.__globals__["RUN_ID"] == module.RUN_ID
    assert module.verify_candidate.__globals__["validate_runtime"] is module.validate_runtime


def test_verifier_requires_hash_bound_independently_strong_runtime() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'policy.get("graph_context_members", [])' in source
    assert 'row.get("selection_gate_passed") is True' in source
    assert 'row.get("audit_gate_passed") is True' in source
    assert 'policy.get("exact_two_t4_required") is True' in source
    assert 'policy.get("external_policy_additive_only") is True' in source
    assert "kaggle competitions submit" not in source
