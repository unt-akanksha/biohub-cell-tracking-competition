from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify-relational-consensus-submission-candidate.py"
SPEC = importlib.util.spec_from_file_location("relational_candidate_verifier", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_verifier_preserves_strict_external_promotion_gate() -> None:
    assert module.MINIMUM_PROXY_GAIN == 0.005
    assert module.MAXIMUM_ADJUSTED_EDGE_REGRESSION == 0.001
    assert module.EXPECTED_PARAMETER_COUNT == 48_313_050
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'candidate_validator["division_tp"] > baseline["division_tp"]' in text
    assert 'candidate_validator["division_jaccard"] > baseline["division_jaccard"]' in text
    assert "KNOWN_PUBLIC_SUBMISSION_SHA256" in text


def test_verifier_requires_dual_gpu_stats_and_additive_integrity() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "ranked_consensus_gpu_groups_used" in text
    assert "expected_gpu_groups" in text
    assert "ranked_consensus_reassignment_performed" in text
    assert "ranked_consensus_node_or_coordinate_changes" in text
    assert '"authorized_for_submission": True' in text
    assert "kaggle competitions submit" not in text
