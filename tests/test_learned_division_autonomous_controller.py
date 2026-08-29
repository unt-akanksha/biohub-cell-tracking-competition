from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "scripts/wait-build-verify-submit-learned-division-candidate.ps1"


def test_controller_has_external_policy_promotion_and_single_submit_gates() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")

    assert "external_policy_rejected" in source
    assert "verify-learned-division-submission-candidate.py" in source
    assert "submit-learned-division-candidate.py" in source
    assert "--execute" in source
    assert "candidate_rejected" in source
    assert "Refusing to risk a duplicate submission" in source
    assert "target_public_score = 0.945" in source
    assert source.count("competitions submit") == 0
