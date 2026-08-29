from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "scripts/wait-run-kaggle-external-division-policy.ps1"


def test_controller_waits_for_v4_and_publishes_no_submission_policy() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")

    assert "zebrahub-multiscale-pretrain-recovery.json" in source
    assert "verify-external-division-policy-output.py" in source
    assert "external-division-recovery-policy-v1.json" in source
    assert "competition_data_read = $false" in source
    assert "submission_created = $false" in source
    assert "kaggle competitions submit" not in source
