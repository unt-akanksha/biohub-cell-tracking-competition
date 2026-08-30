from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "scripts/wait-verify-submit-ranked-consensus-candidate.ps1"


def test_controller_waits_for_promotion_and_submits_once() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")

    assert "verify-ranked-consensus-submission-candidate.py" in source
    assert "submit-ranked-consensus-candidate.py" in source
    assert "ranked-consensus-candidate-submission-receipt.json" in source
    assert "competition_submission_performed = $true" in source
    assert "--execute" in source
    assert "Refusing to reuse controller state" in source
    assert "Start-Sleep -Seconds $PollSeconds" in source
