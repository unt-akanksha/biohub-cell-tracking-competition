from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/wait-evaluate-relational-division-development.ps1"


def test_controller_is_event_driven_and_submission_ineligible() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "harvest-terminal.json" in text
    assert "--extract-to" in text
    assert "evaluate_relational_division_development.py" in text
    assert "build-relational-consensus-division-dataset.py" in text
    assert "RELATIONAL_CONSENSUS_MANIFEST.json" in text
    assert "competition-ranked-consensus-development-baseline-v1.json" in text
    assert "Start-Sleep -Seconds $PollSeconds" in text
    assert 'competition_submission_performed"] = $false' in text
    assert "kaggle competitions submit" not in text
