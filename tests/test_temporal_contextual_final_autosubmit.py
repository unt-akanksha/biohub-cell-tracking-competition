from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "wait-verify-submit-temporal-contextual-final.ps1"


def test_final_autosubmit_waits_for_verified_candidate_and_submits_once() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "MaximumWaitHours = 120.0" in source
    assert "indarkarhana/biohub-temporal-contextual-submission-candidate-v3" in source
    assert "submit-temporal-contextual-kernel.py" in source
    assert "7b43627a89b2d557c69665b3324da02ed8361a9f9a8b32f8df195725269ba6d6" in source
    assert "f5a5e40827c0cd1b846dad0702faa7ae3697288f59c7ce24e8da7797d179db01" in source
    assert "ef010e45a6a10d1f00efee2d696a8c5a218c52b29e039673b32aa128ff248f43" in source
    assert "--kernel-version $version" in source
    assert "--execute" in source
    assert "competition_submission_performed -ne $true" in source
    assert "public_leaderboard_used_for_selection -ne $false" in source
    assert "kaggle kernels status" in source
    assert "kaggle quota" not in source
