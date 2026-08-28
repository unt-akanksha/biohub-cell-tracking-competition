from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "wait-verify-submit-temporal-contextual-final.ps1"


def test_final_autosubmit_waits_for_verified_candidate_and_submits_once() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "MaximumWaitHours = 120.0" in source
    assert "indarkarhana/biohub-temporal-contextual-submission-candidate-v3" in source
    assert "submit-temporal-contextual-kernel.py" in source
    assert "f6122f465c9c39abd6f5e4dbbc476b00fa3c7dd2fe1d256826d22e41ea1be043" in source
    assert "5dad76f56003be6f84e381dbf1a735cb2e091dcedf8f238edb184fd0091c03af" in source
    assert "429ddae35e0633069a5ac44151cd6a1e4782ccd130b1f315d448eb43be8276fe" in source
    assert "--kernel-version $version" in source
    assert "--execute" in source
    assert "competition_submission_performed -ne $true" in source
    assert "public_leaderboard_used_for_selection -ne $false" in source
    assert "kaggle kernels status" in source
    assert "kaggle quota" not in source
