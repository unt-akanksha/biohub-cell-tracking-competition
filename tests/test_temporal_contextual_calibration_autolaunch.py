from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "wait-verify-launch-temporal-contextual-calibration.ps1"


def test_calibration_autolaunch_is_hash_bound_and_evidence_gated() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "MinimumQuotaHours = 6.0" in source
    assert "MaximumWaitHours = 24.0" in source
    assert "indarkarhana/biohub-temporal-contextual-transfer-v3" in source
    assert "indarkarhana/biohub-temporal-contextual-calibration-v3" in source
    assert "NvidiaTeslaT4" in source
    assert "expected_gpu_count = 2" in source
    assert "46f0b73396f50d1e3bebb40ce2f811b20d791347dbf51b0b512e2e71e492b955" in source
    assert "4596cd9fd7211c27f6b437268c8e719847f0e7cce91438cc86195e79aba497e1" in source
    assert "cbe5fe27639155746c95a98d91702d5fbe595172b058e0e9db330374ecfff25d" in source
    assert "--strict-checkpoint" in source
    assert "authorized_for_calibration -ne $true" in source
    assert "authorized_for_submission -ne $false" in source
    assert "public_leaderboard_used_for_selection -ne $false" in source
    assert "submission_created -ne $false" in source
    assert "kaggle kernels output" in source
    assert "kaggle kernels push" in source
    assert "competitions submit" not in source.casefold()
    assert "kaggle competitions" not in source.casefold()
