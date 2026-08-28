from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "wait-verify-launch-temporal-contextual-processed.ps1"


def test_processed_autolaunch_is_hash_bound_and_recomputed_evidence_gated() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "MinimumQuotaHours = 6.0" in source
    assert "MaximumWaitHours = 48.0" in source
    assert "quota_reserve_hours = 0.0" in source
    assert "indarkarhana/biohub-temporal-contextual-calibration-v3" in source
    assert "indarkarhana/biohub-temporal-contextual-processed-acceptance-v3" in source
    assert "NvidiaTeslaT4" in source
    assert "expected_gpu_count = 2" in source
    assert "624a14a202cde15a1bfc4cf942241df1f80801094e60e8886182c8dffc826125" in source
    assert "08c0316da23940e21490d56909d3bb88c690b9103dfb0d2cf57c40de96e8665b" in source
    assert "verify_dual_fold_calibration_output.py" in source
    assert "authorized_for_processed_materialization -ne $true" in source
    assert "authorized_for_submission -ne $false" in source
    assert "public_leaderboard_used_for_selection -ne $false" in source
    assert "submission_created -ne $false" in source
    assert "kaggle kernels output" in source
    assert "kaggle kernels push" in source
    assert "competitions submit" not in source.casefold()
    assert "kaggle competitions" not in source.casefold()
