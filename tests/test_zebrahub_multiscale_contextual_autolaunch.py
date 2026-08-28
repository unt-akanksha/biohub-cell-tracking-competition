from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "wait-launch-zebrahub-multiscale-contextual-pretrain.ps1"


def test_multiscale_autolaunch_is_post_v3_two_gpu_and_zero_reserve() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "MinimumQuotaHours = 7.0" in source
    assert "MaximumWaitHours = 336.0" in source
    assert "quota_reserve_hours = 0.0" in source
    assert "temporal-contextual-submission.json" in source
    assert "temporal-contextual-submission-receipt.json" in source
    assert "$submitted.status -ne 'submitted'" in source
    assert "$submitted.competition_submission_performed -ne $true" in source
    assert "indarkarhana/biohub-zebrahub-multiscale-pretrain-v1" in source
    assert "indarkarhana/biohub-temporal-multiscale-contextual-runtime-v4" in source
    assert "runtime_dataset_version = 3" in source
    assert "shards_dataset_version = 4" in source
    assert "indarkarhana/biohub-zsns001-contextual-gate-v1" in source
    assert "expected_gpu_count = 2" in source
    assert "NvidiaTeslaT4" in source
    assert "kaggle kernels push" in source
    assert "kaggle quota" in source
    assert "competitions submit" not in source.casefold()
    assert "kaggle competitions" not in source.casefold()
