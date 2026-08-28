from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "wait-verify-launch-temporal-multiscale-transfer.ps1"


def test_multiscale_transfer_autolaunch_requires_strict_pretraining_evidence() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "MinimumQuotaHours = 11.0" in source
    assert "MaximumWaitHours = 504.0" in source
    assert "quota_reserve_hours = 0.0" in source
    assert "zebrahub-multiscale-pretrain-launch.json" in source
    assert "verify_multiscale_pretraining_output.py" in source
    assert "$verified.status -ne 'verified'" in source
    assert "$verified.strict_checkpoint_loaded -ne $true" in source
    assert "indarkarhana/biohub-temporal-multiscale-transfer-v4" in source
    assert "indarkarhana/biohub-zsns001-contextual-gate-v1" in source
    assert "expected_gpu_count = 2" in source
    assert "NvidiaTeslaT4" in source
    assert "kaggle kernels output" in source
    assert "kaggle kernels push" in source
    assert "kaggle quota" in source
    assert "competitions submit" not in source.casefold()
    assert "kaggle competitions" not in source.casefold()
