from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "wait-score-stage-launch-temporal-contextual-final.ps1"


def test_final_autolaunch_requires_exact_cpu_gate_and_uses_all_available_quota() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "MinimumQuotaHours = 12.0" in source
    assert "MaximumWaitHours = 96.0" in source
    assert "quota_reserve_hours = 0.0" in source
    assert "indarkarhana/biohub-temporal-contextual-processed-acceptance-v3" in source
    assert "indarkarhana/biohub-temporal-contextual-exact-acceptance-v3" in source
    assert "indarkarhana/biohub-temporal-contextual-submission-candidate-v3" in source
    assert "run-temporal-contextual-exact-acceptance.py" in source
    assert "stage-temporal-contextual-kaggle-artifacts.py" in source
    assert "build-temporal-contextual-submission-candidate-preflight.py" in source
    assert "exact_processed_gate_passed -ne $true" in source
    assert "datasets create" in source
    assert "datasets version" in source
    assert "datasets status" in source
    assert "NvidiaTeslaT4" in source
    assert "expected_gpu_count = 2" in source
    assert "5dad76f56003be6f84e381dbf1a735cb2e091dcedf8f238edb184fd0091c03af" in source
    assert "429ddae35e0633069a5ac44151cd6a1e4782ccd130b1f315d448eb43be8276fe" in source
    assert "kaggle kernels output" in source
    assert "kaggle kernels push" in source
    assert "competitions submit" not in source.casefold()
    assert "kaggle competitions" not in source.casefold()
