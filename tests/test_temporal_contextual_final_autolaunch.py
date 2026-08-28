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
    assert "indarkarhana/biohub-temporal-contextual-final-runtime-v1" in source
    assert "biohub-temporal-contextual-final-runtime-v1-version2" in source
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
    assert "f5a5e40827c0cd1b846dad0702faa7ae3697288f59c7ce24e8da7797d179db01" in source
    assert "ef010e45a6a10d1f00efee2d696a8c5a218c52b29e039673b32aa128ff248f43" in source
    assert "kaggle kernels output" in source
    assert "kaggle kernels push" in source
    assert "competitions submit" not in source.casefold()
    assert "kaggle competitions" not in source.casefold()
