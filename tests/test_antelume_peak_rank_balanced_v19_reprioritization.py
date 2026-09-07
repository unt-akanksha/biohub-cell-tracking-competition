from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run-antelume-peak-rank-expanded-real-balanced-v19.sh"
REPRIORITIZE = (
    ROOT / "scripts/reprioritize-antelume-peak-rank-expanded-real-balanced-v19.ps1"
)
REPAIR = (
    ROOT
    / "scripts/repair-antelume-peak-rank-expanded-real-balanced-v19-after-v3.ps1"
)


def test_after_v3_repair_is_bound_to_the_current_runner() -> None:
    source = REPAIR.read_text(encoding="utf-8")
    digest = hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    assert digest in source
    assert "repaired_waiting_for_verified_v3" in source
    assert "start_after_verified_v3_and_yield_to_all_active_gpu_clients" in source
    assert "active_v3_interrupted" in source
    assert "rsna_or_unrelated_process_interrupted" in source
    assert "ValidateOnly" in source


def test_after_v3_repair_targets_only_the_missing_idle_v19_waiter() -> None:
    source = REPAIR.read_text(encoding="utf-8")
    exact_runner = (
        "/home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-balanced-v19/run\\.sh$"
    )
    assert exact_runner in source
    assert 'test ! -e "$run_root/training.log"' in source
    assert 'test ! -e "$run_root/results"' in source
    assert 'test "${#pids[@]}" -eq 0' in source
    assert "pkill" not in source
    assert "killall" not in source


def test_original_reprioritization_receipt_contract_is_preserved() -> None:
    source = REPRIORITIZE.read_text(encoding="utf-8")
    assert "reprioritized_waiting_for_verified_v2" in source
    assert "start_after_verified_v2_and_yield_to_all_active_gpu_clients" in source
