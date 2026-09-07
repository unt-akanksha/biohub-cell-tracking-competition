from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run-antelume-peak-rank-expanded-real-balanced-v19.sh"
REPRIORITIZE = (
    ROOT / "scripts/reprioritize-antelume-peak-rank-expanded-real-balanced-v19.ps1"
)


def test_reprioritization_is_bound_to_the_priority_runner() -> None:
    source = REPRIORITIZE.read_text(encoding="utf-8")
    digest = hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    assert digest in source
    assert "reprioritized_waiting_for_verified_v2" in source
    assert "start_after_verified_v2_and_yield_to_all_active_gpu_clients" in source
    assert "RecoverExisting" in source
    assert "receipt_recovered_after_successful_remote_launch" in source
    assert "ValidateOnly" in source


def test_reprioritization_targets_only_the_idle_v19_waiter() -> None:
    source = REPRIORITIZE.read_text(encoding="utf-8")
    exact_runner = (
        "/home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-balanced-v19/run\\.sh$"
    )
    assert exact_runner in source
    assert 'test ! -e "$run_root/training.log"' in source
    assert 'test ! -e "$run_root/results"' in source
    assert "kill -TERM '$oldPid'" in source
    assert "pkill" not in source
    assert "killall" not in source
    assert "RSNA" not in source
