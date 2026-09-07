from __future__ import annotations

import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run-antelume-peak-rank-expanded-real-blob-v11.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_runner_waits_for_verified_v9_and_pins_all_sources() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'while test ! -f "$v9_run_root/run.complete"' in source
    assert 'test ! -f "$v9_ack"' in source
    assert "verified_v9_harvest" in source
    assert "while nvidia-smi --query-compute-apps=pid" in source
    assert "--steps 3000" in source
    assert "--real-frequency 2" in source
    assert "--seed 6902243" in source
    assert "--max-wall-seconds 36000" in source
    assert "competition_test_data_read" in source
    assert "public_leaderboard_used_for_selection" in source
    assert "RSNA" not in source

    expected = {
        "trainer_sha256": sha256(
            ROOT / "research/peak_rank_detection/train_expanded_real_blob_detector.py"
        ),
        "blob_model_sha256": sha256(
            ROOT / "research/peak_rank_detection/model_blob.py"
        ),
        "local_shape_trainer_sha256": sha256(
            ROOT
            / "research/peak_rank_detection/train_expanded_real_local_shape_detector.py"
        ),
        "expanded_trainer_sha256": sha256(
            ROOT / "research/peak_rank_detection/train_expanded_real_faint_detector.py"
        ),
        "faint_trainer_sha256": sha256(
            ROOT / "research/peak_rank_detection/train_faint_cell_pu_detector.py"
        ),
        "base_trainer_sha256": sha256(
            ROOT / "research/peak_rank_detection/train_synthetic_real_detector.py"
        ),
        "model_sha256": sha256(ROOT / "research/peak_rank_detection/model.py"),
        "objectives_sha256": sha256(
            ROOT / "research/peak_rank_detection/objectives.py"
        ),
        "synthetic_data_sha256": sha256(
            ROOT / "research/synthetic_pretrain/data.py"
        ),
    }
    for variable, digest in expected.items():
        match = re.search(rf"^{variable}=([0-9a-f]{{64}})$", source, re.MULTILINE)
        assert match and match.group(1) == digest


def test_runner_cleanup_is_narrow_and_yield_guard_compatible() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert (
        "test \"$resolved\" = /home/ubuntu/biohub-peak-rank-detector-v1/"
        "expanded-real-local-shape-v9/results"
    ) in source
    assert 'rm -rf -- "$resolved"' in source
    assert 'trainer="$input_root/train_expanded_real_faint_detector.py"' in source
    assert 'real_root="$v7_run_root/data/' in source
    assert "pkill" not in source
    assert "killall" not in source
    assert "systemctl" not in source
