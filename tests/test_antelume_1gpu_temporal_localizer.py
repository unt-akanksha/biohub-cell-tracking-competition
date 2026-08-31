from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run-antelume-1gpu-temporal-localizer-v2.sh"
HARVESTER = ROOT / "scripts/wait-harvest-antelume-1gpu-temporal-localizer-v2.ps1"


def test_runner_trains_four_large_members_sequentially_on_one_a10g() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert 'gpu_count" -eq 1' in text
    assert "seeds=(41021 41029 41039 41047)" in text
    assert "for gpu_index in 0 1 2 3" in text
    assert "CUDA_VISIBLE_DEVICES=0" in text
    assert "--steps 40000" in text
    assert "--member-max-wall-seconds 11700" in text
    assert "--real-replay-probability 0.25" in text
    assert "score_real_development_probe.py" in text
    assert "sudo shutdown" not in text
    assert "kaggle" not in text.lower()
    assert "submit" not in text.lower()


def test_harvester_preserves_instance_and_hands_off_hash_bound_archive() -> None:
    text = HARVESTER.read_text(encoding="utf-8")
    assert 'status = "harvested"' in text
    assert '"aws-4gpu-temporal-localizer-harvest-v1"' in text
    assert "Get-FileHash -Algorithm SHA256" in text
    assert "recoverable_instance_preserved = $true" in text
    assert "stop-instances" not in text
    assert "terminate-instances" not in text
