from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run-aws-4gpu-temporal-localizer-v1.sh"
CONTROLLER = ROOT / "scripts/wait-launch-harvest-aws-4gpu-temporal-localizer-v1.ps1"


def test_runner_uses_four_independent_large_members_and_auto_stops() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "gpu_count\" -eq 4" in text
    assert "seeds=(41021 41029 41039 41047)" in text
    assert "CUDA_VISIBLE_DEVICES=\"$gpu_index\"" in text
    assert "--steps 20000" in text
    assert "--required-gpu-name A10G" in text
    assert "sudo shutdown -h now" in text
    assert "competition" not in text.lower()


def test_controller_is_hash_bound_single_instance_and_non_submitting() -> None:
    text = CONTROLLER.read_text(encoding="utf-8")
    assert 'InstanceType = "g5.12xlarge"' in text
    assert '"--count", "1"' in text
    assert "__SYNTHETIC16_ARCHIVE_SHA256__" in text
    assert "parameters_per_model = 71249805" in text
    assert "competition_submission_performed = $false" in text
    assert "authorized_for_submission = $false" in text
    assert "/home/ubuntu/antelume" not in text
    assert "terminate-instances" not in text
