from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/wait-build-launch-aws-temporal-localizer-v2.ps1"


def test_overnight_orchestrator_is_bounded_resumable_and_non_submitting() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "InitialKaggleCooldownSeconds = 900" in source
    assert "MaximumDownloadAttempts = 48" in source
    assert "--request-delay-seconds 5.0" in source
    assert "competition-real-localization-replay-v1" in source
    assert "competition-real-localization-shards-v1" in source
    assert "wait-launch-harvest-aws-4gpu-temporal-localizer-v1.ps1" in source
    assert "planned_gpu_count = 4" in source
    assert "planned_model_count = 4" in source
    assert "steps_per_model = 40000" in source
    assert "real_replay_probability = 0.25" in source
    assert "competition_submission_performed = $false" in source
    assert "competitions submit" not in source
