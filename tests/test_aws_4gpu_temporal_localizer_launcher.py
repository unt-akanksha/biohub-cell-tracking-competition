from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run-aws-4gpu-temporal-localizer-v1.sh"
CONTROLLER = ROOT / "scripts/wait-launch-harvest-aws-4gpu-temporal-localizer-v1.ps1"
TRAINER = ROOT / "research/temporal_localization/train_synthetic_localizer.py"


def test_runner_uses_four_independent_large_members_and_auto_stops() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "gpu_count\" -eq 4" in text
    assert "seeds=(41021 41029 41039 41047)" in text
    assert "CUDA_VISIBLE_DEVICES=\"$gpu_index\"" in text
    assert "--steps 20000" in text
    assert "--division-critical-per-batch 4" in text
    assert "--division-validation-examples 512" in text
    assert "--division-audit-examples 512" in text
    assert "--required-gpu-name A10G" in text
    assert "site.ENABLE_USER_SITE" in text
    assert "--disable-pip-version-check --no-input" in text
    assert '"$python_bin" -c \'import scipy, tracksdata, zarr\'' in text
    assert '"$output_root/runtime-environment.json"' in text
    assert '"gpu_names"' in text
    assert "score_real_development_probe.py" in text
    assert "real-development-probe.json" in text
    assert "sudo shutdown -h now" in text
    assert "kaggle" not in text.lower()
    assert "submit" not in text.lower()
    trainer = TRAINER.read_text(encoding="utf-8")
    assert '"serialized_checkpoint_selection_gate_passed"' in trainer
    assert "serialized_state = torch.load(" in trainer
    assert "Audit data is opened only after the exact serialized checkpoint" in trainer


def test_controller_is_hash_bound_single_instance_and_non_submitting() -> None:
    text = CONTROLLER.read_text(encoding="utf-8")
    assert 'InstanceType = "g5.12xlarge"' in text
    assert '"--count", "1"' in text
    assert "__SYNTHETIC16_ARCHIVE_SHA256__" in text
    assert "__DEVELOPMENT_ARCHIVE_SHA256__" in text
    assert "parameters_per_model = 71249805" in text
    assert "division_critical_examples_per_batch = 4" in text
    assert "division_critical_selection_examples = 512" in text
    assert "division_critical_audit_examples = 512" in text
    assert "serialized_checkpoint_selection_gate_required = $true" in text
    assert "rendered_runner_sha256 = $renderedRunnerSha256" in text
    assert "trainer_sha256 = $trainerSha256" in text
    assert "development_probe_runs_only_after_member_audits = $true" in text
    assert "competition_submission_performed = $false" in text
    assert "authorized_for_submission = $false" in text
    assert "/home/ubuntu/antelume" not in text
    assert "terminate-instances" not in text
