from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "wait-launch-evaluate-peak-rank-validation-v1.ps1"


def test_controller_is_gated_by_verified_training_audit() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "harvest-terminal.json" in source
    assert "accepted_for_kaggle_validation" in source
    assert "audit_passed" in source
    assert "checkpoint_sha256" in source
    assert "skipped_after_training_rejection" in source


def test_controller_requires_private_dual_gpu_offline_validation() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'expected_gpu_count = 2' in source
    assert '$metadata.is_private -ne $true' in source
    assert '$metadata.enable_gpu -ne $true' in source
    assert '$metadata.enable_tpu -ne $false' in source
    assert '$metadata.enable_internet -ne $false' in source
    assert '$metadata.machine_shape -ne "NvidiaTeslaT4"' in source
    assert "torch.cuda.device_count() != 2" in source
    assert '"--devices", "0,1"' in source
    assert '"--tta-modes", "none,rot4,d4"' in source
    assert '[ValidateSet("v1", "depth-pu-v2")]' in source
    assert "build-peak-rank-depth-pu-validation-runtime.py" in source
    assert "build-peak-rank-depth-pu-validation-kernel.py" in source


def test_controller_validates_outputs_without_submitting() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "peak_rank_validation.json" in source
    assert "launcher_terminal.json" in source
    assert "validation_result_sha256" in source
    assert "accepted_for_candidate_integration" in source
    assert "selected_tta_mode" in source
    assert "kaggle competitions submit" not in source
    assert "api.competition_submit" not in source
    assert "authorized_for_submission = $false" in source


def test_variants_use_distinct_download_roots() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'output_slug = "peak-rank-validation-v1"' in source
    assert 'output_slug = "peak-rank-depth-pu-validation-v2"' in source
    assert "$variantConfig.output_slug" in source
    assert (
        '".biohub/cache/kernel-outputs/peak-rank-validation-v1-version$expectedVersion"'
        not in source
    )
