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
    assert "expected_gpu_count = 2" in source
    assert "$metadata.is_private -ne $true" in source
    assert "$metadata.enable_gpu -ne $true" in source
    assert "$metadata.enable_tpu -ne $false" in source
    assert "$metadata.enable_internet -ne $false" in source
    assert '$metadata.machine_shape -ne "NvidiaTeslaT4"' in source
    assert "torch.cuda.device_count() != 2" in source
    assert '"--devices", "0,1"' in source
    assert '"--tta-modes", "none,zflip2,rot4,d4"' in source
    assert (
        '[ValidateSet("v1", "depth-pu-v2", "capacity-pu-v3", '
        '"faint-pu-v4", "expanded-real-faint-v7", '
        '"expanded-real-local-shape-v9", '
        '"expanded-real-blob-v11", '
        '"expanded-real-global-v13", '
        '"expanded-real-multiscale-v15", '
        '"expanded-real-safe-rank-v17", '
        '"expanded-real-balanced-v19", '
        '"expanded-real-xl-balanced-v21", '
        '"temporal-min-local-snr-v23", '
        '"capacity-faint-ensemble-v5", '
        '"capacity-faint-confidence-v6", '
        '"capacity-faint-expanded-v8", '
        '"expanded-local-shape-ensemble-v10", '
        '"expanded-blob-ensemble-v12", '
        '"blob-global-ensemble-v14", '
        '"multiscale-triad-v16", '
        '"multiscale-safe-pair-v18", '
        '"safe-balanced-pair-v20", '
        '"balanced-xl-pair-v22", '
        '"xl-temporal-snr-pair-v24", '
        '"logit-ensemble-v4")]' in source
    )
    assert "build-peak-rank-depth-pu-validation-runtime.py" in source
    assert "build-peak-rank-depth-pu-validation-kernel.py" in source
    assert "build-peak-rank-capacity-pu-validation-runtime.py" in source
    assert "build-peak-rank-capacity-pu-validation-kernel.py" in source
    assert "build-peak-rank-faint-pu-validation-runtime.py" in source
    assert "build-peak-rank-faint-pu-validation-kernel.py" in source
    assert "build-peak-rank-expanded-real-faint-validation-runtime-v7.py" in source
    assert "build-peak-rank-expanded-real-faint-validation-kernel-v7.py" in source
    assert (
        "build-peak-rank-expanded-real-local-shape-validation-runtime-v9.py" in source
    )
    assert "build-peak-rank-expanded-real-local-shape-validation-kernel-v9.py" in source
    assert "build-peak-rank-expanded-real-blob-validation-runtime-v11.py" in source
    assert "build-peak-rank-expanded-real-blob-validation-kernel-v11.py" in source
    assert "build-peak-rank-expanded-real-global-validation-runtime-v13.py" in source
    assert "build-peak-rank-expanded-real-global-validation-kernel-v13.py" in source
    assert (
        "build-peak-rank-expanded-real-multiscale-validation-runtime-v15.py" in source
    )
    assert "build-peak-rank-expanded-real-multiscale-validation-kernel-v15.py" in source
    assert "build-peak-rank-expanded-real-safe-rank-validation-runtime-v17.py" in source
    assert "build-peak-rank-expanded-real-safe-rank-validation-kernel-v17.py" in source
    assert "build-peak-rank-expanded-real-balanced-validation-runtime-v19.py" in source
    assert "build-peak-rank-expanded-real-balanced-validation-kernel-v19.py" in source
    assert (
        "build-peak-rank-expanded-real-xl-balanced-validation-runtime-v21.py" in source
    )
    assert (
        "build-peak-rank-expanded-real-xl-balanced-validation-kernel-v21.py" in source
    )
    assert (
        "build-peak-rank-temporal-min-local-snr-balanced-validation-runtime-v23.py"
        in source
    )
    assert (
        "build-peak-rank-temporal-min-local-snr-balanced-validation-kernel-v23.py"
        in source
    )
    assert "build-peak-rank-capacity-faint-ensemble-validation-runtime-v5.py" in source
    assert "build-peak-rank-capacity-faint-ensemble-validation-kernel-v5.py" in source
    assert (
        "build-peak-rank-capacity-faint-confidence-ensemble-validation-runtime-v6.py"
        in source
    )
    assert (
        "build-peak-rank-capacity-faint-confidence-ensemble-validation-kernel-v6.py"
        in source
    )
    assert "build-peak-rank-cfe-ensemble-validation-runtime-v8.py" in source
    assert "build-peak-rank-cfe-ensemble-validation-kernel-v8.py" in source
    assert (
        "build-peak-rank-expanded-local-shape-ensemble-validation-runtime-v10.py"
        in source
    )
    assert (
        "build-peak-rank-expanded-local-shape-ensemble-validation-kernel-v10.py"
        in source
    )
    assert "build-peak-rank-expanded-blob-ensemble-validation-runtime-v12.py" in source
    assert "build-peak-rank-expanded-blob-ensemble-validation-kernel-v12.py" in source
    assert "build-peak-rank-blob-global-ensemble-validation-runtime-v14.py" in source
    assert "build-peak-rank-blob-global-ensemble-validation-kernel-v14.py" in source
    assert "build-peak-rank-multiscale-triad-validation-runtime-v16.py" in source
    assert "build-peak-rank-multiscale-triad-validation-kernel-v16.py" in source
    assert "build-peak-rank-multiscale-safe-pair-validation-runtime-v18.py" in source
    assert "build-peak-rank-multiscale-safe-pair-validation-kernel-v18.py" in source
    assert "build-peak-rank-safe-balanced-pair-validation-runtime-v20.py" in source
    assert "build-peak-rank-safe-balanced-pair-validation-kernel-v20.py" in source
    assert "build-peak-rank-balanced-xl-pair-validation-runtime-v22.py" in source
    assert "build-peak-rank-balanced-xl-pair-validation-kernel-v22.py" in source
    assert "build-peak-rank-xl-temporal-snr-pair-validation-runtime-v24.py" in source
    assert "build-peak-rank-xl-temporal-snr-pair-validation-kernel-v24.py" in source
    assert "parameter_count = 66977670" in source
    assert "build-peak-rank-logit-ensemble-validation-runtime.py" in source
    assert "build-peak-rank-logit-ensemble-validation-kernel.py" in source
    assert "parameter_count = 76762956" in source
    assert "skipped_after_member_rejection" in source
    assert "kaggle quota --format json" in source
    assert '"Global\\BiohubKaggleGpuSessionV1"' in source
    assert "$gpuReserveHours = 8.0" in source
    assert "$declaredWorstCaseGpuHours = 12.0" in source
    assert "gpu_gate_waiting_for_refresh" in source
    assert "$script:gpuMutex.ReleaseMutex()" in source
    assert 'Write-Terminal "skipped_for_gpu_reserve"' in source


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
    assert 'output_slug = "peak-rank-capacity-pu-validation-v3"' in source
    assert 'output_slug = "peak-rank-faint-pu-validation-v4"' in source
    assert 'output_slug = "peak-rank-expanded-real-faint-validation-v7"' in source
    assert 'output_slug = "peak-rank-expanded-real-local-shape-validation-v9"' in source
    assert 'output_slug = "peak-rank-expanded-real-blob-validation-v11"' in source
    assert 'output_slug = "peak-rank-expanded-real-global-validation-v13"' in source
    assert 'output_slug = "peak-rank-expanded-real-balanced-validation-v19"' in source
    assert (
        'output_slug = "peak-rank-expanded-real-xl-balanced-validation-v21"' in source
    )
    assert (
        'output_slug = "peak-rank-temporal-min-local-snr-validation-v23"' in source
    )
    assert 'output_slug = "peak-rank-capacity-faint-ensemble-validation-v5"' in source
    assert 'output_slug = "peak-rank-capacity-faint-confidence-validation-v6"' in source
    assert 'output_slug = "peak-rank-cfe-ensemble-validation-v8"' in source
    assert (
        'output_slug = "peak-rank-expanded-local-shape-ensemble-validation-v10"'
        in source
    )
    assert 'output_slug = "peak-rank-expanded-blob-ensemble-validation-v12"' in source
    assert 'output_slug = "peak-rank-blob-global-ensemble-validation-v14"' in source
    assert 'output_slug = "peak-rank-safe-balanced-pair-validation-v20"' in source
    assert 'output_slug = "peak-rank-balanced-xl-pair-validation-v22"' in source
    assert 'output_slug = "peak-rank-xl-temporal-snr-pair-validation-v24"' in source
    assert 'output_slug = "peak-rank-logit-ensemble-validation-v4"' in source
    assert "$variantConfig.output_slug" in source
    assert (
        '".biohub/cache/kernel-outputs/peak-rank-validation-v1-version$expectedVersion"'
        not in source
    )
