from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "wait-build-verify-submit-peak-rank-candidate.ps1"


def test_controller_waits_for_clean_promotion_then_submits_once() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for required in (
        "accepted_for_candidate_integration",
        "promote-peak-rank-validation-runtime.py",
        "datasets version",
        "build-peak-rank-submission-candidate.py",
        "verify-peak-rank-submission-candidate.py",
        "score_official_candidate.py",
        "submit-peak-rank-candidate.py",
        "--file-pattern",
        "--execute",
        'Write-Terminal "submitted"',
    ):
        assert required in source


def test_controller_is_two_gpu_private_and_fail_closed() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for required in (
        'machine_shape -ne "NvidiaTeslaT4"',
        "device_count() != 2",
        "SEC_EDGE_TTA_ACTIVE",
        "redoctopusk/biohub-948tta2",
        "selected_peak_tta_mode",
        "max_projected_worker_seconds",
        '[ValidateSet("v1", "depth-pu-v2", "capacity-pu-v3", "faint-pu-v4", "expanded-real-faint-v7", "expanded-real-local-shape-v9", "expanded-real-blob-v11", "expanded-real-global-v13", "expanded-real-multiscale-v15", "expanded-real-safe-rank-v17", "expanded-real-balanced-v19", "expanded-real-xl-balanced-v21", "temporal-min-local-snr-v23", "capacity-faint-ensemble-v5", "capacity-faint-confidence-v6", "capacity-faint-expanded-v8", "expanded-local-shape-ensemble-v10", "expanded-blob-ensemble-v12", "blob-global-ensemble-v14", "multiscale-triad-v16", "multiscale-safe-pair-v18", "safe-balanced-pair-v20", "balanced-xl-pair-v22", "logit-ensemble-v4")]',
        "build-peak-rank-depth-pu-submission-candidate.py",
        "build-peak-rank-capacity-pu-submission-candidate.py",
        "build-peak-rank-faint-pu-submission-candidate.py",
        "build-peak-rank-expanded-real-faint-submission-candidate-v7.py",
        "build-peak-rank-expanded-real-local-shape-submission-candidate-v9.py",
        "build-peak-rank-expanded-real-blob-submission-candidate-v11.py",
        "build-peak-rank-expanded-real-global-submission-candidate-v13.py",
        "build-peak-rank-expanded-real-multiscale-submission-candidate-v15.py",
        "build-peak-rank-expanded-real-safe-rank-submission-candidate-v17.py",
        "build-peak-rank-expanded-real-balanced-submission-candidate-v19.py",
        "build-peak-rank-expanded-real-xl-balanced-submission-candidate-v21.py",
        "build-peak-rank-temporal-min-local-snr-balanced-submission-candidate-v23.py",
        "build-peak-rank-capacity-faint-ensemble-submission-candidate-v5.py",
        "build-peak-rank-capacity-faint-confidence-ensemble-submission-candidate-v6.py",
        "build-peak-rank-cfe-ensemble-submission-candidate-v8.py",
        "build-peak-rank-expanded-local-shape-ensemble-submission-candidate-v10.py",
        "build-peak-rank-expanded-blob-ensemble-submission-candidate-v12.py",
        "build-peak-rank-blob-global-ensemble-submission-candidate-v14.py",
        "build-peak-rank-multiscale-triad-submission-candidate-v16.py",
        "build-peak-rank-multiscale-safe-pair-submission-candidate-v18.py",
        "build-peak-rank-safe-balanced-pair-submission-candidate-v20.py",
        "build-peak-rank-balanced-xl-pair-submission-candidate-v22.py",
        "build-peak-rank-logit-ensemble-submission-candidate.py",
        "--expected-run-id",
        "--official-metric-result",
        "official_validator_candidate.csv",
        'Write-Terminal "candidate_rejected_by_patched_official_metric"',
        "enable_internet -ne $false",
        'Write-Terminal "skipped_after_clean_rejection"',
        'Write-Terminal "candidate_rejected"',
        "competition_submission_performed = $false",
        "kaggle quota --format json",
        '"Global\\BiohubKaggleGpuSessionV1"',
        "$gpuReserveHours = 8.0",
        "$declaredWorstCaseGpuHours = 12.0",
        'Write-Terminal "skipped_for_gpu_reserve"',
    ):
        assert required in source
