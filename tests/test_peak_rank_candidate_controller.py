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
        '[ValidateSet("v1", "depth-pu-v2", "capacity-pu-v3", "logit-ensemble-v4")]',
        "build-peak-rank-depth-pu-submission-candidate.py",
        "build-peak-rank-capacity-pu-submission-candidate.py",
        "build-peak-rank-logit-ensemble-submission-candidate.py",
        "--expected-run-id",
        "enable_internet -ne $false",
        'Write-Terminal "skipped_after_clean_rejection"',
        'Write-Terminal "candidate_rejected"',
        "competition_submission_performed = $false",
    ):
        assert required in source
