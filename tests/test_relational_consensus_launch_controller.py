from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "scripts/wait-launch-relational-consensus-candidate.ps1"


def test_launch_waits_for_positive_development_and_requires_dual_t4_execution() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")

    assert 'status -ne "runtime_packaged"' in source
    assert 'runtime_created -ne $true' in source
    assert 'authorized_for_full_candidate_evaluation -ne $true' in source
    assert 'authorized_for_submission -ne $false' in source
    assert '"skipped_after_scientific_rejection"' in source
    assert 'expected_gpu_count = 2' in source
    assert 'machine_shape = "NvidiaTeslaT4"' in source
    assert "Exactly two T4 GPUs are required" in source
    assert '"cuda:0"' in source
    assert '"cuda:1"' in source
    assert '"ThreadPoolExecutor"' in source


def test_launch_publishes_private_runtime_and_hands_off_exact_version() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")

    assert "kaggle datasets create" in source
    assert "kaggle datasets version" in source
    assert "kaggle datasets status" in source
    assert "build-relational-consensus-submission-candidate.py" in source
    assert "kaggle kernels push" in source
    assert "next_version_number" in source
    assert "current_version_number" in source
    assert "wait-verify-submit-strong-member-consensus-candidate.ps1" in source
    assert '"-CandidateKernelRef", $kernelRef' in source
    assert '"-VerifierScript", $verifier' in source
    assert '"-SubmitterScript", $submitter' in source
    assert '"-StatePrefix", "relational-consensus-candidate"' in source
    assert ") + $promotionArguments" in source
    assert "-ArgumentList $promotionProcessArguments" in source
    assert "kaggle competitions submit" not in source


def test_launch_has_validation_only_preflight() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")

    assert "[switch]$ValidateOnly" in source
    assert 'status = "validated"' in source
    assert "python -m py_compile" in source
    assert "candidate_submission_requires_promotion_gate = $true" in source
