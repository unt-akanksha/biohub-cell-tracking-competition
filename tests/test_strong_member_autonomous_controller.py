from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "scripts/wait-verify-submit-strong-member-consensus-candidate.ps1"


def test_controller_requires_runtime_bound_promotion_before_submit() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")

    assert "[Parameter(Mandatory = $true)]" in source
    assert "verify-strong-member-consensus-submission-candidate.py" in source
    assert "submit-strong-member-consensus-candidate.py" in source
    assert "--runtime-manifest $RuntimeManifest" in source
    assert "[string]$StatePrefix = 'strong-member-candidate'" in source
    assert '"$StatePrefix-submission-receipt.json"' in source
    assert "competition_submission_performed = $true" in source
    assert "candidate_rejected" in source
    assert "--execute" in source
    assert "Refusing to reuse controller state" in source
    assert '$versionedKernelRef = "$kernelRef/$KernelVersion"' in source
    assert "--file-pattern $requiredOutputPattern" in source


def test_controller_can_be_parameterized_for_a_sealed_candidate_family() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")

    assert "[string]$CandidateKernelRef" in source
    assert "[string]$VerifierScript" in source
    assert "[string]$SubmitterScript" in source
    assert "[string]$ControllerRunId" in source
    assert "$kernelRef = $CandidateKernelRef" in source
