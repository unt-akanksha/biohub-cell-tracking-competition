from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "scripts/wait-launch-strong-member-consensus-candidate.ps1"


def test_launch_waits_for_scientific_runtime_and_requires_two_t4s() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")

    assert 'status -ne "runtime_packaged"' in source
    assert 'runtime_created -ne $true' in source
    assert '"skipped_after_scientific_rejection"' in source
    assert 'expected_gpu_count = 2' in source
    assert 'machine_shape = "NvidiaTeslaT4"' in source
    assert "Exactly two T4 GPUs are required" in source


def test_launch_publishes_private_runtime_then_hands_off_exact_version() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")

    assert "kaggle datasets create" in source
    assert "kaggle datasets version" in source
    assert "kaggle datasets status" in source
    assert "build-strong-member-consensus-submission-candidate.py" in source
    assert "kaggle kernels push" in source
    assert "next_version_number" in source
    assert "current_version_number" in source
    assert "wait-verify-submit-strong-member-consensus-candidate.ps1" in source
    assert "-RuntimeManifest" in source
    assert "-KernelVersion" in source
    assert "kaggle competitions submit" not in source
