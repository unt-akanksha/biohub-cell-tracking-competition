from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "scripts/build-peak-rank-faint-pu-validation-runtime.py"
KERNEL = ROOT / "scripts/build-peak-rank-faint-pu-validation-kernel.py"
CANDIDATE = ROOT / "scripts/build-peak-rank-faint-pu-submission-candidate.py"


def test_faint_validation_runtime_binds_distinct_harvest() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    assert "antelume-peak-rank-faint-pu-v4" in source
    assert "peak-rank-faint-pu-v4-results.tar.gz" in source
    assert "synthetic256-real-conservative-pu-faint-temporal-peak-rank-v4" in source
    assert "biohub-peak-rank-faint-pu-validation-runtime-v4" in source
    assert '"EXPECTED_PARAMETER_COUNT": 66_977_670' in source


def test_faint_validation_kernel_is_private_controller_target() -> None:
    source = KERNEL.read_text(encoding="utf-8")
    assert "biohub-peak-rank-faint-pu-validation-v4" in source
    assert "biohub-peak-rank-faint-pu-validation-runtime-v4" in source
    assert '"EXPECTED_PARAMETER_COUNT": 66_977_670' in source


def test_faint_candidate_has_distinct_runtime_and_run_id() -> None:
    source = CANDIDATE.read_text(encoding="utf-8")
    assert "indarkarhana/biohub-peak-rank-faint-pu-validation-runtime-v4" in source
    assert "peak-rank-faint-pu-tracking-candidate-v4" in source
    assert "67.0M-parameter capacity-scaled faint-cell" in source

