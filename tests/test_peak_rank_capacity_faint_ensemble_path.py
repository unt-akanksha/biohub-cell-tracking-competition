from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "scripts/build-peak-rank-capacity-faint-ensemble-validation-runtime-v5.py"
KERNEL = ROOT / "scripts/build-peak-rank-capacity-faint-ensemble-validation-kernel-v5.py"
CANDIDATE = ROOT / "scripts/build-peak-rank-capacity-faint-ensemble-submission-candidate-v5.py"
BASE = ROOT / "scripts/build-peak-rank-logit-ensemble-validation-runtime.py"


def test_capacity_faint_runtime_requires_two_large_clean_members() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    assert "peak-rank-capacity-pu-validation-controller-v3.json" in source
    assert "peak-rank-faint-pu-validation-controller-v4.json" in source
    assert source.count('"expected_parameter_count": 66_977_670') == 2
    assert '"PARAMETER_COUNT": 133_955_340' in source
    assert "clean-capacity-faint-equal-logit-peak-rank-ensemble-v5" in source


def test_base_ensemble_builder_supports_member_specific_sizes() -> None:
    source = BASE.read_text(encoding="utf-8")
    assert 'spec.get("expected_parameter_count", 38_381_478)' in source
    assert 'manifest.get("parameter_count") == expected_parameter_count' in source
    assert '"purpose": PURPOSE' in source
    assert '"title": DATASET_TITLE' in source


def test_capacity_faint_kernel_and_candidate_have_distinct_ids() -> None:
    kernel = KERNEL.read_text(encoding="utf-8")
    candidate = CANDIDATE.read_text(encoding="utf-8")
    assert "biohub-peak-rank-capacity-faint-ensemble-validation-v5" in kernel
    assert '"EXPECTED_PARAMETER_COUNT": 133_955_340' in kernel
    assert "peak-rank-capacity-faint-ensemble-tracking-candidate-v5" in candidate
    assert "134.0M-parameter" in candidate

