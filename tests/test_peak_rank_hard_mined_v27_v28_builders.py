from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_v27_runtime_and_kernel_bind_hard_mined_harvest() -> None:
    runtime = _source(
        "scripts/build-peak-rank-hard-mined-temporal-snr-validation-runtime-v27.py"
    )
    kernel = _source(
        "scripts/build-peak-rank-hard-mined-temporal-snr-validation-kernel-v27.py"
    )
    assert "peak-rank-hard-mined-temporal-snr-v27-results.tar.gz" in runtime
    assert "optimization-only hard-example sampling" in runtime
    assert '"EXPECTED_PARAMETER_COUNT": 83_812_614' in runtime
    assert "biohub-peak-rank-hard-mined-temporal-snr-validation-runtime-v27" in kernel
    assert '"EXPECTED_PARAMETER_COUNT": 83_812_614' in kernel


def test_v28_pair_requires_two_individually_promoted_members() -> None:
    runtime = _source(
        "scripts/build-peak-rank-xl-hard-mined-temporal-snr-pair-validation-runtime-v28.py"
    )
    assert "peak-rank-expanded-real-xl-balanced-validation-controller-v21.json" in runtime
    assert "peak-rank-hard-mined-temporal-snr-validation-controller-v27.json" in runtime
    assert runtime.count('"expected_parameter_count"') == 2
    assert '"PARAMETER_COUNT": 213_561_260' in runtime
    assert "equal-logit" in runtime


def test_generic_controllers_route_v27_and_v28() -> None:
    validation = _source("scripts/wait-launch-evaluate-peak-rank-validation-v1.ps1")
    candidate = _source("scripts/wait-build-verify-submit-peak-rank-candidate.ps1")
    for source in (validation, candidate):
        assert '"hard-mined-temporal-snr-v27"' in source
        assert '"xl-hard-mined-temporal-snr-pair-v28"' in source
    assert "peak-rank-hard-mined-temporal-snr-validation-controller-v27.json" in validation
    assert "peak-rank-xl-hard-mined-temporal-snr-pair-validation-controller-v28" in validation
    assert "peak-rank-hard-mined-temporal-snr-candidate-controller-v27" in candidate
    assert "peak-rank-xl-hard-mined-temporal-snr-pair-candidate-controller-v28" in candidate


def test_v27_and_v28_candidate_builders_use_private_runtimes() -> None:
    v27 = _source(
        "scripts/build-peak-rank-hard-mined-temporal-snr-submission-candidate-v27.py"
    )
    v28 = _source(
        "scripts/build-peak-rank-xl-hard-mined-temporal-snr-pair-submission-candidate-v28.py"
    )
    assert "biohub-peak-rank-hard-mined-temporal-snr-validation-" in v27
    assert '"runtime-v27"' in v27
    assert "biohub-peak-rank-xl-hard-mined-temporal-snr-pair-" in v28
    assert '"validation-runtime-v28"' in v28
    assert "213.6M-parameter fixed" in v28
