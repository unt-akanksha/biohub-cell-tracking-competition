from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = (
    ROOT
    / "scripts/build-peak-rank-temporal-min-local-snr-balanced-validation-runtime-v23.py"
)
KERNEL = (
    ROOT
    / "scripts/build-peak-rank-temporal-min-local-snr-balanced-validation-kernel-v23.py"
)
CANDIDATE = (
    ROOT
    / "scripts/build-peak-rank-temporal-min-local-snr-balanced-submission-candidate-v23.py"
)


def test_validation_runtime_carries_temporal_local_snr_model() -> None:
    runtime = RUNTIME.read_text(encoding="utf-8")
    for model in (
        '"model_blob.py"',
        '"model_global.py"',
        '"model_multiscale.py"',
        '"model_safe_rank.py"',
        '"model_temporal_stable.py"',
    ):
        assert model in runtime
    assert "peak-rank-temporal-min-local-snr-balanced-v23-results.tar.gz" in runtime
    assert "83_812_614" in runtime


def test_kernel_and_candidate_are_distinct_and_two_gpu_ready() -> None:
    kernel = KERNEL.read_text(encoding="utf-8")
    candidate = CANDIDATE.read_text(encoding="utf-8")
    assert "temporal-min-local-snr-validation-v23" in kernel
    assert "83_812_614" in kernel
    assert "temporal-min-local-snr-tracking-candidate-v23" in candidate
    assert "83.8M-parameter embryo-balanced temporal-minimum local-SNR" in candidate
