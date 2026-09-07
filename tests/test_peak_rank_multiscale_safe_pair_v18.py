from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "scripts/build-peak-rank-multiscale-safe-pair-validation-runtime-v18.py"
KERNEL = ROOT / "scripts/build-peak-rank-multiscale-safe-pair-validation-kernel-v18.py"
CANDIDATE = ROOT / "scripts/build-peak-rank-multiscale-safe-pair-submission-candidate-v18.py"


def test_runtime_requires_both_individually_clean_members() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    for required in (
        "peak-rank-expanded-real-multiscale-validation-controller-v15.json",
        "peak-rank-expanded-real-safe-rank-validation-controller-v17.json",
        "member-expanded-real-multiscale-v15.pt",
        "member-expanded-real-safe-rank-v17.pt",
        "167_604_492",
        "clean-multiscale-safe-rank-equal-logit-ensemble-v18",
        'sources["model_multiscale.py"]',
        'sources["model_safe_rank.py"]',
    ):
        assert required in source
    assert source.count('"expected_parameter_count": 83_802_246') == 2


def test_validation_and_candidate_have_distinct_v18_identities() -> None:
    kernel = KERNEL.read_text(encoding="utf-8")
    candidate = CANDIDATE.read_text(encoding="utf-8")
    assert "multiscale-safe-pair-validation-v18" in kernel
    assert "multiscale-safe-pair-validation-runtime-v18" in kernel
    assert "167_604_492" in kernel
    assert "multiscale-safe-pair-validation-" in candidate
    assert "runtime-v18" in candidate
    assert "multiscale-safe-pair-tracking-candidate-v18" in candidate
    assert "167.6M-parameter fixed" in candidate
