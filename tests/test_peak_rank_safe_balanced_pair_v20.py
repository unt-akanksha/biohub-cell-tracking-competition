from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "scripts/build-peak-rank-safe-balanced-pair-validation-runtime-v20.py"
KERNEL = ROOT / "scripts/build-peak-rank-safe-balanced-pair-validation-kernel-v20.py"
CANDIDATE = ROOT / "scripts/build-peak-rank-safe-balanced-pair-submission-candidate-v20.py"


def test_runtime_requires_both_individually_clean_members() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    for required in (
        "peak-rank-expanded-real-safe-rank-validation-controller-v17.json",
        "peak-rank-expanded-real-balanced-validation-controller-v19.json",
        "member-expanded-real-safe-rank-v17.pt",
        "member-expanded-real-balanced-v19.pt",
        "167_604_492",
        "clean-safe-rank-balanced-equal-logit-ensemble-v20",
        'sources["model_multiscale.py"]',
        'sources["model_safe_rank.py"]',
    ):
        assert required in source
    assert source.count('"expected_parameter_count": 83_802_246') == 2


def test_validation_and_candidate_have_distinct_v20_identities() -> None:
    kernel = KERNEL.read_text(encoding="utf-8")
    candidate = CANDIDATE.read_text(encoding="utf-8")
    assert "safe-balanced-pair-validation-v20" in kernel
    assert "safe-balanced-pair-validation-runtime-v20" in kernel
    assert "167_604_492" in kernel
    assert "safe-balanced-pair-validation-" in candidate
    assert "runtime-v20" in candidate
    assert "safe-balanced-pair-tracking-candidate-v20" in candidate
    assert "167.6M-parameter fixed" in candidate
