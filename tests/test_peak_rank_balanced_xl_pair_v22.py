from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "scripts/build-peak-rank-balanced-xl-pair-validation-runtime-v22.py"
KERNEL = ROOT / "scripts/build-peak-rank-balanced-xl-pair-validation-kernel-v22.py"
CANDIDATE = ROOT / "scripts/build-peak-rank-balanced-xl-pair-submission-candidate-v22.py"


def test_runtime_requires_two_individually_clean_capacity_diverse_members() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    for required in (
        "peak-rank-expanded-real-balanced-validation-controller-v19.json",
        "peak-rank-expanded-real-xl-balanced-validation-controller-v21.json",
        "member-expanded-real-balanced-v19.pt",
        "member-expanded-real-xl-balanced-v21.pt",
        "213_550_892",
        "clean-balanced-xl-equal-logit-ensemble-v22",
        '"expected_parameter_count": 83_802_246',
        '"expected_parameter_count": 129_748_646',
    ):
        assert required in source


def test_validation_and_candidate_have_distinct_v22_identities() -> None:
    kernel = KERNEL.read_text(encoding="utf-8")
    candidate = CANDIDATE.read_text(encoding="utf-8")
    assert "balanced-xl-pair-validation-v22" in kernel
    assert "213_550_892" in kernel
    assert "balanced-xl-pair-tracking-candidate-v22" in candidate
    assert "213.6M-parameter fixed" in candidate
