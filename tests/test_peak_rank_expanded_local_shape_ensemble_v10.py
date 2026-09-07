from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "scripts/build-peak-rank-expanded-local-shape-ensemble-validation-runtime-v10.py"
KERNEL = ROOT / "scripts/build-peak-rank-expanded-local-shape-ensemble-validation-kernel-v10.py"
CANDIDATE = ROOT / "scripts/build-peak-rank-expanded-local-shape-ensemble-submission-candidate-v10.py"


def test_runtime_requires_two_individually_clean_expanded_members() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    for required in (
        "peak-rank-expanded-real-faint-validation-controller-v7.json",
        "peak-rank-expanded-real-local-shape-validation-controller-v9.json",
        "member-expanded-real-faint-v7.pt",
        "member-expanded-real-local-shape-v9.pt",
        "133_955_340",
        "clean-expanded-local-shape-equal-logit-ensemble-v10",
    ):
        assert required in source
    assert source.count('"expected_parameter_count": 66_977_670') == 2


def test_validation_and_candidate_have_distinct_v10_identities() -> None:
    kernel = KERNEL.read_text(encoding="utf-8")
    candidate = CANDIDATE.read_text(encoding="utf-8")
    assert "expanded-local-shape-ensemble-validation-v10" in kernel
    assert "expanded-local-shape-ensemble-validation-runtime-v10" in kernel
    assert "133_955_340" in kernel
    assert "expanded-local-shape-ensemble-validation-runtime-v10" in candidate
    assert "expanded-local-shape-ensemble-tracking-candidate-v10" in candidate
    assert "two-member 134.0M-parameter" in candidate
