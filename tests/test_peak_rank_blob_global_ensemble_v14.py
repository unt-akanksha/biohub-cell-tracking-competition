from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "scripts/build-peak-rank-blob-global-ensemble-validation-runtime-v14.py"
KERNEL = ROOT / "scripts/build-peak-rank-blob-global-ensemble-validation-kernel-v14.py"
CANDIDATE = ROOT / "scripts/build-peak-rank-blob-global-ensemble-submission-candidate-v14.py"


def test_runtime_requires_both_individually_clean_members() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    for required in (
        "peak-rank-expanded-real-blob-validation-controller-v11.json",
        "peak-rank-expanded-real-global-validation-controller-v13.json",
        "member-expanded-real-blob-v11.pt",
        "member-expanded-real-global-v13.pt",
        "150_773_004",
        "clean-blob-global-context-equal-logit-ensemble-v14",
        'sources["model_blob.py"]',
        'sources["model_global.py"]',
    ):
        assert required in source
    assert source.count('"expected_parameter_count": 66_984_582') == 1
    assert source.count('"expected_parameter_count": 83_788_422') == 1


def test_validation_and_candidate_have_distinct_v14_identities() -> None:
    kernel = KERNEL.read_text(encoding="utf-8")
    candidate = CANDIDATE.read_text(encoding="utf-8")
    assert "blob-global-ensemble-validation-v14" in kernel
    assert "blob-global-ensemble-validation-runtime-v14" in kernel
    assert "150_773_004" in kernel
    assert "blob-global-ensemble-validation-runtime-v14" in candidate
    assert "blob-global-ensemble-tracking-candidate-v14" in candidate
    assert "150.8M-parameter fixed" in candidate
