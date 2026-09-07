from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "scripts/build-peak-rank-expanded-blob-ensemble-validation-runtime-v12.py"
KERNEL = ROOT / "scripts/build-peak-rank-expanded-blob-ensemble-validation-kernel-v12.py"
CANDIDATE = ROOT / "scripts/build-peak-rank-expanded-blob-ensemble-submission-candidate-v12.py"


def test_runtime_requires_all_three_individually_clean_members() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    for required in (
        "peak-rank-expanded-real-faint-validation-controller-v7.json",
        "peak-rank-expanded-real-local-shape-validation-controller-v9.json",
        "peak-rank-expanded-real-blob-validation-controller-v11.json",
        "member-expanded-real-faint-v7.pt",
        "member-expanded-real-local-shape-v9.pt",
        "member-expanded-real-blob-v11.pt",
        "200_939_922",
        "clean-expanded-faint-local-shape-blob-equal-logit-ensemble-v12",
        'sources["model_blob.py"]',
    ):
        assert required in source
    assert source.count('"expected_parameter_count": 66_977_670') == 2
    assert source.count('"expected_parameter_count": 66_984_582') == 1


def test_validation_and_candidate_have_distinct_v12_identities() -> None:
    kernel = KERNEL.read_text(encoding="utf-8")
    candidate = CANDIDATE.read_text(encoding="utf-8")
    assert "expanded-blob-ensemble-validation-v12" in kernel
    assert "expanded-blob-ensemble-validation-runtime-v12" in kernel
    assert "200_939_922" in kernel
    assert "expanded-blob-ensemble-validation-runtime-v12" in candidate
    assert "expanded-blob-ensemble-tracking-candidate-v12" in candidate
    assert "200.9M-parameter fixed" in candidate
