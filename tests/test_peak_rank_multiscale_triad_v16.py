from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "scripts/build-peak-rank-multiscale-triad-validation-runtime-v16.py"
KERNEL = ROOT / "scripts/build-peak-rank-multiscale-triad-validation-kernel-v16.py"
CANDIDATE = ROOT / "scripts/build-peak-rank-multiscale-triad-submission-candidate-v16.py"


def test_runtime_requires_all_three_individually_clean_members() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    for required in (
        "peak-rank-expanded-real-blob-validation-controller-v11.json",
        "peak-rank-expanded-real-global-validation-controller-v13.json",
        "peak-rank-expanded-real-multiscale-validation-controller-v15.json",
        "member-expanded-real-blob-v11.pt",
        "member-expanded-real-global-v13.pt",
        "member-expanded-real-multiscale-v15.pt",
        "234_575_250",
        "clean-local-global-multiscale-equal-logit-ensemble-v16",
        'sources["model_blob.py"]',
        'sources["model_global.py"]',
        'sources["model_multiscale.py"]',
    ):
        assert required in source
    assert source.count('"expected_parameter_count": 66_984_582') == 1
    assert source.count('"expected_parameter_count": 83_788_422') == 1
    assert source.count('"expected_parameter_count": 83_802_246') == 1


def test_validation_and_candidate_have_distinct_v16_identities() -> None:
    kernel = KERNEL.read_text(encoding="utf-8")
    candidate = CANDIDATE.read_text(encoding="utf-8")
    assert "multiscale-triad-validation-v16" in kernel
    assert "multiscale-triad-validation-runtime-v16" in kernel
    assert "234_575_250" in kernel
    assert "multiscale-triad-validation-runtime-v16" in candidate
    assert "multiscale-triad-tracking-candidate-v16" in candidate
    assert "234.6M-parameter fixed" in candidate
