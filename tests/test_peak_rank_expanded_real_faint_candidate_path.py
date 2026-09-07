from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "scripts/build-peak-rank-expanded-real-faint-validation-runtime-v7.py"
KERNEL = ROOT / "scripts/build-peak-rank-expanded-real-faint-validation-kernel-v7.py"
CANDIDATE = ROOT / "scripts/build-peak-rank-expanded-real-faint-submission-candidate-v7.py"


def test_runtime_uses_only_the_verified_v7_archive() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    assert "antelume-peak-rank-expanded-real-faint-v7" in source
    assert "peak-rank-expanded-real-faint-v7-results.tar.gz" in source
    assert "synthetic256-expanded-real-pu-faint-temporal-peak-rank-v7" in source
    assert "66_977_670" in source
    assert "biohub-peak-rank-expanded-real-faint-validation-runtime-v7" in source


def test_validation_kernel_is_distinct_and_dual_gpu_inherited() -> None:
    source = KERNEL.read_text(encoding="utf-8")
    assert "build-peak-rank-validation-kernel.py" in source
    assert "biohub-peak-rank-expanded-real-faint-validation-v7" in source
    assert "biohub-peak-rank-expanded-real-faint-validation-runtime-v7" in source
    assert "66_977_670" in source


def test_candidate_has_distinct_runtime_and_identity() -> None:
    source = CANDIDATE.read_text(encoding="utf-8")
    assert "build-peak-rank-submission-candidate.py" in source
    assert "indarkarhana/biohub-peak-rank-expanded-real-faint-validation-runtime-v7" in source
    assert "peak-rank-expanded-real-faint-tracking-candidate-v7" in source
    assert "expanded-real-coverage faint-cell" in source
