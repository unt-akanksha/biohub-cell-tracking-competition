from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "scripts/deploy-antelume-peak-rank-expanded-real-safe-rank-v17.ps1"
HARVEST = ROOT / "scripts/wait-harvest-antelume-peak-rank-expanded-real-safe-rank-v17.ps1"
VERIFY = ROOT / "scripts/verify-antelume-peak-rank-expanded-real-safe-rank-v17-harvest.py"
RUNTIME = ROOT / "scripts/build-peak-rank-expanded-real-safe-rank-validation-runtime-v17.py"
KERNEL = ROOT / "scripts/build-peak-rank-expanded-real-safe-rank-validation-kernel-v17.py"
CANDIDATE = ROOT / "scripts/build-peak-rank-expanded-real-safe-rank-submission-candidate-v17.py"


def test_deploy_is_hash_bound_and_does_not_control_the_gpu() -> None:
    source = DEPLOY.read_text(encoding="utf-8")
    for digest in (
        "5570ffcb9734623f28db006b5d85452cc0ff7372eee05a6a65231b0104540f93",
        "89a31ed36d172347b1e0f4e13979bbaa60701f7a86defdc2b4140f191cbbfdd8",
        "7e0a8a14da0e2804eab29eb63abcdf9224885c75f190fce06cb51daab2e3538c",
    ):
        assert digest in source
    assert "safe_rank_multiscale_blob_global_temporal_peak_rank_v17" in source
    assert "nohup bash" in source
    assert "nvidia-smi" not in source
    assert "pkill" not in source
    assert "killall" not in source
    assert "systemctl" not in source
    assert "kaggle competitions submit" not in source


def test_harvest_and_verifier_freeze_safe_rank_contract() -> None:
    harvest = HARVEST.read_text(encoding="utf-8")
    verifier = VERIFY.read_text(encoding="utf-8")
    assert "sha256sum -c" in harvest
    assert "harvest.verified" in harvest
    assert "accepted_for_kaggle_validation" in harvest
    assert "83_802_246" in verifier
    assert "3_000" in verifier
    assert "9_235_183" in verifier
    assert '== 0.02811804008908686' in verifier
    assert "expanded_real_optimization_only" in verifier
    assert "safe_rank_multiscale_blob_global_temporal_peak_rank_v17" in verifier


def test_validation_runtime_carries_models_and_candidate_is_distinct() -> None:
    runtime = RUNTIME.read_text(encoding="utf-8")
    kernel = KERNEL.read_text(encoding="utf-8")
    candidate = CANDIDATE.read_text(encoding="utf-8")
    for model in (
        'sources["model_blob.py"]',
        'sources["model_global.py"]',
        'sources["model_multiscale.py"]',
        'sources["model_safe_rank.py"]',
    ):
        assert model in runtime
    assert "peak-rank-expanded-real-safe-rank-v17-results.tar.gz" in runtime
    assert "expanded-real-safe-rank-validation-runtime-v17" in runtime
    assert "expanded-real-safe-rank-validation-v17" in kernel
    assert "83_802_246" in kernel
    assert "expanded-real-safe-rank-tracking-candidate-v17" in candidate
    assert "83.8M-parameter expanded-real multiscale safe-rank" in candidate
