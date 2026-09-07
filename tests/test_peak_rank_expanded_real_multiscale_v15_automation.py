from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "scripts/deploy-antelume-peak-rank-expanded-real-multiscale-v15.ps1"
HARVEST = ROOT / "scripts/wait-harvest-antelume-peak-rank-expanded-real-multiscale-v15.ps1"
VERIFY = ROOT / "scripts/verify-antelume-peak-rank-expanded-real-multiscale-v15-harvest.py"
RUNTIME = ROOT / "scripts/build-peak-rank-expanded-real-multiscale-validation-runtime-v15.py"
KERNEL = ROOT / "scripts/build-peak-rank-expanded-real-multiscale-validation-kernel-v15.py"
CANDIDATE = ROOT / "scripts/build-peak-rank-expanded-real-multiscale-submission-candidate-v15.py"


def test_deploy_is_hash_bound_and_does_not_control_the_gpu() -> None:
    source = DEPLOY.read_text(encoding="utf-8")
    for digest in (
        "ad4a9b496a07afc200ecfbe3c504d6cae5ffff138071f5cbe27c52cb7dcfd008",
        "56cff859acd3c6e3554fb91bb9fb3b4f7c5dca64dbab00bc2a22e8aa5483b9bf",
        "4c6a3bf853d99f18f1112deab62b88476dfc0db4ebce4202d7486457c28d063a",
    ):
        assert digest in source
    assert "multiscale_blob_global_context_temporal_peak_rank_v15" in source
    assert "nohup bash" in source
    assert "nvidia-smi" not in source
    assert "pkill" not in source
    assert "killall" not in source
    assert "systemctl" not in source
    assert "kaggle competitions submit" not in source


def test_harvest_and_verifier_freeze_multiscale_contract() -> None:
    harvest = HARVEST.read_text(encoding="utf-8")
    verifier = VERIFY.read_text(encoding="utf-8")
    assert "sha256sum -c" in harvest
    assert "harvest.verified" in harvest
    assert "accepted_for_kaggle_validation" in harvest
    assert "83_802_246" in verifier
    assert "3_000" in verifier
    assert "8_124_071" in verifier
    assert "[[3, 9], [5, 13], [7, 15]]" in verifier
    assert "expanded_real_optimization_only" in verifier
    assert "multiscale_blob_global_context_temporal_peak_rank_v15" in verifier


def test_validation_runtime_carries_models_and_candidate_is_distinct() -> None:
    runtime = RUNTIME.read_text(encoding="utf-8")
    kernel = KERNEL.read_text(encoding="utf-8")
    candidate = CANDIDATE.read_text(encoding="utf-8")
    assert 'sources["model_blob.py"]' in runtime
    assert 'sources["model_global.py"]' in runtime
    assert 'sources["model_multiscale.py"]' in runtime
    assert "peak-rank-expanded-real-multiscale-v15-results.tar.gz" in runtime
    assert "expanded-real-multiscale-validation-runtime-v15" in runtime
    assert "expanded-real-multiscale-validation-v15" in kernel
    assert "83_802_246" in kernel
    assert "expanded-real-multiscale-tracking-candidate-v15" in candidate
    assert "83.8M-parameter expanded-real multiscale" in candidate
