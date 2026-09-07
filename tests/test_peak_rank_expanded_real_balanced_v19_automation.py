from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "scripts/deploy-antelume-peak-rank-expanded-real-balanced-v19.ps1"
HARVEST = ROOT / "scripts/wait-harvest-antelume-peak-rank-expanded-real-balanced-v19.ps1"
VERIFY = ROOT / "scripts/verify-antelume-peak-rank-expanded-real-balanced-v19-harvest.py"
RUNTIME = ROOT / "scripts/build-peak-rank-expanded-real-balanced-validation-runtime-v19.py"
KERNEL = ROOT / "scripts/build-peak-rank-expanded-real-balanced-validation-kernel-v19.py"
CANDIDATE = ROOT / "scripts/build-peak-rank-expanded-real-balanced-submission-candidate-v19.py"


def test_deploy_is_hash_bound_and_does_not_control_the_gpu() -> None:
    source = DEPLOY.read_text(encoding="utf-8")
    for digest in (
        "854a305143803d2b10116f83d7cf75483d6f91f5930bda8931d95323cdffc802",
        "5570ffcb9734623f28db006b5d85452cc0ff7372eee05a6a65231b0104540f93",
        "57ef88c6756848e83cf8932f2bdee7065633410407acda9bedd16daa0969d248",
    ):
        assert digest in source
    assert "duplicate_44b6_once_balance_embryo_crops" in source
    assert "nohup bash" in source
    assert "nvidia-smi" not in source
    assert "pkill" not in source
    assert "killall" not in source
    assert "systemctl" not in source
    assert "kaggle competitions submit" not in source


def test_harvest_and_verifier_freeze_balanced_contract() -> None:
    harvest = HARVEST.read_text(encoding="utf-8")
    verifier = VERIFY.read_text(encoding="utf-8")
    assert "sha256sum -c" in harvest
    assert "harvest.verified" in harvest
    assert "accepted_for_kaggle_validation" in harvest
    assert "83_802_246" in verifier
    assert "5_000" in verifier
    assert "10_346_297" in verifier
    assert '{"44b6": 150, "6bba": 330}' in verifier
    assert '{"44b6": 300, "6bba": 330}' in verifier
    assert "duplicate_44b6_once_balance_embryo_crops" in verifier


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
    assert "peak-rank-expanded-real-balanced-v19-results.tar.gz" in runtime
    assert "expanded-real-balanced-validation-runtime-v19" in runtime
    assert "expanded-real-balanced-validation-v19" in kernel
    assert "83_802_246" in kernel
    assert "expanded-real-balanced-tracking-candidate-v19" in candidate
    assert "83.8M-parameter embryo-balanced" in candidate
