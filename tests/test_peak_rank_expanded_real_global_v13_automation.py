from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "scripts/deploy-antelume-peak-rank-expanded-real-global-v13.ps1"
HARVEST = ROOT / "scripts/wait-harvest-antelume-peak-rank-expanded-real-global-v13.ps1"
VERIFY = ROOT / "scripts/verify-antelume-peak-rank-expanded-real-global-v13-harvest.py"
RUNTIME = ROOT / "scripts/build-peak-rank-expanded-real-global-validation-runtime-v13.py"
KERNEL = ROOT / "scripts/build-peak-rank-expanded-real-global-validation-kernel-v13.py"
CANDIDATE = ROOT / "scripts/build-peak-rank-expanded-real-global-submission-candidate-v13.py"


def test_deploy_is_hash_bound_and_does_not_control_the_gpu() -> None:
    source = DEPLOY.read_text(encoding="utf-8")
    for digest in (
        "0f348177020a83086d5de89c40f7edb23c5afe9a2d62a4eac8304e3e53657369",
        "ac3f861d7693e760a4da14264ce22cfd55f964ecf5404d16f5ed44bc43acfa82",
        "6728c620a1fd10d4e192f9d1e6c9859971bc335b1bedb50ef54e3d85bc684b3d",
        "837a228aa4a7667bae6d676a5d0e5dac5025985a3d0c8425e740c0db4ce0141b",
    ):
        assert digest in source
    assert "blob_global_context_temporal_peak_rank_v13" in source
    assert "nohup bash" in source
    assert "nvidia-smi" not in source
    assert "pkill" not in source
    assert "killall" not in source
    assert "systemctl" not in source
    assert "kaggle competitions submit" not in source


def test_harvest_and_verifier_freeze_global_contract() -> None:
    harvest = HARVEST.read_text(encoding="utf-8")
    verifier = VERIFY.read_text(encoding="utf-8")
    assert "sha256sum -c" in harvest
    assert "harvest.verified" in harvest
    assert "accepted_for_kaggle_validation" in harvest
    assert "83_788_422" in verifier
    assert "3_000" in verifier
    assert "7_013_267" in verifier
    assert '"EXPECTED_MODEL_FAMILY"' in verifier
    assert "blob_global_context_temporal_peak_rank_v13" in verifier


def test_validation_runtime_carries_models_and_candidate_is_distinct() -> None:
    runtime = RUNTIME.read_text(encoding="utf-8")
    kernel = KERNEL.read_text(encoding="utf-8")
    candidate = CANDIDATE.read_text(encoding="utf-8")
    assert 'sources["model_blob.py"]' in runtime
    assert 'sources["model_global.py"]' in runtime
    assert "peak-rank-expanded-real-global-v13-results.tar.gz" in runtime
    assert "expanded-real-global-validation-runtime-v13" in runtime
    assert "expanded-real-global-validation-v13" in kernel
    assert "83_788_422" in kernel
    assert "expanded-real-global-tracking-candidate-v13" in candidate
    assert "83.8M-parameter expanded-real blob/global-context" in candidate
