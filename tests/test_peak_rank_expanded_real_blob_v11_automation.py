from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "scripts/deploy-antelume-peak-rank-expanded-real-blob-v11.ps1"
HARVEST = ROOT / "scripts/wait-harvest-antelume-peak-rank-expanded-real-blob-v11.ps1"
VERIFY = ROOT / "scripts/verify-antelume-peak-rank-expanded-real-blob-v11-harvest.py"
RUNTIME = ROOT / "scripts/build-peak-rank-expanded-real-blob-validation-runtime-v11.py"
KERNEL = ROOT / "scripts/build-peak-rank-expanded-real-blob-validation-kernel-v11.py"
CANDIDATE = ROOT / "scripts/build-peak-rank-expanded-real-blob-submission-candidate-v11.py"


def test_deploy_is_hash_bound_and_does_not_control_the_gpu() -> None:
    source = DEPLOY.read_text(encoding="utf-8")
    for digest in (
        "381ca892ee0baf0a3485d238b8981ea32706708a52cfbb4d31569b14aedb52c0",
        "6728c620a1fd10d4e192f9d1e6c9859971bc335b1bedb50ef54e3d85bc684b3d",
        "17cdc28e609c0d5ff5d67723bc49657df2fa0a869207fd172af4d56770c384fb",
    ):
        assert digest in source
    assert "blob_aware_temporal_peak_rank_v11" in source
    assert "nohup bash" in source
    assert "nvidia-smi" not in source
    assert "pkill" not in source
    assert "killall" not in source
    assert "systemctl" not in source
    assert "kaggle competitions submit" not in source


def test_harvest_and_verifier_freeze_blob_contract() -> None:
    harvest = HARVEST.read_text(encoding="utf-8")
    verifier = VERIFY.read_text(encoding="utf-8")
    assert "sha256sum -c" in harvest
    assert "harvest.verified" in harvest
    assert "accepted_for_kaggle_validation" in harvest
    assert "66_984_582" in verifier
    assert "3_000" in verifier
    assert "6_902_243" in verifier
    assert "blob_aware_temporal_peak_rank_v11" in verifier
    assert '"EXPECTED_MODEL_FAMILY"' in verifier


def test_validation_runtime_carries_blob_model_and_candidate_is_distinct() -> None:
    runtime = RUNTIME.read_text(encoding="utf-8")
    kernel = KERNEL.read_text(encoding="utf-8")
    candidate = CANDIDATE.read_text(encoding="utf-8")
    assert 'sources["model_blob.py"]' in runtime
    assert "peak-rank-expanded-real-blob-v11-results.tar.gz" in runtime
    assert "expanded-real-blob-validation-runtime-v11" in runtime
    assert "expanded-real-blob-validation-v11" in kernel
    assert "66_984_582" in kernel
    assert "expanded-real-blob-tracking-candidate-v11" in candidate
    assert "expanded-real blob-aware local-shape" in candidate
