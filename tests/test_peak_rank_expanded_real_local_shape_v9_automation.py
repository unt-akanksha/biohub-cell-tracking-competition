from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "scripts/deploy-antelume-peak-rank-expanded-real-local-shape-v9.ps1"
HARVEST = ROOT / "scripts/wait-harvest-antelume-peak-rank-expanded-real-local-shape-v9.ps1"
VERIFY = ROOT / "scripts/verify-antelume-peak-rank-expanded-real-local-shape-v9-harvest.py"
RUNTIME = ROOT / "scripts/build-peak-rank-expanded-real-local-shape-validation-runtime-v9.py"
KERNEL = ROOT / "scripts/build-peak-rank-expanded-real-local-shape-validation-kernel-v9.py"
CANDIDATE = ROOT / "scripts/build-peak-rank-expanded-real-local-shape-submission-candidate-v9.py"


def test_deploy_is_hash_bound_and_only_launches_a_waiting_wrapper() -> None:
    source = DEPLOY.read_text(encoding="utf-8")
    assert "1c3f4fd526aa2126f7b14403d91b4493efbc3dc0d48903992c84bb3aecd8654a" in source
    assert "16f36d6772df38b75e82c53585ee2379d9aa5b55d2b5f774828875684cccda7e" in source
    assert "nohup bash" in source
    assert "nvidia-smi" not in source
    assert "pkill" not in source
    assert "killall" not in source
    assert "systemctl" not in source
    assert "kaggle competitions submit" not in source


def test_harvest_and_verifier_freeze_the_v9_contract() -> None:
    harvest = HARVEST.read_text(encoding="utf-8")
    verifier = VERIFY.read_text(encoding="utf-8")
    assert "sha256sum -c" in harvest
    assert "harvest.verified" in harvest
    assert "accepted_for_kaggle_validation" in harvest
    assert "66_977_670" in verifier
    assert "3_000" in verifier
    assert "5_803_219" in verifier
    assert "synthetic256-expanded-real-pu-faint-local-shape-peak-rank-v9" in verifier


def test_validation_and_candidate_paths_are_distinct() -> None:
    runtime = RUNTIME.read_text(encoding="utf-8")
    kernel = KERNEL.read_text(encoding="utf-8")
    candidate = CANDIDATE.read_text(encoding="utf-8")
    assert "peak-rank-expanded-real-local-shape-v9-results.tar.gz" in runtime
    assert "expanded-real-local-shape-validation-runtime-v9" in runtime
    assert "expanded-real-local-shape-validation-v9" in kernel
    assert "expanded-real-local-shape-tracking-candidate-v9" in candidate
    assert "expanded-real local-shape faint-cell" in candidate
