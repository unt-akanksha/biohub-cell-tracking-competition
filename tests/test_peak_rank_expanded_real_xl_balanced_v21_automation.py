from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "scripts/deploy-antelume-peak-rank-expanded-real-xl-balanced-v21.ps1"
HARVEST = ROOT / "scripts/wait-harvest-antelume-peak-rank-expanded-real-xl-balanced-v21.ps1"
VERIFY = ROOT / "scripts/verify-antelume-peak-rank-expanded-real-xl-balanced-v21-harvest.py"
RUNTIME = ROOT / "scripts/build-peak-rank-expanded-real-xl-balanced-validation-runtime-v21.py"
KERNEL = ROOT / "scripts/build-peak-rank-expanded-real-xl-balanced-validation-kernel-v21.py"
CANDIDATE = ROOT / "scripts/build-peak-rank-expanded-real-xl-balanced-submission-candidate-v21.py"


def test_deploy_is_hash_bound_and_does_not_control_gpu() -> None:
    source = DEPLOY.read_text(encoding="utf-8")
    for digest in (
        "4f3ae24fa996082b976803e473086a94fe3c0f4eb45698c226fac566ab1a4d6e",
        "854a305143803d2b10116f83d7cf75483d6f91f5930bda8931d95323cdffc802",
        "d5e0b93e082a9f42f271fa752b8121b90caee546c6ebfc0bc1414355df02d0ae",
    ):
        assert digest in source
    assert "129748646" in source
    assert "nohup bash" in source
    assert "nvidia-smi" not in source
    assert "pkill" not in source
    assert "kaggle competitions submit" not in source


def test_harvest_and_verifier_freeze_xl_contract() -> None:
    harvest = HARVEST.read_text(encoding="utf-8")
    verifier = VERIFY.read_text(encoding="utf-8")
    assert "sha256sum -c" in harvest
    assert "harvest.verified" in harvest
    assert "accepted_for_kaggle_validation" in harvest
    assert "129_748_646" in verifier
    assert "5_000" in verifier
    assert "11_457_211" in verifier
    assert "duplicate_44b6_once_balance_embryo_crops" in verifier


def test_runtime_and_candidate_freeze_xl_capacity() -> None:
    runtime = RUNTIME.read_text(encoding="utf-8")
    kernel = KERNEL.read_text(encoding="utf-8")
    candidate = CANDIDATE.read_text(encoding="utf-8")
    assert "peak-rank-expanded-real-xl-balanced-v21-results.tar.gz" in runtime
    assert "129_748_646" in runtime
    assert "expanded-real-xl-balanced-validation-v21" in kernel
    assert "129_748_646" in kernel
    assert "expanded-real-xl-balanced-tracking-candidate-v21" in candidate
    assert "129.7M-parameter XL" in candidate
