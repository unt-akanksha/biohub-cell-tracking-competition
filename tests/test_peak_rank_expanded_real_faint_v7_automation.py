from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "scripts/wait-deploy-antelume-peak-rank-expanded-real-faint-v7.ps1"
HARVEST = ROOT / "scripts/wait-harvest-antelume-peak-rank-expanded-real-faint-v7.ps1"
VERIFY = ROOT / "scripts/verify-antelume-peak-rank-expanded-real-faint-v7-harvest.py"


def test_deploy_waits_for_both_verified_inputs_and_yields_gpu() -> None:
    source = DEPLOY.read_text(encoding="utf-8")
    assert "harvest-v$ExpandedKernelVersion-terminal.json" in source
    assert 'if ($expanded.status -ne "verified")' in source
    assert 'if ($v4.status -ne "harvest_verified")' in source
    assert "Get-FileHash" in source
    assert "run-antelume-biohub-yield-guard-v1.sh" in source
    assert 'kill -TERM "$pid"' in source
    assert "pkill" not in source
    assert "killall" not in source
    assert "systemctl" not in source


def test_deploy_is_hash_bound_and_does_not_submit() -> None:
    source = DEPLOY.read_text(encoding="utf-8")
    assert "6eb0f506204c1f30fbee3ad9859215826a07fe7c8d78a2be868371233fbd2ebe" in source
    assert "7f8478c770b5d08f95b310f232714e8ba35a9ca2736d648e1c830ccdba48c5af" in source
    assert "competition_submission_performed" in source
    assert "authorized_for_submission" in source
    assert "kaggle competitions submit" not in source


def test_harvester_verifies_and_acknowledges_exact_archive() -> None:
    source = HARVEST.read_text(encoding="utf-8")
    assert "sha256sum -c" in source
    assert "verify-antelume-peak-rank-expanded-real-faint-v7-harvest.py" in source
    assert "harvest.verified" in source
    assert "accepted_for_kaggle_validation" in source
    assert "competition_submission_performed" in source


def test_verifier_freezes_expanded_capacity_contract() -> None:
    source = VERIFY.read_text(encoding="utf-8")
    assert "66_977_670" in source
    assert "2_000" in source
    assert "4_709_011" in source
    assert "synthetic256-expanded-real-pu-faint-temporal-peak-rank-v7" in source
    assert "capacity_conservative_pu_depth_temporal_fading_expanded_real" in source
