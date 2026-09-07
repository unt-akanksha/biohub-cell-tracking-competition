from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run-antelume-peak-rank-hard-mined-temporal-snr-v27.sh"
HARVEST = ROOT / "scripts/wait-harvest-antelume-peak-rank-hard-mined-temporal-snr-v27.ps1"
VERIFIER = ROOT / "scripts/verify-antelume-peak-rank-hard-mined-temporal-snr-v27-harvest.py"
DEPLOY = ROOT / "scripts/deploy-antelume-peak-rank-hard-mined-temporal-snr-v27.ps1"


def test_runner_is_hash_bound_sequential_and_hard_mined() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "v23_run_root/run.complete" in source
    assert "v23_ack" in source
    assert "verified_v23_harvest" in source
    assert "while nvidia-smi --query-compute-apps=pid" in source
    assert "BIOHUB_HARD_MINING_MANIFEST" in source
    assert "9967efa25021453b4f043a23e47e744153da678f17f5c00180ecd6ebd4dca900" in source
    assert "--steps 6000" in source
    assert "--seed 13679443" in source
    assert "--max-wall-seconds 86400" in source
    assert "RSNA" not in source


def test_harvest_and_verifier_fail_closed() -> None:
    harvest = HARVEST.read_text(encoding="utf-8")
    verifier = VERIFIER.read_text(encoding="utf-8")
    assert "verify-antelume-peak-rank-hard-mined-temporal-snr-v27-harvest.py" in harvest
    assert "harvest.verified" in harvest
    assert "accepted_for_kaggle_validation" in harvest
    assert '"EXPECTED_STEPS": 6_000' in verifier
    assert '"EXPECTED_SEED": 13_679_443' in verifier
    assert '"effective_counts") == {"44b6": 495, "6bba": 495}' in verifier
    assert "public_leaderboard" not in harvest.lower()


def test_deploy_preflights_exact_recipe_and_stages_no_public_teacher() -> None:
    source = DEPLOY.read_text(encoding="utf-8")
    assert "1832e8dc7738ca778b3b4f07159e68741d8d6fef2e30e2a14b1075748851e7c8" in source
    assert "9967efa25021453b4f043a23e47e744153da678f17f5c00180ecd6ebd4dca900" in source
    assert "e3a881ca4be87bbffb0f2772ec4aa844fd46a624c921845359310402070c4490" in source
    assert "assert sum(parameter.numel() for parameter in model.parameters()) == 83812614" in source
    assert "deployed_waiting_for_v23" in source
    assert "public_teacher" not in source
    assert "RSNA" not in source
