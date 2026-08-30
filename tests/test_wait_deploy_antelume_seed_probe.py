from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "scripts/wait-deploy-antelume-real-division-seed-probe.ps1"


def test_deploy_controller_waits_for_credentials_and_is_scope_bound() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")

    assert "aws sts get-caller-identity" in source
    assert '$ErrorActionPreference = "Continue"' in source
    assert "$stsExitCode = $LASTEXITCODE" in source
    assert "Start-Sleep -Seconds $PollSeconds" in source
    assert "send-ssh-public-key" in source
    assert "score_real_division_seed_ensemble_probe.py" in source
    assert "overnight_seed_policy.py" in source
    assert "wait-package-antelume-seed-probe-harvest-v1.sh" in source
    assert "/home/ubuntu/biohub-results/competition-real-division-seed-ensemble-probe-v1" in source
    assert "/home/ubuntu/antelume" not in source
    assert "kaggle competitions submit" not in source
    assert 'status = "deployed"' in source
