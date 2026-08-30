from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run-antelume-graph-context-division-sweep-v1.sh"
CONTROLLER = ROOT / "scripts/wait-deploy-antelume-graph-context-division-sweep.ps1"


def test_remote_runner_is_large_sequential_and_antelume_bound() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "--steps 20000" in source
    assert "613111,713117,813121,913127" in source
    assert source.count('--initial-model "$initial_') == 2
    assert "graph_context_division_model.py" in source
    assert "train_graph_context_division_sweep.py" in source
    assert "prior_terminal=" in source
    assert 'while ! test -f "$prior_terminal"' in source
    assert "while nvidia-smi" in source
    assert "A10G" in source
    assert "__GRAPH_CONTEXT_ARCHIVE_SHA256__" in source
    assert "__GRAPH_CONTEXT_MANIFEST_SHA256__" in source
    assert "kaggle competitions submit" not in source


def test_controller_waits_for_relational_deploy_and_credentials() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")

    assert "deployment-terminal.json" in source
    assert 'status -ne "deployed_and_queued"' in source
    assert '"skipped_after_relational_deploy_failure"' in source
    assert "aws sts get-caller-identity" in source
    assert "ec2-instance-connect send-ssh-public-key" in source
    assert "nohup bash scripts/run-antelume-graph-context-division-sweep-v1.sh" in source
    assert "parameters_per_model = 74732308" in source
    assert "steps_per_model = 20000" in source
    assert "ensemble_members_precommitted_before_audit = $true" in source
    assert "model_subset_searched_on_audit = $false" in source
    assert "competition_submission_performed = $false" in source
    assert "kaggle competitions submit" not in source


def test_controller_has_offline_validation_mode() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")

    assert "[switch]$ValidateOnly" in source
    assert 'status = "validated"' in source
    assert "python -m py_compile" in source
    assert 'bash -n "scripts/run-antelume-graph-context-division-sweep-v1.sh"' in source
