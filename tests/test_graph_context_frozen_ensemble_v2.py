from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run-antelume-graph-context-frozen-ensemble-v2.sh"
DEPLOYER = ROOT / "scripts/deploy-antelume-graph-context-frozen-ensemble-v2.ps1"
HARVESTER = ROOT / "scripts/wait-harvest-antelume-graph-context-frozen-ensemble-v2.ps1"
VERIFIER = ROOT / "scripts/verify-antelume-graph-context-frozen-ensemble-v2-harvest.py"


def test_runner_precommits_fresh_equal_rank_ensemble_after_detector_queue() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "hard-mined-temporal-snr-v27" in source
    assert '"$prior_run_root/harvest.verified"' in source
    assert "while nvidia-smi --query-compute-apps=pid" in source
    assert "--seeds 1013131,1113137,1213139,1313141" in source
    assert "--policy-contract all-selection-admitted-equal-rank-ensemble-v2" in source
    assert "--steps 20000" in source
    assert "competition-graph-context-division-frozen-ensemble-v2" in source
    assert "kaggle competitions submit" not in source


def test_deployment_and_harvest_are_hash_bound_and_credential_tolerant() -> None:
    deployment = DEPLOYER.read_text(encoding="utf-8")
    harvest = HARVESTER.read_text(encoding="utf-8")

    assert "MaximumCredentialPolls = 14400" in deployment
    assert '[string]$DirectRemoteHost = ""' in deployment
    assert 'connection_mode = if ($publishKeyRequired)' in deployment
    assert 'if ($publishKeyRequired) { Publish-Key }' in deployment
    assert "__GRAPH_CONTEXT_TRAINER_SHA256__" in deployment
    assert 'worst_case_gpu_hours = 5' in deployment
    assert 'constituent_audit_gate_required = $false' in deployment
    assert "MaximumPolls = 2880" in harvest
    assert "harvest.verified" in harvest
    assert "verify-antelume-graph-context-frozen-ensemble-v2-harvest.py" in harvest


def test_verifier_accepts_only_a_frozen_audited_ensemble_unit() -> None:
    spec = importlib.util.spec_from_file_location("graph_context_v2_verifier", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = VERIFIER.read_text(encoding="utf-8")

    assert module.TRAIN_RUN_ID == "competition-graph-context-division-frozen-ensemble-v2"
    assert module.metric_gate(
        {
            "average_precision": 0.70,
            "true_positives_before_first_false_positive": 3,
            "by_embryo": {
                "44b6": {"average_precision": 0.50},
                "6bba": {"average_precision": 0.60},
            },
        }
    )
    assert 'terminal.get("policy_unit_audited") is True' in source
    assert 'terminal.get("constituent_audit_gate_required") is False' in source
    assert 'terminal.get("model_subset_searched_on_audit") is False' in source
    assert "competition_submission_performed" in source
