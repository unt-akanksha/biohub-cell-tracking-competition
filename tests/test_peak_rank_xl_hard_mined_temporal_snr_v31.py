from __future__ import annotations

import importlib.util
from pathlib import Path

from research.peak_rank_detection.model_temporal_stable import (
    TemporalMinimumLocalSnrSafeRankDetector,
)


ROOT = Path(__file__).resolve().parents[1]
TRAINER = (
    ROOT
    / "research/peak_rank_detection/train_expanded_real_xl_temporal_stable_hard_mined_safe_rank_detector.py"
)
RUNNER = ROOT / "scripts/run-antelume-peak-rank-xl-hard-mined-temporal-snr-v31.sh"
DEPLOYER = ROOT / "scripts/deploy-antelume-peak-rank-xl-hard-mined-temporal-snr-v31.ps1"
CONDITIONAL = ROOT / "scripts/wait-deploy-antelume-peak-rank-xl-hard-mined-temporal-snr-v31.ps1"
VERIFIER = ROOT / "scripts/verify-antelume-peak-rank-xl-hard-mined-temporal-snr-v31-harvest.py"


def test_xl_temporal_model_capacity_is_frozen() -> None:
    model = TemporalMinimumLocalSnrSafeRankDetector(
        widths=(160, 320, 640, 1280), depths=(3, 3, 9, 3)
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 129_761_606


def test_runner_combines_only_precommitted_clean_changes() -> None:
    trainer = TRAINER.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")

    assert "hard_mined.RUN_ID = RUN_ID" in trainer
    assert "safe-rank-peak-rank-v31" in trainer
    assert 'test ! -f "$graph_run_root/run.complete"' in runner
    assert 'test ! -f "$graph_ack"' in runner
    assert "verified_graph_harvest" in runner
    assert "BIOHUB_HARD_MINING_MANIFEST" in runner
    assert "--widths 160,320,640,1280" in runner
    assert "--steps 6000" in runner
    assert "--max-wall-seconds 108000" in runner
    assert "kaggle competitions submit" not in runner


def test_deploy_is_blocked_on_independent_parent_evidence() -> None:
    deployer = DEPLOYER.read_text(encoding="utf-8")
    conditional = CONDITIONAL.read_text(encoding="utf-8")

    assert "peak-rank-expanded-real-xl-balanced-validation-controller-v21.json" in deployer
    assert "peak-rank-hard-mined-temporal-snr-validation-controller-v27.json" in deployer
    assert "Graph v2 archive was not independently verified and acknowledged" in deployer
    assert '@("accepted", "scientifically_rejected") -contains $graphHarvest.status' in deployer
    assert 'parameter_count = 129761606' in deployer
    assert 'status = "deployed_waiting_for_verified_graph_harvest"' in deployer
    assert '"skipped_after_parent_rejection"' in conditional
    assert '"xl-hard-mined-temporal-snr-v31"' in conditional


def test_verifier_freezes_xl_hard_mined_contract() -> None:
    spec = importlib.util.spec_from_file_location("v31_verifier", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    expected = module.verify.__globals__
    assert expected["EXPECTED_PARAMETER_COUNT"] == 129_761_606
    assert expected["EXPECTED_STEPS"] == 6_000
    assert expected["EXPECTED_SEED"] == 14_790_551
    assert expected["EXPECTED_WIDTHS"] == [160, 320, 640, 1280]


def test_generic_controllers_include_v31_without_relaxing_gpu_reserve() -> None:
    validation = (
        ROOT / "scripts/wait-launch-evaluate-peak-rank-validation-v1.ps1"
    ).read_text(encoding="utf-8")
    candidate = (
        ROOT / "scripts/wait-build-verify-submit-peak-rank-candidate.ps1"
    ).read_text(encoding="utf-8")

    assert 'elseif ($Variant -eq "xl-hard-mined-temporal-snr-v31")' in validation
    assert "parameter_count = 129761606" in validation
    assert 'elseif ($Variant -eq "xl-hard-mined-temporal-snr-v31")' in candidate
    assert '$gpuReserveHours = 8.0' in validation
    assert '$gpuReserveHours = 8.0' in candidate
