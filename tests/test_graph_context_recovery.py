from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run-antelume-graph-context-division-recovery-v1.sh"
HARVESTER = ROOT / "scripts/wait-harvest-antelume-graph-context-recovery-v1.ps1"
DEVELOPMENT = ROOT / "scripts/wait-evaluate-graph-context-recovery-development.ps1"
LAUNCHER = ROOT / "scripts/wait-launch-graph-context-recovery-consensus-candidate.ps1"


def test_recovery_finishes_original_eight_member_policy_after_localizer() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert source.count("seed-613111-init-") == 2
    assert source.count("seed-713117-init-") == 2
    assert source.count("seed-813121-init-1") == 1
    assert "train_synthetic_localizer.py" in source
    assert "while nvidia-smi" in source
    assert "--seeds 613111,713117,813121,913127" in source
    assert "--steps 20000" in source
    assert "--resume-completed" in source
    assert "partial_checkpoint_resumed" in source
    assert "score_graph_context_division_development_probe.py" in source
    assert "kaggle competitions submit" not in source


def test_recovery_harvest_and_candidate_chain_is_distinct_and_gated() -> None:
    harvest = HARVESTER.read_text(encoding="utf-8")
    development = DEVELOPMENT.read_text(encoding="utf-8")
    launch = LAUNCHER.read_text(encoding="utf-8")

    assert "biohub-graph-context-recovery-v1-results.tar.gz" in harvest
    assert "resumed_completed_member_count" in harvest
    assert "verify-antelume-graph-context-division-harvest.py" in harvest
    assert "graph-context-division-recovery-development-v1" in development
    assert "wait-evaluate-graph-context-division-development.ps1" in development
    assert "graph-context-recovery-consensus-candidate" in launch
    assert "wait-launch-relational-consensus-candidate.ps1" in launch
    assert "biohub-ema-graph-context-consensus-v1" in launch
    assert all("kaggle competitions submit" not in source for source in (harvest, development, launch))
