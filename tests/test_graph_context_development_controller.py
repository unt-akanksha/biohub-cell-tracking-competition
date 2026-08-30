from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "scripts/wait-evaluate-graph-context-division-development.ps1"
EVALUATOR = ROOT / "research/evaluate_graph_context_division_development.py"


def test_controller_is_event_driven_and_submission_ineligible() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")

    assert "harvest-terminal.json" in source
    assert "--extract-to" in source
    assert "evaluate_graph_context_division_development.py" in source
    assert "competition-ranked-consensus-development-baseline-v1.json" in source
    assert "Start-Sleep -Seconds $PollSeconds" in source
    assert "build-graph-context-consensus-division-dataset.py" in source
    assert "GRAPH_CONTEXT_CONSENSUS_MANIFEST.json" in source
    assert 'status = "runtime_packaged"' in source
    assert "authorized_for_full_candidate_evaluation = $true" in source
    assert "authorized_for_submission = $false" in source
    assert "kaggle competitions submit" not in source


def test_evaluator_requires_exact_clean_three_of_three_agreement() -> None:
    source = EVALUATOR.read_text(encoding="utf-8")

    assert 'len(selected) == 3' in source
    assert 'true_positives == 3' in source
    assert 'false_positives == 0' in source
    assert 'pooled["edge_after"]["jaccard"] >= frozen["edge_after"]["jaccard"]' in source
    assert 'pooled["division_after"]["tp"] >= frozen["division_after"]["tp"]' in source
    assert 'pooled["division_after"]["fp"] <= frozen["division_after"]["fp"]' in source
    assert 'probe.get("public_leaderboard_used_for_selection") is False' in source
    assert '"authorized_for_submission": False' in source
