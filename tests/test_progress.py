from __future__ import annotations

from pathlib import Path

from biohub_tracker.ledger import EventType, ExperimentEvent, Ledger, registration_payload
from biohub_tracker.progress import render_progress_json, render_progress_markdown


def seeded_events():
    return Ledger(Path("experiments/events.jsonl"), Path.cwd()).read_events()


def test_progress_includes_all_seeded_families_and_failures():
    report = render_progress_markdown(seeded_events())
    for run_id in (
        "prior-dense-motion-state-guard",
        "prior-pairwise-graph-policy",
        "prior-frozen-topology-replication",
        "prior-vjepa-parent-ranking",
        "prior-sea-raft-parent-ranking",
        "prior-zebrahub-selective-ssm-medium",
        "prior-hoct-dense-movie",
        "prior-ranker-coverage-failure",
    ):
        assert run_id in report
    assert "Out of memory on a dense movie" in report
    assert "Output movie coverage changed" in report


def test_zebrahub_is_incomplete_and_selected_as_next_gate():
    projection = render_progress_json(seeded_events())
    zebra = next(
        run for run in projection["runs"] if run["run_id"] == "prior-zebrahub-selective-ssm-medium"
    )
    expected = "reciprocal_competition_calibration_and_complete_graph_exact_metric"
    assert zebra["status"] == "incomplete"
    assert zebra["lifecycle_status"] == "completed"
    assert zebra["next_gate"] == expected
    assert zebra["authorized_for_submission"] is False
    assert projection["next_experiment"]["run_id"] == zebra["run_id"]
    assert projection["next_experiment"]["next_gate"] == expected


def test_state_guard_and_pairwise_division_tradeoff_are_separate():
    projection = render_progress_json(seeded_events())
    by_id = {run["run_id"]: run for run in projection["runs"]}
    dense = by_id["prior-dense-motion-state-guard"]["exact_evidence"]
    pairwise = by_id["prior-pairwise-graph-policy"]["exact_evidence"]
    assert dense["pooled_score_delta"] == "0.000367"
    assert dense["division_delta"] == "0"
    assert pairwise["pooled_score_delta"] == "0.001454"
    assert pairwise["division_jaccard_delta"] == "-0.001364"


def test_reordered_legal_events_produce_identical_progress():
    events = seeded_events()
    assert render_progress_json(events) == render_progress_json(reversed(events))
    assert render_progress_markdown(events) == render_progress_markdown(reversed(events))


def test_progress_escapes_markdown_from_free_text():
    event = ExperimentEvent.create(
        "escape-run",
        EventType.REGISTERED,
        registration_payload(
            hypothesis="unsafe | table\n<node>",
            parent=None,
            config={},
            seeds=[],
            split="test",
            declared_max_runtime_hours="1",
            code={"git_head": "x"},
        ),
        created_at="2026-08-09T00:00:00Z",
        event_id="evt-escape",
    )
    report = render_progress_markdown([event])
    assert "unsafe \\| table &lt;node>" in report
    assert "\n<node>" not in report
