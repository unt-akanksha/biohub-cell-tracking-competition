from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from biohub_tracker.cli import main
from biohub_tracker.ledger import (
    EventType,
    ExperimentEvent,
    Ledger,
    LedgerCorruptionError,
    LedgerLockTimeout,
    RunStatus,
    TransitionError,
    amendment_payload,
    artifact_record,
    completed_payload,
    decision_payload,
    failed_payload,
    reconstruct_runs,
    registration_payload,
    start_payload,
)


def payload(hypothesis="test", parent=None, runtime="1"):
    return registration_payload(
        hypothesis=hypothesis,
        parent=parent,
        config={"lr": "0.001"},
        seeds=[1],
        split="embryo-held-out",
        declared_max_runtime_hours=runtime,
        code={"git_head": "abc", "dirty": False, "dirty_state_sha256": "0" * 64},
    )


def make_ledger(tmp_path, timeout=0.2):
    return Ledger(tmp_path / "experiments" / "events.jsonl", tmp_path, timeout)


def test_register_appends_one_canonical_event(tmp_path):
    ledger = make_ledger(tmp_path)
    ledger.append(ExperimentEvent.create("run-a", EventType.REGISTERED, payload()))
    raw = ledger.path.read_bytes()
    assert raw.count(b"\n") == 1
    event = ledger.read_events()[0]
    assert event.run_id == "run-a"
    assert event.payload["declared_max_runtime_hours"] == "1"
    assert json.loads(raw)["event_type"] == "registered"


def test_register_cli_prints_run_id_and_records_git_state(tmp_path, capsys):
    (tmp_path / "experiments").mkdir()
    exit_code = main(
        [
            "--root",
            str(tmp_path),
            "experiment",
            "register",
            "--run-id",
            "explicit-run",
            "--hypothesis",
            "does this work",
            "--max-runtime-hours",
            "0.5",
            "--seed",
            "7",
        ]
    )
    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "explicit-run"
    event = make_ledger(tmp_path).read_events()[0]
    assert event.payload["code"]["dirty_state_sha256"]


def test_duplicate_nonpositive_unknown_parent_do_not_change_bytes(tmp_path):
    ledger = make_ledger(tmp_path)
    ledger.append(ExperimentEvent.create("run-a", EventType.REGISTERED, payload()))
    original = ledger.path.read_bytes()
    with pytest.raises(TransitionError):
        ledger.append(ExperimentEvent.create("run-a", EventType.REGISTERED, payload()))
    with pytest.raises(ValueError):
        payload(runtime="0")
    with pytest.raises(TransitionError):
        ledger.append(
            ExperimentEvent.create("run-b", EventType.REGISTERED, payload(parent="missing"))
        )
    assert ledger.path.read_bytes() == original


def test_artifact_path_outside_workspace_and_hash_mismatch_fail(tmp_path):
    inside = tmp_path / "artifact.bin"
    inside.write_bytes(b"model")
    outside = tmp_path.parent / "outside-artifact.bin"
    outside.write_bytes(b"outside")
    with pytest.raises(ValueError):
        artifact_record(tmp_path, outside)
    with pytest.raises(ValueError):
        artifact_record(tmp_path, inside, expected_sha256="0" * 64)


def test_truncated_tail_is_quarantined_blocks_append_and_can_be_repaired(tmp_path):
    ledger = make_ledger(tmp_path)
    ledger.append(ExperimentEvent.create("run-a", EventType.REGISTERED, payload()))
    valid_prefix = ledger.path.read_bytes()
    with ledger.path.open("ab") as handle:
        handle.write(b'{"schema_version":1')
    with pytest.raises(LedgerCorruptionError):
        ledger.append(ExperimentEvent.create("run-b", EventType.REGISTERED, payload()))
    quarantine = list((tmp_path / ".biohub" / "quarantine").glob("*.bin"))
    assert len(quarantine) == 1
    ledger.repair_truncated("acknowledged interrupted test write")
    assert ledger.path.read_bytes() == valid_prefix
    ledger.append(ExperimentEvent.create("run-b", EventType.REGISTERED, payload()))
    assert len(ledger.read_events()) == 2


def test_lock_timeout_does_not_change_ledger(tmp_path):
    ledger = make_ledger(tmp_path, timeout=0.02)
    ledger.path.parent.mkdir(parents=True)
    ledger.path.write_bytes(b"")
    ledger.lock_path.write_text("held", encoding="utf-8")
    with pytest.raises(LedgerLockTimeout):
        ledger.append(ExperimentEvent.create("run-a", EventType.REGISTERED, payload()))
    assert ledger.path.read_bytes() == b""


def test_concurrent_append_has_complete_json_lines(tmp_path):
    ledger = make_ledger(tmp_path, timeout=1.0)
    errors = []

    def append(run_id):
        try:
            ledger.append(ExperimentEvent.create(run_id, EventType.REGISTERED, payload()))
        except Exception as exc:  # pragma: no cover - diagnostic capture
            errors.append(exc)

    threads = [threading.Thread(target=append, args=(run_id,)) for run_id in ("run-a", "run-b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert len(ledger.read_events()) == 2
    assert all(json.loads(line) for line in ledger.path.read_text().splitlines())


def complete_metrics():
    return {
        "pooled": {
            "adjusted_edge_jaccard": 0.91,
            "edge_jaccard": 0.90,
            "division_jaccard": 0.10,
            "node_recall": 0.95,
        },
        "division_counts": {"tp": 2, "fp": 1, "fn": 3},
        "by_embryo": {"44b6": {"score": 0.90}, "6bba": {"score": 0.92}},
        "by_fold": {"fold-44b6": {"score": 0.90}},
        "worst_movie_delta": -0.01,
    }


def register_and_start(ledger, run_id="run-a"):
    ledger.append(ExperimentEvent.create(run_id, EventType.REGISTERED, payload()))
    ledger.append(
        ExperimentEvent.create(
            run_id,
            EventType.STARTED,
            start_payload(
                kaggle_ref="owner/kernel",
                authorization_id="auth-1",
                quota_before_hours="30",
            ),
        )
    )


def test_lifecycle_reconstructs_terminal_and_decision(tmp_path):
    ledger = make_ledger(tmp_path)
    register_and_start(ledger)
    ledger.append(
        ExperimentEvent.create(
            "run-a",
            EventType.COMPLETED,
            completed_payload(
                actual_runtime_hours="1.25",
                quota_after_hours="28.75",
                metrics=complete_metrics(),
            ),
        )
    )
    ledger.append(
        ExperimentEvent.create(
            "run-a",
            EventType.DECISION,
            decision_payload("retain", ["exact_oof:report.json"]),
        )
    )
    state = reconstruct_runs(ledger.read_events())["run-a"]
    assert state.status is RunStatus.COMPLETED
    assert state.decision == "retain"
    assert state.terminal["metrics"]["pooled"]["adjusted_edge_jaccard"] == "0.91"


def test_lifecycle_illegal_transition_and_missing_failure_reason_do_not_append(tmp_path):
    ledger = make_ledger(tmp_path)
    ledger.append(ExperimentEvent.create("run-a", EventType.REGISTERED, payload()))
    original = ledger.path.read_bytes()
    with pytest.raises(TransitionError):
        ledger.append(
            ExperimentEvent.create(
                "run-a",
                EventType.COMPLETED,
                completed_payload(
                    actual_runtime_hours="1", quota_after_hours="29", metrics=complete_metrics()
                ),
            )
        )
    with pytest.raises(ValueError):
        failed_payload(actual_runtime_hours="1", quota_after_hours="29", failure_reason="")
    assert ledger.path.read_bytes() == original


def test_hash_missing_artifact_and_mismatch_fail_before_append(tmp_path):
    ledger = make_ledger(tmp_path)
    register_and_start(ledger)
    original = ledger.path.read_bytes()
    with pytest.raises(ValueError):
        artifact_record(tmp_path, "missing.pt")
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"weights")
    with pytest.raises(ValueError):
        artifact_record(tmp_path, artifact, expected_sha256="f" * 64)
    assert ledger.path.read_bytes() == original


def test_promote_rejects_public_only_evidence_without_append(tmp_path):
    ledger = make_ledger(tmp_path)
    register_and_start(ledger)
    ledger.append(
        ExperimentEvent.create(
            "run-a",
            EventType.COMPLETED,
            completed_payload(
                actual_runtime_hours="1", quota_after_hours="29", metrics=complete_metrics()
            ),
        )
    )
    original = ledger.path.read_bytes()
    with pytest.raises(ValueError):
        decision_payload("promote", ["public_score:0.920", "leaderboard:rank-10"])
    assert ledger.path.read_bytes() == original


def test_amendment_preserves_original_line_and_records_both_values(tmp_path):
    ledger = make_ledger(tmp_path)
    registered = ExperimentEvent.create("run-a", EventType.REGISTERED, payload())
    ledger.append(registered)
    original_line = ledger.path.read_bytes()
    amendment = ExperimentEvent.create(
        "run-a",
        EventType.AMENDMENT,
        amendment_payload(
            target_event_id=registered.event_id,
            correction_reason="split label typo",
            replacement_fields={"split": "leave-one-embryo-out"},
        ),
    )
    ledger.append(amendment)
    assert ledger.path.read_bytes().startswith(original_line)
    state = reconstruct_runs(ledger.read_events())["run-a"]
    assert state.registered["split"] == "embryo-held-out"
    assert state.amendments[0]["replacement_fields"]["split"] == "leave-one-embryo-out"


def test_lifecycle_cli_and_amendment_output(tmp_path, capsys):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(complete_metrics()), encoding="utf-8")
    assert main(
        [
            "--root",
            str(tmp_path),
            "experiment",
            "register",
            "--run-id",
            "cli-run",
            "--hypothesis",
            "CLI lifecycle",
            "--max-runtime-hours",
            "2",
        ]
    ) == 0
    registration = make_ledger(tmp_path).read_events()[0]
    assert main(
        [
            "--root",
            str(tmp_path),
            "experiment",
            "start",
            "cli-run",
            "--kaggle-ref",
            "owner/kernel",
            "--authorization-id",
            "auth-1",
            "--quota-before-hours",
            "30",
        ]
    ) == 0
    assert main(
        [
            "--root",
            str(tmp_path),
            "experiment",
            "finish",
            "cli-run",
            "--actual-runtime-hours",
            "1.5",
            "--quota-after-hours",
            "28.5",
            "--metrics-report",
            str(metrics_path),
            "--report",
            str(metrics_path),
        ]
    ) == 0
    assert main(
        [
            "--root",
            str(tmp_path),
            "experiment",
            "decide",
            "cli-run",
            "--decision",
            "retain",
            "--evidence",
            "exact_oof:metrics.json",
        ]
    ) == 0
    assert main(
        [
            "--root",
            str(tmp_path),
            "experiment",
            "amend",
            "cli-run",
            "--target-event-id",
            registration.event_id,
            "--reason",
            "correct split",
            "--replacement",
            '{"split":"leave-one-embryo-out"}',
        ]
    ) == 0
    output = capsys.readouterr().out
    assert '"original"' in output and '"corrected"' in output
    state = reconstruct_runs(make_ledger(tmp_path).read_events())["cli-run"]
    assert state.status is RunStatus.COMPLETED
    assert state.decision == "retain"
    assert len(state.events) == 5
