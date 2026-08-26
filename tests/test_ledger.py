from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from biohub_tracker.cli import main
from biohub_tracker.ledger import (
    CpuAcceptanceStatus,
    EventType,
    ExactEvaluationStatus,
    ExperimentEvent,
    Ledger,
    LedgerCorruptionError,
    LedgerLockTimeout,
    RunStatus,
    TransitionError,
    amendment_payload,
    artifact_record,
    completed_payload,
    cpu_acceptance_completed_payload,
    cpu_acceptance_failed_payload,
    cpu_acceptance_inputs_bound_payload,
    cpu_acceptance_registration_payload,
    cpu_acceptance_started_payload,
    decision_payload,
    failed_payload,
    event_sha256,
    exact_evaluation_completed_payload,
    exact_evaluation_failed_payload,
    exact_evaluation_registration_payload,
    exact_evaluation_started_payload,
    reconstruct_runs,
    reconstruct_cpu_acceptances,
    reconstruct_exact_evaluations,
    registration_payload,
    start_payload,
    validate_producer_registration_evidence,
    validate_producer_terminal_evidence,
    _ExclusiveLock,
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


def test_v1_event_decoder_rejects_unknown_root_field():
    event = ExperimentEvent.create("run-a", EventType.REGISTERED, payload()).to_dict()
    event["future_unhashed_field"] = "must-not-be-discarded"
    with pytest.raises(ValueError, match="unknown or missing fields"):
        ExperimentEvent.from_dict(event)


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


def test_backdated_lifecycle_events_are_rejected_without_changing_bytes(tmp_path):
    def rejected(ledger, event):
        before = ledger.path.read_bytes()
        with pytest.raises(TransitionError):
            ledger.append(event)
        assert ledger.path.read_bytes() == before

    start_ledger = make_ledger(tmp_path / "start")
    start_ledger.append(
        ExperimentEvent.create(
            "run-start", EventType.REGISTERED, payload(), created_at="2099-01-01T00:00:01Z"
        )
    )
    rejected(
        start_ledger,
        ExperimentEvent.create(
            "run-start",
            EventType.STARTED,
            start_payload(
                kaggle_ref="owner/kernel", authorization_id="auth", quota_before_hours="30"
            ),
            created_at="2099-01-01T00:00:00Z",
        ),
    )

    terminal_ledger = make_ledger(tmp_path / "terminal")
    terminal_ledger.append(
        ExperimentEvent.create(
            "run-terminal", EventType.REGISTERED, payload(), created_at="2099-01-01T00:00:01Z"
        )
    )
    terminal_ledger.append(
        ExperimentEvent.create(
            "run-terminal",
            EventType.STARTED,
            start_payload(
                kaggle_ref="owner/kernel", authorization_id="auth", quota_before_hours="30"
            ),
            created_at="2099-01-01T00:00:02Z",
        )
    )
    rejected(
        terminal_ledger,
        ExperimentEvent.create(
            "run-terminal",
            EventType.COMPLETED,
            completed_payload(
                actual_runtime_hours="1", quota_after_hours="29", metrics=complete_metrics()
            ),
            created_at="2099-01-01T00:00:01.500000Z",
        ),
    )

    decision_ledger = make_ledger(tmp_path / "decision")
    decision_ledger.append(
        ExperimentEvent.create(
            "run-decision", EventType.REGISTERED, payload(), created_at="2099-01-01T00:00:01Z"
        )
    )
    decision_ledger.append(
        ExperimentEvent.create(
            "run-decision",
            EventType.STARTED,
            start_payload(
                kaggle_ref="owner/kernel", authorization_id="auth", quota_before_hours="30"
            ),
            created_at="2099-01-01T00:00:02Z",
        )
    )
    decision_ledger.append(
        ExperimentEvent.create(
            "run-decision",
            EventType.COMPLETED,
            completed_payload(
                actual_runtime_hours="1", quota_after_hours="29", metrics=complete_metrics()
            ),
            created_at="2099-01-01T00:00:03Z",
        )
    )
    rejected(
        decision_ledger,
        ExperimentEvent.create(
            "run-decision",
            EventType.DECISION,
            decision_payload("retain", ["exact_oof:report.json"]),
            created_at="2099-01-01T00:00:02.500000Z",
        ),
    )

    cpu_ledger = make_ledger(tmp_path / "cpu")
    cpu_registration = ExperimentEvent.create(
        "cpu-control-a",
        EventType.CPU_ACCEPTANCE_REGISTERED,
        _cpu_registration(),
        created_at="2099-01-01T00:00:01Z",
    )
    cpu_ledger.append(cpu_registration)
    rejected(
        cpu_ledger,
        ExperimentEvent.create(
            "cpu-control-a",
            EventType.CPU_ACCEPTANCE_STARTED,
            cpu_acceptance_started_payload(
                run_id="cpu-control-a",
                registration_event_sha256=event_sha256(cpu_registration),
                kernel_ref="owner/biohub-phase2-cpu-acceptance/1",
                runtime_dataset_ref="owner/biohub-phase2-runtime/1",
            ),
            created_at="2099-01-01T00:00:00Z",
        ),
    )

    exact_ledger = make_ledger(tmp_path / "exact")
    folds = ("fold-44b6-to-6bba", "fold-6bba-to-44b6")
    members = []
    for index, (role, fold) in enumerate(
        (role, fold) for role in ("baseline", "candidate") for fold in folds
    ):
        member = _evidence_producer(exact_ledger, f"producer-{index}", fold, str(index + 2))
        member["role"] = role
        members.append(member)
    exact_registration = exact_evaluation_registration_payload(
        evaluation_run_id="eval-backdated",
        scorer_lock_sha256="a" * 64,
        environment_lock_sha256="b" * 64,
        manifest_sha256="c" * 64,
        evaluation_policy_sha256="d" * 64,
        evidence_kind="synthetic_fixture",
        members=members,
    )
    exact_ledger.append(
        ExperimentEvent.create(
            "eval-backdated",
            EventType.EXACT_EVALUATION_REGISTERED,
            exact_registration,
            created_at="2000-01-01T00:00:00Z",
        )
    )


def test_unrelated_lifecycle_activity_does_not_block_deterministic_cpu_saga(tmp_path):
    ledger = make_ledger(tmp_path)
    registration = ExperimentEvent.create(
        "cpu-control-a",
        EventType.CPU_ACCEPTANCE_REGISTERED,
        _cpu_registration(),
        created_at="2026-08-25T00:00:00Z",
    )
    ledger.append(registration)
    ledger.append(
        ExperimentEvent.create(
            "cpu-control-a",
            EventType.CPU_ACCEPTANCE_STARTED,
            cpu_acceptance_started_payload(
                run_id="cpu-control-a",
                registration_event_sha256=event_sha256(registration),
                kernel_ref="owner/biohub-phase2-cpu-acceptance/1",
                runtime_dataset_ref="owner/biohub-phase2-runtime/1",
            ),
            created_at="2026-08-25T00:00:00Z",
        )
    )
    ledger.append(
        ExperimentEvent.create(
            "unrelated-run",
            EventType.REGISTERED,
            payload(),
            created_at="2026-08-25T00:00:10Z",
        )
    )

    folds = [
        {
            "fold_id": f"fold-{token}",
            "train_membership_sha256": token * 64,
            "calibration_membership_sha256": token * 64,
            "evaluation_membership_sha256": token * 64,
        }
        for token in ("a", "b")
    ]
    saga_event = ExperimentEvent.create(
        "cpu-control-a",
        EventType.CPU_ACCEPTANCE_INPUTS_BOUND,
        cpu_acceptance_inputs_bound_payload(
            run_id="cpu-control-a", manifest_sha256="3" * 64, folds=folds
        ),
        created_at="2026-08-25T00:00:00Z",
    )
    ledger.append(saga_event)
    before = ledger.path.read_bytes()
    with pytest.raises(TransitionError, match="duplicate event ID"):
        ledger.append(saga_event)
    assert ledger.path.read_bytes() == before


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


def test_active_lock_metadata_is_never_stolen(tmp_path):
    ledger = make_ledger(tmp_path, timeout=0.02)
    ledger.path.parent.mkdir(parents=True)
    ledger.path.write_bytes(b"")
    with _ExclusiveLock(ledger.lock_path, 0.2):
        metadata = json.loads(ledger.lock_path.read_bytes())
        assert metadata["pid"] > 0
        assert metadata["process_start"]
        assert metadata["created_at"].endswith("Z")
        assert len(metadata["owner_token"]) == 64
        with pytest.raises(LedgerLockTimeout):
            ledger.append(ExperimentEvent.create("run-a", EventType.REGISTERED, payload()))
        assert ledger.lock_path.exists()
    assert not ledger.lock_path.exists()


def test_stale_path_lock_fails_closed_without_moving_owner_bytes(tmp_path):
    ledger = make_ledger(tmp_path, timeout=0.02)
    ledger.path.parent.mkdir(parents=True)
    ledger.path.write_bytes(b"")
    stale_bytes = json.dumps(
        {
            "schema_version": "biohub.ledger-lock.v1",
            "pid": 999_999_999,
            "process_start": "proc-start-ticks:stale",
            "created_at": "2000-01-01T00:00:00Z",
            "created_at_epoch_seconds": time.time() - 60,
            "owner_token": "a" * 64,
        }
    ).encode()
    ledger.lock_path.write_bytes(stale_bytes)
    with pytest.raises(LedgerLockTimeout):
        ledger.append(ExperimentEvent.create("run-a", EventType.REGISTERED, payload()))
    assert ledger.path.read_bytes() == b""
    assert ledger.lock_path.read_bytes() == stale_bytes
    assert list(ledger.path.parent.glob(".*.stale-*")) == []


def test_stale_recovery_never_renames_a_replaced_live_lock(tmp_path):
    ledger = make_ledger(tmp_path, timeout=0.02)
    ledger.path.parent.mkdir(parents=True)
    live_bytes = json.dumps(
        {
            "schema_version": "biohub.ledger-lock.v1",
            "pid": 123,
            "process_start": "new-owner",
            "created_at": "2026-08-25T00:00:00Z",
            "created_at_epoch_seconds": time.time(),
            "owner_token": "b" * 64,
        }
    ).encode()
    ledger.lock_path.write_bytes(live_bytes)
    assert _ExclusiveLock(ledger.lock_path, 0.02)._recover_stale() is False
    assert ledger.lock_path.read_bytes() == live_bytes


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


def test_optional_producer_evidence_is_additive_and_legacy_bytes_are_unchanged():
    legacy = payload()
    assert validate_producer_registration_evidence(legacy) is None
    legacy_completed = completed_payload(
        actual_runtime_hours="1", quota_after_hours="29", metrics=complete_metrics()
    )
    assert validate_producer_terminal_evidence(legacy_completed) is None
    assert "evidence_eligible" not in legacy_completed

    lineage = {
        "manifest_sha256": "1" * 64,
        "fold_id": "fold-44b6-to-6bba",
        "train_membership_sha256": "2" * 64,
        "calibration_membership_sha256": "3" * 64,
        "evaluation_membership_sha256": "4" * 64,
        "model_sha256": "5" * 64,
        "config_sha256": legacy["config_sha256"],
        "code_sha256": "6" * 64,
        "data_sha256": "7" * 64,
    }
    extended = registration_payload(
        hypothesis="test",
        parent=None,
        config={"lr": "0.001"},
        seeds=[1],
        split="embryo-held-out",
        declared_max_runtime_hours="1",
        code={"git_head": "abc", "dirty": False, "dirty_state_sha256": "0" * 64},
        producer_evidence=lineage,
    )
    assert validate_producer_registration_evidence(extended) == lineage
    terminal = completed_payload(
        actual_runtime_hours="1",
        quota_after_hours="29",
        metrics=complete_metrics(),
        producer_evidence={
            "evidence_eligible": True,
            "graph_inventory_sha256": "8" * 64,
            "artifact_hashes": {"model": "9" * 64},
        },
    )
    assert terminal["actual_runtime_hours"] == legacy_completed["actual_runtime_hours"]
    assert terminal["quota_after_hours"] == legacy_completed["quota_after_hours"]
    assert validate_producer_terminal_evidence(terminal)["evidence_eligible"] is True


def test_partial_producer_evidence_is_rejected_without_changing_gpu_lifecycle(tmp_path):
    ledger = make_ledger(tmp_path)
    incomplete = payload()
    incomplete["manifest_sha256"] = "1" * 64
    with pytest.raises(ValueError):
        ledger.append(ExperimentEvent.create("partial", EventType.REGISTERED, incomplete))
    assert not ledger.path.exists()


def _evidence_producer(ledger, run_id, fold_id, token):
    base = registration_payload(
        hypothesis=run_id,
        parent=None,
        config={"token": token},
        seeds=[1],
        split=fold_id,
        declared_max_runtime_hours="1",
        code={"git_head": token},
    )
    lineage = {
        "manifest_sha256": "1" * 64,
        "fold_id": fold_id,
        "train_membership_sha256": token * 64,
        "calibration_membership_sha256": token * 64,
        "evaluation_membership_sha256": token * 64,
        "model_sha256": token * 64,
        "config_sha256": base["config_sha256"],
        "code_sha256": token * 64,
        "data_sha256": token * 64,
    }
    registration = ExperimentEvent.create(
        run_id,
        EventType.REGISTERED,
        registration_payload(
            hypothesis=run_id,
            parent=None,
            config={"token": token},
            seeds=[1],
            split=fold_id,
            declared_max_runtime_hours="1",
            code={"git_head": token},
            producer_evidence=lineage,
        ),
    )
    ledger.append(registration)
    ledger.append(
        ExperimentEvent.create(
            run_id,
            EventType.STARTED,
            start_payload(kaggle_ref="owner/kernel", authorization_id=run_id, quota_before_hours="30"),
        )
    )
    terminal = ExperimentEvent.create(
        run_id,
        EventType.COMPLETED,
        completed_payload(
            actual_runtime_hours="1",
            quota_after_hours="29",
            metrics=complete_metrics(),
            producer_evidence={
                "evidence_eligible": True,
                "graph_inventory_sha256": token * 64,
                "artifact_hashes": {"predictions": token * 64},
            },
        ),
    )
    ledger.append(terminal)
    return {
        "role": "baseline",
        "fold_id": fold_id,
        "producer_run_id": run_id,
        "producer_registration_event_sha256": event_sha256(registration),
        "producer_terminal_event_sha256": event_sha256(terminal),
        **lineage,
        "graph_inventory_sha256": token * 64,
        "artifact_hashes": {"predictions": token * 64},
    }


def test_exact_evaluation_lifecycle_is_separate_immutable_and_fail_closed(tmp_path):
    ledger = make_ledger(tmp_path)
    folds = ("fold-44b6-to-6bba", "fold-6bba-to-44b6")
    members = []
    for index, (role, fold) in enumerate(
        (role_fold for role in ("baseline", "candidate") for role_fold in ((role, item) for item in folds))
    ):
        token = "2345"[index]
        member = _evidence_producer(ledger, f"{role}-{index}", fold, token)
        member["role"] = role
        members.append(member)
    registered = exact_evaluation_registration_payload(
        evaluation_run_id="eval-a",
        scorer_lock_sha256="a" * 64,
        environment_lock_sha256="b" * 64,
        manifest_sha256="1" * 64,
        evaluation_policy_sha256="c" * 64,
        evidence_kind="synthetic_fixture",
        members=members,
    )
    ledger.append(
        ExperimentEvent.create("eval-a", EventType.EXACT_EVALUATION_REGISTERED, registered)
    )
    ledger.append(
        ExperimentEvent.create(
            "eval-a",
            EventType.EXACT_EVALUATION_STARTED,
            exact_evaluation_started_payload(evaluation_run_id="eval-a"),
        )
    )
    inventories = [
        {
            "role": item["role"],
            "fold_id": item["fold_id"],
            "producer_run_id": item["producer_run_id"],
            "submission_graph_inventory_sha256": item["graph_inventory_sha256"],
            "roundtrip_evidence_sha256": item["model_sha256"],
            "csv_sha256": item["data_sha256"],
        }
        for item in members
    ]
    completed = exact_evaluation_completed_payload(
        evaluation_run_id="eval-a",
        members=members,
        report_core_sha256="d" * 64,
        envelope_sha256="e" * 64,
        artifact_hashes={"core": "d" * 64, "envelope": "e" * 64},
        authoritative_inventories=inventories,
    )
    ledger.append(
        ExperimentEvent.create("eval-a", EventType.EXACT_EVALUATION_COMPLETED, completed)
    )
    state = reconstruct_exact_evaluations(ledger.read_events())["eval-a"]
    assert state.status is ExactEvaluationStatus.COMPLETED
    assert state.terminal["members"] == members
    assert len(reconstruct_runs(ledger.read_events())) == 4
    before = ledger.path.read_bytes()
    with pytest.raises(TransitionError):
        ledger.append(
            ExperimentEvent.create(
                "eval-a",
                EventType.EXACT_EVALUATION_FAILED,
                exact_evaluation_failed_payload(
                    evaluation_run_id="eval-a", reason_code="LATE", detail="late failure"
                ),
            )
        )
    assert ledger.path.read_bytes() == before


def _cpu_registration(
    run_id="cpu-control-a",
    nonce="ab" * 32,
    evaluation_run_id="eval-cpu-control-a",
):
    return cpu_acceptance_registration_payload(
        run_id=run_id,
        purpose="phase2_official_data_control",
        request_nonce=nonce,
        acceptance_request_sha256="1" * 64,
        evaluation_run_id=evaluation_run_id,
        kernel_slug="owner/biohub-phase2-cpu-acceptance",
        runtime_dataset_slug="owner/biohub-phase2-runtime",
        runtime_bundle_name="biohub-runtime-v1.biohubbundle",
        runtime_bundle_sha256="9" * 64,
        runtime_bundle_inventory_sha256="a" * 64,
        runtime_bundle_uncompressed_size_bytes=1024,
        runtime_bundle_file_count=7,
        scorer_lock_sha256="2" * 64,
        environment_lock_sha256="3" * 64,
        manifest_policy_sha256="4" * 64,
        control_model_sha256="5" * 64,
        config_sha256="6" * 64,
        code_sha256="7" * 64,
        data_source_sha256="8" * 64,
        cpu_watchdog_minutes=660,
    )


def _cpu_bind(run_id="cpu-control-a"):
    return cpu_acceptance_inputs_bound_payload(
        run_id=run_id,
        manifest_sha256="9" * 64,
        folds=[
            {
                "fold_id": "fold-44b6-to-6bba",
                "train_membership_sha256": "a" * 64,
                "calibration_membership_sha256": "b" * 64,
                "evaluation_membership_sha256": "c" * 64,
            },
            {
                "fold_id": "fold-6bba-to-44b6",
                "train_membership_sha256": "d" * 64,
                "calibration_membership_sha256": "e" * 64,
                "evaluation_membership_sha256": "f" * 64,
            },
        ],
    )


def test_cpu_acceptance_lifecycle_is_separate_and_does_not_change_phase1_projection(tmp_path):
    ledger = make_ledger(tmp_path)
    ledger.append(ExperimentEvent.create("legacy", EventType.REGISTERED, payload()))
    legacy_events = ledger.read_events()
    legacy_runs = reconstruct_runs(legacy_events)
    registration = ExperimentEvent.create(
        "cpu-control-a", EventType.CPU_ACCEPTANCE_REGISTERED, _cpu_registration()
    )
    ledger.append(registration)
    ledger.append(
        ExperimentEvent.create(
            "cpu-control-a",
            EventType.CPU_ACCEPTANCE_STARTED,
            cpu_acceptance_started_payload(
                run_id="cpu-control-a",
                registration_event_sha256=event_sha256(registration),
                kernel_ref="owner/biohub-phase2-cpu-acceptance/7",
                runtime_dataset_ref="owner/biohub-phase2-runtime/3",
            ),
        )
    )
    binding = ExperimentEvent.create(
        "cpu-control-a", EventType.CPU_ACCEPTANCE_INPUTS_BOUND, _cpu_bind()
    )
    ledger.append(binding)
    ledger.append(
        ExperimentEvent.create(
            "cpu-control-a",
            EventType.CPU_ACCEPTANCE_COMPLETED,
            cpu_acceptance_completed_payload(
                run_id="cpu-control-a",
                actual_cpu_runtime_seconds="12.5",
                peak_memory_mb="256",
                remote_job_identity="owner/kernel/7",
                graph_inventory_sha256="0" * 64,
                artifact_hashes={"graphs": "1" * 64},
                output_inventory_sha256="2" * 64,
                pending_payload_sha256="3" * 64,
                pending_envelope_sha256="4" * 64,
                reconciliation_sha256="5" * 64,
            ),
        )
    )
    state = reconstruct_cpu_acceptances(ledger.read_events())["cpu-control-a"]
    assert state.status is CpuAcceptanceStatus.COMPLETED
    assert state.inputs_bound["manifest_sha256"] == "9" * 64
    assert reconstruct_runs(ledger.read_events()) == legacy_runs
    assert all("quota" not in key and "gpu" not in key for event in state.events for key in event.payload)


@pytest.mark.parametrize(
    "event_type,payload_factory",
    [
        (EventType.CPU_ACCEPTANCE_INPUTS_BOUND, lambda registration: _cpu_bind()),
        (
            EventType.CPU_ACCEPTANCE_COMPLETED,
            lambda registration: cpu_acceptance_completed_payload(
                run_id="cpu-control-a",
                actual_cpu_runtime_seconds="1",
                peak_memory_mb="1",
                remote_job_identity="owner/kernel/1",
                graph_inventory_sha256="0" * 64,
                artifact_hashes={"graphs": "1" * 64},
                output_inventory_sha256="2" * 64,
                pending_payload_sha256="3" * 64,
                pending_envelope_sha256="4" * 64,
                reconciliation_sha256="5" * 64,
            ),
        ),
    ],
)
def test_cpu_acceptance_rejects_skipped_transitions(tmp_path, event_type, payload_factory):
    ledger = make_ledger(tmp_path)
    registration = ExperimentEvent.create(
        "cpu-control-a", EventType.CPU_ACCEPTANCE_REGISTERED, _cpu_registration()
    )
    ledger.append(registration)
    before = ledger.path.read_bytes()
    with pytest.raises(TransitionError):
        ledger.append(ExperimentEvent.create("cpu-control-a", event_type, payload_factory(registration)))
    assert ledger.path.read_bytes() == before


def test_cpu_acceptance_rejects_nonce_reuse_gpu_fields_and_reopen(tmp_path):
    ledger = make_ledger(tmp_path)
    first = ExperimentEvent.create(
        "cpu-control-a", EventType.CPU_ACCEPTANCE_REGISTERED, _cpu_registration()
    )
    ledger.append(first)
    with pytest.raises(TransitionError):
        ledger.append(
            ExperimentEvent.create(
                "cpu-control-b",
                EventType.CPU_ACCEPTANCE_REGISTERED,
                _cpu_registration(run_id="cpu-control-b"),
            )
        )
    invalid = _cpu_registration(run_id="cpu-control-b", nonce="cd" * 32)
    invalid["quota_before_hours"] = "30"
    with pytest.raises(TransitionError):
        ledger.append(
            ExperimentEvent.create(
                "cpu-control-b", EventType.CPU_ACCEPTANCE_REGISTERED, invalid
            )
        )
    ledger.append(
        ExperimentEvent.create(
            "cpu-control-a",
            EventType.CPU_ACCEPTANCE_FAILED,
            cpu_acceptance_failed_payload(
                run_id="cpu-control-a", reason_code="TEST", detail="terminal"
            ),
        )
    )
    with pytest.raises(TransitionError):
        ledger.append(
            ExperimentEvent.create(
                "cpu-control-a",
                EventType.CPU_ACCEPTANCE_STARTED,
                cpu_acceptance_started_payload(
                    run_id="cpu-control-a",
                    registration_event_sha256=event_sha256(first),
                    kernel_ref="owner/kernel/1",
                    runtime_dataset_ref="owner/dataset/1",
                ),
            )
        )


def test_cpu_and_exact_run_ids_collide_symmetrically(tmp_path):
    ledger = make_ledger(tmp_path)
    ledger.append(
        ExperimentEvent.create(
            "cpu-control-a", EventType.CPU_ACCEPTANCE_REGISTERED, _cpu_registration()
        )
    )
    before = ledger.path.read_bytes()
    with pytest.raises(TransitionError, match="duplicate exact evaluation ID"):
        ledger.append(
            ExperimentEvent.create(
                "cpu-control-a",
                EventType.EXACT_EVALUATION_REGISTERED,
                {"evaluation_run_id": "cpu-control-a"},
            )
        )
    assert ledger.path.read_bytes() == before

    other = make_ledger(tmp_path / "other")
    members = []
    for index, (role, fold_id) in enumerate(
        (
            ("baseline", "fold-a"),
            ("baseline", "fold-b"),
            ("candidate", "fold-a"),
            ("candidate", "fold-b"),
        )
    ):
        member = _evidence_producer(
            other, f"producer-{index}", fold_id, str(index + 1)
        )
        member["role"] = role
        members.append(member)
    other.append(
        ExperimentEvent.create(
            "eval-existing",
            EventType.EXACT_EVALUATION_REGISTERED,
            exact_evaluation_registration_payload(
                evaluation_run_id="eval-existing",
                scorer_lock_sha256="a" * 64,
                environment_lock_sha256="b" * 64,
                manifest_sha256="c" * 64,
                evaluation_policy_sha256="d" * 64,
                evidence_kind="synthetic_fixture",
                members=members,
            ),
        )
    )
    with pytest.raises(TransitionError, match="duplicate CPU acceptance ID"):
        other.append(
            ExperimentEvent.create(
                "eval-existing",
                EventType.CPU_ACCEPTANCE_REGISTERED,
                _cpu_registration(
                    run_id="eval-existing",
                    nonce="cd" * 32,
                    evaluation_run_id="eval-future",
                ),
            )
        )


def test_cpu_evaluation_reservations_are_global_and_survive_terminal_state(tmp_path):
    ledger = make_ledger(tmp_path)
    ledger.append(
        ExperimentEvent.create(
            "cpu-control-a", EventType.CPU_ACCEPTANCE_REGISTERED, _cpu_registration()
        )
    )
    ledger.append(
        ExperimentEvent.create(
            "cpu-control-a",
            EventType.CPU_ACCEPTANCE_FAILED,
            cpu_acceptance_failed_payload(
                run_id="cpu-control-a", reason_code="TEST", detail="terminal"
            ),
        )
    )
    for run_id, evaluation_run_id in (
        ("cpu-control-b", "eval-cpu-control-a"),
        ("eval-cpu-control-a", "eval-other"),
    ):
        with pytest.raises(TransitionError):
            ledger.append(
                ExperimentEvent.create(
                    run_id,
                    EventType.CPU_ACCEPTANCE_REGISTERED,
                    _cpu_registration(
                        run_id=run_id,
                        nonce=("cd" if run_id == "cpu-control-b" else "ef") * 32,
                        evaluation_run_id=evaluation_run_id,
                    ),
                )
            )
    with pytest.raises(TransitionError, match="duplicate run ID"):
        ledger.append(
            ExperimentEvent.create(
                "eval-cpu-control-a", EventType.REGISTERED, payload()
            )
        )


def test_cpu_registration_rejects_self_reserved_evaluation_id(tmp_path):
    ledger = make_ledger(tmp_path)
    with pytest.raises(TransitionError, match="must differ"):
        ledger.append(
            ExperimentEvent.create(
                "cpu-control-a",
                EventType.CPU_ACCEPTANCE_REGISTERED,
                _cpu_registration(evaluation_run_id="cpu-control-a"),
            )
        )
