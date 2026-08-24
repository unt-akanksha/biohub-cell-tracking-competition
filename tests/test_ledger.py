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
    TransitionError,
    artifact_record,
    registration_payload,
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
