from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from biohub_tracker.watchdog import BudgetExpired, BudgetWatchdog


def test_watchdog_triggers_callbacks_once_and_persists_terminal_on_error(tmp_path):
    current = {"mono": 0.0}
    calls = []

    def monotonic():
        return current["mono"]

    watchdog = BudgetWatchdog(
        "run-a",
        declared_budget_seconds=100,
        safety_margin_seconds=10,
        output_dir=tmp_path,
        monotonic=monotonic,
        wall_clock=lambda: datetime(2026, 8, 24, 0, 0, tzinfo=timezone.utc),
    )
    watchdog.register_checkpoint("model", lambda: calls.append("model") or "saved.pt")

    def broken_flush():
        calls.append("flush")
        raise RuntimeError("disk fixture failure")

    watchdog.register_flush("metrics", broken_flush)
    current["mono"] = 89.99
    heartbeat = watchdog.pulse()
    assert heartbeat["status"] == "running"
    current["mono"] = 90.0
    with pytest.raises(BudgetExpired) as expired:
        watchdog.pulse()
    terminal = expired.value.terminal
    assert calls == ["model", "flush"]
    assert terminal["status"] == "budget_exhausted"
    assert terminal["checkpoint_results"][0]["ok"] is True
    assert terminal["callback_errors"][0]["name"] == "metrics"
    assert watchdog.shutdown() is terminal
    assert calls == ["model", "flush"]

    heartbeat_file = json.loads(watchdog.heartbeat_path.read_text(encoding="utf-8"))
    terminal_file = json.loads(watchdog.terminal_path.read_text(encoding="utf-8"))
    for payload in (heartbeat_file, terminal_file):
        assert payload["run_id"] == "run-a"
        assert "declared_deadline" in payload
        assert "elapsed_seconds" in payload and "remaining_seconds" in payload
        assert "checkpoint_results" in payload and "callback_errors" in payload


def test_watchdog_rejects_invalid_budget_and_duplicate_callbacks(tmp_path):
    with pytest.raises(ValueError):
        BudgetWatchdog("run-a", 10, tmp_path, safety_margin_seconds=10)
    watchdog = BudgetWatchdog("run-a", 10, tmp_path, safety_margin_seconds=1)
    watchdog.register_checkpoint("model", lambda: None)
    with pytest.raises(ValueError, match="duplicate"):
        watchdog.register_checkpoint("model", lambda: None)
