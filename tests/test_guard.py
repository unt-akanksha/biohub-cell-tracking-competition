from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from biohub_tracker.cli import main
from biohub_tracker.guard import (
    ActiveKernel,
    GuardInputError,
    QuotaSnapshot,
    evaluate_guard,
    hours_text,
    parse_hours,
    read_gpu_quota,
)
from biohub_tracker.kaggle import FixtureRunner
from biohub_tracker.ledger import EventType, ExperimentEvent, Ledger, registration_payload


def quota(remaining: str = "30.00") -> QuotaSnapshot:
    return QuotaSnapshot(
        used=Decimal("0.00"),
        remaining=Decimal(remaining),
        total=Decimal("30.00"),
        refresh_at="2026-08-29T00:00:00Z",
        captured_at="2026-08-24T00:00:00Z",
    )


def decision(runtime: str, **kwargs):
    return evaluate_guard(
        run_id="run-a",
        registered_status="registered",
        declared_max_runtime=runtime,
        quota=quota(),
        active_kernels=[],
        reserve="8.00",
        notebook_maximum="12.00" if Decimal(runtime) <= 12 else "24.00",
        **kwargs,
    )


def test_quota_boundary_uses_decimal_and_two_places():
    exact = evaluate_guard(
        run_id="run-a",
        registered_status="registered",
        declared_max_runtime="22.00",
        quota=quota(),
        active_kernels=[],
        reserve="8.00",
        notebook_maximum="22.00",
    )
    below = evaluate_guard(
        run_id="run-a",
        registered_status="registered",
        declared_max_runtime="22.01",
        quota=quota(),
        active_kernels=[],
        reserve="8.00",
        notebook_maximum="22.01",
    )
    assert exact.authorized and exact.to_dict()["projected_remaining_hours"] == "8.00"
    assert not below.authorized and below.reason_codes == ("GPU_RESERVE_VIOLATION",)
    assert hours_text(parse_hours("30.00h")) == "30.00"


@pytest.mark.parametrize(
    ("status", "runtime", "active", "code"),
    [
        (None, "1", [], "RUN_NOT_REGISTERED"),
        ("registered", "0", [], "RUNTIME_NOT_POSITIVE"),
        ("registered", "12.01", [], "RUNTIME_ABOVE_NOTEBOOK_LIMIT"),
        ("registered", "1", [ActiveKernel("owner/job", "RUNNING")], "ACTIVE_GPU_KERNEL"),
    ],
)
def test_guard_distinct_rejections(status, runtime, active, code):
    result = evaluate_guard(
        run_id="run-a",
        registered_status=status,
        declared_max_runtime=runtime,
        quota=quota(),
        active_kernels=active,
    )
    assert not result.authorized and code in result.reason_codes


def test_quota_reader_rejects_missing_and_malformed_gpu_rows(tmp_path):
    fixture = tmp_path / "fixtures"
    fixture.mkdir()
    (fixture / "quota.json").write_text("[]", encoding="utf-8")
    with pytest.raises(GuardInputError, match="exactly one GPU") as missing:
        read_gpu_quota(FixtureRunner(fixture))
    assert missing.value.reason_code == "GPU_QUOTA_ROW_AMBIGUOUS"
    (fixture / "quota.json").write_text(
        json.dumps([{"resource": "GPU", "used": "0h", "remaining": "?", "total": "30h", "refreshAt": "2026-08-29T00:00:00"}]),
        encoding="utf-8",
    )
    with pytest.raises(GuardInputError) as malformed:
        read_gpu_quota(FixtureRunner(fixture))
    assert malformed.value.reason_code == "GPU_QUOTA_MALFORMED"


def test_guard_cli_is_read_only_and_appends_decision(tmp_path, capsys):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "competition.json").write_text(
        json.dumps({
            "slug": "biohub-cell-tracking-during-development",
            "gpu_reserve_hours": "8.00",
            "notebook_runtime_limit_hours": "12.00",
        }),
        encoding="utf-8",
    )
    fixture = tmp_path / "fixtures"
    fixture.mkdir()
    (fixture / "quota.json").write_text(json.dumps([{
        "resource": "GPU", "used": "0.00h", "remaining": "30.00h", "total": "30.00h", "refreshAt": "2026-08-29T00:00:00"
    }]), encoding="utf-8")
    (fixture / "kernels.json").write_text("[]", encoding="utf-8")
    ledger = Ledger(tmp_path / "experiments" / "events.jsonl", tmp_path)
    payload = registration_payload(
        hypothesis="guard test", parent=None, config={}, seeds=[1], split="test",
        declared_max_runtime_hours="2", code={"git_head": "abc"},
    )
    ledger.append(ExperimentEvent.create("run-a", EventType.REGISTERED, payload))
    assert main([
        "--root", str(tmp_path), "guard", "--run-id", "run-a",
        "--max-runtime-hours", "2", "--fixture-dir", str(fixture), "--json",
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["authorized"] is True
    events = ledger.read_events()
    assert [event.event_type for event in events] == [EventType.REGISTERED, EventType.GUARD_DECISION]
