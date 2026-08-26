from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from biohub_tracker.launch import LaunchError, validate_research_refresh


ROOT = Path(__file__).resolve().parents[1]


def test_research_refresh_is_regular_source_based_and_leaderboard_blind() -> None:
    config = json.loads((ROOT / "config" / "competition.json").read_text())
    policy = config["research_refresh_policy"]

    assert 1 <= policy["max_audit_age_hours"] <= 24
    assert policy["notebook_scan_limit"] >= 50
    assert policy["audit_before_new_model_family"] is True
    assert policy["audit_before_submission_candidate"] is True
    assert policy["require_source_review"] is True
    assert policy["exclude_metric_hacks"] is True
    assert policy["leaderboard_selection_forbidden"] is True
    assert policy["running_kernel_status_checks_do_not_count_as_research"] is True


def test_launch_research_guard_rejects_stale_audit(tmp_path: Path) -> None:
    (tmp_path / "policies").mkdir()
    (tmp_path / "policies" / "notebook_audits.json").write_text(
        json.dumps({"audited_at": "2026-08-24T00:00:00Z"}), encoding="utf-8"
    )
    config = {"research_refresh_policy": {"max_audit_age_hours": 24}}

    with pytest.raises(LaunchError) as error:
        validate_research_refresh(
            tmp_path,
            config,
            now=datetime(2026, 8, 26, tzinfo=timezone.utc),
        )

    assert error.value.reason_code == "RESEARCH_AUDIT_STALE"


def test_launch_research_guard_accepts_recent_source_audit(tmp_path: Path) -> None:
    (tmp_path / "policies").mkdir()
    (tmp_path / "policies" / "notebook_audits.json").write_text(
        json.dumps({"audited_at": "2026-08-25T12:00:00Z"}), encoding="utf-8"
    )
    validate_research_refresh(
        tmp_path,
        {"research_refresh_policy": {"max_audit_age_hours": 24}},
        now=datetime(2026, 8, 26, tzinfo=timezone.utc),
    )
