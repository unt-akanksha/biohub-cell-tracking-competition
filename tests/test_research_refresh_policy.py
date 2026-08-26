from __future__ import annotations

import json
from pathlib import Path


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

