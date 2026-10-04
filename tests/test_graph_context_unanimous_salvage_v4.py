from __future__ import annotations

import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/freeze-graph-context-unanimous-salvage-v4.py"
OUTPUT_ROOT = (
    ROOT
    / ".biohub/cache/kernel-outputs/"
    "biohub-graph-context-fresh-training-v3-version5-20260908/"
    "graph_context_fresh_ensemble_v3"
)


def test_frozen_policy_uses_only_pre_audit_member_thresholds(
    tmp_path: Path,
) -> None:
    values = runpy.run_path(str(SCRIPT), run_name="unanimous_salvage_test")
    report_path = tmp_path / "policy.json"
    report = values["freeze"](
        OUTPUT_ROOT,
        ROOT / "research/graph_context_fresh_split_v3.json",
        report_path,
    )

    assert json.loads(report_path.read_text()) == report
    assert report["status"] == "frozen_before_cross_family_movie_evaluation"
    assert len(report["members"]) == 4
    assert report["minimum_member_agreement"] == 4
    assert report["morphology_top_parent_agreement_required"] is True
    assert report["cross_family_stems_seen_by_graph_training"] is False
    assert report["audit_scores_used_to_set_member_thresholds"] is False
    assert report["leaderboard_used_for_policy_selection"] is False
    assert report["metric_hack_used"] is False
    assert report["authorized_for_submission"] is False
    for member in report["members"]:
        threshold = member["selection_threshold_evidence"]
        assert threshold["threshold"] == member["raw_logit_threshold"]
        assert threshold["fp"] == 0
        assert threshold["tp"] >= 2
        assert threshold["precision"] == 1.0
