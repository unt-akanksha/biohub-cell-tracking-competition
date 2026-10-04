from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRAIN = runpy.run_path(
    str(ROOT / "research/temporal_contrastive/train_graph_context_fresh_ensemble_v3.py"),
    run_name="graph_context_unanimous_salvage_v4_training",
)
V3_VERIFY = runpy.run_path(
    str(ROOT / "scripts/verify-graph-context-fresh-training-v3.py"),
    run_name="graph_context_unanimous_salvage_v4_verifier",
)
RUN_ID = "competition-graph-context-unanimous-salvage-v4"
V3_TERMINAL_SHA256 = (
    "92f1150d5579cdbec41c92e3a04525a67d284975d911e42fb1ed29c54c8cf5dd"
)
CROSS_FAMILY_VALIDATION_STEMS = (
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def freeze(output_root: Path, split_path: Path, report_path: Path) -> dict[str, Any]:
    verification = V3_VERIFY["verify"](output_root)
    if not (
        verification.get("status") == "verified_rejection"
        and verification.get("reason") == "fresh_audit_gate_failed"
        and verification.get("terminal_sha256") == V3_TERMINAL_SHA256
        and verification.get("completed_model_count") == 4
    ):
        raise ValueError("Unexpected v3 source run for unanimous salvage")
    _, roles = TRAIN["load_contract"](split_path)
    overlap = sorted(set(CROSS_FAMILY_VALIDATION_STEMS) & set(roles))
    if overlap:
        raise ValueError(f"Cross-family validation overlaps graph training: {overlap}")
    terminals = TRAIN["load_member_terminals"](output_root)
    members = []
    for terminal in terminals:
        threshold = terminal.get("selection_frozen_threshold")
        if not (
            terminal.get("selection_gate_passed") is True
            and threshold
            and threshold.get("fp") == 0
            and threshold.get("tp", 0) >= 2
            and threshold.get("precision") == 1.0
        ):
            raise ValueError(f"Member lacks a pre-audit precision gate: {terminal['member']}")
        members.append(
            {
                "member": terminal["member"],
                "model_sha256": terminal["model_sha256"],
                "parameter_count": terminal["parameter_count"],
                "raw_logit_threshold": threshold["threshold"],
                "selection_threshold_evidence": threshold,
            }
        )
    report = {
        "schema_version": 1,
        "status": "frozen_before_cross_family_movie_evaluation",
        "run_id": RUN_ID,
        "source_v3_terminal_sha256": V3_TERMINAL_SHA256,
        "source_v3_status": "verified_rejection",
        "retired_v3_policy": "graph-context/morphology top-rank agreement",
        "deployment_policy": (
            "all-four raw logits meet their own pre-audit zero-FP thresholds; "
            "all-four models name the same top parent; morphology names that parent"
        ),
        "members": members,
        "minimum_member_agreement": 4,
        "morphology_top_parent_agreement_required": True,
        "maximum_added_edges_per_movie": 1,
        "biological_geometry_minimum": 3.0,
        "cross_family_validation_stems": list(CROSS_FAMILY_VALIDATION_STEMS),
        "cross_family_stems_seen_by_graph_training": False,
        "audit_scores_used_to_set_member_thresholds": False,
        "leaderboard_used_for_policy_selection": False,
        "metric_hack_used": False,
        "competition_test_data_read": False,
        "public_predictions_copied": False,
        "authorized_for_submission": False,
    }
    atomic_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.output_root, args.split, args.report), indent=2))


if __name__ == "__main__":
    main()
