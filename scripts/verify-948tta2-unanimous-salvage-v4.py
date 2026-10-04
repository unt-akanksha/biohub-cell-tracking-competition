"""Independently verify the paired unanimous-salvage Kaggle output."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


RUN_ID = "948tta2-unanimous-salvage-v4"
EXPECTED_STEMS = {
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
}
EXPECTED_MANIFEST_SHA256 = (
    "f211ef7fe6b328607ee476df3ee1cf424f1c108d07f7968022cae87ce01e690e"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name}, saw {matches}")
    return matches[0]


def close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def aggregate(rows: pd.DataFrame) -> dict[str, float | int]:
    weight = float(rows["weight"].sum())
    if weight <= 0:
        raise ValueError("validator arm has no positive edge weight")
    adjusted = float((rows["adjusted_edge_jaccard"] * rows["weight"]).sum() / weight)
    tp = int(rows["div_tp"].sum())
    fp = int(rows["div_fp"].sum())
    fn = int(rows["div_fn"].sum())
    division = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "adjusted_edge_jaccard": adjusted,
        "division_jaccard": division,
        "proxy_score": adjusted + division * (1.0 / 10.0),
        "div_tp": tp,
        "div_fp": fp,
        "div_fn": fn,
    }


def verify(root: Path) -> dict[str, Any]:
    evidence_path = unique(root, "candidate_evidence.json")
    terminal_path = unique(root, "launcher_terminal.json")
    stats_path = unique(root, "run_stats.csv")
    validator_path = unique(root, "validator_results.csv")
    submission_path = unique(root, "submission.csv")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    validator = pd.read_csv(validator_path)
    stats = pd.read_csv(stats_path)

    if not (
        evidence.get("schema_version") == 1
        and evidence.get("run_id") == RUN_ID
        and evidence.get("source_notebook") == "redoctopusk/biohub-948tta2"
        and evidence.get("public_predictions_copied") is False
        and evidence.get("exact_public_replica") is False
        and evidence.get("metric_hack_used") is False
        and evidence.get("leaderboard_used_for_candidate_selection") is False
        and evidence.get("unanimous_salvage_manifest_sha256") == EXPECTED_MANIFEST_SHA256
        and evidence.get("graph_member_count") == 4
        and evidence.get("graph_parameter_count_per_member") == 74_732_308
        and len(evidence.get("graph_model_sha256", [])) == 4
        and len(set(evidence.get("graph_model_sha256", []))) == 4
        and evidence.get("validation_stems_seen_by_graph_training") is False
        and evidence.get("competition_submission_performed") is False
        and terminal.get("run_id") == RUN_ID
        and terminal.get("status") == "completed"
        and terminal.get("competition_submission_performed") is False
        and terminal.get("evidence_sha256") == sha256_file(evidence_path)
        and terminal.get("submission_sha256") == sha256_file(submission_path)
        and evidence.get("submission_sha256") == sha256_file(submission_path)
    ):
        raise ValueError("terminal, provenance, or immutable artifact gate failed")

    required_columns = {
        "arm", "stem", "weight", "adjusted_edge_jaccard",
        "div_tp", "div_fp", "div_fn", "unanimous_salvage_edges_added",
    }
    if not required_columns <= set(validator.columns):
        raise ValueError("validator result schema changed")
    if set(validator["arm"]) != {"control", "unanimous_salvage"}:
        raise ValueError("paired validator arms changed")
    if set(validator["stem"]) != EXPECTED_STEMS or len(validator) != 8:
        raise ValueError("frozen complete-movie coverage changed")
    by_arm = {
        arm: validator[validator["arm"] == arm].copy()
        for arm in ("control", "unanimous_salvage")
    }
    calculated = {arm: aggregate(rows) for arm, rows in by_arm.items()}
    for arm, evidence_name in (("control", "control"), ("unanimous_salvage", "candidate")):
        recorded = evidence[evidence_name]
        for key, value in calculated[arm].items():
            if isinstance(value, float):
                if not close(recorded[key], value):
                    raise ValueError(f"{arm} aggregate changed: {key}")
            elif recorded[key] != value:
                raise ValueError(f"{arm} aggregate changed: {key}")

    control = by_arm["control"].set_index("stem")
    candidate = by_arm["unanimous_salvage"].set_index("stem")
    deltas = {
        stem: float(candidate.loc[stem, "adjusted_edge_jaccard"] - control.loc[stem, "adjusted_edge_jaccard"])
        for stem in sorted(EXPECTED_STEMS)
    }
    production_edges = int(stats["ranked_consensus_added_edges"].sum())
    integrity = bool(
        int(stats["ranked_consensus_reassignment_performed"].sum()) == 0
        and int(stats["ranked_consensus_node_or_coordinate_changes"].sum()) == 0
    )
    eligible = bool(
        calculated["unanimous_salvage"]["proxy_score"] > calculated["control"]["proxy_score"]
        and min(deltas.values()) >= 0.0
        and calculated["unanimous_salvage"]["division_jaccard"]
            > calculated["control"]["division_jaccard"]
        and production_edges > 0
        and integrity
    )
    if not (
        evidence.get("production_edges_added") == production_edges
        and evidence.get("integrity_passed") is integrity
        and evidence.get("authorized_for_submission") is eligible
        and evidence.get("status") == (
            "eligible_for_submission" if eligible else "rejected_at_complete_movie_gate"
        )
        and close(evidence.get("proxy_score_delta"),
                  calculated["unanimous_salvage"]["proxy_score"] - calculated["control"]["proxy_score"])
        and close(evidence.get("minimum_movie_adjusted_edge_delta"), min(deltas.values()))
        and evidence.get("per_movie_adjusted_edge_delta") == deltas
    ):
        raise ValueError("recomputed promotion decision differs from notebook evidence")
    return {
        "schema_version": 1,
        "status": "verified_eligible" if eligible else "verified_rejection",
        "run_id": RUN_ID,
        "candidate_evidence_sha256": sha256_file(evidence_path),
        "launcher_terminal_sha256": sha256_file(terminal_path),
        "submission_sha256": sha256_file(submission_path),
        "control": calculated["control"],
        "candidate": calculated["unanimous_salvage"],
        "proxy_score_delta": evidence["proxy_score_delta"],
        "minimum_movie_adjusted_edge_delta": min(deltas.values()),
        "production_edges_added": production_edges,
        "authorized_for_submission": eligible,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify(args.output_root)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
