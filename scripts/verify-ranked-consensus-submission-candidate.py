#!/usr/bin/env python
"""Verify and promote an EMA plus ranked-consensus Kaggle output."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import runpy
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
_COMMON = runpy.run_path(
    str(ROOT / "scripts/verify-learned-division-submission-candidate.py")
)
aggregate_validator = _COMMON["aggregate_validator"]
read_csv_rows = _COMMON["read_csv_rows"]
sha256_file = _COMMON["sha256_file"]
unique_file = _COMMON["unique_file"]
validate_submission_csv = _COMMON["validate_submission_csv"]
_assert_close = _COMMON["_assert_close"]

RUN_ID = "ema-ranked-consensus-candidate-v1"
EXPECTED_RUNTIME_MANIFEST_SHA256 = (
    "6e4803ba11cda000c0facf691f86ac2f1c98e9aa8fe9e4aacc720f0c44163aca"
)
EXPECTED_DEEP_PARAMETER_COUNT = 46_386_607
KNOWN_PUBLIC_SUBMISSION_SHA256 = _COMMON["KNOWN_PUBLIC_SUBMISSION_SHA256"]
MINIMUM_PROXY_GAIN = 0.005
MAXIMUM_ADJUSTED_EDGE_REGRESSION = 0.001
PUBLIC_CONTROL_VALIDATOR_SHA256 = _COMMON["PUBLIC_CONTROL_VALIDATOR_SHA256"]


def _sum_integral(rows: list[dict[str, str]], name: str) -> int:
    total = 0
    for row in rows:
        try:
            raw = row[name]
            if raw.lower() in {"true", "false"}:
                total += int(raw.lower() == "true")
            else:
                total += int(float(raw))
        except (KeyError, AttributeError, TypeError, ValueError) as error:
            raise RuntimeError(f"run stats column is invalid: {name}") from error
    return total


def verify_candidate(
    output_root: Path,
    baseline_validator: Path,
    *,
    expected_baseline_sha256: str | None = None,
) -> dict[str, Any]:
    baseline_sha256 = sha256_file(baseline_validator)
    if (
        expected_baseline_sha256 is not None
        and baseline_sha256 != expected_baseline_sha256
    ):
        raise RuntimeError("public-control validator artifact changed")
    terminal_path = unique_file(output_root, "watchdog-terminal.json")
    evidence_path = unique_file(output_root, "candidate_evidence.json")
    submission_path = unique_file(output_root, "submission.csv")
    run_stats_path = unique_file(output_root, "run_stats.csv")
    validator_path = unique_file(output_root, "validator_results.csv")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    submission_sha256 = sha256_file(submission_path)
    if not (
        terminal.get("run_id") == RUN_ID
        and terminal.get("status") == "completed"
        and terminal.get("submission_exists") is True
        and terminal.get("submission_sha256") == submission_sha256
        and float(terminal.get("elapsed_seconds", math.inf)) < 3000.0
    ):
        raise RuntimeError("candidate watchdog terminal is invalid")
    if not (
        evidence.get("schema_version") == 1
        and evidence.get("status") == "completed_pending_external_promotion_gate"
        and evidence.get("run_id") == RUN_ID
        and float(evidence.get("target_public_score", 0.0)) == 0.945
        and evidence.get("public_lineage_attributed") is True
        and evidence.get("public_predictions_copied") is False
        and evidence.get("project_division_components") == 2
        and evidence.get("deep_parameter_count") == EXPECTED_DEEP_PARAMETER_COUNT
        and evidence.get("absolute_threshold_used") is False
        and evidence.get("runtime_manifest_sha256")
        == EXPECTED_RUNTIME_MANIFEST_SHA256
        and evidence.get("safe_division_heuristic_disabled") is True
        and evidence.get("competition_submission_performed") is False
        and evidence.get("authorized_for_submission") is False
        and evidence.get("submission_sha256") == submission_sha256
        and isinstance(evidence.get("deep_model_sha256"), str)
        and len(evidence["deep_model_sha256"]) == 64
        and isinstance(evidence.get("morphology_model_sha256"), str)
        and len(evidence["morphology_model_sha256"]) == 64
    ):
        raise RuntimeError("candidate evidence is invalid")
    if submission_sha256 in KNOWN_PUBLIC_SUBMISSION_SHA256:
        raise RuntimeError("candidate submission is identical to an audited public output")

    submission = validate_submission_csv(submission_path)
    run_rows = read_csv_rows(run_stats_path)
    if not run_rows:
        raise RuntimeError("candidate run stats are empty")
    geometric = _sum_integral(run_rows, "ranked_consensus_geometric_candidates")
    eligible = _sum_integral(run_rows, "ranked_consensus_geometry_eligible_candidates")
    scored = _sum_integral(run_rows, "ranked_consensus_candidate_parents_scored")
    agreements = _sum_integral(run_rows, "ranked_consensus_ranking_agreed")
    added = _sum_integral(run_rows, "ranked_consensus_added_edges")
    maximum_additions = _sum_integral(run_rows, "ranked_consensus_maximum_additions")
    absolute_threshold = _sum_integral(
        run_rows, "ranked_consensus_absolute_threshold_used"
    )
    reassignments = _sum_integral(
        run_rows, "ranked_consensus_reassignment_performed"
    )
    node_changes = _sum_integral(
        run_rows, "ranked_consensus_node_or_coordinate_changes"
    )
    if not (
        geometric > 0
        and 0 < eligible <= geometric
        and scored == eligible
        and 0 < agreements == added <= maximum_additions
        and absolute_threshold == 0
        and reassignments == 0
        and node_changes == 0
    ):
        raise RuntimeError("ranked-consensus integrity gate failed")
    for label, observed, expected in (
        ("ranked_geometric_candidates", evidence.get("ranked_geometric_candidates"), geometric),
        (
            "ranked_geometry_eligible_candidates",
            evidence.get("ranked_geometry_eligible_candidates"),
            eligible,
        ),
        (
            "ranked_candidate_parents_scored",
            evidence.get("ranked_candidate_parents_scored"),
            scored,
        ),
        ("ranked_agreements", evidence.get("ranked_agreements"), agreements),
        ("ranked_edges_added", evidence.get("ranked_edges_added"), added),
        (
            "ranked_reassignments",
            evidence.get("ranked_reassignments"),
            reassignments,
        ),
        (
            "ranked_node_or_coordinate_changes",
            evidence.get("ranked_node_or_coordinate_changes"),
            node_changes,
        ),
    ):
        _assert_close(label, observed, float(expected))

    candidate_validator = aggregate_validator(validator_path)
    baseline = aggregate_validator(baseline_validator)
    if candidate_validator["stems"] != baseline["stems"]:
        raise RuntimeError("candidate and public-control validator stems differ")
    for label, evidence_key, aggregate_key in (
        (
            "validator_adjusted_edge_jaccard",
            "validator_adjusted_edge_jaccard",
            "weighted_adjusted_edge_jaccard",
        ),
        ("validator_division_tp", "validator_division_tp", "division_tp"),
        ("validator_division_fp", "validator_division_fp", "division_fp"),
        ("validator_division_fn", "validator_division_fn", "division_fn"),
        (
            "validator_division_jaccard",
            "validator_division_jaccard",
            "division_jaccard",
        ),
        ("validator_proxy_score", "validator_proxy_score", "proxy_score"),
    ):
        _assert_close(label, evidence.get(evidence_key), candidate_validator[aggregate_key])
    proxy_gain = candidate_validator["proxy_score"] - baseline["proxy_score"]
    adjusted_edge_delta = (
        candidate_validator["weighted_adjusted_edge_jaccard"]
        - baseline["weighted_adjusted_edge_jaccard"]
    )
    if not (
        proxy_gain >= MINIMUM_PROXY_GAIN
        and adjusted_edge_delta >= -MAXIMUM_ADJUSTED_EDGE_REGRESSION
        and candidate_validator["division_tp"] > baseline["division_tp"]
        and candidate_validator["division_jaccard"] > baseline["division_jaccard"]
    ):
        raise RuntimeError(
            "0.945 promotion gate failed: "
            f"proxy_gain={proxy_gain:.6f}, "
            f"adjusted_edge_delta={adjusted_edge_delta:.6f}, "
            f"division_tp={candidate_validator['division_tp']}, "
            f"division_jaccard={candidate_validator['division_jaccard']:.6f}"
        )
    return {
        "schema_version": 1,
        "status": "eligible_for_submission",
        "run_id": RUN_ID,
        "target_public_score": 0.945,
        "submission": submission,
        "submission_path": str(submission_path.resolve()),
        "submission_sha256": submission_sha256,
        "candidate_evidence_sha256": sha256_file(evidence_path),
        "watchdog_terminal_sha256": sha256_file(terminal_path),
        "ranked_edges_added": added,
        "ranked_agreements": agreements,
        "ranked_reassignments": reassignments,
        "ranked_node_or_coordinate_changes": node_changes,
        "public_control": baseline,
        "public_control_validator_sha256": baseline_sha256,
        "candidate_validator": candidate_validator,
        "proxy_gain": proxy_gain,
        "adjusted_edge_delta": adjusted_edge_delta,
        "known_public_hash_match": False,
        "competition_submission_performed": False,
        "authorized_for_submission": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-validator", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify_candidate(
        args.output_root,
        args.baseline_validator,
        expected_baseline_sha256=PUBLIC_CONTROL_VALIDATOR_SHA256,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(args.report.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.report)
    print(rendered, end="")


if __name__ == "__main__":
    main()
