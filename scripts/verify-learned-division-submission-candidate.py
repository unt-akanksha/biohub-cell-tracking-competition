#!/usr/bin/env python
"""Verify and promote an EMA plus learned-division Kaggle kernel output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


RUN_ID = "ema-learned-division-candidate-v1"
EXPECTED_SUBMISSION_HEADER = (
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
)
KNOWN_PUBLIC_SUBMISSION_SHA256 = {
    "e7ee404d499a06aaa5bd5a3f62f291687743685ea5154391487e476dff0e4905",
    "d49f0cfb22eb37ee49ab04a62fb8c79c6800fa013089d84fe6fa366517caaad1",
    "61b57dddab663c68ad19ca8105165b852313ee4ee8487bb4bbc424d2ac634331",
}
MINIMUM_PROXY_GAIN = 0.005
MAXIMUM_ADJUSTED_EDGE_REGRESSION = 0.001
PUBLIC_CONTROL_VALIDATOR_SHA256 = (
    "4dbf2079c1efc1108370f33104a6e80882a851d45dbfaa0540d40e88e4941f4b"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_file(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob(name) if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {name}, saw {matches}")
    return matches[0]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def aggregate_validator(path: Path) -> dict[str, Any]:
    rows = read_csv_rows(path)
    if not rows:
        raise RuntimeError("validator results are empty")
    required = {
        "stem",
        "weight",
        "adjusted_edge_jaccard",
        "div_tp",
        "div_fp",
        "div_fn",
    }
    if not required <= set(rows[0]):
        raise RuntimeError("validator result columns changed")
    stems = [row["stem"] for row in rows]
    if len(stems) != len(set(stems)):
        raise RuntimeError("validator contains duplicate stems")
    weight = sum(float(row["weight"]) for row in rows)
    if not math.isfinite(weight) or weight <= 0:
        raise RuntimeError("validator weight is invalid")
    adjusted_edge = sum(
        float(row["adjusted_edge_jaccard"]) * float(row["weight"])
        for row in rows
    ) / weight
    division_tp = sum(int(row["div_tp"]) for row in rows)
    division_fp = sum(int(row["div_fp"]) for row in rows)
    division_fn = sum(int(row["div_fn"]) for row in rows)
    denominator = division_tp + division_fp + division_fn
    division = division_tp / denominator if denominator else 0.0
    result = {
        "stems": sorted(stems),
        "weighted_adjusted_edge_jaccard": adjusted_edge,
        "division_tp": division_tp,
        "division_fp": division_fp,
        "division_fn": division_fn,
        "division_jaccard": division,
        "proxy_score": adjusted_edge + 0.10 * division,
    }
    if not all(
        math.isfinite(float(result[key]))
        for key in (
            "weighted_adjusted_edge_jaccard",
            "division_jaccard",
            "proxy_score",
        )
    ):
        raise RuntimeError("validator aggregate is not finite")
    return result


def validate_submission_csv(path: Path) -> dict[str, Any]:
    row_count = 0
    datasets: set[str] = set()
    row_types: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        try:
            header = tuple(next(reader))
        except StopIteration as error:
            raise RuntimeError("submission is empty") from error
        if header != EXPECTED_SUBMISSION_HEADER:
            raise RuntimeError(f"submission header changed: {header}")
        for expected_id, row in enumerate(reader):
            if len(row) != len(EXPECTED_SUBMISSION_HEADER):
                raise RuntimeError(f"submission row width changed at {expected_id}")
            if int(row[0]) != expected_id:
                raise RuntimeError(f"submission id sequence changed at {expected_id}")
            if not row[1]:
                raise RuntimeError(f"submission dataset is empty at {expected_id}")
            if row[2] not in {"node", "edge"}:
                raise RuntimeError(f"submission row type changed at {expected_id}")
            datasets.add(row[1])
            row_types.add(row[2])
            row_count += 1
    if row_count == 0 or row_types != {"node", "edge"}:
        raise RuntimeError("submission must contain node and edge rows")
    return {"row_count": row_count, "datasets": sorted(datasets)}


def _sum_int(rows: list[dict[str, str]], name: str) -> int:
    try:
        return sum(int(float(row[name])) for row in rows)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"run stats column is invalid: {name}") from error


def _assert_close(label: str, observed: Any, expected: float) -> None:
    try:
        value = float(observed)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"candidate evidence {label} is invalid") from error
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"candidate evidence {label} mismatch: {value} != {expected}"
        )


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
        and evidence.get("external_training_only") is True
        and evidence.get("safe_division_heuristic_disabled") is True
        and evidence.get("competition_submission_performed") is False
        and evidence.get("authorized_for_submission") is False
        and evidence.get("submission_sha256") == submission_sha256
    ):
        raise RuntimeError("candidate evidence is invalid")
    if submission_sha256 in KNOWN_PUBLIC_SUBMISSION_SHA256:
        raise RuntimeError("candidate submission is identical to an audited public output")

    submission = validate_submission_csv(submission_path)
    run_rows = read_csv_rows(run_stats_path)
    if not run_rows:
        raise RuntimeError("candidate run stats are empty")
    learned_candidates = _sum_int(
        run_rows, "learned_division_geometric_candidates"
    )
    learned_scored = _sum_int(
        run_rows, "learned_division_candidate_parents_scored"
    )
    learned_added = _sum_int(run_rows, "learned_division_added_edges")
    learned_reassignments = _sum_int(
        run_rows, "learned_division_reassignment_performed"
    )
    learned_node_changes = _sum_int(
        run_rows, "learned_division_node_or_coordinate_changes"
    )
    maximum_additions = _sum_int(run_rows, "learned_division_maximum_additions")
    if not (
        learned_candidates > 0
        and learned_scored == learned_candidates
        and 0 < learned_added <= maximum_additions
        and learned_reassignments == 0
        and learned_node_changes == 0
    ):
        raise RuntimeError("learned division recovery integrity gate failed")
    for label, observed, expected in (
        (
            "learned_geometric_candidates",
            evidence.get("learned_geometric_candidates"),
            learned_candidates,
        ),
        (
            "learned_candidate_parents_scored",
            evidence.get("learned_candidate_parents_scored"),
            learned_scored,
        ),
        ("learned_edges_added", evidence.get("learned_edges_added"), learned_added),
        (
            "learned_reassignments",
            evidence.get("learned_reassignments"),
            learned_reassignments,
        ),
        (
            "learned_node_or_coordinate_changes",
            evidence.get("learned_node_or_coordinate_changes"),
            learned_node_changes,
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
        "learned_edges_added": learned_added,
        "learned_reassignments": learned_reassignments,
        "learned_node_or_coordinate_changes": learned_node_changes,
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
