#!/usr/bin/env python
"""Verify the clean 0.948-TTA2 plus LSM coordinate-consensus output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


RUN_ID = "948tta2-lsm-consensus-v1"
SOURCE_NOTEBOOK_SHA256 = (
    "3395f8df72c6d63d243fdb4fede1f1febdd36bfc086b2f0663fec3ccc9dbb189"
)
FEATURE24_MODEL_SHA256 = (
    "1a87e6ed6322e0cac98d9c92f7aade2a07bc5dbd4b322c5947f17227f6e256d1"
)
FEATURE36_MODEL_SHA256 = (
    "adedfaf056ee6e53922df24dd01c5f4cf27d20f313771a37275f335c48093421"
)
EXPECTED_HEADER = (
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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def assert_close(label: str, observed: Any, expected: float) -> None:
    try:
        value = float(observed)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid numeric evidence for {label}") from error
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"Evidence mismatch for {label}: {value} != {expected}")


def aggregate(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("Validator arm is empty")
    stems = [row["stem"] for row in rows]
    if len(stems) != len(set(stems)):
        raise RuntimeError("Validator arm contains duplicate movies")
    weight = sum(float(row["weight"]) for row in rows)
    if not math.isfinite(weight) or weight <= 0:
        raise RuntimeError("Validator arm has invalid weight")
    adjusted = sum(
        float(row["adjusted_edge_jaccard"]) * float(row["weight"])
        for row in rows
    ) / weight
    div_tp = sum(int(row["div_tp"]) for row in rows)
    div_fp = sum(int(row["div_fp"]) for row in rows)
    div_fn = sum(int(row["div_fn"]) for row in rows)
    denominator = div_tp + div_fp + div_fn
    division = div_tp / denominator if denominator else 0.0
    return {
        "stems": sorted(stems),
        "adjusted_edge_jaccard": adjusted,
        "division_jaccard": division,
        "proxy_score": adjusted + 0.10 * division,
        "div_tp": div_tp,
        "div_fp": div_fp,
        "div_fn": div_fn,
    }


def validate_submission(path: Path) -> dict[str, Any]:
    nodes: dict[str, dict[int, tuple[int, int, int, int]]] = {}
    edges: dict[str, list[tuple[int, int]]] = {}
    row_count = 0
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        try:
            header = tuple(next(reader))
        except StopIteration as error:
            raise RuntimeError("Submission is empty") from error
        if header != EXPECTED_HEADER:
            raise RuntimeError(f"Submission header changed: {header}")
        for expected_id, row in enumerate(reader):
            if len(row) != len(EXPECTED_HEADER) or int(row[0]) != expected_id:
                raise RuntimeError(f"Submission row contract failed at {expected_id}")
            dataset, row_type = row[1], row[2]
            if not dataset or row_type not in {"node", "edge"}:
                raise RuntimeError(f"Invalid submission row at {expected_id}")
            if row_type == "node":
                node_id, timepoint = int(row[3]), int(row[4])
                coordinate = (int(row[5]), int(row[6]), int(row[7]))
                if (
                    node_id < 0
                    or timepoint < 0
                    or min(coordinate) < 0
                    or int(row[8]) != -1
                    or int(row[9]) != -1
                    or node_id in nodes.setdefault(dataset, {})
                ):
                    raise RuntimeError(f"Invalid node row at {expected_id}")
                nodes[dataset][node_id] = (timepoint, *coordinate)
            else:
                if any(int(row[index]) != -1 for index in range(3, 8)):
                    raise RuntimeError(f"Invalid edge sentinels at {expected_id}")
                edges.setdefault(dataset, []).append((int(row[8]), int(row[9])))
            row_count += 1
    if not row_count or set(nodes) != set(edges):
        raise RuntimeError("Submission dataset coverage is incomplete")
    for dataset, pairs in edges.items():
        indegree: dict[int, int] = {}
        outdegree: dict[int, int] = {}
        if len(pairs) != len(set(pairs)):
            raise RuntimeError(f"Duplicate edges in {dataset}")
        for source, target in pairs:
            if source not in nodes[dataset] or target not in nodes[dataset]:
                raise RuntimeError(f"Dangling edge in {dataset}: {source}->{target}")
            if nodes[dataset][target][0] != nodes[dataset][source][0] + 1:
                raise RuntimeError(f"Nonconsecutive edge in {dataset}: {source}->{target}")
            outdegree[source] = outdegree.get(source, 0) + 1
            indegree[target] = indegree.get(target, 0) + 1
        if max(outdegree.values(), default=0) > 2 or max(indegree.values(), default=0) > 1:
            raise RuntimeError(f"Invalid lineage degree in {dataset}")
    return {
        "row_count": row_count,
        "datasets": sorted(nodes),
        "node_count": sum(len(rows) for rows in nodes.values()),
        "edge_count": sum(len(rows) for rows in edges.values()),
    }


def verify_candidate(output_root: Path) -> dict[str, Any]:
    terminal_path = unique_file(output_root, "launcher_terminal.json")
    evidence_path = unique_file(output_root, "candidate_evidence.json")
    submission_path = unique_file(output_root, "submission.csv")
    run_stats_path = unique_file(output_root, "run_stats.csv")
    validator_path = unique_file(output_root, "validator_results.csv")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    submission_sha256 = sha256_file(submission_path)

    if not (
        terminal.get("schema_version") == 1
        and terminal.get("run_id") == RUN_ID
        and terminal.get("status") == "completed"
        and terminal.get("metric_hack_used") is False
        and terminal.get("public_predictions_copied") is False
        and terminal.get("competition_submission_performed") is False
        and terminal.get("submission_exists") is True
        and terminal.get("evidence_exists") is True
        and terminal.get("submission_sha256") == submission_sha256
        and terminal.get("evidence_sha256") == sha256_file(evidence_path)
        and float(terminal.get("elapsed_seconds", math.inf)) < 42000.0
    ):
        raise RuntimeError("Candidate launcher terminal is invalid")
    if not (
        evidence.get("schema_version") == 1
        and evidence.get("status") == "eligible_for_submission"
        and evidence.get("run_id") == RUN_ID
        and evidence.get("source_notebook") == "redoctopusk/biohub-948tta2"
        and evidence.get("source_notebook_sha256") == SOURCE_NOTEBOOK_SHA256
        and evidence.get("public_lineage_attributed") is True
        and evidence.get("public_predictions_copied") is False
        and evidence.get("exact_public_replica") is False
        and evidence.get("metric_hack_used") is False
        and evidence.get("estimated_node_count_used_by_candidate") is False
        and evidence.get("coordinate_policy")
        == "two_lsm_exact_final_integer_agreement_r2_p2_b025"
        and evidence.get("feature24_model_sha256") == FEATURE24_MODEL_SHA256
        and evidence.get("feature36_model_sha256") == FEATURE36_MODEL_SHA256
        and int(evidence.get("production_coordinate_changes", 0)) > 0
        and evidence.get("integrity_passed") is True
        and evidence.get("competition_submission_performed") is False
        and evidence.get("authorized_for_submission") is True
        and evidence.get("submission_sha256") == submission_sha256
    ):
        raise RuntimeError("Candidate evidence did not pass the clean promotion gate")
    if submission_sha256 in KNOWN_PUBLIC_SUBMISSION_SHA256:
        raise RuntimeError("Candidate is byte-identical to a known public submission")

    submission = validate_submission(submission_path)
    run_rows = read_rows(run_stats_path)
    required_stats = {
        "lsm_consensus_nodes_seen",
        "lsm_consensus_nodes_agreed",
        "lsm_consensus_nodes_moved",
        "lsm_consensus_nodes_disagreed",
        "lsm_consensus_node_count_changes",
        "lsm_consensus_time_or_id_changes",
        "lsm_consensus_edge_changes",
        "lsm_consensus_out_of_bounds",
    }
    if not run_rows or not required_stats <= set(run_rows[0]):
        raise RuntimeError("Production consensus statistics are incomplete")
    moves = sum(int(float(row["lsm_consensus_nodes_moved"])) for row in run_rows)
    integrity = sum(
        abs(int(float(row[name])))
        for row in run_rows
        for name in (
            "lsm_consensus_node_count_changes",
            "lsm_consensus_time_or_id_changes",
            "lsm_consensus_edge_changes",
            "lsm_consensus_out_of_bounds",
        )
    )
    if moves <= 0 or moves != int(evidence["production_coordinate_changes"]) or integrity:
        raise RuntimeError("Production coordinate-consensus integrity gate failed")

    validator_rows = read_rows(validator_path)
    required_validator = {
        "stem",
        "arm",
        "weight",
        "adjusted_edge_jaccard",
        "div_tp",
        "div_fp",
        "div_fn",
    }
    if not validator_rows or not required_validator <= set(validator_rows[0]):
        raise RuntimeError("Validator comparison columns are incomplete")
    by_arm = {
        arm: [row for row in validator_rows if row["arm"] == arm]
        for arm in ("control", "lsm_consensus")
    }
    if any(len(rows) < 4 for rows in by_arm.values()):
        raise RuntimeError("Complete-movie validator coverage is insufficient")
    control = aggregate(by_arm["control"])
    candidate = aggregate(by_arm["lsm_consensus"])
    if control["stems"] != candidate["stems"]:
        raise RuntimeError("Candidate/control movie sets differ")
    for label, observed, expected in (
        ("control adjusted", evidence["control"]["adjusted_edge_jaccard"], control["adjusted_edge_jaccard"]),
        ("control division", evidence["control"]["division_jaccard"], control["division_jaccard"]),
        ("control proxy", evidence["control"]["proxy_score"], control["proxy_score"]),
        ("candidate adjusted", evidence["candidate"]["adjusted_edge_jaccard"], candidate["adjusted_edge_jaccard"]),
        ("candidate division", evidence["candidate"]["division_jaccard"], candidate["division_jaccard"]),
        ("candidate proxy", evidence["candidate"]["proxy_score"], candidate["proxy_score"]),
    ):
        assert_close(label, observed, expected)
    control_movie = {row["stem"]: float(row["adjusted_edge_jaccard"]) for row in by_arm["control"]}
    candidate_movie = {row["stem"]: float(row["adjusted_edge_jaccard"]) for row in by_arm["lsm_consensus"]}
    deltas = {stem: candidate_movie[stem] - control_movie[stem] for stem in sorted(control_movie)}
    proxy_gain = candidate["proxy_score"] - control["proxy_score"]
    if not (
        proxy_gain > 0.0
        and min(deltas.values()) >= 0.0
        and candidate["division_jaccard"] >= control["division_jaccard"]
    ):
        raise RuntimeError("External complete-movie promotion gate failed")
    assert_close("proxy delta", evidence["proxy_score_delta"], proxy_gain)
    assert_close("minimum movie delta", evidence["minimum_movie_adjusted_edge_delta"], min(deltas.values()))

    return {
        "schema_version": 1,
        "status": "eligible_for_submission",
        "run_id": RUN_ID,
        "target_public_score": 0.945,
        "submission": submission,
        "submission_path": str(submission_path.resolve()),
        "submission_sha256": submission_sha256,
        "candidate_evidence_sha256": sha256_file(evidence_path),
        "launcher_terminal_sha256": sha256_file(terminal_path),
        "production_coordinate_changes": moves,
        "control_validator": control,
        "candidate_validator": candidate,
        "proxy_gain": proxy_gain,
        "minimum_movie_adjusted_edge_delta": min(deltas.values()),
        "known_public_hash_match": False,
        "metric_hack_used": False,
        "competition_submission_performed": False,
        "authorized_for_submission": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify_candidate(args.output_root)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(args.report.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.report)
    print(rendered, end="")


if __name__ == "__main__":
    main()
