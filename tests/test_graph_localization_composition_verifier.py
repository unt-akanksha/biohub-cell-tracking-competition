from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify-graph-localization-composition-candidate.py"
MODULE = runpy.run_path(str(SCRIPT))
BUILDER_FIXTURE = runpy.run_path(
    str(ROOT / "tests/test_graph_localization_composition_builder.py")
)


def _promotion(run_id: str, runtime_sha256: str) -> dict:
    return {
        "schema_version": 1,
        "status": "eligible_for_submission",
        "run_id": run_id,
        "target_public_score": 0.945,
        "runtime_manifest_sha256": runtime_sha256,
        "submission_sha256": "a" * 64,
        "candidate_validator": {"proxy_score": 0.95},
        "known_public_hash_match": False,
        "competition_submission_performed": False,
        "authorized_for_submission": True,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validator_rows(kind: str) -> list[dict]:
    values = {
        "baseline": (0.900, 5, 1, 2, (10, 10)),
        "graph": (0.905, 6, 1, 1, (10, 10)),
        "localization": (0.910, 5, 1, 2, (8, 10)),
        "composition": (0.912, 6, 1, 1, (7, 10)),
    }
    adjusted, div_tp, div_fp, div_fn, missed = values[kind]
    return [
        {
            "stem": stem,
            "weight": 100,
            "adjusted_edge_jaccard": adjusted,
            "div_tp": div_tp,
            "div_fp": div_fp,
            "div_fn": div_fn,
            "missed_gt_nodes": missed[index],
            "spurious_pred_nodes": 3,
        }
        for index, stem in enumerate(("a", "b"))
    ]


def _aggregate(rows: list[dict]) -> dict:
    weight = sum(row["weight"] for row in rows)
    adjusted = sum(row["adjusted_edge_jaccard"] * row["weight"] for row in rows) / weight
    div_tp = sum(row["div_tp"] for row in rows)
    div_fp = sum(row["div_fp"] for row in rows)
    div_fn = sum(row["div_fn"] for row in rows)
    division = div_tp / (div_tp + div_fp + div_fn)
    return {
        "stems": [row["stem"] for row in rows],
        "weighted_adjusted_edge_jaccard": adjusted,
        "division_tp": div_tp,
        "division_fp": div_fp,
        "division_fn": div_fn,
        "division_jaccard": division,
        "proxy_score": adjusted + 0.10 * division,
    }


def _write_promotion(
    path: Path,
    *,
    run_id: str,
    runtime_sha256: str,
    rows: list[dict],
    include_node_errors: bool,
) -> Path:
    payload = _promotion(run_id, runtime_sha256)
    payload["candidate_validator"] = _aggregate(rows)
    if include_node_errors:
        payload["candidate_node_errors"] = {
            "by_stem": {
                row["stem"]: {
                    "missed_gt_nodes": row["missed_gt_nodes"],
                    "spurious_pred_nodes": row["spurious_pred_nodes"],
                }
                for row in rows
            },
            "missed_gt_nodes": sum(row["missed_gt_nodes"] for row in rows),
            "spurious_pred_nodes": sum(row["spurious_pred_nodes"] for row in rows),
        }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _composition_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    graph_root = BUILDER_FIXTURE["_graph_runtime"](tmp_path / "graph-runtime")
    localization_root = BUILDER_FIXTURE["_localization_runtime"](
        tmp_path / "localization-runtime"
    )
    graph_manifest = graph_root / "GRAPH_CONTEXT_CONSENSUS_MANIFEST.json"
    localization_manifest = (
        localization_root / "TEMPORAL_LOCALIZATION_CONSENSUS_MANIFEST.json"
    )
    graph_runtime = MODULE["GRAPH"]["validate_runtime"](graph_manifest)
    localization_runtime = MODULE["LOCALIZATION"]["validate_runtime"](
        localization_manifest
    )
    graph_promotion = _write_promotion(
        tmp_path / "graph-promotion.json",
        run_id=MODULE["GRAPH_RUN_ID"],
        runtime_sha256=graph_runtime["manifest_sha256"],
        rows=_validator_rows("graph"),
        include_node_errors=False,
    )
    localization_promotion = _write_promotion(
        tmp_path / "localization-promotion.json",
        run_id=MODULE["LOCALIZATION_RUN_ID"],
        runtime_sha256=localization_runtime["manifest_sha256"],
        rows=_validator_rows("localization"),
        include_node_errors=True,
    )
    baseline = tmp_path / "baseline.csv"
    _write_csv(baseline, _validator_rows("baseline"))
    output = tmp_path / "output"
    output.mkdir()
    submission = output / "submission.csv"
    submission.write_text(
        "id,dataset,row_type,node_id,t,z,y,x,source_id,target_id\n"
        "0,test,node,1,0,1,2,3,-1,-1\n"
        "1,test,node,2,1,1,3,3,-1,-1\n"
        "2,test,edge,-1,-1,-1,-1,-1,1,2\n",
        encoding="utf-8",
    )
    submission_hash = hashlib.sha256(submission.read_bytes()).hexdigest()
    (output / "watchdog-terminal.json").write_text(
        json.dumps(
            {
                "run_id": MODULE["RUN_ID"],
                "status": "completed",
                "submission_exists": True,
                "submission_sha256": submission_hash,
                "declared_budget_seconds": 43_200,
                "safety_margin_seconds": 1_200,
                "elapsed_seconds": 1_000,
            }
        ),
        encoding="utf-8",
    )
    run_row = {
        "ranked_consensus_geometric_candidates": 5,
        "ranked_consensus_geometry_eligible_candidates": 4,
        "ranked_consensus_candidate_parents_scored": 4,
        "ranked_consensus_ranking_agreed": 1,
        "ranked_consensus_added_edges": 1,
        "ranked_consensus_maximum_additions": 1,
        "ranked_consensus_absolute_threshold_used": 0,
        "ranked_consensus_reassignment_performed": 0,
        "ranked_consensus_node_or_coordinate_changes": 0,
        "ranked_consensus_gpu_groups_used": 1,
        "ranked_consensus_localization_candidate_nodes": 100,
        "ranked_consensus_localization_nodes_moved": 2,
        "ranked_consensus_localization_boundary_rejected": 0,
        "ranked_consensus_localization_rounded_coordinate_changes": 2,
        "ranked_consensus_localization_global_gate_failures": 0,
        "ranked_consensus_localization_node_count_changes": 0,
        "ranked_consensus_localization_edge_changes": 0,
        "ranked_consensus_localization_gpu_groups_used": 2,
    }
    _write_csv(output / "run_stats.csv", [run_row])
    candidate_rows = _validator_rows("composition")
    _write_csv(output / "validator_results.csv", candidate_rows)
    candidate = _aggregate(candidate_rows)
    graph_policy = json.loads(
        (graph_root / "graph-context-consensus-policy.json").read_text()
    )
    localization_policy = json.loads(
        (localization_root / "temporal-localization-consensus-policy.json").read_text()
    )
    evidence = {
        "schema_version": 1,
        "status": "completed_pending_external_promotion_gate",
        "run_id": MODULE["RUN_ID"],
        "target_public_score": 0.945,
        "public_lineage_attributed": True,
        "public_predictions_copied": False,
        "component_order": ["graph_context_division", "temporal_localization"],
        "independent_component_promotion_required": True,
        "graph_runtime_manifest_sha256": graph_runtime["manifest_sha256"],
        "graph_policy": graph_runtime["deep_policy"],
        "graph_member_count": graph_runtime["deep_member_count"],
        "graph_parameters_per_member": 74_732_308,
        "graph_model_sha256": graph_runtime["deep_model_sha256"],
        "morphology_model_sha256": graph_runtime["morphology_model_sha256"],
        "localization_runtime_manifest_sha256": localization_runtime[
            "manifest_sha256"
        ],
        "localization_member_count": localization_runtime["member_count"],
        "localization_parameters_per_member": 71_249_805,
        "localization_model_sha256": localization_runtime["model_sha256"],
        "absolute_threshold_used": False,
        "node_count_preserving": True,
        "model_subset_searched_on_audit": False,
        "weights_searched_on_development": False,
        "threshold_searched_on_development": False,
        "ranked_geometric_candidates": 5,
        "ranked_geometry_eligible_candidates": 4,
        "ranked_candidate_parents_scored": 4,
        "ranked_agreements": 1,
        "ranked_edges_added": 1,
        "ranked_reassignments": 0,
        "ranked_node_or_coordinate_changes": 0,
        "graph_gpu_groups_used": 1,
        "localization_candidate_nodes": 100,
        "localization_nodes_moved": 2,
        "localization_boundary_rejected": 0,
        "localization_rounded_coordinate_changes": 2,
        "localization_global_gate_failures": 0,
        "localization_node_count_changes": 0,
        "localization_edge_changes": 0,
        "localization_gpu_groups_used": 2,
        "validator_adjusted_edge_jaccard": candidate[
            "weighted_adjusted_edge_jaccard"
        ],
        "validator_division_tp": candidate["division_tp"],
        "validator_division_fp": candidate["division_fp"],
        "validator_division_fn": candidate["division_fn"],
        "validator_division_jaccard": candidate["division_jaccard"],
        "validator_proxy_score": candidate["proxy_score"],
        "validator_missed_gt_nodes": sum(
            row["missed_gt_nodes"] for row in candidate_rows
        ),
        "validator_spurious_pred_nodes": sum(
            row["spurious_pred_nodes"] for row in candidate_rows
        ),
        "submission_sha256": submission_hash,
        "competition_submission_performed": False,
        "authorized_for_submission": False,
    }
    assert graph_policy["graph_context_member_count"] == 1
    assert localization_policy["localization_member_count"] == 3
    (output / "candidate_evidence.json").write_text(
        json.dumps(evidence), encoding="utf-8"
    )
    return (
        output,
        baseline,
        graph_manifest,
        localization_manifest,
        graph_promotion,
        localization_promotion,
    )


def test_standalone_promotion_is_required_and_hash_bound(tmp_path: Path) -> None:
    path = tmp_path / "promotion.json"
    payload = _promotion("component-v1", "b" * 64)
    path.write_text(json.dumps(payload), encoding="utf-8")
    observed = MODULE["validate_component_promotion"](
        path,
        component="fixture",
        expected_run_id="component-v1",
        expected_runtime_sha256="b" * 64,
    )
    assert observed == payload
    payload["authorized_for_submission"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="standalone promotion"):
        MODULE["validate_component_promotion"](
            path,
            component="fixture",
            expected_run_id="component-v1",
            expected_runtime_sha256="b" * 64,
        )


def test_composition_gate_must_beat_both_components_without_metric_hacks() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert MODULE["MINIMUM_PROXY_GAIN"] == 0.005
    assert MODULE["MAXIMUM_ADJUSTED_EDGE_REGRESSION"] == 0.001
    assert MODULE["MINIMUM_COMPOSITION_GAIN_OVER_BEST_COMPONENT"] == 0.001
    assert "best_component_proxy = max(" in source
    assert "composition_gain >= MINIMUM_COMPOSITION_GAIN_OVER_BEST_COMPONENT" in source
    assert 'candidate_validator["division_jaccard"] >= graph_validator["division_jaccard"]' in source
    assert "node_nonregressive_to_localization" in source
    assert "public" in source
    assert "leaderboard" not in source.lower()
    assert "kaggle competitions submit" not in source


def test_full_composition_is_promoted_only_when_stronger_than_both_components(
    tmp_path: Path,
) -> None:
    inputs = _composition_fixture(tmp_path)
    result = MODULE["verify_candidate"](*inputs)
    assert result["status"] == "eligible_for_submission"
    assert result["composition_gain_over_best_component"] >= 0.001
    assert result["ranked_edges_added"] == 1
    assert result["localization_nodes_moved"] == 2


def test_full_composition_rejects_when_it_does_not_beat_graph_component(
    tmp_path: Path,
) -> None:
    inputs = _composition_fixture(tmp_path)
    graph_promotion = inputs[4]
    payload = json.loads(graph_promotion.read_text())
    candidate_rows = _validator_rows("composition")
    payload["candidate_validator"] = _aggregate(candidate_rows)
    graph_promotion.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="composition promotion gate failed"):
        MODULE["verify_candidate"](*inputs)
