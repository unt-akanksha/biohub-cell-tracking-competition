from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "scripts/verify-strong-member-consensus-submission-candidate.py"
MODULE = runpy.run_path(str(VERIFIER))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_runtime(tmp_path: Path) -> Path:
    root = tmp_path / "runtime"
    root.mkdir()
    model = root / "deep_division_model_00.pt"
    morphology = root / "morphology_division_model.joblib"
    model.write_bytes(b"independent-strong-model")
    morphology.write_bytes(b"independent-morphology")
    policy = root / "strong-member-consensus-policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "development_accepted",
                "run_id": MODULE["POLICY_RUN_ID"],
                "deep_policy": "strongest_individual_rank",
                "deep_member_count": 1,
                "deep_members": [
                    {
                        "path": model.name,
                        "model_sha256": sha256(model),
                        "parameter_count": 46_386_607,
                        "trainable_parameters": 25_178_047,
                    }
                ],
                "morphology_model_sha256": sha256(morphology),
                "base_safe_division_heuristic_enabled": True,
                "external_policy_additive_only": True,
                "maximum_added_edges_per_movie": 1,
                "absolute_threshold_used": False,
                "authorized_for_full_candidate_evaluation": True,
                "authorized_for_submission": False,
            }
        ),
        encoding="utf-8",
    )
    manifest = root / "STRONG_MEMBER_CONSENSUS_MANIFEST.json"
    files = {
        path.name: {"path": path.name, "sha256": sha256(path)}
        for path in (model, morphology, policy)
    }
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "run_id": MODULE["RUNTIME_RUN_ID"],
                "competition_test_data_read": False,
                "public_predictions_copied": False,
                "public_leaderboard_used_for_selection": False,
                "submission_command_included": False,
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def make_output(
    tmp_path: Path, runtime_manifest: Path, *, proxy_gain: bool = True
) -> tuple[Path, Path]:
    runtime = MODULE["validate_runtime"](runtime_manifest)
    output = tmp_path / "output"
    output.mkdir()
    submission = output / "submission.csv"
    submission.write_text(
        "id,dataset,row_type,node_id,t,z,y,x,source_id,target_id\n"
        "0,test_a,node,1,0,1,2,3,-1,-1\n"
        "1,test_a,node,2,1,1,2,3,-1,-1\n"
        "2,test_a,edge,-1,-1,-1,-1,-1,1,2\n",
        encoding="utf-8",
    )
    write_csv(
        output / "run_stats.csv",
        [
            "ranked_consensus_geometric_candidates",
            "ranked_consensus_geometry_eligible_candidates",
            "ranked_consensus_candidate_parents_scored",
            "ranked_consensus_ranking_agreed",
            "ranked_consensus_added_edges",
            "ranked_consensus_maximum_additions",
            "ranked_consensus_absolute_threshold_used",
            "ranked_consensus_reassignment_performed",
            "ranked_consensus_node_or_coordinate_changes",
        ],
        [
            {
                "ranked_consensus_geometric_candidates": 5,
                "ranked_consensus_geometry_eligible_candidates": 4,
                "ranked_consensus_candidate_parents_scored": 4,
                "ranked_consensus_ranking_agreed": True,
                "ranked_consensus_added_edges": 1,
                "ranked_consensus_maximum_additions": 1,
                "ranked_consensus_absolute_threshold_used": False,
                "ranked_consensus_reassignment_performed": 0,
                "ranked_consensus_node_or_coordinate_changes": 0,
            }
        ],
    )
    fields = ["stem", "weight", "adjusted_edge_jaccard", "div_tp", "div_fp", "div_fn"]
    baseline = tmp_path / "baseline.csv"
    write_csv(
        baseline,
        fields,
        [
            {
                "stem": "held_out",
                "weight": 100,
                "adjusted_edge_jaccard": 0.918,
                "div_tp": 1,
                "div_fp": 2,
                "div_fn": 4,
            }
        ],
    )
    candidate = {
        "stem": "held_out",
        "weight": 100,
        "adjusted_edge_jaccard": 0.919,
        "div_tp": 3 if proxy_gain else 1,
        "div_fp": 1 if proxy_gain else 2,
        "div_fn": 2 if proxy_gain else 4,
    }
    write_csv(output / "validator_results.csv", fields, [candidate])
    division = candidate["div_tp"] / (
        candidate["div_tp"] + candidate["div_fp"] + candidate["div_fn"]
    )
    evidence = {
        "schema_version": 1,
        "status": "completed_pending_external_promotion_gate",
        "run_id": MODULE["RUN_ID"],
        "target_public_score": 0.945,
        "public_lineage_attributed": True,
        "public_predictions_copied": False,
        "deep_policy": runtime["deep_policy"],
        "deep_member_count": runtime["deep_member_count"],
        "deep_parameter_count_per_member": 46_386_607,
        "deep_model_sha256": runtime["deep_model_sha256"],
        "morphology_model_sha256": runtime["morphology_model_sha256"],
        "absolute_threshold_used": False,
        "runtime_manifest_sha256": runtime["manifest_sha256"],
        "base_safe_division_heuristic_enabled": True,
        "external_policy_additive_only": True,
        "ranked_geometric_candidates": 5,
        "ranked_geometry_eligible_candidates": 4,
        "ranked_candidate_parents_scored": 4,
        "ranked_agreements": 1,
        "ranked_edges_added": 1,
        "ranked_reassignments": 0,
        "ranked_node_or_coordinate_changes": 0,
        "validator_adjusted_edge_jaccard": 0.919,
        "validator_division_tp": candidate["div_tp"],
        "validator_division_fp": candidate["div_fp"],
        "validator_division_fn": candidate["div_fn"],
        "validator_division_jaccard": division,
        "validator_proxy_score": 0.919 + 0.10 * division,
        "submission_sha256": sha256(submission),
        "competition_submission_performed": False,
        "authorized_for_submission": False,
    }
    (output / "candidate_evidence.json").write_text(json.dumps(evidence))
    (output / "watchdog-terminal.json").write_text(
        json.dumps(
            {
                "run_id": MODULE["RUN_ID"],
                "status": "completed",
                "submission_exists": True,
                "submission_sha256": sha256(submission),
                "elapsed_seconds": 1200,
            }
        )
    )
    return output, baseline


def test_promotes_only_additive_strong_member_candidate(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    output, baseline = make_output(tmp_path, runtime)

    result = MODULE["verify_candidate"](output, baseline, runtime)

    assert result["status"] == "eligible_for_submission"
    assert result["base_safe_division_heuristic_enabled"] is True
    assert result["deep_member_count"] == 1


def test_rejects_candidate_that_disables_clean_base_rule(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    output, baseline = make_output(tmp_path, runtime)
    evidence_path = output / "candidate_evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["base_safe_division_heuristic_enabled"] = False
    evidence_path.write_text(json.dumps(evidence))

    with pytest.raises(RuntimeError, match="candidate evidence"):
        MODULE["verify_candidate"](output, baseline, runtime)


def test_rejects_additive_policy_without_end_to_end_gain(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    output, baseline = make_output(tmp_path, runtime, proxy_gain=False)

    with pytest.raises(RuntimeError, match="additive promotion gate"):
        MODULE["verify_candidate"](output, baseline, runtime)
