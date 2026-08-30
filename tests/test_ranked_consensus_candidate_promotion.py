from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "scripts/verify-ranked-consensus-submission-candidate.py"
SUBMITTER = ROOT / "scripts/submit-ranked-consensus-candidate.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_output(tmp_path: Path, *, proxy_gain: bool = True) -> tuple[Path, Path]:
    module = runpy.run_path(str(VERIFIER))
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
    baseline = tmp_path / "baseline.csv"
    validator_fields = [
        "stem",
        "weight",
        "adjusted_edge_jaccard",
        "div_tp",
        "div_fp",
        "div_fn",
    ]
    write_csv(
        baseline,
        validator_fields,
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
    candidate_row = {
        "stem": "held_out",
        "weight": 100,
        "adjusted_edge_jaccard": 0.918,
        "div_tp": 3 if proxy_gain else 1,
        "div_fp": 1 if proxy_gain else 2,
        "div_fn": 2 if proxy_gain else 4,
    }
    write_csv(output / "validator_results.csv", validator_fields, [candidate_row])
    division = candidate_row["div_tp"] / (
        candidate_row["div_tp"] + candidate_row["div_fp"] + candidate_row["div_fn"]
    )
    evidence = {
        "schema_version": 1,
        "status": "completed_pending_external_promotion_gate",
        "run_id": "ema-ranked-consensus-candidate-v1",
        "target_public_score": 0.945,
        "public_lineage_attributed": True,
        "public_predictions_copied": False,
        "project_division_components": 2,
        "deep_parameter_count": 46_386_607,
        "absolute_threshold_used": False,
        "runtime_manifest_sha256": module["EXPECTED_RUNTIME_MANIFEST_SHA256"],
        "deep_model_sha256": "1" * 64,
        "morphology_model_sha256": "2" * 64,
        "safe_division_heuristic_disabled": True,
        "ranked_geometric_candidates": 5,
        "ranked_geometry_eligible_candidates": 4,
        "ranked_candidate_parents_scored": 4,
        "ranked_agreements": 1,
        "ranked_edges_added": 1,
        "ranked_reassignments": 0,
        "ranked_node_or_coordinate_changes": 0,
        "validator_adjusted_edge_jaccard": 0.918,
        "validator_division_tp": candidate_row["div_tp"],
        "validator_division_fp": candidate_row["div_fp"],
        "validator_division_fn": candidate_row["div_fn"],
        "validator_division_jaccard": division,
        "validator_proxy_score": 0.918 + 0.10 * division,
        "submission_sha256": sha256(submission),
        "competition_submission_performed": False,
        "authorized_for_submission": False,
    }
    (output / "candidate_evidence.json").write_text(json.dumps(evidence))
    (output / "watchdog-terminal.json").write_text(
        json.dumps(
            {
                "run_id": "ema-ranked-consensus-candidate-v1",
                "status": "completed",
                "submission_exists": True,
                "submission_sha256": sha256(submission),
                "elapsed_seconds": 1200,
            }
        )
    )
    return output, baseline


def test_promotes_threshold_free_consensus_candidate(tmp_path: Path) -> None:
    output, baseline = make_output(tmp_path)
    module = runpy.run_path(str(VERIFIER))

    result = module["verify_candidate"](output, baseline)

    assert result["status"] == "eligible_for_submission"
    assert result["ranked_edges_added"] == result["ranked_agreements"] == 1
    assert result["proxy_gain"] >= 0.005


def test_rejects_consensus_candidate_without_proxy_gain(tmp_path: Path) -> None:
    output, baseline = make_output(tmp_path, proxy_gain=False)
    module = runpy.run_path(str(VERIFIER))

    try:
        module["verify_candidate"](output, baseline)
    except RuntimeError as error:
        assert "0.945 promotion gate failed" in str(error)
    else:
        raise AssertionError("non-improving candidate unexpectedly passed")


def test_submitter_requires_unchanged_consensus_submission(tmp_path: Path) -> None:
    submission = tmp_path / "submission.csv"
    submission.write_text("id,dataset,row_type\n", encoding="utf-8")
    promotion = tmp_path / "promotion.json"
    promotion.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "eligible_for_submission",
                "run_id": "ema-ranked-consensus-candidate-v1",
                "target_public_score": 0.945,
                "known_public_hash_match": False,
                "ranked_edges_added": 2,
                "ranked_agreements": 2,
                "ranked_reassignments": 0,
                "ranked_node_or_coordinate_changes": 0,
                "proxy_gain": 0.01,
                "competition_submission_performed": False,
                "authorized_for_submission": True,
                "submission_path": str(submission),
                "submission_sha256": sha256(submission),
            }
        ),
        encoding="utf-8",
    )
    module = runpy.run_path(str(SUBMITTER))

    payload, observed = module["validate_promotion"](promotion)

    assert observed == submission.resolve()
    assert payload["ranked_agreements"] == 2
