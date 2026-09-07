#!/usr/bin/env python
"""Verify and promote a clean temporal peak-ranking tracking candidate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import runpy
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COMMON = runpy.run_path(str(ROOT / "scripts/verify-learned-division-submission-candidate.py"))
aggregate_validator = COMMON["aggregate_validator"]
read_csv_rows = COMMON["read_csv_rows"]
sha256_file = COMMON["sha256_file"]
unique_file = COMMON["unique_file"]
validate_submission_csv = COMMON["validate_submission_csv"]
_assert_close = COMMON["_assert_close"]
KNOWN_PUBLIC_SUBMISSION_SHA256 = COMMON["KNOWN_PUBLIC_SUBMISSION_SHA256"]
PUBLIC_CONTROL_VALIDATOR_SHA256 = COMMON["PUBLIC_CONTROL_VALIDATOR_SHA256"]

RUN_ID = "peak-rank-tracking-candidate-v1"
SOURCE_PUBLIC_KERNEL_REF = "redoctopusk/biohub-948tta2"
SOURCE_PUBLIC_NOTEBOOK_SHA256 = "3395f8df72c6d63d243fdb4fede1f1febdd36bfc086b2f0663fec3ccc9dbb189"
OFFICIAL_SCORER_LOCK_SHA256 = (
    "1db65dee620059f19bf16633aa54a9f3379eb5d5bdff148a4037b949393b7a9c"
)
OFFICIAL_METRIC_RUN_ID = "peak-rank-patched-official-complete-movie-v1"
OFFICIAL_EVALUATION_KIND = (
    "patched_official_complete_movie_candidate_vs_frozen_control"
)
EXPECTED_CONTROL_SCORE = 0.9343483108193262
MINIMUM_EXACT_SCORE_GAIN = 0.003
MAXIMUM_EXACT_EDGE_REGRESSION = 0.001
MAXIMUM_EXACT_MOVIE_REGRESSION = 0.005
SUPPORTED_ARCHITECTURES = {
    "independent temporal 3D ConvNeXt U-Net peak ranker",
    "equal-logit ensemble of independent temporal 3D ConvNeXt U-Net peak rankers",
    "confidence-selective ensemble of independent temporal 3D ConvNeXt U-Net peak rankers",
}


def validate_runtime(runtime_manifest: Path) -> dict[str, Any]:
    root = runtime_manifest.parent
    manifest = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    validation_path = root / "clean_validation.json"
    checkpoint = root / "peak_rank_detector.pt"
    training = json.loads((root / "training_terminal.json").read_text(encoding="utf-8"))
    parameter_count = manifest.get("parameter_count")
    ensemble_size = manifest.get("ensemble_size")
    ensemble_members = training.get("ensemble_members")
    expected_ensemble_size = len(ensemble_members) if ensemble_members is not None else 1
    ensemble_fusion = training.get("ensemble_fusion")
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("architecture") in SUPPORTED_ARCHITECTURES
        and isinstance(parameter_count, int)
        and not isinstance(parameter_count, bool)
        and parameter_count > 0
        and parameter_count == training.get("parameter_count")
        and manifest.get("widths") == training.get("widths")
        and manifest.get("depths") == training.get("depths")
        and ensemble_size == expected_ensemble_size
        and (
            ensemble_members is None
            or (
                ensemble_fusion
                in {
                    "equal_logit_and_offset_mean",
                    "confidence_max_logit_with_winner_offset",
                }
                and manifest.get("ensemble_fusion") == ensemble_fusion
            )
        )
        and isinstance(ensemble_size, int)
        and ensemble_size > 0
        and manifest.get("training_audit_passed") is True
        and manifest.get("clean_validation_promotion_passed") is True
        and manifest.get("checkpoint_sha256") == sha256_file(checkpoint)
        and manifest.get("clean_validation_sha256") == sha256_file(validation_path)
        and "predict_with_official_linker.py" in files
        and "clean_validation.json" in files
    ):
        raise RuntimeError("peak-ranking promoted runtime is invalid")
    for name, record in files.items():
        path = root / name
        if not path.is_file() or record.get("sha256") != sha256_file(path):
            raise RuntimeError(f"peak-ranking runtime file changed: {name}")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not (
        validation.get("run_id") == "temporal-peak-rank-clean-validation-v1"
        and validation.get("selection_passed") is True
        and validation.get("acceptance_opened") is True
        and validation.get("promotion_passed") is True
        and validation.get("selected_tta_mode") in {"none", "rot4", "d4"}
        and manifest.get("selected_peak_tta_mode") == validation.get("selected_tta_mode")
        and manifest.get("selected_peak_tta_views") == validation.get("selected_tta_views")
        and validation.get("competition_test_data_read") is False
        and validation.get("competition_submission_performed") is False
        and validation.get("provenance", {}).get("checkpoint_sha256")
            == manifest["checkpoint_sha256"]
    ):
        raise RuntimeError("peak-ranking clean validation is invalid")
    return {
        "manifest_sha256": sha256_file(runtime_manifest),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "clean_validation_sha256": manifest["clean_validation_sha256"],
        "selected_peak_tta_mode": manifest["selected_peak_tta_mode"],
        "selected_peak_tta_views": manifest["selected_peak_tta_views"],
        "parameter_count": parameter_count,
        "ensemble_size": ensemble_size,
    }


def movie_proxy_rows(path: Path) -> dict[str, float]:
    result = {}
    for row in read_csv_rows(path):
        denominator = int(row["div_tp"]) + int(row["div_fp"]) + int(row["div_fn"])
        division = int(row["div_tp"]) / denominator if denominator else 0.0
        value = float(row["adjusted_edge_jaccard"]) + 0.10 * division
        if not math.isfinite(value) or row["stem"] in result:
            raise RuntimeError("per-movie validator rows are invalid")
        result[row["stem"]] = value
    return result


def verify_candidate(
    output_root: Path,
    baseline_validator: Path,
    runtime_manifest: Path,
    official_metric_result: Path,
    *,
    expected_baseline_sha256: str | None = None,
    expected_run_id: str = RUN_ID,
) -> dict[str, Any]:
    runtime = validate_runtime(runtime_manifest)
    baseline_sha256 = sha256_file(baseline_validator)
    if expected_baseline_sha256 and baseline_sha256 != expected_baseline_sha256:
        raise RuntimeError("public-control validator artifact changed")
    terminal_path = unique_file(output_root, "launcher_terminal.json")
    evidence_path = unique_file(output_root, "candidate_evidence.json")
    submission_path = unique_file(output_root, "submission.csv")
    validator_path = unique_file(output_root, "validator_results.csv")
    official_validator_path = unique_file(
        output_root, "official_validator_candidate.csv"
    )
    exact_path = unique_file(output_root, "official_metric_result.json")
    if exact_path.resolve() != official_metric_result.resolve():
        raise RuntimeError("official metric result path is not the candidate result")
    unique_file(output_root, "run_stats.csv")
    worker_paths = sorted(path for path in output_root.rglob("worker-*.json") if path.is_file())
    if len(worker_paths) != 2:
        raise RuntimeError(f"expected two downloaded worker manifests, saw {worker_paths}")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    exact = json.loads(exact_path.read_text(encoding="utf-8"))
    workers = [json.loads(path.read_text(encoding="utf-8")) for path in worker_paths]
    submission_sha256 = sha256_file(submission_path)
    if not (
        terminal.get("run_id") == expected_run_id
        and terminal.get("status") == "completed"
        and terminal.get("submission_exists") is True
        and terminal.get("evidence_exists") is True
        and terminal.get("competition_submission_performed") is False
        and terminal.get("authorized_for_submission") is False
        and float(terminal.get("elapsed_seconds", math.inf)) < 41_400.0
    ):
        raise RuntimeError("peak-ranking candidate launcher terminal is invalid")
    if not (
        evidence.get("schema_version") == 1
        and evidence.get("run_id") == expected_run_id
        and evidence.get("status") == "completed_pending_external_promotion_gate"
        and float(evidence.get("target_public_score", 0.0)) == 0.945
        and evidence.get("source_public_lineage_attributed") is True
        and evidence.get("source_public_kernel_ref") == SOURCE_PUBLIC_KERNEL_REF
        and evidence.get("source_public_notebook_sha256")
            == SOURCE_PUBLIC_NOTEBOOK_SHA256
        and evidence.get("secondary_edge_feature_tta") is True
        and evidence.get("subvoxel_association_coordinates") is True
        and evidence.get("dual_association_models_verified") is True
        and evidence.get("source_advertised_score_used_as_evidence") is False
        and evidence.get("public_predictions_copied") is False
        and evidence.get("checkpoint_sha256") == runtime["checkpoint_sha256"]
        and evidence.get("runtime_manifest_sha256") == runtime["manifest_sha256"]
        and evidence.get("clean_validation_sha256") == runtime["clean_validation_sha256"]
        and evidence.get("selected_peak_tta_mode") == runtime["selected_peak_tta_mode"]
        and evidence.get("selected_peak_tta_views") == runtime["selected_peak_tta_views"]
        and evidence.get("parameter_count") == runtime["parameter_count"]
        and evidence.get("ensemble_size") == runtime["ensemble_size"]
        and float(evidence.get("max_worker_elapsed_seconds", math.inf)) < 31_500.0
        and float(evidence.get("max_projected_worker_seconds", math.inf)) <= 31_500.0
        and float(evidence.get("worker_budget_seconds", 0.0)) == 31_500.0
        and evidence.get("worker_count") == 2
        and evidence.get("competition_submission_performed") is False
        and evidence.get("authorized_for_submission") is False
        and evidence.get("submission_sha256") == submission_sha256
    ):
        raise RuntimeError("peak-ranking candidate evidence is invalid")
    worker_movies = [movie["dataset"] for row in workers for movie in row["movies"]]
    if not (
        {row.get("worker_index") for row in workers} == {0, 1}
        and all(
            row.get("run_id") == "peak-rank-official-linker-production-v1"
            and row.get("worker_count") == 2
            and row.get("checkpoint_sha256") == runtime["checkpoint_sha256"]
            and row.get("parameter_count") == runtime["parameter_count"]
            and row.get("ensemble_size") == runtime["ensemble_size"]
            and row.get("peak_tta_mode") == runtime["selected_peak_tta_mode"]
            and row.get("peak_tta_views") == runtime["selected_peak_tta_views"]
            and row.get("association", {}).get("edge_feature_tta") is True
            and row.get("association", {}).get("secondary_link_mode")
                == "low_margin_consensus"
            and float(
                row.get("association", {}).get("bidirectional_edge_weight", math.nan)
            ) == 0.15
            and float(row.get("worker_elapsed_seconds", math.inf)) < 31_500.0
            and float(row.get("projected_worker_seconds", math.inf)) <= 31_500.0
            and float(row.get("worker_budget_seconds", 0.0)) == 31_500.0
            and row.get("input_partition") == "test"
            and row.get("competition_train_labels_read") is False
            and row.get("competition_test_labels_read") is False
            and row.get("public_predictions_copied") is False
            and row.get("public_leaderboard_used_for_selection") is False
            and row.get("submission_created") is False
            for row in workers
        )
        and len(worker_movies) == len(set(worker_movies))
    ):
        raise RuntimeError("peak-ranking worker evidence is invalid")
    submission = validate_submission_csv(submission_path)
    if sorted(worker_movies) != submission["datasets"]:
        raise RuntimeError("worker and submission movie inventories differ")
    if evidence.get("complete_test_movie_count") != len(worker_movies):
        raise RuntimeError("candidate evidence test movie count changed")
    if submission_sha256 in KNOWN_PUBLIC_SUBMISSION_SHA256:
        raise RuntimeError("peak-ranking candidate is identical to an audited public output")
    candidate = aggregate_validator(validator_path)
    baseline = aggregate_validator(baseline_validator)
    if candidate["stems"] != baseline["stems"]:
        raise RuntimeError("candidate and clean control validator stems differ")
    for label, key, aggregate_key in (
        ("validator_proxy_score", "validator_proxy_score", "proxy_score"),
        ("validator_adjusted_edge_jaccard", "validator_adjusted_edge_jaccard", "weighted_adjusted_edge_jaccard"),
        ("validator_division_jaccard", "validator_division_jaccard", "division_jaccard"),
    ):
        _assert_close(label, evidence.get(key), candidate[aggregate_key])
    if evidence.get("complete_validator_movie_count") != len(candidate["stems"]):
        raise RuntimeError("candidate evidence validator movie count changed")
    if not (
        evidence.get("official_metric_status")
        == "pending_external_patched_official_scoring"
        and evidence.get("official_scorer_lock_sha256")
        == OFFICIAL_SCORER_LOCK_SHA256
        and evidence.get("official_validator_candidate_sha256")
        == sha256_file(official_validator_path)
    ):
        raise RuntimeError("candidate official-validator materialization is invalid")
    proxy_gain = candidate["proxy_score"] - baseline["proxy_score"]
    proxy_edge_delta = (
        candidate["weighted_adjusted_edge_jaccard"]
        - baseline["weighted_adjusted_edge_jaccard"]
    )
    candidate_movies = movie_proxy_rows(validator_path)
    baseline_movies = movie_proxy_rows(baseline_validator)
    worst_movie_proxy_delta = min(
        candidate_movies[stem] - baseline_movies[stem] for stem in candidate["stems"]
    )
    exact_gate = exact.get("gate", {})
    exact_checks = exact_gate.get("checks", {})
    exact_control = exact.get("control", {})
    exact_candidate = exact.get("candidate", {})
    exact_score_gain = float(exact_gate.get("score_gain", math.nan))
    exact_edge_delta = float(exact_gate.get("adjusted_edge_delta", math.nan))
    exact_worst_movie_delta = float(
        exact_gate.get("worst_movie_score_delta", math.nan)
    )
    if not (
        exact.get("schema_version") == 1
        and exact.get("run_id") == OFFICIAL_METRIC_RUN_ID
        and exact.get("status") == "accepted"
        and exact.get("evaluation_kind") == OFFICIAL_EVALUATION_KIND
        and exact.get("exact_official_gate_passed") is True
        and exact.get("official_scorer_source_verified") is True
        and exact.get("scorer_lock_sha256") == OFFICIAL_SCORER_LOCK_SHA256
        and exact.get("candidate_validator_sha256")
        == sha256_file(official_validator_path)
        and exact.get("complete_movie_count") == 4
        and exact.get("embryo_prefixes") == ["44b6", "6bba"]
        and len(exact.get("by_movie", [])) == 4
        and bool(exact_checks)
        and all(value is True for value in exact_checks.values())
        and math.isclose(
            float(exact_control.get("score", math.nan)),
            EXPECTED_CONTROL_SCORE,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and exact_score_gain >= MINIMUM_EXACT_SCORE_GAIN
        and exact_edge_delta >= -MAXIMUM_EXACT_EDGE_REGRESSION
        and exact_worst_movie_delta >= -MAXIMUM_EXACT_MOVIE_REGRESSION
        and math.isclose(
            float(exact_candidate.get("score", math.nan))
            - float(exact_control.get("score", math.nan)),
            exact_score_gain,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and exact.get("competition_test_labels_read") is False
        and exact.get("public_leaderboard_used_for_selection") is False
        and exact.get("competition_submission_performed") is False
        and exact.get("authorized_for_submission") is True
    ):
        raise RuntimeError(
            "0.945 peak-ranking patched-official promotion gate failed: "
            f"exact_score_gain={exact_score_gain:.6f}, "
            f"exact_edge_delta={exact_edge_delta:.6f}, "
            f"exact_worst_movie_delta={exact_worst_movie_delta:.6f}"
        )
    return {
        "schema_version": 1,
        "status": "eligible_for_submission",
        "run_id": expected_run_id,
        "target_public_score": 0.945,
        "source_public_kernel_ref": SOURCE_PUBLIC_KERNEL_REF,
        "source_public_notebook_sha256": SOURCE_PUBLIC_NOTEBOOK_SHA256,
        "secondary_edge_feature_tta": True,
        "subvoxel_association_coordinates": True,
        "dual_association_models_verified": True,
        "submission": submission,
        "submission_path": str(submission_path.resolve()),
        "submission_sha256": submission_sha256,
        "candidate_evidence_sha256": sha256_file(evidence_path),
        "launcher_terminal_sha256": sha256_file(terminal_path),
        "runtime_manifest_sha256": runtime["manifest_sha256"],
        "checkpoint_sha256": runtime["checkpoint_sha256"],
        "worker_count": 2,
        "selected_peak_tta_mode": runtime["selected_peak_tta_mode"],
        "selected_peak_tta_views": runtime["selected_peak_tta_views"],
        "public_control": baseline,
        "public_control_validator_sha256": baseline_sha256,
        "candidate_validator": candidate,
        "diagnostic_proxy_gain": proxy_gain,
        "diagnostic_proxy_adjusted_edge_delta": proxy_edge_delta,
        "diagnostic_worst_movie_proxy_delta": worst_movie_proxy_delta,
        "public_validator_proxy_used_for_promotion": False,
        "official_metric_result": exact,
        "official_metric_result_sha256": sha256_file(exact_path),
        "official_validator_candidate_sha256": sha256_file(
            official_validator_path
        ),
        "official_scorer_lock_sha256": OFFICIAL_SCORER_LOCK_SHA256,
        "exact_control_score": float(exact_control["score"]),
        "exact_candidate_score": float(exact_candidate["score"]),
        "exact_score_gain": exact_score_gain,
        "adjusted_edge_delta": exact_edge_delta,
        "worst_movie_score_delta": exact_worst_movie_delta,
        "known_public_hash_match": False,
        "competition_submission_performed": False,
        "authorized_for_submission": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-validator", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--official-metric-result", type=Path, required=True)
    parser.add_argument("--expected-run-id", default=RUN_ID)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify_candidate(
        args.output_root,
        args.baseline_validator,
        args.runtime_manifest,
        args.official_metric_result,
        expected_baseline_sha256=PUBLIC_CONTROL_VALIDATOR_SHA256,
        expected_run_id=args.expected_run_id,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(args.report.suffix + ".partial")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.report)
    print(rendered, end="")


if __name__ == "__main__":
    main()
