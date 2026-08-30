#!/usr/bin/env python
"""Score the frozen CPU morphology gate on the held-out complete-movie probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import zarr

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.score_competition_division_probe import (
    BIOLOGICAL_GEOMETRY_MINIMUM,
    atomic_json,
    decision_metrics,
    geometry_division_score,
    ranking_metrics,
    validate_inputs,
    verify_cache,
)
from research.temporal_contrastive.patch_model import sample_physical_patches
from research.train_handcrafted_division_gate import (
    FEATURE_FAMILY,
    RUN_ID as TRAINING_RUN_ID,
    patch_features,
    sha256_file,
)


RUN_ID = "competition-real-handcrafted-division-probe-v1"
POLICY_RUN_ID = "competition-real-handcrafted-division-policy-v1"
VOXEL_SIZE_ZYX_UM = (1.625, 0.40625, 0.40625)


def validate_model_and_policy(
    training_root: Path, policy_path: Path
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    terminal_path = training_root / "handcrafted_division_gate_terminal.json"
    model_path = training_root / "handcrafted_division_gate.joblib"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    threshold = policy.get("frozen_division_probability_threshold")
    if not (
        terminal.get("schema_version") == 1
        and terminal.get("status") == "completed"
        and terminal.get("run_id") == TRAINING_RUN_ID
        and terminal.get("feature_family") == FEATURE_FAMILY
        and terminal.get("final_probe_opened") is False
        and terminal.get("competition_test_data_read") is False
        and terminal.get("public_leaderboard_used_for_selection") is False
        and terminal.get("submission_created") is False
        and model_path.is_file()
        and terminal.get("model_sha256") == sha256_file(model_path)
        and policy.get("schema_version") == 1
        and policy.get("status") == "accepted_at_selection"
        and policy.get("run_id") == POLICY_RUN_ID
        and policy.get("model_training_run_id") == TRAINING_RUN_ID
        and policy.get("model_sha256") == terminal.get("model_sha256")
        and policy.get("training_terminal_sha256") == sha256_file(terminal_path)
        and policy.get("selection_gate_passed") is True
        and policy.get("final_probe_opened") is False
        and policy.get("competition_test_data_read") is False
        and policy.get("public_leaderboard_used_for_selection") is False
        and policy.get("submission_created") is False
        and policy.get("authorized_for_final_probe") is True
        and policy.get("authorized_for_submission") is False
        and isinstance(threshold, (int, float))
        and np.isfinite(threshold)
    ):
        raise ValueError("handcrafted division policy is ineligible")
    return terminal, policy, model_path


def score_candidates(
    inventory: dict[str, Any],
    cache_root: Path,
    model_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for movie in inventory["movies"]:
        stem = str(movie["stem"])
        array = zarr.open_group(
            str(cache_root / "train" / f"{stem}.zarr"), mode="r"
        )["0"]
        by_timepoint: dict[int, list[dict[str, Any]]] = {}
        for candidate in movie["candidates"]:
            by_timepoint.setdefault(int(candidate["timepoint"]), []).append(candidate)
        for timepoint, candidates in sorted(by_timepoint.items()):
            temporal = np.stack(
                [np.asarray(array[timepoint + offset]) for offset in (-1, 0, 1)]
            )
            centers = np.asarray(
                [candidate["center_zyx_voxel"] for candidate in candidates],
                dtype=np.float32,
            )
            patches = sample_physical_patches(
                temporal,
                centers,
                voxel_size_zyx_um=VOXEL_SIZE_ZYX_UM,
                output_shape_zyx=(17, 17, 17),
                half_extent_zyx_um=(8.0, 8.0, 8.0),
                chunk_size=64,
            ).numpy()
            features = patch_features(patches)
            parts = [
                record["model"].predict_proba(features)[:, 1]
                for record in model_payload["models"]
            ]
            probabilities = np.mean(np.stack(parts), axis=0)
            for candidate, probability in zip(candidates, probabilities, strict=True):
                row = {
                    **candidate,
                    "stem": stem,
                    "ensemble_logit": float(probability),
                }
                row["geometry_division_score"] = geometry_division_score(row)
                rows.append(row)
    if len(rows) != 225 or sum(row["safe_recovery_positive"] for row in rows) != 3:
        raise RuntimeError("handcrafted division probe inventory changed")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    cache_manifest = json.loads(
        (args.cache_root / "probe_cache_manifest.json").read_text(encoding="utf-8")
    )
    validate_inputs(inventory, cache_manifest)
    verify_cache(args.cache_root, cache_manifest)
    terminal, policy, model_path = validate_model_and_policy(
        args.training_root, args.policy
    )
    payload = joblib.load(model_path)
    if not (
        payload.get("run_id") == TRAINING_RUN_ID
        and payload.get("feature_family") == FEATURE_FAMILY
        and payload.get("feature_count") == 132
        and len(payload.get("models", [])) == 2
    ):
        raise ValueError("handcrafted division model payload changed")
    rows = score_candidates(inventory, args.cache_root, payload)
    threshold = float(policy["frozen_division_probability_threshold"])
    threshold_only = decision_metrics(
        rows, model_threshold=threshold, geometry_minimum=None
    )
    conjunctive = decision_metrics(
        rows,
        model_threshold=threshold,
        geometry_minimum=BIOLOGICAL_GEOMETRY_MINIMUM,
    )
    accepted = bool(
        conjunctive["tp"] >= 2
        and conjunctive["precision"] >= 0.75
        and conjunctive["jaccard"] > 0.0
    )
    result = {
        "schema_version": 1,
        "status": "accepted" if accepted else "rejected",
        "run_id": RUN_ID,
        "feature_family": FEATURE_FAMILY,
        "model_sha256": sha256_file(model_path),
        "training_terminal_sha256": sha256_file(
            args.training_root / "handcrafted_division_gate_terminal.json"
        ),
        "selection_policy_sha256": sha256_file(args.policy),
        "frozen_division_probability_threshold": threshold,
        "biological_geometry_minimum": BIOLOGICAL_GEOMETRY_MINIMUM,
        "ranking": ranking_metrics(rows, "ensemble_logit"),
        "threshold_only": threshold_only,
        "conjunctive": conjunctive,
        "rows": rows,
        "final_probe_opened": True,
        "final_probe_opened_after_threshold_freeze": True,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_competition_graph_evaluation": accepted,
        "authorized_for_submission": False,
    }
    atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("status", "ranking", "threshold_only", "conjunctive")}, indent=2, sort_keys=True))
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
