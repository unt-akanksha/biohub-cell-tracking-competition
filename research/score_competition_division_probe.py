#!/usr/bin/env python
"""Measure focused-division transfer on a tiny competition-train probe.

This is a diagnostic only.  It scores all geometric candidates in the five
known competition-train division frames and never reads competition test data,
chooses a submission threshold, or creates a submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import zarr

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.temporal_contrastive.multiscale_contextual_pair_fusion import (
    EXPECTED_PARAMETER_COUNT,
    MultiscaleContextualPairFusionAssociationModel,
)
from research.temporal_contrastive.patch_model import sample_physical_patches
from research.temporal_contrastive.train_focused_division_gate import division_logits


RUN_ID = "competition-train-focused-division-transfer-probe-v1"
VOXEL_SIZE_ZYX_UM = (1.625, 0.40625, 0.40625)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_inputs(inventory: dict[str, Any], manifest: dict[str, Any]) -> None:
    summary = inventory.get("summary", {})
    cache_summary = manifest.get("summary", {})
    if not (
        inventory.get("schema_version") == 1
        and inventory.get("status") == "competition_train_probe_only"
        and inventory.get("competition_train_data_read") is True
        and inventory.get("competition_test_data_read") is False
        and inventory.get("public_leaderboard_used_for_selection") is False
        and inventory.get("authorized_for_submission") is False
        and summary.get("movies") == 4
        and summary.get("event_frames") == 5
        and summary.get("geometric_candidates") == 225
        and summary.get("safe_recovery_positives") == 3
        and manifest.get("schema_version") == 1
        and manifest.get("status") == "complete"
        and manifest.get("run_id") == "competition-division-probe-frame-cache-v1"
        and manifest.get("competition_train_data_read") is True
        and manifest.get("competition_test_data_read") is False
        and manifest.get("authorized_for_submission") is False
        and cache_summary.get("movies") == 4
        and cache_summary.get("frames") == 15
        and cache_summary.get("files") == 23
    ):
        raise ValueError("competition division probe inputs are ineligible")


def verify_cache(cache_root: Path, manifest: dict[str, Any]) -> None:
    for record in manifest["files"]:
        remote = str(record["remote_path"])
        relative = str(record["local_relative_path"])
        path = cache_root / relative
        if (
            not remote.startswith("train/")
            or relative != remote
            or not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"competition train cache verification failed: {relative}")


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if labels.ndim != 1 or scores.shape != labels.shape or positives <= 0:
        raise ValueError("average precision requires aligned rows and positives")
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].mean())


def geometry_division_score(row: dict[str, Any]) -> float:
    """Fixed branch-symmetry score; higher means a more division-like fork."""
    minimum_step = min(
        float(row["existing_distance_um"]), float(row["parent_distance_um"])
    )
    return float(
        minimum_step
        * float(row["daughter_step_ratio"])
        * (1.0 - float(row["daughter_opposition_cosine"]))
        / (1.0 + float(row["daughter_midpoint_distance_um"]) / 4.0)
    )


def ranking_metrics(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    labels = np.asarray([row["safe_recovery_positive"] for row in rows], dtype=bool)
    scores = np.asarray([row[score_key] for row in rows], dtype=np.float64)
    positive_count = int(labels.sum())
    order = np.argsort(-scores, kind="stable")
    precision_at_k = float(labels[order[:positive_count]].mean())
    event_ranks: list[dict[str, Any]] = []
    groups = sorted({(str(row["stem"]), int(row["timepoint"])) for row in rows})
    for stem, timepoint in groups:
        event_rows = [
            row
            for row in rows
            if row["stem"] == stem and int(row["timepoint"]) == timepoint
        ]
        positive_rows = [row for row in event_rows if row["safe_recovery_positive"]]
        if not positive_rows:
            continue
        event_order = sorted(
            event_rows,
            key=lambda row: (-float(row[score_key]), int(row["parent_id"])),
        )
        positive = positive_rows[0]
        rank = 1 + next(
            index
            for index, row in enumerate(event_order)
            if int(row["parent_id"]) == int(positive["parent_id"])
        )
        event_ranks.append(
            {
                "stem": stem,
                "timepoint": timepoint,
                "parent_id": int(positive["parent_id"]),
                "rank": rank,
                "candidates": len(event_rows),
                "top_fraction": float(rank / len(event_rows)),
                "score": float(positive[score_key]),
            }
        )
    return {
        "rows": len(rows),
        "positives": positive_count,
        "average_precision": average_precision(labels, scores),
        "precision_at_positive_count": precision_at_k,
        "event_ranks": event_ranks,
        "mean_positive_event_top_fraction": float(
            np.mean([row["top_fraction"] for row in event_ranks])
        ),
        "all_positive_events_top_10_percent": all(
            row["top_fraction"] <= 0.10 for row in event_ranks
        ),
    }


@torch.inference_mode()
def score_patches(
    models: list[torch.nn.Module],
    patches: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> list[np.ndarray]:
    fold_scores: list[np.ndarray] = []
    for model in models:
        pieces: list[torch.Tensor] = []
        for start in range(0, len(patches), batch_size):
            batch = patches[start : start + batch_size].to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                pieces.append(division_logits(model, batch).float().cpu())
        fold_scores.append(torch.cat(pieces).numpy())
    return fold_scores


def load_models(paths: list[Path], device: torch.device) -> tuple[list[torch.nn.Module], list[str]]:
    if len(paths) != 2:
        raise ValueError("the focused transfer probe requires two independent models")
    models: list[torch.nn.Module] = []
    hashes: list[str] = []
    for path in paths:
        model = MultiscaleContextualPairFusionAssociationModel().to(device)
        model.load_state_dict(
            torch.load(path, map_location=device, weights_only=True), strict=True
        )
        if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETER_COUNT:
            raise RuntimeError("focused model parameter inventory changed")
        models.append(model.requires_grad_(False).eval())
        hashes.append(sha256_file(path))
    if len(set(hashes)) != len(hashes):
        raise ValueError("focused transfer models must be independently optimized")
    return models, hashes


def build_rows(
    inventory: dict[str, Any],
    cache_root: Path,
    models: list[torch.nn.Module],
    *,
    device: torch.device,
    batch_size: int,
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
            context = np.stack(
                [np.asarray(array[timepoint + offset]) for offset in (-1, 0, 1)]
            )
            centers = np.asarray(
                [candidate["center_zyx_voxel"] for candidate in candidates],
                dtype=np.float32,
            )
            patches = sample_physical_patches(
                context,
                centers,
                voxel_size_zyx_um=VOXEL_SIZE_ZYX_UM,
                output_shape_zyx=(17, 17, 17),
                half_extent_zyx_um=(8.0, 8.0, 8.0),
                chunk_size=64,
            )
            fold_scores = score_patches(
                models, patches, device=device, batch_size=batch_size
            )
            ensemble = np.mean(np.stack(fold_scores), axis=0)
            for index, candidate in enumerate(candidates):
                row = {
                        "stem": stem,
                        "timepoint": timepoint,
                        "parent_id": int(candidate["parent_id"]),
                        "existing_child_id": int(candidate["existing_child_id"]),
                        "second_child_id": int(candidate["second_child_id"]),
                        "parent_distance_um": float(candidate["parent_distance_um"]),
                        "sister_distance_um": float(candidate["sister_distance_um"]),
                        "existing_distance_um": float(
                            candidate["existing_distance_um"]
                        ),
                        "daughter_midpoint_distance_um": float(
                            candidate["daughter_midpoint_distance_um"]
                        ),
                        "daughter_opposition_cosine": float(
                            candidate["daughter_opposition_cosine"]
                        ),
                        "daughter_step_ratio": float(
                            candidate["daughter_step_ratio"]
                        ),
                        "parent_velocity_um": candidate["parent_velocity_um"],
                        "constant_velocity_midpoint_error_um": candidate[
                            "constant_velocity_midpoint_error_um"
                        ],
                        "safe_recovery_positive": bool(
                            candidate["safe_recovery_positive"]
                        ),
                        "model_0_logit": float(fold_scores[0][index]),
                        "model_1_logit": float(fold_scores[1][index]),
                        "ensemble_logit": float(ensemble[index]),
                    }
                row["geometry_division_score"] = geometry_division_score(row)
                rows.append(row)
    if len(rows) != 225 or sum(row["safe_recovery_positive"] for row in rows) != 3:
        raise RuntimeError("competition division probe row inventory changed")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--required-gpu-name", default="A10G")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("competition transfer probe requires one Antelume GPU")
    gpu_name = torch.cuda.get_device_name(0)
    if args.required_gpu_name.lower() not in gpu_name.lower():
        raise RuntimeError(
            f"required Antelume GPU {args.required_gpu_name!r}, saw {gpu_name!r}"
        )

    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    manifest = json.loads(
        (args.cache_root / "probe_cache_manifest.json").read_text(encoding="utf-8")
    )
    validate_inputs(inventory, manifest)
    verify_cache(args.cache_root, manifest)
    device = torch.device("cuda:0")
    models, model_hashes = load_models(args.model_path, device)
    rows = build_rows(
        inventory,
        args.cache_root,
        models,
        device=device,
        batch_size=args.batch_size,
    )
    metrics = {
        key: ranking_metrics(rows, key)
        for key in (
            "model_0_logit",
            "model_1_logit",
            "ensemble_logit",
            "geometry_division_score",
        )
    }
    result = {
        "schema_version": 1,
        "status": "diagnostic_complete",
        "run_id": RUN_ID,
        "gpu_name": gpu_name,
        "model_sha256": model_hashes,
        "metrics": metrics,
        "rows": rows,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "threshold_selected": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }
    atomic_json(args.output, result)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
