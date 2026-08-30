#!/usr/bin/env python
"""Admit only independently strong models from the overnight seed sweep.

The selection split is reused only to gate the independently trained models
and to evaluate one precommitted equal-rank ensemble.  No weight search, final
probe, competition test data, leaderboard result, or submission is used.
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

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.temporal_contrastive.multiscale_contextual_pair_fusion import (
    EXPECTED_PARAMETER_COUNT,
    MultiscaleContextualPairFusionAssociationModel,
)
from research.temporal_contrastive.train_real_division_gate import (
    load_role,
    predict,
    select_frozen_threshold,
    threshold_metrics,
)


RUN_ID = "competition-real-division-seed-ensemble-v1"
SWEEP_RUN_ID = "competition-real-division-seed-sweep-v1"
TRAIN_RUN_ID = "competition-real-division-gate-v1"
MANIFEST_SHA256 = "943717472518b917175312bd4ada9e12660d31d3ebf0bf7afd5672ab40442e1e"
FOLDS = ("target_44b6", "target_6bba")
MINIMUM_POOLED_AP = 0.55
MINIMUM_EMBRYO_AP = 0.40
REFERENCE_MODEL_SHA256 = (
    "4f2d0d4b6db1851543fd0652c7d5d245fad91f3c8a37ea1054501fee357042dc"
)
REFERENCE_SELECTION_AP = 0.5659425288793085
MINIMUM_INDIVIDUAL_AP_GAIN = 0.01
MINIMUM_ENSEMBLE_AP_GAIN = 0.01
MINIMUM_ENSEMBLE_EMBRYO_AP = 0.45


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


def grouped_indices(inventory: list[dict[str, Any]], key: str) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(inventory):
        groups.setdefault(str(row[key]), []).append(index)
    return groups


def movie_rank_percentiles(
    scores: torch.Tensor, inventory: list[dict[str, Any]]
) -> torch.Tensor:
    """Map each model to a calibration-free within-movie ranking score."""
    if scores.ndim != 1 or len(scores) != len(inventory):
        raise ValueError("rank scores and inventory must be aligned vectors")
    ranked = torch.empty_like(scores, dtype=torch.float32)
    for indices in grouped_indices(inventory, "stem").values():
        local = scores[indices]
        order = torch.argsort(local, descending=True, stable=True)
        if len(indices) == 1:
            values = torch.ones(1, dtype=torch.float32, device=scores.device)
        else:
            values = torch.linspace(
                1.0, 0.0, len(indices), dtype=torch.float32, device=scores.device
            )
        local_rank = torch.empty_like(values)
        local_rank[order] = values
        ranked[indices] = local_rank
    return ranked


def movie_top_metrics(
    targets: torch.Tensor, scores: torch.Tensor, inventory: list[dict[str, Any]]
) -> dict[str, float | int]:
    if targets.shape != scores.shape or len(scores) != len(inventory):
        raise ValueError("movie-top inputs are not aligned")
    positives = 0
    selected = 0
    positive_movies = 0
    for indices in grouped_indices(inventory, "stem").values():
        local_targets = targets[indices] > 0.5
        positive_movies += int(torch.any(local_targets))
        top = int(torch.argmax(scores[indices]))
        positives += int(local_targets[top])
        selected += 1
    return {
        "selected_movies": selected,
        "positive_movies": positive_movies,
        "true_positive_tops": positives,
        "false_positive_tops": selected - positives,
        "precision": positives / selected if selected else 0.0,
        "positive_movie_recall": positives / positive_movies if positive_movies else 0.0,
    }


def individual_admitted(
    pooled: dict[str, Any],
    by_embryo: dict[str, dict[str, Any]],
    frozen: dict[str, Any] | None,
) -> bool:
    return bool(
        float(pooled["average_precision"]) >= MINIMUM_POOLED_AP
        and set(by_embryo) == {"44b6", "6bba"}
        and all(
            float(metrics["average_precision"]) >= MINIMUM_EMBRYO_AP
            for metrics in by_embryo.values()
        )
        and frozen is not None
        and int(frozen["fp"]) == 0
        and int(frozen["tp"]) >= 2
    )


def stronger_than_reference(candidate: dict[str, Any]) -> bool:
    return bool(
        candidate.get("status") == "admitted"
        and float(candidate["selection"]["average_precision"])
        >= REFERENCE_SELECTION_AP + MINIMUM_INDIVIDUAL_AP_GAIN
    )


def embryo_metrics(
    targets: torch.Tensor,
    scores: torch.Tensor,
    inventory: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result = {}
    for embryo, indices in grouped_indices(inventory, "embryo").items():
        result[embryo] = threshold_metrics(targets[indices], scores[indices])
    return result


def validate_sweep_terminal(payload: dict[str, Any]) -> list[int]:
    seeds = payload.get("seeds", [])
    if not (
        payload.get("schema_version") == 1
        and payload.get("status") == "completed"
        and payload.get("run_id") == SWEEP_RUN_ID
        and payload.get("planned_model_count") == 16
        and payload.get("steps_per_model") == 50_000
        and len(seeds) == 8
        and len(set(seeds)) == len(seeds)
        and payload.get("competition_test_data_read") is False
        and payload.get("final_probe_opened") is False
        and payload.get("public_leaderboard_used_for_selection") is False
        and payload.get("submission_created") is False
    ):
        raise ValueError("overnight seed-sweep terminal is ineligible")
    return [int(seed) for seed in seeds]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--required-gpu-name", default="A10G")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("seed-ensemble evaluation requires one Antelume GPU")
    gpu_name = torch.cuda.get_device_name(0)
    if args.required_gpu_name.lower() not in gpu_name.lower():
        raise RuntimeError(f"required GPU {args.required_gpu_name!r}, saw {gpu_name!r}")

    sweep_terminal_path = args.sweep_root / "seed_sweep_terminal.json"
    seeds = validate_sweep_terminal(json.loads(sweep_terminal_path.read_text()))
    manifest_path = args.data_root / "real_division_patch_manifest.json"
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise ValueError("real-division manifest changed")
    manifest = json.loads(manifest_path.read_text())
    device = torch.device("cuda:0")
    selection_patches, selection_targets, _, selection_inventory = load_role(
        args.data_root, manifest, "selection", device
    )

    candidates: list[dict[str, Any]] = []
    admitted_scores: list[torch.Tensor] = []
    for seed in seeds:
        run_root = args.sweep_root / f"seed-{seed}"
        terminal_path = run_root / "real_division_gate_terminal.json"
        if not terminal_path.is_file():
            candidates.append({"seed": seed, "status": "missing_run_terminal"})
            continue
        terminal = json.loads(terminal_path.read_text())
        if not (
            terminal.get("schema_version") == 1
            and terminal.get("run_id") == TRAIN_RUN_ID
            and terminal.get("manifest_sha256") == MANIFEST_SHA256
            and terminal.get("competition_test_data_read") is False
            and terminal.get("final_probe_opened") is False
            and terminal.get("public_leaderboard_used_for_selection") is False
            and terminal.get("submission_created") is False
        ):
            raise ValueError(f"seed {seed} run terminal is ineligible")
        for fold in FOLDS:
            worker = terminal.get("folds", {}).get(fold, {})
            checkpoint = run_root / fold / "division_model.pt"
            if not (
                worker.get("status") == "completed"
                and worker.get("parameter_count") == EXPECTED_PARAMETER_COUNT
                and worker.get("trainable_parameters") == 25_178_047
                and worker.get("final_probe_opened") is False
                and worker.get("model_sha256") == sha256_file(checkpoint)
            ):
                raise ValueError(f"seed {seed} fold {fold} checkpoint is ineligible")
            model = MultiscaleContextualPairFusionAssociationModel().to(device)
            model.load_state_dict(
                torch.load(checkpoint, map_location=device, weights_only=True), strict=True
            )
            model.requires_grad_(False).eval()
            raw_scores = predict(
                model, selection_patches, batch_size=args.batch_size
            ).detach()
            pooled = threshold_metrics(selection_targets, raw_scores)
            by_embryo = embryo_metrics(
                selection_targets, raw_scores, selection_inventory
            )
            try:
                frozen = select_frozen_threshold(selection_targets, raw_scores)
            except RuntimeError:
                frozen = None
            admitted = individual_admitted(pooled, by_embryo, frozen)
            rank_scores = movie_rank_percentiles(raw_scores, selection_inventory)
            row = {
                "seed": seed,
                "fold": fold,
                "status": "admitted" if admitted else "rejected",
                "model_sha256": sha256_file(checkpoint),
                "parameter_count": EXPECTED_PARAMETER_COUNT,
                "trainable_parameters": 25_178_047,
                "selection": pooled,
                "selection_by_embryo": by_embryo,
                "selection_frozen_threshold_diagnostic": frozen,
                "movie_top": movie_top_metrics(
                    selection_targets, raw_scores, selection_inventory
                ),
                "absolute_threshold_authorized": False,
            }
            candidates.append(row)
            if admitted:
                admitted_scores.append(rank_scores)
            del model, raw_scores
            torch.cuda.empty_cache()

    stronger_individuals = [
        {
            "seed": int(row["seed"]),
            "fold": str(row["fold"]),
            "model_sha256": str(row["model_sha256"]),
            "selection_average_precision": float(
                row["selection"]["average_precision"]
            ),
            "selection_gain_vs_reference": float(
                row["selection"]["average_precision"]
            )
            - REFERENCE_SELECTION_AP,
        }
        for row in candidates
        if stronger_than_reference(row)
    ]
    stronger_individuals.sort(
        key=lambda row: (
            -float(row["selection_average_precision"]),
            int(row["seed"]),
            str(row["fold"]),
        )
    )
    ensemble = None
    ensemble_eligible = False
    if len(admitted_scores) >= 2:
        scores = torch.stack(admitted_scores).mean(dim=0)
        pooled = threshold_metrics(selection_targets, scores)
        by_embryo = embryo_metrics(selection_targets, scores, selection_inventory)
        try:
            frozen = select_frozen_threshold(selection_targets, scores)
        except RuntimeError:
            frozen = None
        best_individual_ap = max(
            float(row["selection"]["average_precision"])
            for row in candidates
            if row.get("status") == "admitted"
        )
        ensemble_eligible = bool(
            float(pooled["average_precision"])
            >= best_individual_ap + MINIMUM_ENSEMBLE_AP_GAIN
            and all(
                float(metrics["average_precision"])
                >= MINIMUM_ENSEMBLE_EMBRYO_AP
                for metrics in by_embryo.values()
            )
            and frozen is not None
            and int(frozen["fp"]) == 0
            and int(frozen["tp"]) >= 2
        )
        ensemble = {
            "policy": "equal average of within-movie percentile ranks from every independently admitted model",
            "model_count": len(admitted_scores),
            "selection": pooled,
            "selection_by_embryo": by_embryo,
            "selection_frozen_threshold_diagnostic": frozen,
            "movie_top": movie_top_metrics(
                selection_targets, scores, selection_inventory
            ),
            "best_individual_average_precision": best_individual_ap,
            "minimum_required_average_precision": (
                best_individual_ap + MINIMUM_ENSEMBLE_AP_GAIN
            ),
            "absolute_threshold_authorized": False,
        }

    development_eligible = bool(stronger_individuals or ensemble_eligible)
    if ensemble_eligible:
        status = "ensemble_eligible_for_development_probe"
    elif stronger_individuals:
        status = "stronger_individuals_eligible_for_development_probe"
    else:
        status = "rejected_at_selection"
    payload = {
        "schema_version": 1,
        "status": status,
        "run_id": RUN_ID,
        "gpu_name": gpu_name,
        "sweep_terminal_sha256": sha256_file(sweep_terminal_path),
        "manifest_sha256": MANIFEST_SHA256,
        "candidate_model_count": sum(
            1 for row in candidates if "selection" in row
        ),
        "admitted_model_count": len(admitted_scores),
        "individual_admission": {
            "minimum_pooled_average_precision": MINIMUM_POOLED_AP,
            "minimum_each_embryo_average_precision": MINIMUM_EMBRYO_AP,
            "minimum_zero_false_positive_true_positives": 2,
            "reference_model_sha256": REFERENCE_MODEL_SHA256,
            "reference_selection_average_precision": REFERENCE_SELECTION_AP,
            "minimum_gain_vs_reference_for_development_probe": MINIMUM_INDIVIDUAL_AP_GAIN,
        },
        "ensemble_admission": {
            "minimum_average_precision_gain_vs_best_individual": MINIMUM_ENSEMBLE_AP_GAIN,
            "minimum_each_embryo_average_precision": MINIMUM_ENSEMBLE_EMBRYO_AP,
            "weights_searched": False,
            "equal_rank_policy_precommitted": True,
        },
        "candidates": candidates,
        "stronger_individuals": stronger_individuals,
        "ensemble": ensemble,
        "ensemble_eligible_for_development_probe": ensemble_eligible,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "final_probe_opened": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
        "authorized_for_development_probe": development_eligible,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if not development_eligible:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
