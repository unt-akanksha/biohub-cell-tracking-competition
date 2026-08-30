#!/usr/bin/env python
"""Score one precommitted overnight seed policy on the train-only probe.

The selection evaluator decides the policy before this script reads any probe
frame. If its equal-rank ensemble qualifies, every independently admitted
member is used. Otherwise only the strongest qualifying individual is used.
No model weights, thresholds, or subsets are searched on the probe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import zarr

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.score_competition_division_probe import (
    VOXEL_SIZE_ZYX_UM,
    atomic_json,
    geometry_division_score,
    ranking_metrics,
    score_patches,
    validate_inputs,
    verify_cache,
)
from research.temporal_contrastive.evaluate_real_division_seed_ensemble import (
    REFERENCE_SELECTION_AP,
    RUN_ID as ENSEMBLE_RUN_ID,
    individual_admitted,
    sha256_file,
    stronger_than_reference,
)
from research.temporal_contrastive.multiscale_contextual_pair_fusion import (
    EXPECTED_PARAMETER_COUNT,
    MultiscaleContextualPairFusionAssociationModel,
)
from research.temporal_contrastive.patch_model import sample_physical_patches


RUN_ID = "competition-real-division-seed-ensemble-probe-v1"
ELIGIBLE_STATUSES = {
    "ensemble_eligible_for_development_probe",
    "stronger_individuals_eligible_for_development_probe",
}


def _candidate_identity(row: dict[str, Any]) -> tuple[int, str, str]:
    return int(row["seed"]), str(row["fold"]), str(row["model_sha256"])


def _candidate_is_admitted(row: dict[str, Any]) -> bool:
    return bool(
        row.get("status") == "admitted"
        and individual_admitted(
            row.get("selection", {}),
            row.get("selection_by_embryo", {}),
            row.get("selection_frozen_threshold_diagnostic"),
        )
    )


def select_precommitted_members(
    terminal: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Return the one policy authorized before the development probe opens."""
    if not (
        terminal.get("schema_version") == 1
        and terminal.get("run_id") == ENSEMBLE_RUN_ID
        and terminal.get("status") in ELIGIBLE_STATUSES
        and terminal.get("authorized_for_development_probe") is True
        and terminal.get("competition_test_data_read") is False
        and terminal.get("final_probe_opened") is False
        and terminal.get("public_code_copied") is False
        and terminal.get("public_predictions_copied") is False
        and terminal.get("public_leaderboard_used_for_selection") is False
        and terminal.get("submission_created") is False
        and terminal.get("authorized_for_submission") is False
    ):
        raise ValueError("overnight ensemble terminal is ineligible for the probe")

    admitted = [
        row for row in terminal.get("candidates", []) if _candidate_is_admitted(row)
    ]
    identities = [_candidate_identity(row) for row in admitted]
    if len(identities) != len(set(identities)):
        raise ValueError("admitted overnight member identities are not unique")
    hashes = [identity[2] for identity in identities]
    if len(hashes) != len(set(hashes)):
        raise ValueError("byte-identical checkpoints cannot both vote")

    if terminal.get("ensemble_eligible_for_development_probe") is True:
        ensemble = terminal.get("ensemble") or {}
        if not (
            terminal.get("status") == "ensemble_eligible_for_development_probe"
            and len(admitted) >= 2
            and ensemble.get("model_count") == len(admitted)
            and ensemble.get("absolute_threshold_authorized") is False
            and ensemble.get("policy")
            == "equal average of within-movie percentile ranks from every independently admitted model"
        ):
            raise ValueError("authorized equal-rank ensemble changed")
        return "equal_rank_admitted_ensemble", admitted

    stronger = [row for row in admitted if stronger_than_reference(row)]
    stronger.sort(
        key=lambda row: (
            -float(row["selection"]["average_precision"]),
            int(row["seed"]),
            str(row["fold"]),
        )
    )
    declared = terminal.get("stronger_individuals", [])
    if not (
        terminal.get("status")
        == "stronger_individuals_eligible_for_development_probe"
        and terminal.get("ensemble_eligible_for_development_probe") is False
        and stronger
        and declared
        and _candidate_identity(stronger[0])
        == (
            int(declared[0]["seed"]),
            str(declared[0]["fold"]),
            str(declared[0]["model_sha256"]),
        )
    ):
        raise ValueError("strongest authorized individual changed")
    return "strongest_individual_rank", [stronger[0]]


def movie_rank_percentiles(raw_scores: np.ndarray, stems: list[str]) -> np.ndarray:
    """Convert every member to a calibration-free rank within each movie."""
    values = np.asarray(raw_scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(stems):
        raise ValueError("member scores and movie stems must be aligned")
    ranked = np.empty_like(values)
    for stem in sorted(set(stems)):
        indices = np.asarray(
            [index for index, value in enumerate(stems) if value == stem]
        )
        for member_index in range(values.shape[0]):
            order = np.argsort(-values[member_index, indices], kind="stable")
            percentiles = (
                np.ones(1, dtype=np.float64)
                if len(indices) == 1
                else np.linspace(1.0, 0.0, len(indices), dtype=np.float64)
            )
            local = np.empty(len(indices), dtype=np.float64)
            local[order] = percentiles
            ranked[member_index, indices] = local
    return ranked


def resolve_member_paths(
    members: list[dict[str, Any]], sweep_root: Path
) -> list[Path]:
    paths = []
    for member in members:
        path = (
            sweep_root
            / f"seed-{int(member['seed'])}"
            / str(member["fold"])
            / "division_model.pt"
        )
        if not path.is_file() or sha256_file(path) != member["model_sha256"]:
            raise ValueError(f"overnight checkpoint changed: {path}")
        paths.append(path)
    return paths


def load_models(paths: list[Path], device: torch.device) -> list[torch.nn.Module]:
    models = []
    for path in paths:
        model = MultiscaleContextualPairFusionAssociationModel().to(device)
        model.load_state_dict(
            torch.load(path, map_location=device, weights_only=True), strict=True
        )
        if (
            sum(parameter.numel() for parameter in model.parameters())
            != EXPECTED_PARAMETER_COUNT
        ):
            raise RuntimeError("overnight model parameter inventory changed")
        models.append(model.requires_grad_(False).eval())
    return models


def build_probe_rows(
    inventory: dict[str, Any],
    cache_root: Path,
    models: list[torch.nn.Module],
    *,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    member_scores: list[list[float]] = [[] for _ in models]
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
            scores = score_patches(
                models, patches, device=device, batch_size=batch_size
            )
            for member_index, values in enumerate(scores):
                member_scores[member_index].extend(float(value) for value in values)
            for candidate in candidates:
                row = {
                    "stem": stem,
                    "timepoint": timepoint,
                    "parent_id": int(candidate["parent_id"]),
                    "existing_child_id": int(candidate["existing_child_id"]),
                    "second_child_id": int(candidate["second_child_id"]),
                    "parent_distance_um": float(candidate["parent_distance_um"]),
                    "sister_distance_um": float(candidate["sister_distance_um"]),
                    "existing_distance_um": float(candidate["existing_distance_um"]),
                    "daughter_midpoint_distance_um": float(
                        candidate["daughter_midpoint_distance_um"]
                    ),
                    "daughter_opposition_cosine": float(
                        candidate["daughter_opposition_cosine"]
                    ),
                    "daughter_step_ratio": float(candidate["daughter_step_ratio"]),
                    "parent_velocity_um": candidate["parent_velocity_um"],
                    "constant_velocity_midpoint_error_um": candidate[
                        "constant_velocity_midpoint_error_um"
                    ],
                    "safe_recovery_positive": bool(
                        candidate["safe_recovery_positive"]
                    ),
                }
                row["geometry_division_score"] = geometry_division_score(row)
                rows.append(row)
            del patches

    raw = np.asarray(member_scores, dtype=np.float64)
    ranked = movie_rank_percentiles(raw, [str(row["stem"]) for row in rows])
    ensemble = ranked.mean(axis=0)
    for index, row in enumerate(rows):
        row["member_logits"] = [
            float(raw[member, index]) for member in range(len(models))
        ]
        row["member_rank_percentiles"] = [
            float(ranked[member, index]) for member in range(len(models))
        ]
        row["ensemble_logit"] = float(ensemble[index])
    if len(rows) != 225 or sum(row["safe_recovery_positive"] for row in rows) != 3:
        raise RuntimeError("competition division probe row inventory changed")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--ensemble-terminal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--required-gpu-name", default="A10G")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("overnight development probe requires one Antelume GPU")
    gpu_name = torch.cuda.get_device_name(0)
    if args.required_gpu_name.lower() not in gpu_name.lower():
        raise RuntimeError(f"required GPU {args.required_gpu_name!r}, saw {gpu_name!r}")

    terminal = json.loads(args.ensemble_terminal.read_text(encoding="utf-8"))
    evaluator_source = Path(__file__).with_name(
        "evaluate_real_division_seed_ensemble.py"
    )
    sweep_terminal = args.sweep_root / "seed_sweep_terminal.json"
    if not (
        terminal.get("evaluation_source_sha256") == sha256_file(evaluator_source)
        and terminal.get("sweep_terminal_sha256") == sha256_file(sweep_terminal)
    ):
        raise ValueError("overnight selection evidence or evaluator changed")
    policy, members = select_precommitted_members(terminal)
    paths = resolve_member_paths(members, args.sweep_root)
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    cache_manifest = json.loads(
        (args.cache_root / "probe_cache_manifest.json").read_text(encoding="utf-8")
    )
    validate_inputs(inventory, cache_manifest)
    verify_cache(args.cache_root, cache_manifest)
    device = torch.device("cuda:0")
    models = load_models(paths, device)
    rows = build_probe_rows(
        inventory,
        args.cache_root,
        models,
        device=device,
        batch_size=args.batch_size,
    )
    result = {
        "schema_version": 1,
        "status": "development_probe_complete",
        "run_id": RUN_ID,
        "gpu_name": gpu_name,
        "selection_terminal_sha256": sha256_file(args.ensemble_terminal),
        "selection_policy": policy,
        "member_count": len(members),
        "members": [
            {
                "seed": int(row["seed"]),
                "fold": str(row["fold"]),
                "model_sha256": str(row["model_sha256"]),
                "selection_average_precision": float(
                    row["selection"]["average_precision"]
                ),
            }
            for row in members
        ],
        "metrics": {
            "equal_rank_ensemble": ranking_metrics(rows, "ensemble_logit"),
            "geometry": ranking_metrics(rows, "geometry_division_score"),
        },
        "rows": rows,
        "absolute_threshold_used": False,
        "weights_searched_on_probe": False,
        "model_subset_searched_on_probe": False,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "final_probe_opened": True,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_ranked_consensus_development_evaluation": True,
        "authorized_for_submission": False,
    }
    atomic_json(args.output, result)
    print(json.dumps(result["metrics"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
