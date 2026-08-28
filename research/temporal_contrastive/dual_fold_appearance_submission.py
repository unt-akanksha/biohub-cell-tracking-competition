#!/usr/bin/env python
"""Build an accepted appearance candidate with two transition-safe GPU shards.

The program creates a local ``submission.csv`` candidate but has no Kaggle
submission command. It requires hash-bound exact acceptance evidence and fails
unless exactly two CUDA devices are visible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import rerank_submission as rerank
    from appearance_blend import (
        blend_movie_pair_scores,
        division_logits_for_movie,
        extract_movie_embeddings,
        extract_reciprocal_movie_embeddings,
        reciprocal_movie_evidence,
    )
    from appearance_family import (
        CONTEXTUAL_PAIR_FUSION_FAMILY,
        MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        PAIR_FUSION_FAMILY,
        appearance_evidence_for_movie,
        build_appearance_model,
        candidate_appearance_family,
        verify_appearance_metadata,
    )
    from dual_fold_processed_acceptance import FROZEN_ASSOCIATION_CONFIGURATION
    from dual_fold_rerank_submission import embryo_prefix
    from submission_sharding import (
        DEFAULT_INFERENCE_HARD_STOP_SECONDS,
        KAGGLE_GPU_NOTEBOOK_MAX_SECONDS,
        WORKER_TERMINATION_GRACE_SECONDS,
        build_movie_shards,
        terminate_and_reap_processes,
        validate_inference_hard_stop,
        visible_cuda_tokens,
    )
except ModuleNotFoundError:
    from research.submission_sharding import (
        DEFAULT_INFERENCE_HARD_STOP_SECONDS,
        KAGGLE_GPU_NOTEBOOK_MAX_SECONDS,
        WORKER_TERMINATION_GRACE_SECONDS,
        build_movie_shards,
        terminate_and_reap_processes,
        validate_inference_hard_stop,
        visible_cuda_tokens,
    )
    from research.temporal_contrastive.appearance_blend import (
        blend_movie_pair_scores,
        division_logits_for_movie,
        extract_movie_embeddings,
        extract_reciprocal_movie_embeddings,
        reciprocal_movie_evidence,
    )
    from research.temporal_contrastive.appearance_family import (
        CONTEXTUAL_PAIR_FUSION_FAMILY,
        MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        PAIR_FUSION_FAMILY,
        appearance_evidence_for_movie,
        build_appearance_model,
        candidate_appearance_family,
        verify_appearance_metadata,
    )
    from research.trackastra_graph import rerank_submission as rerank
    from research.trackastra_graph.dual_fold_processed_acceptance import (
        FROZEN_ASSOCIATION_CONFIGURATION,
    )
    from research.trackastra_graph.dual_fold_rerank_submission import (
        embryo_prefix,
    )


FOLD_BY_PREFIX = {"44b6": "target_44b6", "6bba": "target_6bba"}
# The 19.2M-parameter encoder is about 2.56x the original staged 7.5M model.
# Keep the whole-movie LPT estimate conservative so one GPU is not assigned
# most of the node-encoding work even when pair-product counts look balanced.
APPEARANCE_NODE_COST = 12_288.0
PAIR_FUSION_PAIR_COST = 1_024.0


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_acceptance(
    path: Path,
    trackastra_dirs: dict[str, Path],
    appearance_models: dict[str, Path],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        model_family = candidate_appearance_family(
            str(payload.get("candidate_family", ""))
        )
    except ValueError as error:
        raise RuntimeError("appearance evidence is not an accepted exact candidate") from error
    if not (
        payload.get("status") == "accepted"
        and payload.get("evaluation_kind") == "exact_processed_dual_fold_acceptance"
        and payload.get("exact_processed_gate_passed") is True
        and payload.get("appearance_family", model_family) == model_family
        and payload.get("public_leaderboard_used_for_selection") is False
        and payload.get("competition_submission_performed") is False
    ):
        raise RuntimeError("appearance evidence is not an accepted exact candidate")
    if payload.get("association_configuration") != FROZEN_ASSOCIATION_CONFIGURATION:
        raise RuntimeError("accepted association configuration changed")
    expected_trackastra = payload.get("models")
    expected_appearance = payload.get("appearance_models")
    blends = payload.get("appearance_blend")
    folds = set(FOLD_BY_PREFIX.values())
    if not (
        isinstance(expected_trackastra, dict)
        and set(expected_trackastra) == folds
        and isinstance(expected_appearance, dict)
        and set(expected_appearance) == folds
        and isinstance(blends, dict)
        and set(blends) == folds
    ):
        raise RuntimeError("accepted evidence omits a reciprocal model or blend")
    for fold in folds:
        if sha256_file(trackastra_dirs[fold] / "model.pt") != expected_trackastra[
            fold
        ].get("model_sha256"):
            raise RuntimeError(f"accepted Trackastra hash mismatch: {fold}")
        if sha256_file(appearance_models[fold]) != expected_appearance[fold].get(
            "model_sha256"
        ):
            raise RuntimeError(f"accepted appearance hash mismatch: {fold}")
        trackastra_source_valid = bool(
            int(expected_trackastra[fold].get("best_step", 0)) > 0
            or (
                int(expected_trackastra[fold].get("best_step", -1)) == 0
                and expected_trackastra[fold].get("source_policy")
                == "predeclared_pretrained_control"
            )
        )
        try:
            fold_family = verify_appearance_metadata(expected_appearance[fold])
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"accepted reciprocal evidence is invalid: {fold}"
            ) from error
        if not (
            trackastra_source_valid
            and int(expected_appearance[fold].get("best_step", 0)) > 0
            and fold_family == model_family
            and float(blends[fold].get("appearance_weight", 0.0)) > 0.0
            and float(blends[fold].get("division_weight", 0.0)) >= 0.0
            and blends[fold].get("ensemble_mode")
            in {"target_only", "reciprocal_mean"}
            and float(blends[fold].get("appearance_temperature", 0.0)) == 0.10
        ):
            raise RuntimeError(f"accepted reciprocal evidence is invalid: {fold}")
    payload["appearance_family"] = model_family
    return payload


def hybrid_configuration(selected: dict[str, Any]) -> rerank.HybridLinkConfig:
    return rerank.HybridLinkConfig(
        edge_threshold=float(selected["edge_threshold"]),
        base_lock_probability=float(selected["base_lock_probability"]),
        base_keep_probability=float(selected["base_keep_probability"]),
        base_bonus=float(selected["base_bonus"]),
        division_threshold=float(selected["division_threshold"]),
        division_ratio=float(selected["division_ratio"]),
        base_division_keep_probability=float(
            selected["base_division_keep_probability"]
        ),
    )


def appearance_movie_inference_weight(
    video: Any,
    *,
    encoder_count: int = 1,
    pair_fusion: bool = False,
) -> float:
    """Estimate transformer pairs plus the per-node 3D encoder workload."""

    if encoder_count not in {1, 2}:
        raise ValueError("appearance encoder_count must be one or two")
    pair_products = 0
    for timepoint, source_ids in video.ids_by_time.items():
        targets = video.ids_by_time.get(timepoint + 1)
        if targets is not None:
            pair_products += len(source_ids) * len(targets)
    pair_cost = PAIR_FUSION_PAIR_COST if pair_fusion else 1.0
    return float(
        pair_cost * max(pair_products, 1)
        + encoder_count * APPEARANCE_NODE_COST * len(video.node_ids)
    )


def movie_transition_starts(video: Any) -> tuple[int, ...]:
    """Return every consecutive transition represented by one movie."""

    return tuple(
        int(timepoint)
        for timepoint in sorted(video.ids_by_time)
        if timepoint + 1 in video.ids_by_time
        and len(video.ids_by_time[timepoint]) > 0
        and len(video.ids_by_time[timepoint + 1]) > 0
    )


def transition_block_weight(
    video: Any,
    transition_starts: tuple[int, ...],
    *,
    encoder_count: int,
    pair_fusion: bool,
) -> float:
    """Estimate one contiguous transition block without double-counting frames."""

    starts = tuple(map(int, transition_starts))
    available = set(movie_transition_starts(video))
    if not starts or len(set(starts)) != len(starts) or not set(starts) <= available:
        raise ValueError("transition block is empty, duplicated, or outside the movie")
    frames = set(starts) | {timepoint + 1 for timepoint in starts}
    nodes = sum(len(video.ids_by_time[timepoint]) for timepoint in frames)
    pair_products = sum(
        len(video.ids_by_time[timepoint])
        * len(video.ids_by_time[timepoint + 1])
        for timepoint in starts
    )
    pair_cost = PAIR_FUSION_PAIR_COST if pair_fusion else 1.0
    return float(
        pair_cost * max(pair_products, 1)
        + encoder_count * APPEARANCE_NODE_COST * nodes
    )


def transition_subvideo(video: Any, transition_starts: tuple[int, ...]) -> Any:
    """Slice a movie to complete frames for disjoint consecutive transitions."""

    starts = tuple(sorted(map(int, transition_starts)))
    available = set(movie_transition_starts(video))
    if not starts or len(set(starts)) != len(starts) or not set(starts) <= available:
        raise ValueError("transition slice is empty, duplicated, or outside the movie")
    frames = set(starts) | {timepoint + 1 for timepoint in starts}
    node_mask = np.asarray([int(value) in frames for value in video.times], dtype=bool)
    edge_mask = np.asarray(
        [
            video.time_by_id[int(source)] in starts
            and video.time_by_id[int(target)]
            == video.time_by_id[int(source)] + 1
            for source, target in video.edges.tolist()
        ],
        dtype=bool,
    )
    sliced = type(video)(
        stem=video.stem,
        node_ids=video.node_ids[node_mask],
        times=video.times[node_mask],
        coords_voxel=video.coords_voxel[node_mask],
        edges=video.edges[edge_mask],
        edge_probabilities=video.edge_probabilities[edge_mask],
    )
    if set(movie_transition_starts(sliced)) != set(starts):
        raise RuntimeError("transition slice did not preserve its exact inventory")
    return sliced


def _work_plan_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_transition_work_plan(
    videos: dict[str, Any],
    cuda_tokens: tuple[str, ...],
    *,
    encoder_count_by_stem: dict[str, int],
    pair_fusion: bool,
    dominant_fraction: float = 0.60,
) -> dict[str, Any]:
    """Split one dominant movie at a frame boundary, then LPT-place others."""

    if len(cuda_tokens) != 2 or len(set(cuda_tokens)) != 2:
        raise ValueError("transition inference requires two unique CUDA tokens")
    if set(encoder_count_by_stem) != set(videos):
        raise ValueError("encoder counts must cover every movie")
    inventories = {
        stem: movie_transition_starts(video) for stem, video in videos.items()
    }
    if any(not starts for starts in inventories.values()):
        raise ValueError("every submission movie must contain a transition")
    movie_weights = {
        stem: appearance_movie_inference_weight(
            video,
            encoder_count=encoder_count_by_stem[stem],
            pair_fusion=pair_fusion,
        )
        for stem, video in videos.items()
    }
    dominant = max(sorted(videos), key=lambda stem: movie_weights[stem])
    total_weight = sum(movie_weights.values())

    def whole_unit(stem: str) -> dict[str, Any]:
        return {
            "unit_id": f"{stem}@all",
            "stem": stem,
            "transition_starts": list(inventories[stem]),
            "weight": movie_weights[stem],
        }

    partition_kind = "whole_movie_lpt_v1"
    dominant_split: dict[str, Any] | None = None
    if (
        movie_weights[dominant] > dominant_fraction * total_weight
        and len(inventories[dominant]) >= 2
    ):
        best: tuple[tuple[float, float, int, int], list[list[dict[str, Any]]]] | None = None
        starts = inventories[dominant]
        other_stems = sorted(
            (stem for stem in videos if stem != dominant),
            key=lambda stem: (-movie_weights[stem], stem),
        )
        for cut in range(1, len(starts)):
            blocks = (starts[:cut], starts[cut:])
            block_weights = tuple(
                transition_block_weight(
                    videos[dominant],
                    block,
                    encoder_count=encoder_count_by_stem[dominant],
                    pair_fusion=pair_fusion,
                )
                for block in blocks
            )
            for orientation in (0, 1):
                ordered = (
                    (blocks[0], block_weights[0]),
                    (blocks[1], block_weights[1]),
                )
                if orientation:
                    ordered = (ordered[1], ordered[0])
                units = [
                    [
                        {
                            "unit_id": f"{dominant}@{block[0]}-{block[-1]}",
                            "stem": dominant,
                            "transition_starts": list(block),
                            "weight": weight,
                        }
                    ]
                    for block, weight in ordered
                ]
                loads = [ordered[0][1], ordered[1][1]]
                for stem in other_stems:
                    shard_index = min(range(2), key=lambda index: (loads[index], index))
                    units[shard_index].append(whole_unit(stem))
                    loads[shard_index] += movie_weights[stem]
                objective = (
                    max(loads),
                    abs(loads[0] - loads[1]),
                    cut,
                    orientation,
                )
                if best is None or objective < best[0]:
                    best = (objective, units)
        if best is None:
            raise RuntimeError("dominant transition split search produced no plan")
        assigned_units = best[1]
        partition_kind = "dominant_movie_transition_split_v1"
        dominant_split = {
            "stem": dominant,
            "movie_weight": movie_weights[dominant],
            "movie_fraction": movie_weights[dominant] / total_weight,
        }
    else:
        shards = build_movie_shards(
            sorted(videos), cuda_tokens, movie_weights=movie_weights
        )
        assigned_units = [
            [whole_unit(stem) for stem in shard.movie_ids] for shard in shards
        ]

    planned = {
        (str(unit["stem"]), int(timepoint))
        for units in assigned_units
        for unit in units
        for timepoint in unit["transition_starts"]
    }
    expected = {
        (stem, timepoint)
        for stem, starts in inventories.items()
        for timepoint in starts
    }
    if planned != expected or sum(
        len(unit["transition_starts"])
        for units in assigned_units
        for unit in units
    ) != len(expected):
        raise RuntimeError("transition work plan does not cover every transition once")
    plan = {
        "schema_version": 1,
        "partition_kind": partition_kind,
        "dominant_split": dominant_split,
        "movie_weights": movie_weights,
        "movie_transition_inventory": {
            stem: list(starts) for stem, starts in inventories.items()
        },
        "shards": [
            {
                "shard_index": index,
                "cuda_token": cuda_tokens[index],
                "units": units,
                "total_weight": sum(float(unit["weight"]) for unit in units),
            }
            for index, units in enumerate(assigned_units)
        ],
    }
    plan["work_plan_sha256"] = _work_plan_sha256(plan)
    return plan


def worker(args: argparse.Namespace) -> None:
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"isolated appearance inference worker requires one GPU, saw {torch.cuda.device_count()}"
        )
    all_videos = rerank.read_submission(args.base_submission)
    started = time.monotonic()
    if args.work_plan is not None:
        plan = json.loads(args.work_plan.read_text(encoding="utf-8"))
        recorded_hash = str(plan.get("work_plan_sha256", ""))
        unhashed = dict(plan)
        unhashed.pop("work_plan_sha256", None)
        if recorded_hash != _work_plan_sha256(unhashed):
            raise RuntimeError("worker transition plan hash mismatch")
        matching = [
            row
            for row in plan.get("shards", [])
            if int(row.get("shard_index", -1)) == args.shard_index
        ]
        if len(matching) != 1:
            raise ValueError("worker transition shard is missing or ambiguous")
        work_units = matching[0].get("units")
        if not isinstance(work_units, list) or not work_units:
            raise ValueError("worker transition shard has no work units")
    else:
        requested_legacy = tuple(item for item in args.datasets.split(",") if item)
        if not requested_legacy or len(requested_legacy) != len(
            set(requested_legacy)
        ):
            raise ValueError("worker movies must be non-empty and unique")
        work_units = [
            {
                "unit_id": f"{stem}@all",
                "stem": stem,
                "transition_starts": list(movie_transition_starts(all_videos[stem])),
            }
            for stem in requested_legacy
        ]
    unit_ids = [str(unit.get("unit_id", "")) for unit in work_units]
    requested = tuple(sorted({str(unit.get("stem", "")) for unit in work_units}))
    if (
        not requested
        or "" in requested
        or not set(requested).issubset(all_videos)
        or "" in unit_ids
        or len(set(unit_ids)) != len(unit_ids)
    ):
        raise ValueError("worker received an invalid transition work inventory")
    videos = {stem: all_videos[stem] for stem in requested}
    raw_videos = rerank.read_raw_graphs(args.base_graph_root, set(videos))
    transfer = rerank.transfer_raw_edge_probabilities(videos, raw_videos)
    acceptance = json.loads(args.acceptance_evidence.read_text(encoding="utf-8"))
    model_family = candidate_appearance_family(acceptance["candidate_family"])
    if acceptance.get("appearance_family", model_family) != model_family:
        raise RuntimeError("worker acceptance evidence mixes appearance families")
    device = torch.device("cuda:0")
    sys.path.insert(0, str(args.trackastra_dir.resolve()))
    from trackastra.model.model import TrackingTransformer

    trackastra_dirs = {
        "target_44b6": args.trackastra_44b6_dir,
        "target_6bba": args.trackastra_6bba_dir,
    }
    appearance_paths = {
        "target_44b6": args.appearance_44b6_model,
        "target_6bba": args.appearance_6bba_model,
    }
    trackastra_models: dict[str, torch.nn.Module] = {}
    appearance_models: dict[str, torch.nn.Module] = {}
    config = hybrid_configuration(acceptance["association_configuration"])
    import zarr

    candidate_edges: dict[str, list[tuple[int, int]]] = {}
    dataset_stats: dict[str, dict[str, Any]] = {}
    extraction: dict[str, Any] = {}

    def appearance_model_for(fold_name: str) -> torch.nn.Module:
        if fold_name not in appearance_models:
            loaded = build_appearance_model(model_family).to(device)
            loaded.load_state_dict(
                torch.load(
                    appearance_paths[fold_name], map_location="cpu", weights_only=True
                ),
                strict=True,
            )
            loaded.eval()
            appearance_models[fold_name] = loaded
        return appearance_models[fold_name]

    for unit in work_units:
        unit_id = str(unit["unit_id"])
        stem = str(unit["stem"])
        transitions = tuple(map(int, unit.get("transition_starts", [])))
        full_video = videos[stem]
        video = transition_subvideo(full_video, transitions)
        fold = FOLD_BY_PREFIX[embryo_prefix(stem)]
        if fold not in trackastra_models:
            trackastra_models[fold] = TrackingTransformer.from_folder(
                trackastra_dirs[fold], map_location="cpu"
            ).to(device)
            trackastra_models[fold].eval()
        track_scores = rerank.predict_movie_scores(
            trackastra_models[fold],
            video,
            device,
            max_tokens=args.max_tokens,
            candidate_radius=args.candidate_radius,
        )
        image = zarr.open_group(
            str(args.image_root / f"{stem}.zarr"), mode="r"
        )["0"]
        blend = acceptance["appearance_blend"][fold]
        ensemble_mode = str(blend["ensemble_mode"])
        if ensemble_mode == "reciprocal_mean":
            peer_fold = next(
                candidate for candidate in FOLD_BY_PREFIX.values() if candidate != fold
            )
            (
                embeddings,
                division_logits,
                peer_embeddings,
                peer_logits,
                shared_extraction,
            ) = extract_reciprocal_movie_embeddings(
                appearance_model_for(fold),
                appearance_model_for(peer_fold),
                video,
                image,
                device,
                node_batch_size=args.node_batch_size,
            )
        else:
            embeddings, division_logits, shared_extraction = extract_movie_embeddings(
                appearance_model_for(fold),
                video,
                image,
                device,
                node_batch_size=args.node_batch_size,
            )
            peer_embeddings = embeddings
            peer_logits = division_logits
        primary_scores = appearance_evidence_for_movie(
            model_family,
            appearance_model_for(fold),
            video,
            embeddings,
            division_logits,
            track_scores,
            image=image,
        )
        primary_divisions = division_logits_for_movie(
            video, division_logits, track_scores
        )
        if ensemble_mode == "reciprocal_mean":
            peer_scores = appearance_evidence_for_movie(
                model_family,
                appearance_model_for(peer_fold),
                video,
                peer_embeddings,
                peer_logits,
                track_scores,
                image=image,
            )
            peer_divisions = division_logits_for_movie(
                video, peer_logits, track_scores
            )
        else:
            peer_scores = primary_scores
            peer_divisions = primary_divisions
        appearance_scores, source_divisions = reciprocal_movie_evidence(
            primary_scores,
            primary_divisions,
            peer_scores,
            peer_divisions,
            mode=ensemble_mode,
        )
        extraction[unit_id] = {
            "stem": stem,
            "transition_starts": list(transitions),
            "ensemble_mode": ensemble_mode,
            "shared": shared_extraction,
        }
        pair_scores = blend_movie_pair_scores(
            track_scores,
            appearance_scores,
            appearance_weight=float(blend["appearance_weight"]),
            appearance_temperature=float(blend["appearance_temperature"]),
            source_division_logits=source_divisions,
            division_weight=float(blend.get("division_weight", 0.0)),
        )
        selected = acceptance["association_configuration"]
        edges = rerank.hybrid_link_movie(
            video,
            pair_scores,
            config,
            submission_edge_probability=float(selected["base_pseudo_probability"]),
            use_stored_edge_probabilities=True,
        )
        rerank.validate_edges(video, edges)
        candidate_edges[unit_id] = edges
        base_set = set(map(tuple, video.edges.tolist()))
        candidate_set = set(edges)
        dataset_stats[unit_id] = {
            "stem": stem,
            "transition_starts": list(transitions),
            "fold": fold,
            "nodes": len(video.node_ids),
            "base_edges": len(base_set),
            "candidate_edges": len(candidate_set),
            "retained_edges": len(base_set & candidate_set),
            "removed_edges": len(base_set - candidate_set),
            "new_edges": len(candidate_set - base_set),
        }
    atomic_json(
        args.worker_output,
        {
            "schema_version": 1,
            "status": "completed",
            "elapsed_seconds": time.monotonic() - started,
            "datasets": list(requested),
            "work_units": work_units,
            "appearance_family": model_family,
            "candidate_edges": {
                unit_id: [[int(source), int(target)] for source, target in edges]
                for unit_id, edges in candidate_edges.items()
            },
            "dataset_stats": dataset_stats,
            "extraction": extraction,
            "edge_probability_transfer": transfer,
        },
    )


def orchestrate(args: argparse.Namespace) -> None:
    started = time.monotonic()
    validate_inference_hard_stop(args.hard_stop_seconds)
    detected = torch.cuda.device_count()
    if detected != 2:
        raise RuntimeError(
            f"appearance submission inference requires exactly 2 GPUs, saw {detected}"
        )
    trackastra_dirs = {
        "target_44b6": args.trackastra_44b6_dir,
        "target_6bba": args.trackastra_6bba_dir,
    }
    appearance_models = {
        "target_44b6": args.appearance_44b6_model,
        "target_6bba": args.appearance_6bba_model,
    }
    acceptance = load_acceptance(
        args.acceptance_evidence, trackastra_dirs, appearance_models
    )
    videos = rerank.read_submission(args.base_submission)
    encoder_count_by_stem = {
        stem: (
            2
            if acceptance["appearance_blend"][
                FOLD_BY_PREFIX[embryo_prefix(stem)]
            ]["ensemble_mode"]
            == "reciprocal_mean"
            else 1
        )
        for stem in videos
    }
    pair_fusion = acceptance["appearance_family"] in {
        PAIR_FUSION_FAMILY,
        CONTEXTUAL_PAIR_FUSION_FAMILY,
        MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
    }
    plan = build_transition_work_plan(
        videos,
        visible_cuda_tokens(detected_devices=detected),
        encoder_count_by_stem=encoder_count_by_stem,
        pair_fusion=pair_fusion,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    work_plan_path = args.output_dir / "transition_work_plan.json"
    atomic_json(work_plan_path, plan)
    processes = []
    for shard in plan["shards"]:
        shard_index = int(shard["shard_index"])
        output = args.output_dir / f"shard_{shard_index}.json"
        handle = (args.output_dir / f"shard_{shard_index}.log").open(
            "w", encoding="utf-8"
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--base-submission",
            str(args.base_submission),
            "--base-graph-root",
            str(args.base_graph_root),
            "--image-root",
            str(args.image_root),
            "--trackastra-44b6-dir",
            str(args.trackastra_44b6_dir),
            "--trackastra-6bba-dir",
            str(args.trackastra_6bba_dir),
            "--appearance-44b6-model",
            str(args.appearance_44b6_model),
            "--appearance-6bba-model",
            str(args.appearance_6bba_model),
            "--acceptance-evidence",
            str(args.acceptance_evidence),
            "--trackastra-dir",
            str(args.trackastra_dir),
            "--output-dir",
            str(args.output_dir),
            "--work-plan",
            str(work_plan_path),
            "--shard-index",
            str(shard_index),
            "--worker-output",
            str(output),
            "--max-tokens",
            str(args.max_tokens),
            "--candidate-radius",
            str(args.candidate_radius),
            "--node-batch-size",
            str(args.node_batch_size),
        ]
        environment = dict(os.environ)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(shard["cuda_token"]),
                "BIOHUB_SHARD_INDEX": str(shard_index),
                "BIOHUB_SHARD_COUNT": "2",
                "BIOHUB_SHARD_MOVIES": ",".join(
                    sorted({str(unit["stem"]) for unit in shard["units"]})
                ),
                "BIOHUB_SHARD_UNITS": ",".join(
                    str(unit["unit_id"]) for unit in shard["units"]
                ),
            }
        )
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((shard_index, process, handle, output))
    return_codes = {}
    try:
        while len(return_codes) < len(processes):
            for index, process, _handle, _output in processes:
                code = process.poll()
                if code is not None and index not in return_codes:
                    return_codes[index] = int(code)
            if time.monotonic() - started >= args.hard_stop_seconds:
                raise TimeoutError("two-GPU appearance inference exceeded hard stop")
            if len(return_codes) < len(processes):
                time.sleep(3)
    finally:
        terminate_and_reap_processes(
            [process for _index, process, _handle, _output in processes]
        )
        for _index, process, handle, _output in processes:
            handle.close()
    failures = {index: code for index, code in return_codes.items() if code != 0}
    if failures:
        raise RuntimeError(f"appearance inference workers failed: {failures}")
    expected_units_by_shard = {
        int(shard["shard_index"]): {
            str(unit["unit_id"]): unit for unit in shard["units"]
        }
        for shard in plan["shards"]
    }
    candidate_edges_by_stem: dict[str, list[tuple[int, int]]] = {
        stem: [] for stem in videos
    }
    observed_units: set[str] = set()
    extraction: dict[str, Any] = {}
    transfers: dict[str, Any] = {}
    for index, _process, _handle, output in processes:
        payload = json.loads(output.read_text(encoding="utf-8"))
        if not (
            payload.get("schema_version") == 1
            and payload.get("status") == "completed"
            and payload.get("appearance_family") == acceptance["appearance_family"]
        ):
            raise RuntimeError(f"appearance shard family mismatch: {index}")
        expected_units = expected_units_by_shard[index]
        payload_units = {
            str(unit.get("unit_id", "")): unit
            for unit in payload.get("work_units", [])
        }
        if payload_units != expected_units:
            raise RuntimeError(f"appearance shard work inventory mismatch: {index}")
        for unit_id, edges in payload["candidate_edges"].items():
            if unit_id in observed_units or unit_id not in expected_units:
                raise RuntimeError(f"duplicate or unknown transition unit: {unit_id}")
            unit = expected_units[unit_id]
            stem = str(unit["stem"])
            transitions = set(map(int, unit["transition_starts"]))
            normalized = [tuple(map(int, edge)) for edge in edges]
            if any(videos[stem].time_by_id[source] not in transitions for source, _target in normalized):
                raise RuntimeError(f"transition unit emitted an edge outside its block: {unit_id}")
            candidate_edges_by_stem[stem].extend(normalized)
            observed_units.add(unit_id)
        extraction.update(payload["extraction"])
        transfers[str(index)] = payload["edge_probability_transfer"]
    expected_unit_ids = set().union(*[set(rows) for rows in expected_units_by_shard.values()])
    if observed_units != expected_unit_ids:
        raise RuntimeError("appearance shards do not cover every transition unit")
    coverage = tuple(sorted(videos))
    dataset_stats: dict[str, Any] = {}
    for stem, video in videos.items():
        edges = candidate_edges_by_stem[stem]
        if len(edges) != len(set(edges)):
            raise RuntimeError(f"transition shards emitted duplicate edges: {stem}")
        rerank.validate_edges(video, edges)
        base_set = set(map(tuple, video.edges.tolist()))
        candidate_set = set(edges)
        dataset_stats[stem] = {
            "fold": FOLD_BY_PREFIX[embryo_prefix(stem)],
            "nodes": len(video.node_ids),
            "base_edges": len(base_set),
            "candidate_edges": len(candidate_set),
            "retained_edges": len(base_set & candidate_set),
            "removed_edges": len(base_set - candidate_set),
            "new_edges": len(candidate_set - base_set),
        }
    output_path = args.output_dir / "submission.csv"
    rerank.write_submission(output_path, videos, candidate_edges_by_stem)
    changed = sum(
        row["removed_edges"] + row["new_edges"] for row in dataset_stats.values()
    )
    if changed <= 0 or sha256_file(output_path) == sha256_file(args.base_submission):
        raise RuntimeError("appearance candidate is an exact public-base replica")
    report = {
        "schema_version": 1,
        "status": "completed",
        "candidate_family": acceptance["candidate_family"],
        "appearance_family": acceptance["appearance_family"],
        "elapsed_seconds": time.monotonic() - started,
        "inference_hard_stop_seconds": args.hard_stop_seconds,
        "notebook_runtime_reserve_seconds": (
            KAGGLE_GPU_NOTEBOOK_MAX_SECONDS - args.hard_stop_seconds
        ),
        "worker_termination_grace_seconds": WORKER_TERMINATION_GRACE_SECONDS,
        "gpu_count": 2,
        "whole_movie_coverage": list(coverage),
        "transition_partitioned_inference": True,
        "transition_partition_kind": plan["partition_kind"],
        "transition_work_plan_sha256": plan["work_plan_sha256"],
        "shard_plan_sha256": plan["work_plan_sha256"],
        "dominant_movie_split": plan["dominant_split"],
        "shard_estimated_weights": {
            str(shard["shard_index"]): shard["total_weight"]
            for shard in plan["shards"]
        },
        "movie_transition_inventory": plan["movie_transition_inventory"],
        "appearance_node_cost_weight": APPEARANCE_NODE_COST,
        "pair_fusion_pair_cost_weight": (
            PAIR_FUSION_PAIR_COST
            if acceptance["appearance_family"]
            in {
                PAIR_FUSION_FAMILY,
                CONTEXTUAL_PAIR_FUSION_FAMILY,
                MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
            }
            else 1.0
        ),
        "base_submission_sha256": sha256_file(args.base_submission),
        "candidate_submission_sha256": sha256_file(output_path),
        "acceptance_evidence_sha256": sha256_file(args.acceptance_evidence),
        "models": acceptance["models"],
        "appearance_models": acceptance["appearance_models"],
        "appearance_blend": acceptance["appearance_blend"],
        "association_configuration": acceptance["association_configuration"],
        "datasets": dataset_stats,
        "extraction": extraction,
        "edge_probability_transfer_by_shard": transfers,
        "total_changed_edges": changed,
        "nodes_preserved_exactly": True,
        "public_leaderboard_used_for_selection": False,
        "competition_submission_performed": False,
    }
    atomic_json(args.output_dir / "candidate_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    role = result.add_mutually_exclusive_group(required=True)
    role.add_argument("--orchestrate", action="store_true")
    role.add_argument("--worker", action="store_true")
    result.add_argument("--base-submission", type=Path, required=True)
    result.add_argument("--base-graph-root", type=Path, required=True)
    result.add_argument("--image-root", type=Path, required=True)
    result.add_argument("--trackastra-44b6-dir", type=Path, required=True)
    result.add_argument("--trackastra-6bba-dir", type=Path, required=True)
    result.add_argument("--appearance-44b6-model", type=Path, required=True)
    result.add_argument("--appearance-6bba-model", type=Path, required=True)
    result.add_argument("--acceptance-evidence", type=Path, required=True)
    result.add_argument("--trackastra-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--datasets", default="")
    result.add_argument("--work-plan", type=Path)
    result.add_argument("--shard-index", type=int, default=-1)
    result.add_argument("--worker-output", type=Path)
    result.add_argument("--max-tokens", type=int, default=512)
    result.add_argument("--candidate-radius", type=float, default=80.0)
    result.add_argument("--node-batch-size", type=int, default=64)
    result.add_argument(
        "--hard-stop-seconds",
        type=int,
        default=DEFAULT_INFERENCE_HARD_STOP_SECONDS,
        help=(
            "two-GPU inference ceiling; values above 36000 are rejected so the "
            "12-hour Kaggle notebook retains at least two hours for setup and "
            "finalization"
        ),
    )
    return result


def main() -> None:
    args = parser().parse_args()
    if args.worker:
        if args.worker_output is None:
            raise ValueError("--worker-output is required for a worker")
        worker(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
