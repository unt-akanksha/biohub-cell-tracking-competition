#!/usr/bin/env python
"""Build an accepted appearance candidate with two whole-movie GPU shards.

The program creates a local ``submission.csv`` candidate but has no Kaggle
submission command. It requires hash-bound exact acceptance evidence and fails
unless exactly two CUDA devices are visible.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

try:
    import rerank_submission as rerank
    from appearance_blend import (
        appearance_scores_for_movie,
        blend_movie_pair_scores,
        division_logits_for_movie,
        extract_movie_embeddings,
        extract_reciprocal_movie_embeddings,
        reciprocal_movie_evidence,
    )
    from dual_fold_processed_acceptance import FROZEN_ASSOCIATION_CONFIGURATION
    from dual_fold_rerank_submission import embryo_prefix
    from patch_model import PhysicalPatchAssociationModel
    from submission_sharding import (
        build_movie_shards,
        shard_plan_sha256,
        validate_shard_outputs,
        visible_cuda_tokens,
        worker_environment,
    )
except ModuleNotFoundError:
    from research.submission_sharding import (
        build_movie_shards,
        shard_plan_sha256,
        validate_shard_outputs,
        visible_cuda_tokens,
        worker_environment,
    )
    from research.temporal_contrastive.appearance_blend import (
        appearance_scores_for_movie,
        blend_movie_pair_scores,
        division_logits_for_movie,
        extract_movie_embeddings,
        extract_reciprocal_movie_embeddings,
        reciprocal_movie_evidence,
    )
    from research.temporal_contrastive.patch_model import PhysicalPatchAssociationModel
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
    if not (
        payload.get("status") == "accepted"
        and payload.get("evaluation_kind") == "exact_processed_dual_fold_acceptance"
        and payload.get("exact_processed_gate_passed") is True
        and payload.get("candidate_family") == "trackastra_appearance_blend"
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
        if not (
            int(expected_trackastra[fold].get("best_step", 0)) > 0
            and int(expected_appearance[fold].get("best_step", 0)) > 0
            and int(expected_appearance[fold].get("parameter_count", 0))
            == 19_221_954
            and expected_appearance[fold].get("input_channels") == 3
            and expected_appearance[fold].get("temporal_frame_offsets")
            == [-1, 0, 1]
            and expected_appearance[fold].get("checkpoint_weight_source")
            == "optimizer-step exponential moving average"
            and expected_appearance[fold].get("ema_decay") == 0.997
            and expected_appearance[fold].get("link_loss_policy")
            == "all-positive supervised contrastive mean-log-probability"
            and float(blends[fold].get("appearance_weight", 0.0)) > 0.0
            and float(blends[fold].get("division_weight", 0.0)) >= 0.0
            and blends[fold].get("ensemble_mode")
            in {"target_only", "reciprocal_mean"}
            and float(blends[fold].get("appearance_temperature", 0.0)) == 0.10
        ):
            raise RuntimeError(f"accepted reciprocal evidence is invalid: {fold}")
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


def appearance_movie_inference_weight(video: Any, *, encoder_count: int = 1) -> float:
    """Estimate transformer pairs plus the per-node 3D encoder workload."""

    if encoder_count not in {1, 2}:
        raise ValueError("appearance encoder_count must be one or two")
    pair_products = 0
    for timepoint, source_ids in video.ids_by_time.items():
        targets = video.ids_by_time.get(timepoint + 1)
        if targets is not None:
            pair_products += len(source_ids) * len(targets)
    return float(
        max(pair_products, 1)
        + encoder_count * APPEARANCE_NODE_COST * len(video.node_ids)
    )


def worker(args: argparse.Namespace) -> None:
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"isolated appearance inference worker requires one GPU, saw {torch.cuda.device_count()}"
        )
    started = time.monotonic()
    requested = tuple(item for item in args.datasets.split(",") if item)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("worker movies must be non-empty and unique")
    all_videos = rerank.read_submission(args.base_submission)
    if not set(requested).issubset(all_videos):
        raise ValueError("worker received an unknown movie")
    videos = {stem: all_videos[stem] for stem in requested}
    raw_videos = rerank.read_raw_graphs(args.base_graph_root, set(videos))
    transfer = rerank.transfer_raw_edge_probabilities(videos, raw_videos)
    acceptance = json.loads(args.acceptance_evidence.read_text(encoding="utf-8"))
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
    appearance_models: dict[str, PhysicalPatchAssociationModel] = {}
    config = hybrid_configuration(acceptance["association_configuration"])
    import zarr

    candidate_edges: dict[str, list[tuple[int, int]]] = {}
    dataset_stats: dict[str, dict[str, Any]] = {}
    extraction: dict[str, Any] = {}

    def appearance_model_for(fold_name: str) -> PhysicalPatchAssociationModel:
        if fold_name not in appearance_models:
            loaded = PhysicalPatchAssociationModel().to(device)
            loaded.load_state_dict(
                torch.load(
                    appearance_paths[fold_name], map_location="cpu", weights_only=True
                ),
                strict=True,
            )
            loaded.eval()
            appearance_models[fold_name] = loaded
        return appearance_models[fold_name]

    for stem, video in videos.items():
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
        primary_scores = appearance_scores_for_movie(video, embeddings, track_scores)
        primary_divisions = division_logits_for_movie(
            video, division_logits, track_scores
        )
        if ensemble_mode == "reciprocal_mean":
            peer_scores = appearance_scores_for_movie(
                video, peer_embeddings, track_scores
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
        extraction[stem] = {
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
        candidate_edges[stem] = edges
        base_set = set(map(tuple, video.edges.tolist()))
        candidate_set = set(edges)
        dataset_stats[stem] = {
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
            "candidate_edges": {
                stem: [[int(source), int(target)] for source, target in edges]
                for stem, edges in candidate_edges.items()
            },
            "dataset_stats": dataset_stats,
            "extraction": extraction,
            "edge_probability_transfer": transfer,
        },
    )


def orchestrate(args: argparse.Namespace) -> None:
    started = time.monotonic()
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
    weights = {
        stem: appearance_movie_inference_weight(
            video,
            encoder_count=(
                2
                if acceptance["appearance_blend"][
                    FOLD_BY_PREFIX[embryo_prefix(stem)]
                ]["ensemble_mode"]
                == "reciprocal_mean"
                else 1
            ),
        )
        for stem, video in videos.items()
    }
    shards = build_movie_shards(
        sorted(videos),
        visible_cuda_tokens(detected_devices=detected),
        movie_weights=weights,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": 1,
        "shard_plan_sha256": shard_plan_sha256(shards),
        "weights": weights,
        "shards": [
            {
                "shard_index": shard.shard_index,
                "cuda_token": shard.cuda_token,
                "movie_ids": list(shard.movie_ids),
                "total_weight": sum(weights[movie] for movie in shard.movie_ids),
            }
            for shard in shards
        ],
    }
    atomic_json(args.output_dir / "shard_plan.json", plan)
    processes = []
    for shard in shards:
        output = args.output_dir / f"shard_{shard.shard_index}.json"
        handle = (args.output_dir / f"shard_{shard.shard_index}.log").open(
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
            "--datasets",
            ",".join(shard.movie_ids),
            "--worker-output",
            str(output),
            "--max-tokens",
            str(args.max_tokens),
            "--candidate-radius",
            str(args.candidate_radius),
            "--node-batch-size",
            str(args.node_batch_size),
        ]
        process = subprocess.Popen(
            command,
            env=worker_environment(shard),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((shard.shard_index, process, handle, output))
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
        for _index, process, handle, _output in processes:
            if process.poll() is None:
                process.terminate()
            handle.close()
    failures = {index: code for index, code in return_codes.items() if code != 0}
    if failures:
        raise RuntimeError(f"appearance inference workers failed: {failures}")
    observed_by_shard = {}
    candidate_edges: dict[str, list[tuple[int, int]]] = {}
    dataset_stats: dict[str, Any] = {}
    extraction: dict[str, Any] = {}
    transfers: dict[str, Any] = {}
    for index, _process, _handle, output in processes:
        payload = json.loads(output.read_text(encoding="utf-8"))
        observed_by_shard[index] = list(payload["datasets"])
        for stem, edges in payload["candidate_edges"].items():
            if stem in candidate_edges:
                raise RuntimeError(f"duplicate shard movie: {stem}")
            candidate_edges[stem] = [tuple(map(int, edge)) for edge in edges]
        dataset_stats.update(payload["dataset_stats"])
        extraction.update(payload["extraction"])
        transfers[str(index)] = payload["edge_probability_transfer"]
    coverage = validate_shard_outputs(shards, observed_by_shard)
    if set(candidate_edges) != set(videos) or set(coverage) != set(videos):
        raise RuntimeError("appearance shards do not cover every movie exactly once")
    output_path = args.output_dir / "submission.csv"
    rerank.write_submission(output_path, videos, candidate_edges)
    changed = sum(
        row["removed_edges"] + row["new_edges"] for row in dataset_stats.values()
    )
    if changed <= 0 or sha256_file(output_path) == sha256_file(args.base_submission):
        raise RuntimeError("appearance candidate is an exact public-base replica")
    report = {
        "schema_version": 1,
        "status": "completed",
        "candidate_family": "trackastra_appearance_blend",
        "elapsed_seconds": time.monotonic() - started,
        "gpu_count": 2,
        "whole_movie_coverage": list(coverage),
        "shard_plan_sha256": plan["shard_plan_sha256"],
        "appearance_node_cost_weight": APPEARANCE_NODE_COST,
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
    result.add_argument("--worker-output", type=Path)
    result.add_argument("--max-tokens", type=int, default=512)
    result.add_argument("--candidate-radius", type=float, default=80.0)
    result.add_argument("--node-batch-size", type=int, default=64)
    result.add_argument("--hard-stop-seconds", type=int, default=39000)
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
