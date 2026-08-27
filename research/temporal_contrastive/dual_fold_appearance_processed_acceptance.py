#!/usr/bin/env python
"""Materialize one frozen appearance-blended processed candidate.

This exactly-two-GPU inference stage consumes reciprocal models and blend
weights selected on reserved movies. It reads the four processed movie images
but never their ground truth, never tunes a parameter, and never submits.
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
        blend_movie_pair_scores,
        division_logits_for_movie,
        extract_movie_embeddings,
        extract_reciprocal_movie_embeddings,
        reciprocal_movie_evidence,
    )
    from appearance_family import (
        CALIBRATION_RUN_BY_FAMILY,
        CANDIDATE_FAMILY_BY_APPEARANCE,
        COSINE_FAMILY,
        FAMILIES,
        PROCESSED_RUN_BY_FAMILY,
        TRAINING_RUN_BY_FAMILY,
        appearance_evidence_for_movie,
        appearance_metadata,
        build_appearance_model,
        verify_appearance_metadata,
    )
    from dual_fold_processed_acceptance import (
        EXPECTED_STEMS,
        FOLD_BY_PREFIX,
        FROZEN_ASSOCIATION_CONFIGURATION,
        atomic_json,
        configuration_sha256,
        sha256_file,
    )
except ModuleNotFoundError:
    from research.temporal_contrastive.appearance_blend import (
        blend_movie_pair_scores,
        division_logits_for_movie,
        extract_movie_embeddings,
        extract_reciprocal_movie_embeddings,
        reciprocal_movie_evidence,
    )
    from research.temporal_contrastive.appearance_family import (
        CALIBRATION_RUN_BY_FAMILY,
        CANDIDATE_FAMILY_BY_APPEARANCE,
        COSINE_FAMILY,
        FAMILIES,
        PROCESSED_RUN_BY_FAMILY,
        TRAINING_RUN_BY_FAMILY,
        appearance_evidence_for_movie,
        appearance_metadata,
        build_appearance_model,
        verify_appearance_metadata,
    )
    from research.trackastra_graph import rerank_submission as rerank
    from research.trackastra_graph.dual_fold_processed_acceptance import (
        EXPECTED_STEMS,
        FOLD_BY_PREFIX,
        FROZEN_ASSOCIATION_CONFIGURATION,
        atomic_json,
        configuration_sha256,
        sha256_file,
    )

try:
    from submission_sharding import terminate_and_reap_processes
except ModuleNotFoundError:
    from research.submission_sharding import terminate_and_reap_processes

try:
    from verify_trackastra_output import terminal_source_policy
except ModuleNotFoundError:
    from research.trackastra_graph.verify_dual_fold_training_output import (
        terminal_source_policy,
    )


APPEARANCE_TEMPERATURE = 0.10


def verify_sources(
    trackastra_root: Path,
    appearance_root: Path,
    calibration_terminal_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    trackastra_terminal_path = trackastra_root / "training_terminal.json"
    trackastra_terminal = json.loads(
        trackastra_terminal_path.read_text(encoding="utf-8")
    )
    trackastra_folds = trackastra_terminal.get("folds", {})
    trackastra_source_policy = terminal_source_policy(trackastra_terminal)
    adapted_trackastra = trackastra_source_policy == "adapted_dual_fold"
    pretrained_control = (
        trackastra_source_policy == "predeclared_pretrained_control"
    )
    if not (
        trackastra_terminal.get("status") == "completed"
        and trackastra_terminal.get("gpu_count") == 2
        and (adapted_trackastra or pretrained_control)
        and trackastra_terminal.get("submission_created") is False
    ):
        raise RuntimeError("reciprocal Trackastra source is not accepted")
    appearance_aggregate = json.loads(
        (appearance_root / "training_terminal.json").read_text(encoding="utf-8")
    )
    calibration = json.loads(calibration_terminal_path.read_text(encoding="utf-8"))
    model_family = str(calibration.get("appearance_family", COSINE_FAMILY))
    if model_family not in FAMILIES:
        raise RuntimeError("appearance calibration declares an unknown model family")
    if not (
        calibration.get("status") == "completed"
        and calibration.get("run_id") == CALIBRATION_RUN_BY_FAMILY[model_family]
        and calibration.get("gpu_count") == 2
        and calibration.get("both_folds_improved") is True
        and calibration.get("processed_acceptance_ground_truth_read") is False
        and calibration.get("public_leaderboard_used_for_selection") is False
        and calibration.get("submission_created") is False
    ):
        raise RuntimeError("appearance blend calibration is not eligible")
    if set(calibration.get("folds", {})) != set(FOLD_BY_PREFIX.values()):
        raise RuntimeError("appearance calibration does not cover both folds")
    aggregate_family = str(appearance_aggregate.get("appearance_family", COSINE_FAMILY))
    if not (
        appearance_aggregate.get("status") == "completed"
        and appearance_aggregate.get("run_id")
        == TRAINING_RUN_BY_FAMILY[model_family]
        and appearance_aggregate.get("gpu_count") == 2
        and appearance_aggregate.get("both_folds_trained") is True
        and appearance_aggregate.get("public_predictions_copied") is False
        and appearance_aggregate.get("public_leaderboard_used_for_selection") is False
        and appearance_aggregate.get("submission_created") is False
        and aggregate_family == model_family
        and set(appearance_aggregate.get("folds", {}))
        == set(FOLD_BY_PREFIX.values())
    ):
        raise RuntimeError("reciprocal appearance source is not accepted")

    verified_folds: dict[str, dict[str, Any]] = {}
    for fold in sorted(FOLD_BY_PREFIX.values()):
        trackastra_fold = trackastra_terminal["folds"][fold]
        trackastra_worker = json.loads(
            (trackastra_root / fold / "worker_terminal.json").read_text(
                encoding="utf-8"
            )
        )
        if trackastra_worker != trackastra_fold:
            raise RuntimeError(f"Trackastra worker/aggregate mismatch: {fold}")
        trackastra_model = trackastra_root / fold / "model.pt"
        valid_trackastra_step = bool(
            int(trackastra_fold.get("best_step", 0)) > 0
            if adapted_trackastra
            else int(trackastra_fold.get("best_step", -1)) == 0
        )
        if not (
            valid_trackastra_step
            and sha256_file(trackastra_model) == trackastra_fold.get("model_sha256")
        ):
            raise RuntimeError(f"Trackastra fold hash or gain is invalid: {fold}")
        appearance_terminal_path = appearance_root / fold / "worker_terminal.json"
        appearance_terminal = json.loads(
            appearance_terminal_path.read_text(encoding="utf-8")
        )
        if appearance_terminal != appearance_aggregate["folds"].get(fold):
            raise RuntimeError(f"appearance worker/aggregate mismatch: {fold}")
        appearance_model = appearance_root / fold / "appearance_model.pt"
        try:
            fold_family = verify_appearance_metadata(
                appearance_terminal, require_training_run=True
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"appearance fold hash or gain is invalid: {fold}"
            ) from error
        if not (
            appearance_terminal.get("status") == "completed"
            and appearance_terminal.get("fold") == fold
            and int(appearance_terminal.get("best_step", 0)) > 0
            and fold_family == model_family
            and appearance_terminal.get("public_predictions_copied") is False
            and appearance_terminal.get("public_leaderboard_used_for_selection") is False
            and appearance_terminal.get("submission_created") is False
            and sha256_file(appearance_model)
            == appearance_terminal.get("model_sha256")
        ):
            raise RuntimeError(f"appearance fold hash or gain is invalid: {fold}")
        calibration_fold = calibration["folds"][fold]
        selection = calibration_fold.get("selection", {})
        selected_weight = float(selection.get("selected_weight", 0.0))
        selected_division_weight = float(
            selection.get("selected_division_weight", 0.0)
        )
        selected_ensemble_mode = str(
            selection.get("selected_ensemble_mode", "")
        )
        peer_fold = next(
            candidate for candidate in FOLD_BY_PREFIX.values() if candidate != fold
        )
        peer_terminal = json.loads(
            (appearance_root / peer_fold / "worker_terminal.json").read_text(
                encoding="utf-8"
            )
        )
        peer_model = appearance_root / peer_fold / "appearance_model.pt"
        try:
            peer_family = verify_appearance_metadata(
                peer_terminal, require_training_run=True
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"appearance calibration/model binding is invalid: {fold}"
            ) from error
        if not (
            calibration_fold.get("status") == "completed"
            and calibration_fold.get("fold") == fold
            and calibration_fold.get("appearance_family", model_family)
            == model_family
            and peer_family == model_family
            and calibration_fold.get("processed_acceptance_ground_truth_read") is False
            and calibration_fold.get("public_leaderboard_used_for_selection") is False
            and calibration_fold.get("submission_created") is False
            and selection.get("improved") is True
            and selected_weight > 0.0
            and selected_division_weight >= 0.0
            and selected_ensemble_mode in {"target_only", "reciprocal_mean"}
            and calibration_fold.get("appearance_temperature")
            == APPEARANCE_TEMPERATURE
            and calibration_fold.get("appearance_model_sha256")
            == appearance_terminal["model_sha256"]
            and calibration_fold.get("peer_fold") == peer_fold
            and calibration_fold.get("peer_appearance_model_sha256")
            == peer_terminal.get("model_sha256")
            and sha256_file(peer_model) == peer_terminal.get("model_sha256")
            and calibration_fold.get("trackastra_model_sha256")
            == trackastra_fold["model_sha256"]
            and calibration_fold.get("trackastra_source_policy")
            == (
                "adapted_dual_fold"
                if adapted_trackastra
                else "predeclared_pretrained_control"
            )
        ):
            raise RuntimeError(f"appearance calibration/model binding is invalid: {fold}")
        verified_folds[fold] = {
            "trackastra_model_dir": trackastra_root / fold,
            "trackastra_model_sha256": trackastra_fold["model_sha256"],
            "trackastra_best_step": trackastra_fold["best_step"],
            "trackastra_source_policy": (
                "adapted_dual_fold"
                if adapted_trackastra
                else "predeclared_pretrained_control"
            ),
            "appearance_model": appearance_model,
            "peer_appearance_model": peer_model,
            "peer_appearance_model_sha256": peer_terminal["model_sha256"],
            "appearance_model_sha256": appearance_terminal["model_sha256"],
            "appearance_best_step": appearance_terminal["best_step"],
            "appearance_metadata": appearance_metadata(appearance_terminal),
            "appearance_family": model_family,
            "appearance_weight": selected_weight,
            "division_weight": selected_division_weight,
            "ensemble_mode": selected_ensemble_mode,
        }
    return calibration, verified_folds


def worker(args: argparse.Namespace) -> None:
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"isolated processed worker requires one GPU, saw {torch.cuda.device_count()}"
        )
    started = time.monotonic()
    requested = tuple(item for item in args.datasets.split(",") if item)
    expected = {stem for stem in EXPECTED_STEMS if stem.startswith(args.prefix)}
    if set(requested) != expected:
        raise RuntimeError("appearance worker prefix coverage is not the frozen set")
    videos = rerank.read_submission(args.processed_control_csv)
    videos = {stem: videos[stem] for stem in requested}
    raw_videos = rerank.read_raw_graphs(args.raw_graph_root, set(videos))
    transfer = rerank.transfer_raw_edge_probabilities(videos, raw_videos)
    device = torch.device("cuda:0")

    sys.path.insert(0, str(args.trackastra_dir.resolve()))
    from trackastra.model.model import TrackingTransformer

    trackastra = TrackingTransformer.from_folder(
        args.trackastra_model_dir, map_location="cpu"
    ).to(device)
    trackastra.eval()
    appearance = build_appearance_model(args.appearance_family).to(device)
    appearance.load_state_dict(
        torch.load(args.appearance_model, map_location="cpu", weights_only=True),
        strict=True,
    )
    appearance.eval()
    peer_appearance = None
    if args.ensemble_mode == "reciprocal_mean":
        peer_appearance = build_appearance_model(args.appearance_family).to(device)
        peer_appearance.load_state_dict(
            torch.load(
                args.peer_appearance_model, map_location="cpu", weights_only=True
            ),
            strict=True,
        )
        peer_appearance.eval()
    config = rerank.HybridLinkConfig(
        edge_threshold=float(FROZEN_ASSOCIATION_CONFIGURATION["edge_threshold"]),
        base_lock_probability=float(
            FROZEN_ASSOCIATION_CONFIGURATION["base_lock_probability"]
        ),
        base_keep_probability=float(
            FROZEN_ASSOCIATION_CONFIGURATION["base_keep_probability"]
        ),
        base_bonus=float(FROZEN_ASSOCIATION_CONFIGURATION["base_bonus"]),
        division_threshold=float(
            FROZEN_ASSOCIATION_CONFIGURATION["division_threshold"]
        ),
        division_ratio=float(FROZEN_ASSOCIATION_CONFIGURATION["division_ratio"]),
        base_division_keep_probability=float(
            FROZEN_ASSOCIATION_CONFIGURATION[
                "base_division_keep_probability"
            ]
        ),
    )
    import zarr

    candidate_edges: dict[str, list[tuple[int, int]]] = {}
    dataset_stats: dict[str, dict[str, Any]] = {}
    extraction: dict[str, Any] = {}
    for stem, video in videos.items():
        image = zarr.open_group(
            str(args.competition_dir / "train" / f"{stem}.zarr"), mode="r"
        )["0"]
        trackastra_scores = rerank.predict_movie_scores(
            trackastra,
            video,
            device,
            max_tokens=args.max_tokens,
            candidate_radius=args.candidate_radius,
        )
        if peer_appearance is None:
            embeddings, division_logits, shared_extraction = extract_movie_embeddings(
                appearance,
                video,
                image,
                device,
                node_batch_size=args.node_batch_size,
            )
            peer_embeddings = embeddings
            peer_logits = division_logits
        else:
            (
                embeddings,
                division_logits,
                peer_embeddings,
                peer_logits,
                shared_extraction,
            ) = extract_reciprocal_movie_embeddings(
                appearance,
                peer_appearance,
                video,
                image,
                device,
                node_batch_size=args.node_batch_size,
            )
        primary_scores = appearance_evidence_for_movie(
            args.appearance_family,
            appearance,
            video,
            embeddings,
            division_logits,
            trackastra_scores,
            image=image,
        )
        primary_divisions = division_logits_for_movie(
            video, division_logits, trackastra_scores
        )
        if peer_appearance is None:
            peer_scores = primary_scores
            peer_divisions = primary_divisions
        else:
            peer_scores = appearance_evidence_for_movie(
                args.appearance_family,
                peer_appearance,
                video,
                peer_embeddings,
                peer_logits,
                trackastra_scores,
                image=image,
            )
            peer_divisions = division_logits_for_movie(
                video, peer_logits, trackastra_scores
            )
        appearance_scores, source_divisions = reciprocal_movie_evidence(
            primary_scores,
            primary_divisions,
            peer_scores,
            peer_divisions,
            mode=args.ensemble_mode,
        )
        extraction[stem] = {
            "ensemble_mode": args.ensemble_mode,
            "shared": shared_extraction,
        }
        blended = blend_movie_pair_scores(
            trackastra_scores,
            appearance_scores,
            appearance_weight=args.appearance_weight,
            appearance_temperature=APPEARANCE_TEMPERATURE,
            source_division_logits=source_divisions,
            division_weight=args.division_weight,
        )
        edges = rerank.hybrid_link_movie(
            video,
            blended,
            config,
            submission_edge_probability=float(
                FROZEN_ASSOCIATION_CONFIGURATION["base_pseudo_probability"]
            ),
            use_stored_edge_probabilities=True,
        )
        rerank.validate_edges(video, edges)
        candidate_edges[stem] = edges
        base_set = set(map(tuple, video.edges.tolist()))
        candidate_set = set(edges)
        dataset_stats[stem] = {
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
            "appearance_family": args.appearance_family,
            "appearance_weight": args.appearance_weight,
            "division_weight": args.division_weight,
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
    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"processed appearance acceptance requires exactly two GPUs, saw {torch.cuda.device_count()}"
        )
    calibration, folds = verify_sources(
        args.trackastra_output_root,
        args.appearance_output_root,
        args.calibration_terminal,
    )
    videos = rerank.read_submission(args.processed_control_csv)
    if set(videos) != EXPECTED_STEMS:
        raise RuntimeError("processed control does not contain the frozen four movies")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for gpu_index, prefix in enumerate(sorted(FOLD_BY_PREFIX)):
        fold = FOLD_BY_PREFIX[prefix]
        fold_source = folds[fold]
        datasets = sorted(stem for stem in EXPECTED_STEMS if stem.startswith(prefix))
        output = args.output_dir / f"{fold}.json"
        handle = (args.output_dir / f"{fold}.log").open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--processed-control-csv",
            str(args.processed_control_csv),
            "--raw-graph-root",
            str(args.raw_graph_root),
            "--competition-dir",
            str(args.competition_dir),
            "--trackastra-output-root",
            str(args.trackastra_output_root),
            "--appearance-output-root",
            str(args.appearance_output_root),
            "--calibration-terminal",
            str(args.calibration_terminal),
            "--trackastra-dir",
            str(args.trackastra_dir),
            "--output-dir",
            str(args.output_dir),
            "--trackastra-model-dir",
            str(fold_source["trackastra_model_dir"]),
            "--appearance-model",
            str(fold_source["appearance_model"]),
            "--peer-appearance-model",
            str(fold_source["peer_appearance_model"]),
            "--appearance-family",
            str(fold_source["appearance_family"]),
            "--appearance-weight",
            str(fold_source["appearance_weight"]),
            "--division-weight",
            str(fold_source["division_weight"]),
            "--ensemble-mode",
            str(fold_source["ensemble_mode"]),
            "--prefix",
            prefix,
            "--datasets",
            ",".join(datasets),
            "--worker-output",
            str(output),
            "--max-tokens",
            str(args.max_tokens),
            "--candidate-radius",
            str(args.candidate_radius),
            "--node-batch-size",
            str(args.node_batch_size),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((fold, process, handle, output))
    return_codes = {}
    try:
        while len(return_codes) < len(processes):
            for fold, process, _handle, _output in processes:
                code = process.poll()
                if code is not None and fold not in return_codes:
                    return_codes[fold] = int(code)
            if time.monotonic() - started >= args.hard_stop_seconds:
                raise TimeoutError("processed appearance acceptance exceeded hard stop")
            if len(return_codes) < len(processes):
                time.sleep(3)
    finally:
        terminate_and_reap_processes(
            [process for _fold, process, _handle, _output in processes]
        )
        for _fold, process, handle, _output in processes:
            handle.close()
    failures = {fold: code for fold, code in return_codes.items() if code != 0}
    if failures:
        raise RuntimeError(f"processed appearance workers failed: {failures}")

    candidate_edges: dict[str, list[tuple[int, int]]] = {}
    dataset_stats: dict[str, Any] = {}
    transfers: dict[str, Any] = {}
    extraction: dict[str, Any] = {}
    for fold, _process, _handle, output in processes:
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("appearance_family") != folds[fold]["appearance_family"]:
            raise RuntimeError(f"processed worker family mismatch: {fold}")
        for stem, edges in payload["candidate_edges"].items():
            if stem in candidate_edges:
                raise RuntimeError(f"duplicate acceptance movie: {stem}")
            candidate_edges[stem] = [tuple(map(int, edge)) for edge in edges]
        dataset_stats.update(payload["dataset_stats"])
        extraction.update(payload["extraction"])
        transfers[fold] = payload["edge_probability_transfer"]
    if set(candidate_edges) != EXPECTED_STEMS:
        raise RuntimeError("appearance candidate does not cover all four movies")
    candidate_csv = args.output_dir / "processed_candidate.csv"
    rerank.write_submission(candidate_csv, videos, candidate_edges)
    changed_edges = sum(
        row["removed_edges"] + row["new_edges"] for row in dataset_stats.values()
    )
    if changed_edges <= 0 or sha256_file(candidate_csv) == sha256_file(
        args.processed_control_csv
    ):
        raise RuntimeError("processed appearance candidate is an exact control replica")

    model_families = {source["appearance_family"] for source in folds.values()}
    if len(model_families) != 1:
        raise RuntimeError("processed sources mix appearance model families")
    model_family = model_families.pop()
    result = {
        "schema_version": 1,
        "status": "completed",
        "run_id": PROCESSED_RUN_BY_FAMILY[model_family],
        "candidate_family": CANDIDATE_FAMILY_BY_APPEARANCE[model_family],
        "appearance_family": model_family,
        "evaluation_kind": "predeclared_processed_candidate_materialization",
        "elapsed_seconds": time.monotonic() - started,
        "gpu_count": 2,
        "whole_movie_sharding": True,
        "processed_control_sha256": sha256_file(args.processed_control_csv),
        "processed_candidate_sha256": sha256_file(candidate_csv),
        "calibration_terminal_sha256": sha256_file(args.calibration_terminal),
        "models": {
            fold: {
                "model_sha256": source["trackastra_model_sha256"],
                "best_step": source["trackastra_best_step"],
                "source_policy": source["trackastra_source_policy"],
            }
            for fold, source in folds.items()
        },
        "appearance_models": {
            fold: {
                "model_sha256": source["appearance_model_sha256"],
                "best_step": source["appearance_best_step"],
                **source["appearance_metadata"],
            }
            for fold, source in folds.items()
        },
        "appearance_blend": {
            fold: {
                "appearance_weight": source["appearance_weight"],
                "division_weight": source["division_weight"],
                "ensemble_mode": source["ensemble_mode"],
                "appearance_temperature": APPEARANCE_TEMPERATURE,
            }
            for fold, source in folds.items()
        },
        "association_configuration": FROZEN_ASSOCIATION_CONFIGURATION,
        "association_configuration_sha256": configuration_sha256(
            FROZEN_ASSOCIATION_CONFIGURATION
        ),
        "configuration_selection": "association preset was predeclared; appearance weights were frozen on reserved movies before this one-shot materialization",
        "datasets": dataset_stats,
        "extraction": extraction,
        "edge_probability_transfer": transfers,
        "total_changed_edges": changed_edges,
        "ground_truth_read": False,
        "public_leaderboard_used_for_selection": False,
        "hyperparameter_selection_performed": False,
        "exact_processed_scoring_performed": False,
        "competition_submission_performed": False,
        "authorized_for_submission": False,
    }
    atomic_json(args.output_dir / "materialization_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    role = result.add_mutually_exclusive_group(required=True)
    role.add_argument("--orchestrate", action="store_true")
    role.add_argument("--worker", action="store_true")
    result.add_argument("--processed-control-csv", type=Path, required=True)
    result.add_argument("--raw-graph-root", type=Path, required=True)
    result.add_argument("--competition-dir", type=Path, required=True)
    result.add_argument("--trackastra-output-root", type=Path, required=True)
    result.add_argument("--appearance-output-root", type=Path, required=True)
    result.add_argument("--calibration-terminal", type=Path, required=True)
    result.add_argument("--trackastra-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--trackastra-model-dir", type=Path)
    result.add_argument("--appearance-model", type=Path)
    result.add_argument("--peer-appearance-model", type=Path)
    result.add_argument("--appearance-family", choices=FAMILIES)
    result.add_argument("--appearance-weight", type=float)
    result.add_argument("--division-weight", type=float)
    result.add_argument(
        "--ensemble-mode", choices=("target_only", "reciprocal_mean")
    )
    result.add_argument("--prefix", choices=sorted(FOLD_BY_PREFIX))
    result.add_argument("--datasets", default="")
    result.add_argument("--worker-output", type=Path)
    result.add_argument("--max-tokens", type=int, default=512)
    result.add_argument("--candidate-radius", type=float, default=80.0)
    result.add_argument("--node-batch-size", type=int, default=64)
    result.add_argument("--hard-stop-seconds", type=int, default=10800)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.worker:
        required = (
            args.trackastra_model_dir,
            args.appearance_model,
            args.peer_appearance_model,
            args.appearance_family,
            args.appearance_weight,
            args.division_weight,
            args.ensemble_mode,
            args.prefix,
            args.worker_output,
        )
        if any(value is None for value in required):
            raise ValueError("appearance worker arguments are incomplete")
        worker(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
