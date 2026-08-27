#!/usr/bin/env python
"""Calibrate reciprocal appearance/Trackastra blending on reserved movies.

This two-GPU stage consumes only the calibration stems reserved before patch
training. It ranks a fixed grid including exact zero-weight control, never opens
the four processed-acceptance labels, and creates no competition submission.
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
from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:
    import trainer as graph_base
    from appearance_blend import (
        blend_movie_pair_scores,
        division_logits_for_movie,
        extract_reciprocal_movie_embeddings,
        reciprocal_movie_evidence,
    )
    from appearance_family import (
        CALIBRATION_RUN_BY_FAMILY,
        COSINE_FAMILY,
        appearance_evidence_for_movie,
        build_appearance_model,
        verify_appearance_metadata,
    )
except ModuleNotFoundError:
    from research.temporal_contrastive.appearance_blend import (
        blend_movie_pair_scores,
        division_logits_for_movie,
        extract_reciprocal_movie_embeddings,
        reciprocal_movie_evidence,
    )
    from research.temporal_contrastive.appearance_family import (
        CALIBRATION_RUN_BY_FAMILY,
        COSINE_FAMILY,
        appearance_evidence_for_movie,
        build_appearance_model,
        verify_appearance_metadata,
    )
    from research.trackastra_graph import train_biohub_graph_transformer as graph_base

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


FOLDS = ("target_44b6", "target_6bba")
PREFIX_BY_FOLD = {"target_44b6": "44b6", "target_6bba": "6bba"}
OPENED_ACCEPTANCE_STEMS = frozenset(
    {
        "44b6_12dfb391",
        "44b6_267148e4",
        "6bba_062c8d37",
        "6bba_07e24132",
    }
)
APPEARANCE_WEIGHTS = (0.0, 0.05, 0.10, 0.20, 0.35)
DIVISION_WEIGHTS = (0.0, 0.05, 0.10, 0.20)
ENSEMBLE_MODES = ("target_only", "reciprocal_mean")
APPEARANCE_TEMPERATURE = 0.10
MINIMUM_POOLED_GAIN = 0.001
MAXIMUM_MOVIE_REGRESSION = 0.002
FROZEN_LINK_CONFIGURATION = {
    "edge_threshold": 0.08,
    "division_threshold": 0.18,
    "division_ratio": 0.50,
}


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sources(
    fold: str, appearance_root: Path, trackastra_root: Path
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[str],
    Path,
    Path,
    str,
    str,
]:
    appearance_dir = appearance_root / fold
    appearance_terminal_path = appearance_dir / "worker_terminal.json"
    appearance_config_path = appearance_dir / "training_config.json"
    appearance_terminal = json.loads(
        appearance_terminal_path.read_text(encoding="utf-8")
    )
    appearance_config = json.loads(appearance_config_path.read_text(encoding="utf-8"))
    appearance_aggregate = json.loads(
        (appearance_root / "training_terminal.json").read_text(encoding="utf-8")
    )
    appearance_model = appearance_dir / "appearance_model.pt"
    aggregate_family = str(
        appearance_aggregate.get(
            "appearance_family",
            appearance_terminal.get("appearance_family", COSINE_FAMILY),
        )
    )
    if not (
        appearance_aggregate.get("status") == "completed"
        and appearance_aggregate.get("run_id")
        == appearance_terminal.get("run_id")
        and appearance_aggregate.get("gpu_count") == 2
        and appearance_aggregate.get("both_folds_trained") is True
        and appearance_aggregate.get("public_predictions_copied") is False
        and appearance_aggregate.get("public_leaderboard_used_for_selection")
        is False
        and appearance_aggregate.get("submission_created") is False
        and aggregate_family
        == appearance_terminal.get("appearance_family", COSINE_FAMILY)
        and appearance_terminal.get("status") == "completed"
        and appearance_terminal.get("fold") == fold
        and int(appearance_terminal.get("best_step", 0)) > 0
        and appearance_terminal.get("public_predictions_copied") is False
        and appearance_terminal.get("public_leaderboard_used_for_selection") is False
        and appearance_terminal.get("submission_created") is False
    ):
        raise RuntimeError(f"appearance source is not eligible: {fold}")
    if appearance_terminal != appearance_aggregate.get("folds", {}).get(fold):
        raise RuntimeError(f"appearance worker/aggregate mismatch: {fold}")
    if sha256_file(appearance_model) != appearance_terminal.get("model_sha256"):
        raise RuntimeError(f"appearance model hash mismatch: {fold}")
    try:
        model_family = verify_appearance_metadata(
            appearance_terminal, require_training_run=True
        )
        config_family = verify_appearance_metadata(
            appearance_config, require_training_run=True
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"appearance architecture or split boundary changed: {fold}"
        ) from error
    if not (
        config_family == model_family
        and appearance_config.get("base_channels") == 64
        and appearance_config.get("embedding_channels") == 256
        and appearance_config.get("calibration_ground_truth_read") is False
    ):
        raise RuntimeError(f"appearance architecture or split boundary changed: {fold}")
    stems = appearance_config.get("real_calibration_stems_reserved")
    if not isinstance(stems, list) or len(stems) != 12:
        raise RuntimeError(f"appearance calibration inventory is invalid: {fold}")
    if len(set(stems)) != len(stems) or set(stems) & OPENED_ACCEPTANCE_STEMS:
        raise RuntimeError(f"appearance calibration split leaks acceptance: {fold}")
    prefix = PREFIX_BY_FOLD[fold]
    if any(not str(stem).startswith(f"{prefix}_") for stem in stems):
        raise RuntimeError(f"appearance calibration prefix mismatch: {fold}")

    trackastra_dir = trackastra_root / fold
    trackastra_aggregate = json.loads(
        (trackastra_root / "training_terminal.json").read_text(encoding="utf-8")
    )
    if not (
        trackastra_aggregate.get("status") == "completed"
        and trackastra_aggregate.get("gpu_count") == 2
        and trackastra_aggregate.get("submission_created") is False
    ):
        raise RuntimeError("aggregate Trackastra source is not eligible")
    trackastra_source_policy = terminal_source_policy(trackastra_aggregate)
    trackastra_terminal = json.loads(
        (trackastra_dir / "worker_terminal.json").read_text(encoding="utf-8")
    )
    if trackastra_terminal != trackastra_aggregate["folds"].get(fold):
        raise RuntimeError(f"Trackastra worker/aggregate mismatch: {fold}")
    trackastra_model = trackastra_dir / "model.pt"
    adapted_trackastra = bool(
        int(trackastra_terminal.get("best_step", 0)) > 0
        and trackastra_terminal.get("pretrained_initialization_retained") is False
    )
    pretrained_control = bool(
        int(trackastra_terminal.get("best_step", -1)) == 0
        and trackastra_terminal.get("pretrained_initialization_retained") is True
        and trackastra_terminal.get("best_real")
        == trackastra_terminal.get("initial_real")
        and trackastra_terminal.get("best_synthetic")
        == trackastra_terminal.get("initial_synthetic")
        and float(trackastra_terminal.get("best_selection_score", float("nan")))
        == float(trackastra_terminal.get("initial_selection_score", float("nan")))
    )
    if not (
        trackastra_terminal.get("status") == "completed"
        and trackastra_terminal.get("fold") == fold
        and (adapted_trackastra or pretrained_control)
        and trackastra_terminal.get("submission_created") is False
    ):
        raise RuntimeError(f"Trackastra source is not eligible: {fold}")
    if sha256_file(trackastra_model) != trackastra_terminal.get("model_sha256"):
        raise RuntimeError(f"Trackastra model hash mismatch: {fold}")
    local_source_policy = (
        "adapted_dual_fold" if adapted_trackastra else "predeclared_pretrained_control"
    )
    if local_source_policy != trackastra_source_policy:
        raise RuntimeError(f"Trackastra source policy mismatch: {fold}")
    return (
        appearance_terminal,
        appearance_config,
        trackastra_terminal,
        [str(stem) for stem in stems],
        appearance_model,
        trackastra_dir,
        model_family,
        trackastra_source_policy,
    )


def ranking_metrics_for_video(
    video: Any,
    pair_scores: Mapping[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, float | int]:
    outgoing: dict[int, set[int]] = {}
    for source, target in video.edges.tolist():
        outgoing.setdefault(int(source), set()).add(int(target))
    reciprocal_ranks: list[float] = []
    top1: list[float] = []
    division_top2: list[float] = []
    for _timepoint, (source_ids, target_ids, score_matrix) in sorted(pair_scores.items()):
        target_to_col = {int(node): col for col, node in enumerate(target_ids.tolist())}
        for row, source in enumerate(source_ids.tolist()):
            positives = {
                target_to_col[target]
                for target in outgoing.get(int(source), set())
                if target in target_to_col
            }
            if not positives:
                continue
            available = np.flatnonzero(np.isfinite(score_matrix[row]))
            if not positives.issubset(set(map(int, available.tolist()))):
                raise RuntimeError("Trackastra candidates omitted a calibration positive")
            if len(available) <= len(positives):
                continue
            ordered = available[
                np.argsort(-score_matrix[row, available], kind="stable")
            ]
            positive_ranks = [
                rank
                for rank, target_col in enumerate(ordered.tolist(), start=1)
                if target_col in positives
            ]
            if not positive_ranks:
                raise RuntimeError("calibration ranking lost every positive")
            reciprocal_ranks.append(1.0 / min(positive_ranks))
            top1.append(float(int(ordered[0]) in positives))
            if len(positives) >= 2:
                division_top2.append(
                    len(set(map(int, ordered[:2].tolist())) & positives) / len(positives)
                )
    if not reciprocal_ranks:
        raise RuntimeError(f"no rankable calibration edges for {video.stem}")
    edge_top1 = float(np.mean(top1))
    edge_mrr = float(np.mean(reciprocal_ranks))
    division = float(np.mean(division_top2)) if division_top2 else 0.0
    composite = 0.65 * edge_top1 + 0.25 * edge_mrr + 0.10 * division
    return {
        "stem": video.stem,
        "composite": composite,
        "top1": edge_top1,
        "mrr": edge_mrr,
        "division_top2": division,
        "edge_rows": len(reciprocal_ranks),
        "division_rows": len(division_top2),
    }


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    edge_rows = sum(int(row["edge_rows"]) for row in rows)
    division_rows = sum(int(row["division_rows"]) for row in rows)
    if edge_rows <= 0:
        raise ValueError("calibration metrics contain no edges")
    top1 = sum(float(row["top1"]) * int(row["edge_rows"]) for row in rows) / edge_rows
    mrr = sum(float(row["mrr"]) * int(row["edge_rows"]) for row in rows) / edge_rows
    division = (
        sum(
            float(row["division_top2"]) * int(row["division_rows"])
            for row in rows
        )
        / division_rows
        if division_rows
        else 0.0
    )
    return {
        "composite": 0.65 * top1 + 0.25 * mrr + 0.10 * division,
        "top1": top1,
        "mrr": mrr,
        "division_top2": division,
        "edge_rows": edge_rows,
        "division_rows": division_rows,
        "movies": len(rows),
    }


def association_metrics_for_video(
    video: Any,
    pair_scores: Mapping[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, float | int]:
    """Score the actual frozen clean linker, including daughter recovery."""

    predicted = set(
        graph_base.link_movie(
            dict(pair_scores),
            edge_threshold=FROZEN_LINK_CONFIGURATION["edge_threshold"],
            division_threshold=FROZEN_LINK_CONFIGURATION["division_threshold"],
            division_ratio=FROZEN_LINK_CONFIGURATION["division_ratio"],
        )
    )
    truth = {(int(source), int(target)) for source, target in video.edges.tolist()}
    edge_tp = len(predicted & truth)
    edge_fp = len(predicted - truth)
    edge_fn = len(truth - predicted)

    node_ids = {int(node_id) for node_id in video.node_ids.tolist()}
    identity = {node_id: node_id for node_id in node_ids}
    division_tp, division_fp, division_fn = graph_base.compute_division_confusion(
        node_ids,
        predicted,
        node_ids,
        truth,
        identity,
        identity,
    )

    def jaccard(tp: int, fp: int, fn: int) -> float:
        denominator = tp + fp + fn
        return float(tp / denominator) if denominator else 1.0

    edge_jaccard = jaccard(edge_tp, edge_fp, edge_fn)
    division_jaccard = jaccard(division_tp, division_fp, division_fn)
    return {
        "stem": video.stem,
        "composite": edge_jaccard + 0.10 * division_jaccard,
        "edge_jaccard": edge_jaccard,
        "adjusted_edge_jaccard": edge_jaccard,
        "division_jaccard": division_jaccard,
        "edge_tp": edge_tp,
        "edge_fp": edge_fp,
        "edge_fn": edge_fn,
        "division_tp": division_tp,
        "division_fp": division_fp,
        "division_fn": division_fn,
        "predicted_edges": len(predicted),
        "true_edges": len(truth),
    }


def aggregate_association_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    totals = {
        key: sum(int(row[key]) for row in rows)
        for key in (
            "edge_tp",
            "edge_fp",
            "edge_fn",
            "division_tp",
            "division_fp",
            "division_fn",
            "predicted_edges",
            "true_edges",
        )
    }

    def jaccard(prefix: str) -> float:
        denominator = (
            totals[f"{prefix}_tp"]
            + totals[f"{prefix}_fp"]
            + totals[f"{prefix}_fn"]
        )
        return float(totals[f"{prefix}_tp"] / denominator) if denominator else 1.0

    edge = jaccard("edge")
    division = jaccard("division")
    return {
        **totals,
        "edge_jaccard": edge,
        "adjusted_edge_jaccard": edge,
        "division_jaccard": division,
        "composite": edge + 0.10 * division,
        "movies": len(rows),
    }


def select_weight(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_weight = {
        (
            str(row.get("ensemble_mode", "target_only")),
            float(row["appearance_weight"]),
            float(row.get("division_weight", 0.0)),
        ): row
        for row in rows
    }
    expected = {
        (ensemble_mode, appearance_weight, division_weight)
        for ensemble_mode in ENSEMBLE_MODES
        for appearance_weight in APPEARANCE_WEIGHTS
        for division_weight in DIVISION_WEIGHTS
    }
    if set(by_weight) != expected:
        raise ValueError("calibration grid is incomplete")
    control = by_weight[("target_only", 0.0, 0.0)]
    control_movies = {
        str(row["stem"]): float(row["composite"]) for row in control["by_movie"]
    }
    evaluated = []
    for ensemble_mode in ENSEMBLE_MODES:
        for appearance_weight in APPEARANCE_WEIGHTS:
            for division_weight in DIVISION_WEIGHTS:
                row = by_weight[(ensemble_mode, appearance_weight, division_weight)]
                movie_deltas = {
                    str(movie["stem"]): float(movie["composite"])
                    - control_movies[str(movie["stem"])]
                    for movie in row["by_movie"]
                }
                pooled_gain = float(row["pooled"]["composite"]) - float(
                    control["pooled"]["composite"]
                )
                worst_delta = min(movie_deltas.values())
                eligible = bool(
                    appearance_weight > 0
                    and pooled_gain >= MINIMUM_POOLED_GAIN
                    and worst_delta >= -MAXIMUM_MOVIE_REGRESSION
                )
                evaluated.append(
                    {
                        **dict(row),
                        "pooled_gain_vs_zero": pooled_gain,
                        "worst_movie_delta_vs_zero": worst_delta,
                        "movie_deltas_vs_zero": movie_deltas,
                        "eligible": eligible,
                    }
                )
    eligible = [row for row in evaluated if row["eligible"]]
    selected = (
        max(
            eligible,
            key=lambda row: (
                float(row["worst_movie_delta_vs_zero"]),
                float(row["pooled_gain_vs_zero"]),
                1 if row["ensemble_mode"] == "target_only" else 0,
                -float(row["division_weight"]),
                -float(row["appearance_weight"]),
            ),
        )
        if eligible
        else next(
            row
            for row in evaluated
            if float(row["appearance_weight"]) == 0.0
            and float(row["division_weight"]) == 0.0
            and row["ensemble_mode"] == "target_only"
        )
    )
    return {
        "selected_weight": float(selected["appearance_weight"]),
        "selected_division_weight": float(selected["division_weight"]),
        "selected_ensemble_mode": str(selected["ensemble_mode"]),
        "improved": bool(selected["eligible"]),
        "selected": selected,
        "control": next(
            row
            for row in evaluated
            if float(row["appearance_weight"]) == 0.0
            and float(row["division_weight"]) == 0.0
            and row["ensemble_mode"] == "target_only"
        ),
        "grid": evaluated,
    }


def worker(args: argparse.Namespace) -> None:
    if args.fold not in FOLDS:
        raise ValueError(f"unknown fold: {args.fold}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"isolated calibration worker requires one GPU, saw {torch.cuda.device_count()}"
        )
    started = time.monotonic()
    device = torch.device("cuda:0")
    (
        appearance_terminal,
        appearance_config,
        trackastra_terminal,
        stems,
        appearance_model_path,
        trackastra_model_dir,
        model_family,
        trackastra_source_policy,
    ) = verify_sources(args.fold, args.appearance_output_root, args.trackastra_output_root)
    peer_fold = next(fold for fold in FOLDS if fold != args.fold)
    (
        peer_appearance_terminal,
        peer_appearance_config,
        _peer_trackastra_terminal,
        _peer_stems,
        peer_appearance_model_path,
        _peer_trackastra_model_dir,
        peer_model_family,
        _peer_trackastra_source_policy,
    ) = verify_sources(
        peer_fold, args.appearance_output_root, args.trackastra_output_root
    )
    if set(stems) & set(peer_appearance_config.get("real_train_stems", [])):
        raise RuntimeError("peer appearance model trained on a calibration movie")
    if appearance_config.get("real_split_policy") != peer_appearance_config.get(
        "real_split_policy"
    ):
        raise RuntimeError("reciprocal appearance split policies differ")
    if peer_model_family != model_family:
        raise RuntimeError("reciprocal appearance model families differ")

    appearance_model = build_appearance_model(model_family).to(device)
    appearance_model.load_state_dict(
        torch.load(appearance_model_path, map_location="cpu", weights_only=True),
        strict=True,
    )
    appearance_model.eval()
    peer_appearance_model = build_appearance_model(model_family).to(device)
    peer_appearance_model.load_state_dict(
        torch.load(peer_appearance_model_path, map_location="cpu", weights_only=True),
        strict=True,
    )
    peer_appearance_model.eval()
    sys.path.insert(0, str(args.trackastra_dir.resolve()))
    from trackastra.model.model import TrackingTransformer

    trackastra_model = TrackingTransformer.from_folder(
        trackastra_model_dir, map_location="cpu"
    ).to(device)
    trackastra_model.eval()
    import zarr

    rows_by_weight: dict[tuple[str, float, float], list[dict[str, Any]]] = {
        (ensemble_mode, appearance_weight, division_weight): []
        for ensemble_mode in ENSEMBLE_MODES
        for appearance_weight in APPEARANCE_WEIGHTS
        for division_weight in DIVISION_WEIGHTS
    }
    extraction = {}
    for stem in stems:
        if time.monotonic() - started >= args.max_wall_seconds:
            raise TimeoutError(f"calibration exceeded worker deadline: {args.fold}")
        video = graph_base.read_graph_video(
            args.competition_dir / "train" / f"{stem}.geff"
        )
        image = zarr.open_group(
            str(args.competition_dir / "train" / f"{stem}.zarr"), mode="r"
        )["0"]
        pair_scores = graph_base.predict_movie_scores(
            trackastra_model,
            video,
            device,
            max_tokens=args.max_tokens,
            candidate_radius=args.candidate_radius,
        )
        (
            embeddings,
            division_logits,
            peer_embeddings,
            peer_division_logits,
            shared_extraction,
        ) = extract_reciprocal_movie_embeddings(
            appearance_model,
            peer_appearance_model,
            video,
            image,
            device,
            node_batch_size=args.node_batch_size,
        )
        extraction[stem] = shared_extraction
        primary_scores = appearance_evidence_for_movie(
            model_family,
            appearance_model,
            video,
            embeddings,
            division_logits,
            pair_scores,
        )
        primary_divisions = division_logits_for_movie(
            video, division_logits, pair_scores
        )
        peer_scores = appearance_evidence_for_movie(
            model_family,
            peer_appearance_model,
            video,
            peer_embeddings,
            peer_division_logits,
            pair_scores,
        )
        peer_divisions = division_logits_for_movie(
            video, peer_division_logits, pair_scores
        )
        for ensemble_mode in ENSEMBLE_MODES:
            appearance_scores, source_divisions = reciprocal_movie_evidence(
                primary_scores,
                primary_divisions,
                peer_scores,
                peer_divisions,
                mode=ensemble_mode,
            )
            for appearance_weight in APPEARANCE_WEIGHTS:
                for division_weight in DIVISION_WEIGHTS:
                    blended = blend_movie_pair_scores(
                        pair_scores,
                        appearance_scores,
                        appearance_weight=appearance_weight,
                        appearance_temperature=APPEARANCE_TEMPERATURE,
                        source_division_logits=source_divisions,
                        division_weight=division_weight,
                    )
                    association = association_metrics_for_video(video, blended)
                    ranking = ranking_metrics_for_video(video, blended)
                    rows_by_weight[
                        (ensemble_mode, appearance_weight, division_weight)
                    ].append(
                        {
                            **association,
                            "ranking": ranking,
                        }
                    )
    calibration_rows = [
        {
            "ensemble_mode": ensemble_mode,
            "appearance_weight": appearance_weight,
            "division_weight": division_weight,
            "appearance_temperature": APPEARANCE_TEMPERATURE,
            "pooled": aggregate_association_metrics(
                rows_by_weight[(ensemble_mode, appearance_weight, division_weight)]
            ),
            "ranking_pooled": aggregate_metrics(
                [
                    row["ranking"]
                    for row in rows_by_weight[
                        (ensemble_mode, appearance_weight, division_weight)
                    ]
                ]
            ),
            "by_movie": rows_by_weight[
                (ensemble_mode, appearance_weight, division_weight)
            ],
        }
        for ensemble_mode in ENSEMBLE_MODES
        for appearance_weight in APPEARANCE_WEIGHTS
        for division_weight in DIVISION_WEIGHTS
    ]
    selection = select_weight(calibration_rows)
    output = {
        "schema_version": 1,
        "status": "completed",
        "run_id": CALIBRATION_RUN_BY_FAMILY[model_family],
        "appearance_family": model_family,
        "fold": args.fold,
        "elapsed_seconds": time.monotonic() - started,
        "calibration_stems": stems,
        "selection": selection,
        "extraction": extraction,
        "appearance_model_sha256": appearance_terminal["model_sha256"],
        "peer_fold": peer_fold,
        "peer_appearance_model_sha256": peer_appearance_terminal["model_sha256"],
        "trackastra_model_sha256": trackastra_terminal["model_sha256"],
        "trackastra_source_policy": trackastra_source_policy,
        "ensemble_mode_grid": list(ENSEMBLE_MODES),
        "appearance_weight_grid": list(APPEARANCE_WEIGHTS),
        "division_weight_grid": list(DIVISION_WEIGHTS),
        "appearance_temperature": APPEARANCE_TEMPERATURE,
        "frozen_clean_link_configuration": FROZEN_LINK_CONFIGURATION,
        "minimum_pooled_gain": MINIMUM_POOLED_GAIN,
        "maximum_movie_regression": MAXIMUM_MOVIE_REGRESSION,
        "processed_acceptance_ground_truth_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    atomic_json(args.output_dir / args.fold / "calibration_result.json", output)
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)


def orchestrate(args: argparse.Namespace) -> None:
    started = time.monotonic()
    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"blend calibration requires exactly two GPUs, saw {torch.cuda.device_count()}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for gpu_index, fold in enumerate(FOLDS):
        fold_dir = args.output_dir / fold
        fold_dir.mkdir(parents=True, exist_ok=True)
        handle = (fold_dir / "calibration.log").open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *[item for item in sys.argv[1:] if item != "--orchestrate"],
            "--worker",
            "--fold",
            fold,
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
        processes.append((fold, process, handle))
    return_codes = {}
    try:
        while len(return_codes) < len(processes):
            for fold, process, _handle in processes:
                code = process.poll()
                if code is not None and fold not in return_codes:
                    return_codes[fold] = int(code)
            if time.monotonic() - started >= args.orchestrator_hard_stop_seconds:
                raise TimeoutError("blend calibration orchestrator exceeded its hard stop")
            if len(return_codes) < len(processes):
                time.sleep(5)
    finally:
        terminate_and_reap_processes(
            [process for _fold, process, _handle in processes]
        )
        for _fold, process, handle in processes:
            handle.close()
    failures = {fold: code for fold, code in return_codes.items() if code != 0}
    if failures:
        raise RuntimeError(f"blend calibration workers failed: {failures}")
    folds = {
        fold: json.loads(
            (args.output_dir / fold / "calibration_result.json").read_text(
                encoding="utf-8"
            )
        )
        for fold in FOLDS
    }
    model_families = {row.get("appearance_family") for row in folds.values()}
    if len(model_families) != 1 or None in model_families:
        raise RuntimeError("calibration workers produced mixed appearance families")
    model_family = model_families.pop()
    terminal = {
        "schema_version": 1,
        "status": "completed",
        "run_id": CALIBRATION_RUN_BY_FAMILY[model_family],
        "appearance_family": model_family,
        "elapsed_seconds": time.monotonic() - started,
        "gpu_count": 2,
        "folds": folds,
        "both_folds_improved": all(
            row["selection"]["improved"] is True for row in folds.values()
        ),
        "processed_acceptance_ground_truth_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    atomic_json(args.output_dir / "calibration_terminal.json", terminal)
    print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    role = result.add_mutually_exclusive_group(required=True)
    role.add_argument("--orchestrate", action="store_true")
    role.add_argument("--worker", action="store_true")
    result.add_argument("--fold", choices=list(FOLDS))
    result.add_argument("--appearance-output-root", type=Path, required=True)
    result.add_argument("--trackastra-output-root", type=Path, required=True)
    result.add_argument("--competition-dir", type=Path, required=True)
    result.add_argument("--trackastra-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--max-tokens", type=int, default=512)
    result.add_argument("--candidate-radius", type=float, default=80.0)
    result.add_argument("--node-batch-size", type=int, default=64)
    result.add_argument("--max-wall-seconds", type=int, default=10800)
    result.add_argument("--orchestrator-hard-stop-seconds", type=int, default=12000)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.worker:
        if args.fold is None:
            raise ValueError("--fold is required for a worker")
        worker(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
