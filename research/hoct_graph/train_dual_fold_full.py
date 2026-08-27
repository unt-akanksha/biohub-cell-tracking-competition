#!/usr/bin/env python
"""Full-backbone reciprocal HOCT adaptation on exactly two GPUs.

This fallback trains the released 6.25M-parameter JIT backbone rather than the
previous 289-parameter probe. Corrected synthetic geometry and opposite-embryo
real replay are selected without the four processed acceptance movies. The
pretrained initialization remains an explicit checkpoint candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:
    from biohub_adapter import HOCTTile, iter_pair_tiles
    import trainer as graph_base
    from dual_trainer import (
        FOLD_SPECS,
        OPENED_ACCEPTANCE_STEMS,
        select_real_paths,
        split_synthetic_paths,
        synthetic_video_native_geometry,
    )
except ModuleNotFoundError:
    from research.hoct_graph.biohub_adapter import HOCTTile, iter_pair_tiles
    from research.trackastra_graph import train_biohub_graph_transformer as graph_base
    from research.trackastra_graph.train_dual_fold_synthetic import (
        FOLD_SPECS,
        OPENED_ACCEPTANCE_STEMS,
        select_real_paths,
        split_synthetic_paths,
        synthetic_video_native_geometry,
    )


RUN_ID = "hoct-dual-fold-full-v1"
EXPECTED_PARAMETERS = 6_252_593


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def forward_window(
    model: torch.nn.Module,
    tile: HOCTTile,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    window = tile.window
    if not len(window.edge_indices):
        raise ValueError("cannot train an empty HOCT window")
    node_features = torch.from_numpy(window.node_features).unsqueeze(0).to(device)
    node_positions = torch.from_numpy(window.node_positions).unsqueeze(0).to(device)
    edge_positions = torch.from_numpy(window.edge_positions).unsqueeze(0).to(device)
    edge_indices = torch.from_numpy(window.edge_indices).unsqueeze(0).to(device)
    node_mask = torch.ones(
        (1, len(window.node_ids)), dtype=torch.bool, device=device
    )
    edge_mask = torch.ones(
        (1, len(window.edge_indices)), dtype=torch.bool, device=device
    )
    edge_logits, _node_features, _edge_features, orphan_logits = model.forward(
        node_features,
        node_positions,
        edge_positions,
        edge_indices,
        node_mask,
        edge_mask,
    )
    return edge_logits[0, :, 0].float(), orphan_logits[0, :, 0].float()


def core_parent_groups(tile: HOCTTile) -> list[tuple[int, np.ndarray, int]]:
    """Return target-local, candidate-edge indices, positive local edge index."""

    labels = tile.window.edge_labels
    if labels is None:
        raise ValueError("HOCT training tile has no edge labels")
    result = []
    core_edges = np.flatnonzero(tile.core_edge_mask)
    targets = tile.window.edge_indices[core_edges, 1]
    for target in np.unique(targets):
        indices = core_edges[targets == target]
        positives = indices[labels[indices] > 0.5]
        if len(positives) > 1:
            raise RuntimeError("a target has multiple labeled parents")
        if len(positives) == 1:
            positive_local = int(np.flatnonzero(indices == positives[0])[0])
            result.append((int(target), indices, positive_local))
    return result


def parental_training_loss(
    edge_logits: torch.Tensor,
    orphan_logits: torch.Tensor,
    tile: HOCTTile,
    *,
    bce_weight: float = 0.20,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Train labeled parents against every candidate plus the orphan option."""

    if edge_logits.shape != (len(tile.window.edge_indices),):
        raise ValueError("edge logits do not align with the HOCT tile")
    if orphan_logits.shape != (len(tile.window.node_ids),):
        raise ValueError("orphan logits do not align with the HOCT tile")
    if not 0.0 <= bce_weight <= 1.0:
        raise ValueError("bce_weight must lie in [0, 1]")
    groups = core_parent_groups(tile)
    if not groups:
        raise ValueError("HOCT tile has no supervised core target")
    parental_losses = []
    for target, indices, positive_local in groups:
        candidates = edge_logits[torch.as_tensor(indices, device=edge_logits.device)]
        values = torch.cat((candidates, orphan_logits[target : target + 1]))
        parental_losses.append(
            F.cross_entropy(
                values.unsqueeze(0),
                torch.tensor([positive_local], device=edge_logits.device),
            )
        )
    parental = torch.stack(parental_losses).mean()
    core = torch.as_tensor(
        np.flatnonzero(tile.core_edge_mask), device=edge_logits.device
    )
    labels = torch.as_tensor(
        tile.window.edge_labels[tile.core_edge_mask],
        dtype=edge_logits.dtype,
        device=edge_logits.device,
    )
    positives = int(torch.count_nonzero(labels).item())
    negatives = len(labels) - positives
    positive_weight = min(20.0, negatives / max(positives, 1))
    bce = F.binary_cross_entropy_with_logits(
        edge_logits[core],
        labels,
        pos_weight=torch.tensor(positive_weight, device=edge_logits.device),
    )
    loss = (1.0 - bce_weight) * parental + bce_weight * bce
    return loss, {
        "parental_loss": float(parental.detach()),
        "balanced_edge_bce": float(bce.detach()),
        "supervised_targets": len(groups),
        "positive_edges": positives,
        "negative_edges": negatives,
    }


def l2sp_penalty(
    model: torch.nn.Module, anchors: MappingLike
) -> torch.Tensor:
    total = None
    parameters = 0
    for name, parameter in model.named_parameters():
        anchor = anchors[name]
        term = torch.sum((parameter.float() - anchor.float()) ** 2)
        total = term if total is None else total + term
        parameters += parameter.numel()
    if total is None or parameters <= 0:
        raise ValueError("model has no parameters for L2-SP")
    return total / parameters


MappingLike = dict[str, torch.Tensor]


def model_anchors(model: torch.nn.Module) -> MappingLike:
    return {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }


def true_edges(video: graph_base.GraphVideo) -> set[tuple[int, int]]:
    return {(int(source), int(target)) for source, target in video.edges.tolist()}


def eligible_times(video: graph_base.GraphVideo) -> list[int]:
    return sorted(
        {
            int(video.time_by_id[int(source)])
            for source, target in video.edges.tolist()
            if video.time_by_id[int(target)]
            == video.time_by_id[int(source)] + 1
        }
    )


def validate_tile_candidate_recall(tile: HOCTTile, truth: set[tuple[int, int]]) -> None:
    window = tile.window
    core_targets = set(
        map(int, window.node_ids[window.edge_indices[tile.core_edge_mask, 1]].tolist())
    )
    expected = {edge for edge in truth if edge[1] in core_targets}
    captured = {
        (int(window.node_ids[source]), int(window.node_ids[target]))
        for index, (source, target) in enumerate(window.edge_indices.tolist())
        if tile.core_edge_mask[index]
        and window.edge_labels is not None
        and window.edge_labels[index] > 0.5
    }
    if expected != captured:
        raise RuntimeError(
            f"HOCT candidate graph omitted labeled core edges: {sorted(expected - captured)[:5]}"
        )


def validate_frame_candidate_recall(
    tiles: Sequence[HOCTTile],
    video: graph_base.GraphVideo,
    timepoint: int,
    truth: set[tuple[int, int]],
) -> None:
    """Prove tiled candidates cover every labeled edge in the frame pair."""

    expected = {
        (source, target)
        for source, target in truth
        if video.time_by_id[source] == timepoint
        and video.time_by_id[target] == timepoint + 1
    }
    captured: set[tuple[int, int]] = set()
    for tile in tiles:
        window = tile.window
        if window.edge_labels is None:
            raise ValueError("HOCT candidate tile has no labels")
        captured.update(
            (int(window.node_ids[source]), int(window.node_ids[target]))
            for index, (source, target) in enumerate(window.edge_indices.tolist())
            if tile.core_edge_mask[index] and window.edge_labels[index] > 0.5
        )
    if expected != captured:
        raise RuntimeError(
            f"HOCT tiled candidates omitted frame edges: {sorted(expected - captured)[:5]}"
        )


def sample_tile(
    videos: Sequence[graph_base.GraphVideo],
    rng: np.random.Generator,
    *,
    core_size: np.ndarray,
    neighbors: int,
    max_distance: float,
    prefer_division_probability: float,
) -> HOCTTile:
    for _attempt in range(50):
        video = videos[int(rng.integers(0, len(videos)))]
        times = eligible_times(video)
        if not times:
            continue
        truth = true_edges(video)
        division_sources = {
            source
            for source in set(source for source, _target in truth)
            if sum(edge_source == source for edge_source, _target in truth) >= 2
        }
        if division_sources and rng.random() < prefer_division_probability:
            division_times = sorted(
                {
                    video.time_by_id[source]
                    for source in division_sources
                    if video.time_by_id[source] in times
                }
            )
            timepoint = int(rng.choice(division_times or times))
        else:
            timepoint = int(rng.choice(times))
        tiles = iter_pair_tiles(
            video.node_ids,
            video.times,
            video.coords_voxel,
            source_t=timepoint,
            core_size=core_size,
            context_halo=max_distance,
            neighbors=neighbors,
            max_distance=max_distance,
            true_edges=truth,
        )
        validate_frame_candidate_recall(tiles, video, timepoint, truth)
        supervised = []
        for tile in tiles:
            validate_tile_candidate_recall(tile, truth)
            if core_parent_groups(tile):
                supervised.append(tile)
        if supervised:
            division_tiles = [
                tile
                for tile in supervised
                if any(
                    int(tile.window.node_ids[source]) in division_sources
                    and tile.window.edge_labels is not None
                    and tile.window.edge_labels[index] > 0.5
                    for index, (source, _target) in enumerate(
                        tile.window.edge_indices.tolist()
                    )
                    if tile.core_edge_mask[index]
                )
            ]
            choices = (
                division_tiles
                if division_tiles and rng.random() < prefer_division_probability
                else supervised
            )
            return choices[int(rng.integers(0, len(choices)))]
    raise RuntimeError("could not sample a supervised HOCT tile")


@torch.no_grad()
def tile_metrics(
    model: torch.nn.Module, tile: HOCTTile, device: torch.device
) -> dict[str, float | int]:
    model.eval()
    edge_logits, orphan_logits = forward_window(model, tile, device)
    groups = core_parent_groups(tile)
    reciprocal_ranks = []
    top1 = []
    predicted_parent_by_target: dict[int, int | None] = {}
    true_parent_by_target: dict[int, int] = {}
    for target, indices, positive_local in groups:
        candidate_logits = edge_logits[
            torch.as_tensor(indices, device=edge_logits.device)
        ]
        values = torch.cat((candidate_logits, orphan_logits[target : target + 1]))
        order = torch.argsort(values, descending=True, stable=True).tolist()
        rank = order.index(positive_local) + 1
        reciprocal_ranks.append(1.0 / rank)
        top1.append(float(order[0] == positive_local))
        target_id = int(tile.window.node_ids[target])
        true_edge = int(indices[positive_local])
        true_parent = int(
            tile.window.node_ids[tile.window.edge_indices[true_edge, 0]]
        )
        true_parent_by_target[target_id] = true_parent
        if order[0] == len(indices):
            predicted_parent_by_target[target_id] = None
        else:
            predicted_edge = int(indices[order[0]])
            predicted_parent_by_target[target_id] = int(
                tile.window.node_ids[tile.window.edge_indices[predicted_edge, 0]]
            )
    daughters_by_parent: dict[int, list[int]] = {}
    for target, parent in true_parent_by_target.items():
        daughters_by_parent.setdefault(parent, []).append(target)
    divisions = [targets for targets in daughters_by_parent.values() if len(targets) >= 2]
    division_both = [
        float(all(predicted_parent_by_target[target] == true_parent_by_target[target] for target in targets))
        for targets in divisions
    ]
    return {
        "top1": float(np.mean(top1)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "division_both": float(np.mean(division_both)) if division_both else 0.0,
        "rows": len(groups),
        "division_rows": len(division_both),
    }


def aggregate_metrics(rows: Sequence[dict[str, float | int]]) -> dict[str, float | int]:
    total = sum(int(row["rows"]) for row in rows)
    division_total = sum(int(row["division_rows"]) for row in rows)
    if total <= 0:
        raise ValueError("HOCT validation has no supervised rows")
    top1 = sum(float(row["top1"]) * int(row["rows"]) for row in rows) / total
    mrr = sum(float(row["mrr"]) * int(row["rows"]) for row in rows) / total
    division = (
        sum(
            float(row["division_both"]) * int(row["division_rows"])
            for row in rows
        )
        / division_total
        if division_total
        else 0.0
    )
    return {
        "composite": 0.65 * top1 + 0.25 * mrr + 0.10 * division,
        "top1": top1,
        "mrr": mrr,
        "division_both": division,
        "rows": total,
        "division_rows": division_total,
        "tiles": len(rows),
    }


def fixed_tiles(
    videos: Sequence[graph_base.GraphVideo],
    *,
    count: int,
    seed: int,
    args: argparse.Namespace,
) -> list[HOCTTile]:
    rng = np.random.default_rng(seed)
    return [
        sample_tile(
            videos,
            rng,
            core_size=np.asarray((args.core_size,) * 3, dtype=np.float32),
            neighbors=args.neighbors,
            max_distance=args.max_distance,
            prefer_division_probability=1.0,
        )
        for _ in range(count)
    ]


@torch.no_grad()
def validate_model(
    model: torch.nn.Module, tiles: Sequence[HOCTTile], device: torch.device
) -> dict[str, float | int]:
    return aggregate_metrics([tile_metrics(model, tile, device) for tile in tiles])


def train_worker(args: argparse.Namespace) -> None:
    started = time.monotonic()
    if args.fold not in FOLD_SPECS:
        raise ValueError(f"unknown reciprocal fold: {args.fold}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"isolated HOCT worker requires one GPU, saw {torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    spec = FOLD_SPECS[args.fold]
    seed = args.seed + int(spec["seed_offset"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    output_dir = args.output_dir / args.fold
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest, synthetic_train_paths, synthetic_validation_paths = split_synthetic_paths(
        args.synthetic_root,
        validation_count=args.synthetic_validation_graphs,
        train_limit=args.synthetic_train_graphs,
    )
    train_dir = args.competition_dir / "train"
    real_train_paths = select_real_paths(
        train_dir, prefix=str(spec["source_prefix"]), limit=args.real_train_graphs
    )
    real_validation_paths = select_real_paths(
        train_dir, prefix=str(spec["target_prefix"]), limit=args.real_validation_graphs
    )
    if set(path.stem for path in real_train_paths + real_validation_paths) & set(
        OPENED_ACCEPTANCE_STEMS
    ):
        raise RuntimeError("opened processed movies leaked into HOCT full training")
    synthetic_train = [
        synthetic_video_native_geometry(path) for path in synthetic_train_paths
    ]
    synthetic_validation = [
        synthetic_video_native_geometry(path) for path in synthetic_validation_paths
    ]
    real_train = [graph_base.read_graph_video(path) for path in real_train_paths]
    real_validation = [
        graph_base.read_graph_video(path) for path in real_validation_paths
    ]

    model = torch.jit.load(str(args.pretrained_model), map_location=device).train()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(f"unexpected HOCT parameter count: {parameter_count}")
    anchors = model_anchors(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda")
    real_fixed = fixed_tiles(
        real_validation,
        count=args.real_validation_tiles,
        seed=seed + 101,
        args=args,
    )
    synthetic_fixed = fixed_tiles(
        synthetic_validation,
        count=args.synthetic_validation_tiles,
        seed=seed + 202,
        args=args,
    )
    initial_real = validate_model(model, real_fixed, device)
    initial_synthetic = validate_model(model, synthetic_fixed, device)
    model.train()
    best_score = 0.85 * float(initial_real["composite"]) + 0.15 * float(
        initial_synthetic["composite"]
    )
    best_step = 0
    best_real = initial_real
    best_synthetic = initial_synthetic
    best_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    history = [
        {
            "step": 0,
            "score": best_score,
            "eligible": True,
            "real": initial_real,
            "synthetic": initial_synthetic,
        }
    ]
    atomic_json(
        output_dir / "training_config.json",
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "fold": args.fold,
            "seed": seed,
            "parameter_count": parameter_count,
            "pretrained_sha256": sha256_file(args.pretrained_model),
            "synthetic_manifest_sha256": sha256_file(manifest),
            "synthetic_train_names": [path.name for path in synthetic_train_paths],
            "synthetic_validation_names": [
                path.name for path in synthetic_validation_paths
            ],
            "real_train_stems": [path.stem for path in real_train_paths],
            "real_validation_stems": [path.stem for path in real_validation_paths],
            "opened_acceptance_stems_excluded": sorted(OPENED_ACCEPTANCE_STEMS),
            "full_backbone_trainable_parameters": parameter_count,
            "pretrained_initialization_is_candidate": True,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
        },
    )

    optimizer.zero_grad(set_to_none=True)
    completed_step = 0
    rolling = []
    skipped = 0
    for step in range(1, args.steps + 1):
        if time.monotonic() - started >= args.max_wall_seconds - args.finalization_reserve_seconds:
            break
        use_real = bool(rng.random() < args.real_replay_probability)
        try:
            tile = sample_tile(
                real_train if use_real else synthetic_train,
                rng,
                core_size=np.asarray((args.core_size,) * 3, dtype=np.float32),
                neighbors=args.neighbors,
                max_distance=args.max_distance,
                prefer_division_probability=args.prefer_division_probability,
            )
        except RuntimeError:
            skipped += 1
            continue
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            edge_logits, orphan_logits = forward_window(model, tile, device)
            supervised_loss, stats = parental_training_loss(
                edge_logits, orphan_logits, tile, bce_weight=args.bce_weight
            )
            anchor_loss = l2sp_penalty(model, anchors)
            loss = supervised_loss + args.l2sp_weight * anchor_loss
            scaled = loss / args.gradient_accumulation
        scaler.scale(scaled).backward()
        if step % args.gradient_accumulation == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        completed_step = step
        rolling.append(float(loss.detach()))
        progress = min(step / args.steps, 1.0)
        learning_rate = args.minimum_learning_rate + 0.5 * (
            args.learning_rate - args.minimum_learning_rate
        ) * (1.0 + np.cos(np.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = float(learning_rate)
        if step % args.log_every == 0:
            print(
                f"{args.fold} step={step} loss={np.mean(rolling[-args.log_every:]):.6f} "
                f"skipped={skipped} stats={stats}",
                flush=True,
            )
        if step in {500, 1500, 3000} or step % args.validation_every == 0 or step == args.steps:
            real_metrics = validate_model(model, real_fixed, device)
            synthetic_metrics = validate_model(model, synthetic_fixed, device)
            score = 0.85 * float(real_metrics["composite"]) + 0.15 * float(
                synthetic_metrics["composite"]
            )
            eligible = bool(
                float(real_metrics["composite"])
                >= float(initial_real["composite"]) + args.minimum_real_gain
                and float(synthetic_metrics["composite"])
                >= float(initial_synthetic["composite"])
                - args.maximum_synthetic_regression
            )
            row = {
                "step": step,
                "score": score,
                "eligible": eligible,
                "real": real_metrics,
                "synthetic": synthetic_metrics,
            }
            history.append(row)
            atomic_json(output_dir / "validation_latest.json", row)
            if eligible and score > best_score:
                best_score = score
                best_step = step
                best_real = real_metrics
                best_synthetic = synthetic_metrics
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            model.train()

    if completed_step % args.gradient_accumulation:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
    model.load_state_dict(best_state)
    model.eval()
    model_path = output_dir / "hoct_adapted.pt"
    torch.jit.save(model, str(model_path))
    reloaded = torch.jit.load(str(model_path), map_location="cpu")
    if sum(parameter.numel() for parameter in reloaded.parameters()) != EXPECTED_PARAMETERS:
        raise RuntimeError("saved adapted HOCT model failed reload verification")
    atomic_json(output_dir / "validation_history.json", {"rows": history})
    terminal = {
        "schema_version": 1,
        "status": "completed",
        "run_id": RUN_ID,
        "fold": args.fold,
        "elapsed_seconds": time.monotonic() - started,
        "completed_step": completed_step,
        "skipped_tiles": skipped,
        "best_step": best_step,
        "pretrained_initialization_retained": best_step == 0,
        "initial_real": initial_real,
        "best_real": best_real,
        "initial_synthetic": initial_synthetic,
        "best_synthetic": best_synthetic,
        "model_sha256": sha256_file(model_path),
        "parameter_count": parameter_count,
        "full_backbone_trained": True,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    atomic_json(output_dir / "worker_terminal.json", terminal)
    print(json.dumps(_plain(terminal), indent=2), flush=True)


def orchestrate(args: argparse.Namespace) -> None:
    started = time.monotonic()
    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"full HOCT training requires exactly two GPUs, saw {torch.cuda.device_count()}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for gpu_index, fold in enumerate(FOLD_SPECS):
        handle = (args.output_dir / f"{fold}.log").open("w", encoding="utf-8")
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
                raise TimeoutError("full HOCT orchestrator exceeded its hard stop")
            if len(return_codes) < len(processes):
                time.sleep(5)
    finally:
        for _fold, process, handle in processes:
            if process.poll() is None:
                process.terminate()
            handle.close()
    failures = {fold: code for fold, code in return_codes.items() if code != 0}
    if failures:
        raise RuntimeError(f"full HOCT workers failed: {failures}")
    folds = {
        fold: json.loads(
            (args.output_dir / fold / "worker_terminal.json").read_text(encoding="utf-8")
        )
        for fold in FOLD_SPECS
    }
    terminal = {
        "schema_version": 1,
        "status": "completed",
        "run_id": RUN_ID,
        "elapsed_seconds": time.monotonic() - started,
        "gpu_count": 2,
        "folds": folds,
        "both_folds_improved": all(int(row["best_step"]) > 0 for row in folds.values()),
        "full_backbone_trained": True,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    atomic_json(args.output_dir / "training_terminal.json", terminal)
    print(json.dumps(_plain(terminal), indent=2), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    role = result.add_mutually_exclusive_group(required=True)
    role.add_argument("--orchestrate", action="store_true")
    role.add_argument("--worker", action="store_true")
    result.add_argument("--fold", choices=sorted(FOLD_SPECS))
    result.add_argument("--competition-dir", type=Path, required=True)
    result.add_argument("--synthetic-root", type=Path, required=True)
    result.add_argument("--pretrained-model", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--seed", type=int, default=51031)
    result.add_argument("--steps", type=int, default=20000)
    result.add_argument("--max-wall-seconds", type=int, default=36000)
    result.add_argument("--orchestrator-hard-stop-seconds", type=int, default=37800)
    result.add_argument("--finalization-reserve-seconds", type=int, default=1200)
    result.add_argument("--synthetic-train-graphs", type=int, default=1900)
    result.add_argument("--synthetic-validation-graphs", type=int, default=128)
    result.add_argument("--real-train-graphs", type=int, default=96)
    result.add_argument("--real-validation-graphs", type=int, default=12)
    result.add_argument("--real-validation-tiles", type=int, default=48)
    result.add_argument("--synthetic-validation-tiles", type=int, default=32)
    result.add_argument("--real-replay-probability", type=float, default=0.10)
    result.add_argument("--core-size", type=float, default=128.0)
    result.add_argument("--neighbors", type=int, default=8)
    result.add_argument("--max-distance", type=float, default=80.0)
    result.add_argument("--prefer-division-probability", type=float, default=0.25)
    result.add_argument("--bce-weight", type=float, default=0.20)
    result.add_argument("--l2sp-weight", type=float, default=0.001)
    result.add_argument("--learning-rate", type=float, default=0.000002)
    result.add_argument("--minimum-learning-rate", type=float, default=0.0000001)
    result.add_argument("--weight-decay", type=float, default=0.000001)
    result.add_argument("--gradient-accumulation", type=int, default=4)
    result.add_argument("--gradient-clip", type=float, default=0.5)
    result.add_argument("--minimum-real-gain", type=float, default=0.001)
    result.add_argument("--maximum-synthetic-regression", type=float, default=0.005)
    result.add_argument("--validation-every", type=int, default=2500)
    result.add_argument("--log-every", type=int, default=100)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.worker:
        if args.fold is None:
            raise ValueError("--fold is required for a worker")
        train_worker(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
