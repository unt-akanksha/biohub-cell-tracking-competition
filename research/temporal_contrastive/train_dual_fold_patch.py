#!/usr/bin/env python
"""Train reciprocal Biohub 3D appearance models on exactly two GPUs.

Each isolated worker learns from the public physical synthetic movies plus one
real embryo prefix and selects checkpoints on the opposite, unopened prefix.
The four processed-acceptance movies are excluded. This module creates model
evidence only; it contains no competition artifact or submission path.
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
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    import trainer as graph_base
    from synthetic_data import corrected_sequence_sample, division_prior_weight
except ModuleNotFoundError:
    from research.synthetic_pretrain.data import (
        corrected_sequence_sample,
        division_prior_weight,
    )
    from research.trackastra_graph import train_biohub_graph_transformer as graph_base

try:
    from model import masked_multi_positive_info_nce
    from patch_model import (
        PhysicalPatchAssociationModel,
        physical_candidate_masks,
        sample_physical_patches,
    )
except ModuleNotFoundError:
    from research.temporal_contrastive.model import masked_multi_positive_info_nce
    from research.temporal_contrastive.patch_model import (
        PhysicalPatchAssociationModel,
        physical_candidate_masks,
        sample_physical_patches,
    )


RUN_ID = "temporal-patch-dual-fold-v1"
OPENED_ACCEPTANCE_STEMS = frozenset(
    {
        "44b6_12dfb391",
        "44b6_267148e4",
        "6bba_062c8d37",
        "6bba_07e24132",
    }
)
FOLD_SPECS = {
    "target_44b6": {
        "target_prefix": "44b6",
        "source_prefix": "6bba",
        "seed_offset": 4400,
    },
    "target_6bba": {
        "target_prefix": "6bba",
        "source_prefix": "44b6",
        "seed_offset": 6600,
    },
}
REAL_VOXEL_SIZE_UM = (1.625, 0.40625, 0.40625)


@dataclass(frozen=True)
class TransitionBatch:
    source_coords: np.ndarray
    target_coords: np.ndarray
    candidate_mask: np.ndarray
    positive_mask: np.ndarray
    division_target: np.ndarray
    timepoint: int


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


def select_paths(
    paths: list[Path],
    *,
    salt: str,
    prefix: str | None = None,
    limit: int | None = None,
) -> list[Path]:
    eligible = [
        path
        for path in paths
        if (prefix is None or path.stem.startswith(f"{prefix}_"))
        and path.stem not in OPENED_ACCEPTANCE_STEMS
    ]
    ranked = sorted(
        eligible,
        key=lambda path: hashlib.sha256(f"{salt}:{path.name}".encode()).hexdigest(),
    )
    if limit is not None:
        ranked = ranked[:limit]
    if not ranked:
        raise FileNotFoundError(f"no eligible paths for prefix={prefix!r}")
    return ranked


def synthetic_split(
    root: Path, *, validation_count: int, training_count: int
) -> tuple[Path, list[Path], list[Path]]:
    manifests: list[tuple[Path, dict[str, Any]]] = []
    for path in (
        [root / "manifest.json"]
        if (root / "manifest.json").is_file()
        else root.rglob("manifest.json")
    ):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("sequences"):
            manifests.append((path, payload))
    if len(manifests) != 1:
        raise FileNotFoundError(
            f"expected one sequence manifest below {root}, found "
            f"{[str(path) for path, _payload in manifests]}"
        )
    manifest, payload = manifests[0]
    records = payload.get("sequences")
    if not isinstance(records, list):
        raise ValueError("synthetic manifest has no sequence inventory")
    records = [
        record
        for record in records
        if int(record.get("T", 0)) >= 4 and int(record.get("n_edges", 0)) > 0
    ]
    paths = [manifest.parent / str(record["file"]) for record in records]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("synthetic sequence inventory is incomplete")
    ranked = select_paths(paths, salt="temporal-patch-v1")
    if validation_count <= 0 or training_count <= 0:
        raise ValueError("synthetic split counts must be positive")
    if validation_count + training_count > len(ranked):
        raise ValueError("synthetic split exceeds available movies")
    return manifest, ranked[validation_count : validation_count + training_count], ranked[:validation_count]


def graph_arrays(video: graph_base.GraphVideo) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    id_to_row = {
        int(node_id): row for row, node_id in enumerate(video.node_ids.tolist())
    }
    edges = np.asarray(
        [(id_to_row[int(source)], id_to_row[int(target)]) for source, target in video.edges],
        dtype=np.int64,
    ).reshape(-1, 2)
    return video.times.astype(np.int32), video.coords_voxel.astype(np.float32), edges


def prepare_transition(
    times: np.ndarray,
    coords_zyx: np.ndarray,
    edges: np.ndarray,
    *,
    timepoint: int,
    voxel_size_zyx_um: tuple[float, float, float],
    radius_um: float,
    max_sources: int,
    max_targets: int,
    rng: np.random.Generator,
) -> TransitionBatch:
    """Subsample one labeled transition while preserving every chosen positive."""

    times = np.asarray(times, dtype=np.int32).reshape(-1)
    coords = np.asarray(coords_zyx, dtype=np.float32)
    links = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if coords.shape != (len(times), 3):
        raise ValueError("times and coordinates are misaligned")
    if links.size and (links.min() < 0 or links.max() >= len(times)):
        raise ValueError("edge references an unavailable node row")
    transition = links[
        (times[links[:, 0]] == timepoint)
        & (times[links[:, 1]] == timepoint + 1)
    ]
    if not len(transition):
        raise ValueError("transition has no labeled consecutive-frame edges")
    source_global = np.unique(transition[:, 0])
    if len(source_global) > max_sources:
        outgoing = {
            int(source): int(np.sum(transition[:, 0] == source))
            for source in source_global
        }
        divisions = np.asarray(
            [source for source in source_global if outgoing[int(source)] >= 2],
            dtype=np.int64,
        )
        remaining = np.asarray(
            [source for source in source_global if outgoing[int(source)] < 2],
            dtype=np.int64,
        )
        rng.shuffle(divisions)
        rng.shuffle(remaining)
        source_global = np.concatenate((divisions, remaining))[:max_sources]
    source_global = np.sort(source_global)
    target_global = np.flatnonzero(times == timepoint + 1)
    source_to_local = {int(row): index for index, row in enumerate(source_global)}
    target_to_local = {int(row): index for index, row in enumerate(target_global)}
    selected_links = [
        (source_to_local[int(source)], target_to_local[int(target)])
        for source, target in transition
        if int(source) in source_to_local
    ]
    candidates, positives = physical_candidate_masks(
        coords[source_global],
        coords[target_global],
        np.asarray(selected_links, dtype=np.int64),
        voxel_size_zyx_um=voxel_size_zyx_um,
        radius_um=radius_um,
    )
    required_targets = np.flatnonzero(positives.any(axis=0))
    candidate_targets = np.flatnonzero(candidates.any(axis=0))
    if len(required_targets) > max_targets:
        raise ValueError("positive targets exceed max_targets")
    if len(candidate_targets) > max_targets:
        optional = np.setdiff1d(candidate_targets, required_targets, assume_unique=True)
        source_um = coords[source_global] * np.asarray(voxel_size_zyx_um)
        target_um = coords[target_global[optional]] * np.asarray(voxel_size_zyx_um)
        nearest = np.min(
            np.linalg.norm(source_um[:, None] - target_um[None], axis=2), axis=0
        )
        optional = optional[np.argsort(nearest, kind="stable")]
        keep_targets = np.sort(
            np.concatenate((required_targets, optional[: max_targets - len(required_targets)]))
        )
    else:
        keep_targets = candidate_targets
    candidates = candidates[:, keep_targets]
    positives = positives[:, keep_targets]
    valid_sources = positives.any(axis=1) & (
        candidates.sum(axis=1) > positives.sum(axis=1)
    )
    if not np.any(valid_sources):
        raise ValueError("transition contains no labeled source with a hard negative")
    candidates = candidates[valid_sources]
    positives = positives[valid_sources]
    division = positives.sum(axis=1) >= 2
    return TransitionBatch(
        source_coords=coords[source_global[valid_sources]],
        target_coords=coords[target_global[keep_targets]],
        candidate_mask=candidates,
        positive_mask=positives,
        division_target=division.astype(np.float32),
        timepoint=int(timepoint),
    )


def transition_metrics(
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    candidate_mask: torch.Tensor,
    positive_mask: torch.Tensor,
) -> dict[str, float | int]:
    logits = source_embeddings @ target_embeddings.transpose(0, 1)
    logits = logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)
    order = torch.argsort(logits, dim=1, descending=True, stable=True)
    top1 = positive_mask.gather(1, order[:, :1]).any(dim=1).float()
    ranked_positive = positive_mask.gather(1, order)
    ranks = torch.argmax(ranked_positive.to(torch.int64), dim=1) + 1
    divisions = positive_mask.sum(dim=1) >= 2
    if torch.any(divisions):
        top2_division = (
            positive_mask[divisions].gather(1, order[divisions, :2]).sum(dim=1) >= 2
        ).float()
        division_recall = float(top2_division.mean())
        division_rows = int(divisions.sum())
    else:
        division_recall = 0.0
        division_rows = 0
    return {
        "top1": float(top1.mean()),
        "mrr": float((1.0 / ranks.float()).mean()),
        "division_top2": division_recall,
        "rows": len(source_embeddings),
        "division_rows": division_rows,
    }


def aggregate_metrics(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("no validation metric rows")
    edge_rows = sum(int(row["rows"]) for row in rows)
    division_rows = sum(int(row["division_rows"]) for row in rows)
    top1 = sum(float(row["top1"]) * int(row["rows"]) for row in rows) / edge_rows
    mrr = sum(float(row["mrr"]) * int(row["rows"]) for row in rows) / edge_rows
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
        "rows": edge_rows,
        "division_rows": division_rows,
        "transitions": len(rows),
    }


class MovieStore:
    def __init__(self, competition_dir: Path):
        self.train_dir = competition_dir / "train"

    @lru_cache(maxsize=8)
    def synthetic(self, path: Path):
        return corrected_sequence_sample(path)

    @lru_cache(maxsize=8)
    def real_graph(self, path: Path):
        return graph_base.read_graph_video(path)

    @lru_cache(maxsize=8)
    def real_image(self, stem: str):
        import zarr

        return zarr.open_group(str(self.train_dir / f"{stem}.zarr"), mode="r")["0"]


def load_random_transition(
    store: MovieStore,
    path: Path,
    *,
    synthetic: bool,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, TransitionBatch, tuple[float, float, float]]:
    if synthetic:
        sample = store.synthetic(path)
        volumes = sample.volumes
        times = sample.nodes[:, 0].astype(np.int32)
        coords = sample.nodes[:, 1:4].astype(np.float32)
        edges = sample.edges
        voxel_size = tuple(float(value) for value in sample.voxel_um)
    else:
        video = store.real_graph(path)
        times, coords, edges = graph_arrays(video)
        volumes = store.real_image(path.stem)
        voxel_size = REAL_VOXEL_SIZE_UM
    valid_times = np.unique(times[edges[:, 0]])
    valid_times = valid_times[
        np.isin(valid_times + 1, np.unique(times[edges[:, 1]]))
    ]
    if not len(valid_times):
        raise ValueError("movie has no consecutive labeled transition")
    timepoint = int(rng.choice(valid_times))
    batch = prepare_transition(
        times,
        coords,
        edges,
        timepoint=timepoint,
        voxel_size_zyx_um=voxel_size,
        radius_um=args.candidate_radius_um,
        max_sources=args.max_sources,
        max_targets=args.max_targets,
        rng=rng,
    )
    return (
        np.asarray(volumes[timepoint]),
        np.asarray(volumes[timepoint + 1]),
        batch,
        voxel_size,
    )


def encode_transition(
    model: PhysicalPatchAssociationModel,
    source_volume: np.ndarray,
    target_volume: np.ndarray,
    batch: TransitionBatch,
    voxel_size: tuple[float, float, float],
    device: torch.device,
    *,
    augment: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_patches = sample_physical_patches(
        torch.as_tensor(source_volume, device=device),
        batch.source_coords,
        voxel_size_zyx_um=voxel_size,
    )
    target_patches = sample_physical_patches(
        torch.as_tensor(target_volume, device=device),
        batch.target_coords,
        voxel_size_zyx_um=voxel_size,
    )
    patches = torch.cat((source_patches, target_patches), dim=0)
    if augment:
        contrast = torch.empty((len(patches), 1, 1, 1, 1), device=device).uniform_(0.8, 1.2)
        noise = torch.randn_like(patches) * 0.03
        patches = patches * contrast + noise
        for axis in (2, 3, 4):
            if bool(torch.rand((), device=device) < 0.5):
                patches = torch.flip(patches, dims=(axis,))
    embeddings, division_logits = model(patches)
    source_count = len(batch.source_coords)
    return (
        embeddings[:source_count],
        embeddings[source_count:],
        division_logits[:source_count],
    )


@torch.no_grad()
def validate_model(
    model: PhysicalPatchAssociationModel,
    examples: list[tuple[np.ndarray, np.ndarray, TransitionBatch, tuple[float, float, float]]],
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    rows = []
    for source_volume, target_volume, batch, voxel_size in examples:
        source, target, _division = encode_transition(
            model,
            source_volume,
            target_volume,
            batch,
            voxel_size,
            device,
            augment=False,
        )
        rows.append(
            transition_metrics(
                source,
                target,
                torch.as_tensor(batch.candidate_mask, device=device),
                torch.as_tensor(batch.positive_mask, device=device),
            )
        )
    model.train()
    return aggregate_metrics(rows)


def fixed_examples(
    store: MovieStore,
    paths: list[Path],
    *,
    synthetic: bool,
    count: int,
    seed: int,
    args: argparse.Namespace,
) -> list[tuple[np.ndarray, np.ndarray, TransitionBatch, tuple[float, float, float]]]:
    rng = np.random.default_rng(seed)
    examples = []
    attempts = 0
    while len(examples) < count and attempts < count * 20:
        path = paths[int(rng.integers(0, len(paths)))]
        attempts += 1
        try:
            examples.append(
                load_random_transition(
                    store, path, synthetic=synthetic, args=args, rng=rng
                )
            )
        except ValueError:
            continue
    if len(examples) != count:
        raise RuntimeError(f"built only {len(examples)}/{count} fixed examples")
    return examples


def state_dict_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


@torch.no_grad()
def update_ema_model(
    ema_model: torch.nn.Module,
    model: torch.nn.Module,
    *,
    decay: float,
) -> None:
    """Update a full-model exponential average after an optimizer step."""

    if not 0.0 <= decay < 1.0:
        raise ValueError("EMA decay must lie in [0, 1)")
    model_state = model.state_dict()
    ema_state = ema_model.state_dict()
    if model_state.keys() != ema_state.keys():
        raise ValueError("EMA and training model state inventories differ")
    for name, averaged in ema_state.items():
        current = model_state[name].detach()
        if averaged.is_floating_point():
            averaged.mul_(decay).add_(current, alpha=1.0 - decay)
        else:
            averaged.copy_(current)


def train_worker(args: argparse.Namespace) -> None:
    started = time.monotonic()
    if args.fold not in FOLD_SPECS:
        raise ValueError(f"unknown fold: {args.fold}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"isolated appearance worker requires one GPU, saw {torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    spec = FOLD_SPECS[args.fold]
    seed = args.seed + int(spec["seed_offset"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rng = np.random.default_rng(seed)
    output_dir = args.output_dir / args.fold
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest, synthetic_train, synthetic_validation = synthetic_split(
        args.synthetic_root,
        validation_count=args.synthetic_validation_movies,
        training_count=args.synthetic_train_movies,
    )
    real_paths = list((args.competition_dir / "train").glob("*.geff"))
    real_train = select_paths(
        real_paths,
        salt=f"{RUN_ID}:train",
        prefix=str(spec["source_prefix"]),
        limit=args.real_train_movies,
    )
    target_pool = select_paths(
        real_paths,
        salt=f"{RUN_ID}:target",
        prefix=str(spec["target_prefix"]),
        limit=args.real_validation_movies + args.real_calibration_movies,
    )
    real_validation = target_pool[: args.real_validation_movies]
    real_calibration = target_pool[args.real_validation_movies :]
    if len(real_calibration) != args.real_calibration_movies:
        raise RuntimeError("reciprocal target pool cannot fill the calibration split")
    if set(path.stem for path in real_train + target_pool) & OPENED_ACCEPTANCE_STEMS:
        raise RuntimeError("opened acceptance labels entered appearance training")
    store = MovieStore(args.competition_dir)
    real_fixed = fixed_examples(
        store,
        real_validation,
        synthetic=False,
        count=args.real_validation_transitions,
        seed=seed + 101,
        args=args,
    )
    synthetic_fixed = fixed_examples(
        store,
        synthetic_validation,
        synthetic=True,
        count=args.synthetic_validation_transitions,
        seed=seed + 202,
        args=args,
    )

    model = PhysicalPatchAssociationModel(
        base_channels=args.base_channels,
        embedding_channels=args.embedding_channels,
    ).to(device)
    ema_model = PhysicalPatchAssociationModel(
        base_channels=args.base_channels,
        embedding_channels=args.embedding_channels,
    ).to(device)
    ema_model.load_state_dict(model.state_dict(), strict=True)
    ema_model.requires_grad_(False)
    ema_model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count < 5_000_000:
        raise RuntimeError(f"appearance network is unexpectedly small: {parameter_count}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda")
    initial_real = validate_model(ema_model, real_fixed, device)
    initial_synthetic = validate_model(ema_model, synthetic_fixed, device)
    ema_model.eval()
    best_state = state_dict_cpu(ema_model)
    best_step = 0
    best_real = initial_real
    best_synthetic = initial_synthetic
    best_score = 0.85 * float(initial_real["composite"]) + 0.15 * float(
        initial_synthetic["composite"]
    )
    history = [
        {
            "step": 0,
            "score": best_score,
            "eligible": False,
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
            "base_channels": args.base_channels,
            "embedding_channels": args.embedding_channels,
            "checkpoint_weight_source": "optimizer-step exponential moving average",
            "ema_decay": args.ema_decay,
            "source_prefix": spec["source_prefix"],
            "target_prefix": spec["target_prefix"],
            "opened_acceptance_stems_excluded": sorted(OPENED_ACCEPTANCE_STEMS),
            "synthetic_manifest_sha256": sha256_file(manifest),
            "synthetic_train_names": [path.name for path in synthetic_train],
            "synthetic_validation_names": [path.name for path in synthetic_validation],
            "real_train_stems": [path.stem for path in real_train],
            "real_validation_stems": [path.stem for path in real_validation],
            "real_calibration_stems_reserved": [path.stem for path in real_calibration],
            "calibration_ground_truth_read": False,
            "candidate_radius_um": args.candidate_radius_um,
            "patch_shape": [17, 17, 17],
            "patch_half_extent_um": [8.0, 8.0, 8.0],
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
        },
    )

    optimizer.zero_grad(set_to_none=True)
    completed_step = 0
    skipped_batches = 0
    rolling: list[float] = []
    for step in range(1, args.steps + 1):
        if time.monotonic() - started >= args.max_wall_seconds - args.finalization_reserve_seconds:
            break
        use_real = bool(rng.random() < args.real_replay_probability)
        paths = real_train if use_real else synthetic_train
        try:
            example = load_random_transition(
                store,
                paths[int(rng.integers(0, len(paths)))],
                synthetic=not use_real,
                args=args,
                rng=rng,
            )
        except ValueError:
            skipped_batches += 1
            continue
        source_volume, target_volume, batch, voxel_size = example
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            source, target, division_logits = encode_transition(
                model,
                source_volume,
                target_volume,
                batch,
                voxel_size,
                device,
                augment=True,
            )
            link_loss = masked_multi_positive_info_nce(
                source,
                target,
                torch.as_tensor(batch.positive_mask, device=device),
                torch.as_tensor(batch.candidate_mask, device=device),
                temperature=args.temperature,
            )
            division_loss = F.binary_cross_entropy_with_logits(
                division_logits.float(),
                torch.as_tensor(batch.division_target, device=device),
            )
            loss = link_loss + args.division_loss_weight * division_loss * (
                1.0 if use_real else division_prior_weight()
            )
            scaled_loss = loss / args.gradient_accumulation
        scaler.scale(scaled_loss).backward()
        if step % args.gradient_accumulation == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            update_ema_model(ema_model, model, decay=args.ema_decay)
            optimizer.zero_grad(set_to_none=True)
        completed_step = step
        rolling.append(float(loss))
        progress = min(step / args.steps, 1.0)
        learning_rate = args.minimum_learning_rate + 0.5 * (
            args.learning_rate - args.minimum_learning_rate
        ) * (1.0 + np.cos(np.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = float(learning_rate)
        if step % args.log_every == 0:
            print(
                f"{args.fold} step={step} loss={np.mean(rolling[-args.log_every:]):.6f} "
                f"skipped={skipped_batches}",
                flush=True,
            )
        if step in {500, 1500, 3000} or step % args.validation_every == 0 or step == args.steps:
            real_metrics = validate_model(ema_model, real_fixed, device)
            synthetic_metrics = validate_model(ema_model, synthetic_fixed, device)
            ema_model.eval()
            score = 0.85 * float(real_metrics["composite"]) + 0.15 * float(
                synthetic_metrics["composite"]
            )
            eligible = bool(
                float(real_metrics["top1"]) >= args.minimum_real_top1
                and float(synthetic_metrics["top1"]) >= args.minimum_synthetic_top1
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
                best_step = step
                best_score = score
                best_real = real_metrics
                best_synthetic = synthetic_metrics
                best_state = state_dict_cpu(ema_model)
            model.train()

    if completed_step % args.gradient_accumulation:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        update_ema_model(ema_model, model, decay=args.ema_decay)
    model.load_state_dict(best_state)
    model_path = output_dir / "appearance_model.pt"
    torch.save(model.state_dict(), model_path)
    atomic_json(output_dir / "validation_history.json", {"rows": history})
    terminal = {
        "schema_version": 1,
        "status": "completed",
        "run_id": RUN_ID,
        "fold": args.fold,
        "elapsed_seconds": time.monotonic() - started,
        "completed_step": completed_step,
        "skipped_batches": skipped_batches,
        "best_step": best_step,
        "best_score": best_score,
        "initial_real": initial_real,
        "initial_synthetic": initial_synthetic,
        "best_real": best_real,
        "best_synthetic": best_synthetic,
        "model_sha256": sha256_file(model_path),
        "parameter_count": parameter_count,
        "checkpoint_weight_source": "optimizer-step exponential moving average",
        "ema_decay": args.ema_decay,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    atomic_json(output_dir / "worker_terminal.json", terminal)
    print(json.dumps(_plain(terminal), indent=2), flush=True)


def orchestrate(args: argparse.Namespace) -> None:
    started = time.monotonic()
    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"appearance training requires exactly two GPUs, saw {torch.cuda.device_count()}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for gpu_index, fold in enumerate(FOLD_SPECS):
        log_handle = (args.output_dir / f"{fold}.log").open("w", encoding="utf-8")
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
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((fold, process, log_handle))
    return_codes: dict[str, int] = {}
    try:
        while len(return_codes) < len(processes):
            for fold, process, _handle in processes:
                code = process.poll()
                if code is not None and fold not in return_codes:
                    return_codes[fold] = int(code)
            if time.monotonic() - started >= args.orchestrator_hard_stop_seconds:
                raise TimeoutError("appearance orchestrator exceeded its hard stop")
            if len(return_codes) < len(processes):
                time.sleep(5)
    finally:
        for _fold, process, handle in processes:
            if process.poll() is None:
                process.terminate()
            handle.close()
    failures = {fold: code for fold, code in return_codes.items() if code != 0}
    if failures:
        raise RuntimeError(f"appearance workers failed: {failures}")
    terminals = {
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
        "folds": terminals,
        "both_folds_trained": all(int(row["best_step"]) > 0 for row in terminals.values()),
        "public_predictions_copied": False,
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
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--seed", type=int, default=41027)
    result.add_argument("--steps", type=int, default=30000)
    result.add_argument("--max-wall-seconds", type=int, default=36000)
    result.add_argument("--orchestrator-hard-stop-seconds", type=int, default=37800)
    result.add_argument("--finalization-reserve-seconds", type=int, default=1200)
    result.add_argument("--synthetic-train-movies", type=int, default=1900)
    result.add_argument("--synthetic-validation-movies", type=int, default=128)
    result.add_argument("--real-train-movies", type=int, default=96)
    result.add_argument("--real-validation-movies", type=int, default=12)
    result.add_argument("--real-calibration-movies", type=int, default=12)
    result.add_argument("--real-validation-transitions", type=int, default=48)
    result.add_argument("--synthetic-validation-transitions", type=int, default=32)
    result.add_argument("--real-replay-probability", type=float, default=0.20)
    result.add_argument("--base-channels", type=int, default=64)
    result.add_argument("--embedding-channels", type=int, default=256)
    result.add_argument("--candidate-radius-um", type=float, default=32.0)
    result.add_argument("--max-sources", type=int, default=48)
    result.add_argument("--max-targets", type=int, default=128)
    result.add_argument("--temperature", type=float, default=0.10)
    result.add_argument("--division-loss-weight", type=float, default=0.20)
    result.add_argument("--learning-rate", type=float, default=2e-4)
    result.add_argument("--minimum-learning-rate", type=float, default=2e-6)
    result.add_argument("--weight-decay", type=float, default=1e-5)
    result.add_argument("--ema-decay", type=float, default=0.997)
    result.add_argument("--gradient-accumulation", type=int, default=2)
    result.add_argument("--minimum-real-top1", type=float, default=0.70)
    result.add_argument("--minimum-synthetic-top1", type=float, default=0.85)
    result.add_argument("--validation-every", type=int, default=3000)
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
