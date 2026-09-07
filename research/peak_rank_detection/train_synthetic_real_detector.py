#!/usr/bin/env python
"""Train and gate an independent temporal 3D peak-ranking detector.

Synthetic256 provides complete detection supervision.  Competition-train
event crops provide positive-only local ranking and subvoxel supervision.  The
optimization, selection, and sealed-audit inventories are disjoint; audit
files are not opened until a serialized checkpoint passes every selection
gate.  Competition test data, leaderboard scores, and public notebook weights
are never read.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.peak_rank_detection.model import (
    DEFAULT_DEPTHS,
    DEFAULT_WIDTHS,
    TemporalPeakRankDetector,
    count_parameters,
)
from research.peak_rank_detection.objectives import (
    focal_heatmap_loss,
    points_to_gaussian_heatmap,
    sparse_peak_ranking_loss,
    subvoxel_offset_loss,
)
from research.synthetic_pretrain.data import SequenceSample, corrected_sequence_sample


RUN_ID = "synthetic256-real-positive-temporal-peak-rank-v1"
TRAIN_INDICES = tuple(range(240))
SELECTION_INDICES = tuple(range(240, 248))
AUDIT_INDICES = tuple(range(248, 256))
EXPECTED_REAL_ROLE_COUNTS = {
    "optimization": 146,
    "selection": 17,
    "sealed_audit": 14,
}
EXPECTED_EXCLUDED_PROBE_STEMS = (
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
)


@dataclass(frozen=True)
class TrainingExample:
    frames: np.ndarray
    points: np.ndarray
    division_points: np.ndarray
    source: str
    identity: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sequence_paths(root: Path) -> list[Path]:
    paths = [root / "sequences" / f"seq_{index:04d}.npz" for index in range(256)]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("Synthetic256 sequence inventory is incomplete")
    if sorted((root / "sequences").glob("seq_*.npz")) != paths:
        raise ValueError("Synthetic256 sequence inventory changed")
    return paths


def validate_real_manifest(root: Path, expected_sha256: str) -> dict[str, Any]:
    manifest_path = root / "real_localization_shard_manifest.json"
    if sha256_file(manifest_path) != expected_sha256.lower():
        raise ValueError("real localization manifest hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("status") == "complete"
        and manifest.get("run_id") == "competition-real-localization-shards-v1"
        and manifest.get("competition_train_data_read") is True
        and manifest.get("competition_test_data_read") is False
        and manifest.get("public_leaderboard_used_for_selection") is False
        and manifest.get("submission_created") is False
        and manifest.get("authorized_for_submission") is False
        and tuple(manifest.get("excluded_final_probe_stems", ()))
        == EXPECTED_EXCLUDED_PROBE_STEMS
    ):
        raise ValueError("real localization manifest violates clean-training policy")
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise ValueError("real localization manifest has no file inventory")
    for role, expected_count in EXPECTED_REAL_ROLE_COUNTS.items():
        role_rows = [row for row in rows if row.get("role") == role]
        if len(role_rows) != expected_count:
            raise ValueError(f"unexpected {role} shard count")
    role_stems = {
        role: {str(row["stem"]) for row in rows if row.get("role") == role}
        for role in EXPECTED_REAL_ROLE_COUNTS
    }
    for left_index, left in enumerate(role_stems):
        for right in tuple(role_stems)[left_index + 1 :]:
            if role_stems[left] & role_stems[right]:
                raise ValueError(f"real stems overlap between {left} and {right}")
    for row in rows:
        path = root / str(row["path"])
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"real localization shard changed: {path}")
        if row["role"] != "sealed_audit" and sha256_file(path) != row["sha256"]:
            raise ValueError(f"real localization shard hash changed: {path}")
    return manifest


def real_paths(manifest: dict[str, Any], root: Path, role: str) -> list[Path]:
    return [
        root / str(row["path"])
        for row in manifest["files"]
        if row["role"] == role
    ]


def center_points(sample: SequenceSample, center: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.flatnonzero(sample.nodes[:, 0].astype(np.int64) == int(center))
    points = sample.nodes[rows, 1:4].astype(np.float32, copy=True)
    division_rows = set(int(value) for value in sample.divisions.tolist())
    division = sample.nodes[
        [int(row) for row in rows if int(row) in division_rows], 1:4
    ].astype(np.float32, copy=True)
    return points.reshape(-1, 3), division.reshape(-1, 3)


def make_example(path: Path, *, center: int, source: str) -> TrainingExample:
    sample = corrected_sequence_sample(path)
    if sample.volumes.shape[1:] != (64, 64, 64):
        raise ValueError(f"unexpected volume shape in {path}")
    if not 1 <= center < sample.volumes.shape[0] - 1:
        raise ValueError("center frame must have temporal context")
    points, divisions = center_points(sample, center)
    if not len(points):
        raise ValueError(f"training example has no center-frame points: {path}")
    return TrainingExample(
        frames=np.ascontiguousarray(sample.volumes[center - 1 : center + 2]),
        points=points,
        division_points=divisions,
        source=source,
        identity=f"{path.stem}:t{center}",
    )


def load_synthetic_examples(paths: Sequence[Path], indices: Sequence[int]) -> list[TrainingExample]:
    examples = []
    for index in indices:
        sample = corrected_sequence_sample(paths[int(index)])
        for center in range(1, sample.volumes.shape[0] - 1):
            points, divisions = center_points(sample, center)
            examples.append(
                TrainingExample(
                    np.ascontiguousarray(sample.volumes[center - 1 : center + 2]),
                    points,
                    divisions,
                    "synthetic",
                    f"seq_{int(index):04d}:t{center}",
                )
            )
    return examples


def load_real_examples(paths: Sequence[Path], *, role: str) -> list[TrainingExample]:
    examples = []
    for path in paths:
        # Reading sealed paths occurs only in the explicitly gated audit call.
        examples.append(make_example(path, center=1, source=f"real_{role}"))
    return examples


def normalize_frames(frames: np.ndarray) -> np.ndarray:
    values = np.asarray(frames, dtype=np.float32)
    low, high = np.quantile(values, (0.001, 0.999))
    if not np.isfinite(low + high) or high <= low:
        raise ValueError("volume has no finite robust intensity range")
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def augment_example(
    example: TrainingExample, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    frames = normalize_frames(example.frames)
    points = example.points.copy()
    shape = np.asarray(frames.shape[-3:], dtype=np.float32)
    for spatial_axis in range(3):
        if rng.random() < 0.5:
            frames = np.flip(frames, axis=spatial_axis + 1)
            points[:, spatial_axis] = shape[spatial_axis] - 1 - points[:, spatial_axis]
    rotation = int(rng.integers(0, 4))
    for _ in range(rotation):
        old_y = points[:, 1].copy()
        old_x = points[:, 2].copy()
        points[:, 1] = shape[2] - 1 - old_x
        points[:, 2] = old_y
        frames = np.rot90(frames, k=1, axes=(-2, -1))
        shape[1], shape[2] = shape[2], shape[1]
    if rng.random() < 0.5:
        frames = frames[::-1]
    gamma = float(rng.uniform(0.7, 1.5))
    frames = np.power(frames, gamma)
    frames = frames * float(rng.uniform(0.85, 1.15)) + float(
        rng.uniform(-0.08, 0.08)
    )
    noise = rng.normal(0.0, rng.uniform(0.0, 0.035), size=frames.shape)
    frames = np.clip(frames + noise, 0.0, 1.0).astype(np.float32)
    return np.ascontiguousarray(frames), np.ascontiguousarray(points)


def positive_logit_loss(logits: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    centers = points.round().long()
    shape = torch.tensor(logits.shape[-3:], device=logits.device)
    centers = centers[((centers >= 0) & (centers < shape)).all(dim=1)]
    if not len(centers):
        return logits.sum() * 0.0
    values = logits[0, 0, centers[:, 0], centers[:, 1], centers[:, 2]]
    return -F.logsigmoid(values).mean()


def training_loss(
    prediction: dict[str, Any],
    points: torch.Tensor,
    *,
    complete_labels: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = prediction["logits"]
    offsets = prediction["offsets"]
    rank_points = points
    if len(points) > 64:
        rank_indices = torch.linspace(
            0, len(points) - 1, steps=64, device=points.device
        ).round().long()
        rank_points = points[rank_indices]
    rank = sparse_peak_ranking_loss(logits, [rank_points])
    offset = subvoxel_offset_loss(offsets, [points])
    target = torch.from_numpy(
        points_to_gaussian_heatmap(points.detach().cpu().numpy(), logits.shape[-3:])
    )[None, None].to(device=logits.device, dtype=logits.dtype)
    if complete_labels:
        heatmap = focal_heatmap_loss(logits, target)
        auxiliary = logits.sum() * 0.0
        for auxiliary_logits in prediction["auxiliary_logits"]:
            scale = logits.shape[-1] // auxiliary_logits.shape[-1]
            auxiliary_target = F.max_pool3d(target, kernel_size=scale, stride=scale)
            auxiliary = auxiliary + focal_heatmap_loss(
                auxiliary_logits, auxiliary_target
            )
        total = heatmap + 0.35 * rank + 0.20 * offset + 0.08 * auxiliary
    else:
        heatmap = positive_logit_loss(logits, points)
        auxiliary = logits.sum() * 0.0
        for auxiliary_logits in prediction["auxiliary_logits"]:
            scaled = points * (
                auxiliary_logits.shape[-1] / float(logits.shape[-1])
            )
            auxiliary = auxiliary + positive_logit_loss(auxiliary_logits, scaled)
        total = heatmap + 0.75 * rank + 0.20 * offset + 0.10 * auxiliary
    return total, {
        "heatmap": float(heatmap.detach()),
        "rank": float(rank.detach()),
        "offset": float(offset.detach()),
        "auxiliary": float(auxiliary.detach()),
    }


@torch.no_grad()
def prediction_peaks(
    model: TemporalPeakRankDetector,
    example: TrainingExample,
    *,
    device: torch.device,
    maximum_predictions: int,
) -> tuple[np.ndarray, np.ndarray]:
    frames = torch.from_numpy(normalize_frames(example.frames))[None].to(device)
    output = model(frames)
    probabilities = torch.sigmoid(output["logits"])[0, 0]
    local = probabilities == F.max_pool3d(
        probabilities[None, None], kernel_size=3, stride=1, padding=1
    )[0, 0]
    coords = torch.nonzero(local, as_tuple=False)
    scores = probabilities[local]
    keep = scores.topk(min(maximum_predictions, len(scores))).indices
    coords, scores = coords[keep], scores[keep]
    learned_offsets = output["offsets"][
        0, :, coords[:, 0], coords[:, 1], coords[:, 2]
    ].transpose(0, 1)
    return (
        (coords.float() + learned_offsets).cpu().numpy(),
        scores.cpu().numpy(),
    )


def detection_average_precision(
    predicted: np.ndarray,
    scores: np.ndarray,
    truth: np.ndarray,
    *,
    radius: float = 2.5,
) -> tuple[float, float]:
    truth = np.asarray(truth, dtype=np.float32).reshape(-1, 3)
    predicted = np.asarray(predicted, dtype=np.float32).reshape(-1, 3)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(scores) != len(predicted):
        raise ValueError("prediction scores and coordinates differ")
    if not len(truth):
        return 0.0, 0.0
    matched: set[int] = set()
    true_positive = np.zeros(len(predicted), dtype=bool)
    order = np.argsort(-scores, kind="stable")
    for rank, prediction_index in enumerate(order):
        distances = np.linalg.norm(truth - predicted[prediction_index], axis=1)
        for truth_index in np.argsort(distances, kind="stable"):
            if float(distances[truth_index]) > radius:
                break
            if int(truth_index) not in matched:
                matched.add(int(truth_index))
                true_positive[rank] = True
                break
    precision = np.cumsum(true_positive) / np.arange(1, len(predicted) + 1)
    ap = float(precision[true_positive].sum() / len(truth))
    return ap, len(matched) / len(truth)


def local_positive_metrics(
    predicted: np.ndarray,
    scores: np.ndarray,
    truth: np.ndarray,
    *,
    search_radius: float = 6.0,
    success_radius: float = 2.5,
) -> tuple[list[float], list[bool]]:
    del scores
    distances, successes = [], []
    for point in np.asarray(truth, dtype=np.float32).reshape(-1, 3):
        separation = np.linalg.norm(predicted - point, axis=1)
        local = separation[separation <= search_radius]
        distance = float(local.min()) if len(local) else float(search_radius)
        distances.append(distance)
        successes.append(distance <= success_radius)
    return distances, successes


@torch.no_grad()
def evaluate(
    model: TemporalPeakRankDetector,
    synthetic: Sequence[TrainingExample],
    real: Sequence[TrainingExample],
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    synthetic_rows = []
    for example in synthetic:
        predicted, scores = prediction_peaks(
            model,
            example,
            device=device,
            maximum_predictions=max(32, len(example.points) * 3),
        )
        ap, recall = detection_average_precision(predicted, scores, example.points)
        synthetic_rows.append(
            {"identity": example.identity, "average_precision": ap, "recall": recall}
        )
    real_rows, real_distances, real_successes = [], [], []
    for example in real:
        predicted, scores = prediction_peaks(
            model,
            example,
            device=device,
            maximum_predictions=max(64, len(example.points) * 32),
        )
        distances, successes = local_positive_metrics(
            predicted, scores, example.points
        )
        real_distances.extend(distances)
        real_successes.extend(successes)
        real_rows.append(
            {
                "identity": example.identity,
                "points": len(example.points),
                "mean_distance_voxels": float(np.mean(distances)),
                "positive_peak_recall": float(np.mean(successes)),
            }
        )
    result = {
        "synthetic": {
            "movies": len(synthetic_rows),
            "mean_average_precision": float(
                np.mean([row["average_precision"] for row in synthetic_rows])
            ),
            "worst_average_precision": float(
                min(row["average_precision"] for row in synthetic_rows)
            ),
            "mean_recall": float(np.mean([row["recall"] for row in synthetic_rows])),
            "rows": synthetic_rows,
        },
        "real_positive_only": {
            "crops": len(real_rows),
            "points": len(real_distances),
            "positive_peak_recall": float(np.mean(real_successes)),
            "mean_distance_voxels": float(np.mean(real_distances)),
            "p90_distance_voxels": float(np.quantile(real_distances, 0.9)),
            "rows": real_rows,
        },
    }
    model.train()
    return result


def selection_passed(metrics: dict[str, Any]) -> bool:
    synthetic = metrics["synthetic"]
    real = metrics["real_positive_only"]
    return bool(
        synthetic["mean_average_precision"] >= 0.80
        and synthetic["worst_average_precision"] >= 0.65
        and synthetic["mean_recall"] >= 0.90
        and real["positive_peak_recall"] >= 0.85
        and real["mean_distance_voxels"] <= 2.25
        and real["p90_distance_voxels"] <= 3.5
    )


def audit_passed(metrics: dict[str, Any]) -> bool:
    synthetic = metrics["synthetic"]
    real = metrics["real_positive_only"]
    return bool(
        synthetic["mean_average_precision"] >= 0.78
        and synthetic["worst_average_precision"] >= 0.60
        and synthetic["mean_recall"] >= 0.88
        and real["positive_peak_recall"] >= 0.80
        and real["mean_distance_voxels"] <= 2.50
        and real["p90_distance_voxels"] <= 4.0
    )


def parse_ints(value: str, *, count: int, name: str) -> tuple[int, ...]:
    result = tuple(int(part) for part in value.split(","))
    if len(result) != count or min(result) <= 0:
        raise ValueError(f"{name} must contain {count} positive integers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--real-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12_000)
    parser.add_argument("--validation-every", type=int, default=1_000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--real-frequency", type=int, default=4)
    parser.add_argument("--widths", default=",".join(map(str, DEFAULT_WIDTHS)))
    parser.add_argument("--depths", default=",".join(map(str, DEFAULT_DEPTHS)))
    parser.add_argument("--seed", type=int, default=1_041_729)
    parser.add_argument("--max-wall-seconds", type=float, default=25_200.0)
    return parser.parse_args()


def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_value, value in zip(
            ema.state_dict().values(), model.state_dict().values(), strict=True
        ):
            if ema_value.is_floating_point():
                ema_value.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                ema_value.copy_(value)


def checkpoint_state(model: torch.nn.Module, *, step: int) -> dict[str, Any]:
    return {
        "step": int(step),
        "state_dict": {
            name: value.detach().cpu().half()
            if value.is_floating_point()
            else value.detach().cpu()
            for name, value in model.state_dict().items()
        },
    }


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or not 0 < args.validation_every <= args.steps:
        raise ValueError("invalid step or validation interval")
    if args.log_every <= 0 or not 0 < args.learning_rate <= 1e-3:
        raise ValueError("invalid logging interval or learning rate")
    if not 0 < args.minimum_learning_rate <= args.learning_rate:
        raise ValueError("minimum learning rate is invalid")
    if not 0.9 <= args.ema_decay < 1.0 or args.real_frequency < 2:
        raise ValueError("EMA decay or real frequency is invalid")
    if args.max_wall_seconds <= 0:
        raise ValueError("wall limit must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("temporal peak-ranking training requires CUDA")

    widths = parse_ints(args.widths, count=4, name="widths")
    depths = parse_ints(args.depths, count=4, name="depths")
    args.output_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(args.seed)

    paths = sequence_paths(args.synthetic_root)
    manifest = validate_real_manifest(args.real_root, args.real_manifest_sha256)
    print("Loading optimization and selection images; sealed audit remains closed", flush=True)
    synthetic_train = load_synthetic_examples(paths, TRAIN_INDICES)
    synthetic_selection_all = load_synthetic_examples(paths, SELECTION_INDICES)
    synthetic_selection = synthetic_selection_all[::4]
    real_train = load_real_examples(
        real_paths(manifest, args.real_root, "optimization"), role="optimization"
    )
    real_selection = load_real_examples(
        real_paths(manifest, args.real_root, "selection"), role="selection"
    )

    device = torch.device("cuda")
    model = TemporalPeakRankDetector(widths=widths, depths=depths).to(device)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    parameter_count = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda")
    history: list[dict[str, Any]] = []
    best_composite = -math.inf
    best_step = 0
    checkpoint_path = args.output_root / "peak_rank_detector.pt"
    last_checkpoint_path = args.output_root / "last_peak_rank_detector.pt"

    for step in range(1, args.steps + 1):
        if time.monotonic() - started > args.max_wall_seconds:
            raise TimeoutError("temporal peak-ranking run reached its wall guard")
        use_real = step % args.real_frequency == 0
        pool = real_train if use_real else synthetic_train
        example = pool[int(rng.integers(0, len(pool)))]
        frames_array, points_array = augment_example(example, rng)
        frames = torch.from_numpy(frames_array)[None].to(device)
        points = torch.from_numpy(points_array).to(device)
        progress = step / args.steps
        learning_rate = args.minimum_learning_rate + 0.5 * (
            args.learning_rate - args.minimum_learning_rate
        ) * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            prediction = model(frames)
            loss, components = training_loss(
                prediction, points, complete_labels=not use_real
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        scaler.step(optimizer)
        scaler.update()
        decay = min(args.ema_decay, (step + 1) / (step + 10))
        update_ema(ema, model, decay)

        if step % args.log_every == 0:
            print(
                "TRAIN",
                json.dumps(
                    {
                        "step": step,
                        "source": "real_positive_only" if use_real else "synthetic_complete",
                        "loss": float(loss.detach()),
                        "gradient_norm": gradient_norm,
                        "learning_rate": learning_rate,
                        **components,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if step % args.validation_every == 0:
            metrics = evaluate(
                ema,
                synthetic_selection,
                real_selection,
                device=device,
            )
            passed = selection_passed(metrics)
            composite = (
                metrics["synthetic"]["mean_average_precision"]
                + 0.5 * metrics["synthetic"]["mean_recall"]
                + metrics["real_positive_only"]["positive_peak_recall"]
                - 0.1 * metrics["real_positive_only"]["mean_distance_voxels"]
            )
            row = {
                "step": step,
                "metrics": metrics,
                "selection_passed": passed,
                "composite": composite,
            }
            history.append(row)
            atomic_json(args.output_root / "selection_history.json", {"rows": history})
            print("SELECTION", json.dumps(row, sort_keys=True), flush=True)
            frozen = checkpoint_state(ema, step=step)
            atomic_torch_save(frozen, last_checkpoint_path)
            if passed and composite > best_composite:
                atomic_torch_save(frozen, checkpoint_path)
                best_composite = composite
                best_step = step

    terminal: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "rejected_at_selection",
        "seed": args.seed,
        "completed_steps": args.steps,
        "best_step": best_step,
        "parameter_count": parameter_count,
        "widths": widths,
        "depths": depths,
        "selection_history": history,
        "selection_passed": best_step > 0,
        "audit_opened": False,
        "audit": None,
        "audit_passed": False,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_predictions_read": False,
        "public_notebook_weights_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
        "elapsed_seconds": time.monotonic() - started,
        "last_checkpoint_sha256": sha256_file(last_checkpoint_path),
    }
    if best_step > 0:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        ema.load_state_dict(state["state_dict"], strict=True)
        ema.to(device)
        # Only now are sealed files hashed and opened.
        audit_rows = [
            row for row in manifest["files"] if row["role"] == "sealed_audit"
        ]
        for row in audit_rows:
            path = args.real_root / str(row["path"])
            if sha256_file(path) != row["sha256"]:
                raise ValueError(f"sealed audit shard hash changed: {path}")
        synthetic_audit_all = load_synthetic_examples(paths, AUDIT_INDICES)
        synthetic_audit = synthetic_audit_all[::4]
        real_audit = load_real_examples(
            real_paths(manifest, args.real_root, "sealed_audit"), role="sealed_audit"
        )
        audit_metrics = evaluate(
            ema, synthetic_audit, real_audit, device=device
        )
        passed = audit_passed(audit_metrics)
        checkpoint_sha256 = sha256_file(checkpoint_path)
        terminal.update(
            {
                "status": "accepted_at_audit" if passed else "rejected_at_audit",
                "audit_opened": True,
                "audit": audit_metrics,
                "audit_passed": passed,
                "checkpoint_sha256": checkpoint_sha256,
            }
        )
    atomic_json(args.output_root / "terminal.json", terminal)
    print("PEAK RANK TERMINAL", json.dumps(terminal, sort_keys=True), flush=True)
    if terminal["status"] != "accepted_at_audit":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
