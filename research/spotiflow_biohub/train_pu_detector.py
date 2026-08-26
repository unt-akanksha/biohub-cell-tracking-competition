#!/usr/bin/env python
"""Adapt official 35.5M-parameter Spotiflow to positive-unlabeled Biohub data.

Two frozen public TemporalUNet seeds provide conservative consensus positives.
Sparse organizer annotations are forced positive, teacher disagreements are
ignored, and only voxels outside both teachers' buffered support receive a
small background loss.  The twelve clean selection/acceptance movies are
excluded from every training image and label read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from pu_targets import YXTransform, build_pu_targets, weighted_pu_bce_with_logits
    from public_teacher import load_public_teacher, teacher_probabilities
except ModuleNotFoundError:
    from research.spotiflow_biohub.pu_targets import (
        YXTransform,
        build_pu_targets,
        weighted_pu_bce_with_logits,
    )
    from research.spotiflow_biohub.public_teacher import (
        load_public_teacher,
        teacher_probabilities,
    )


VALIDATION_STEMS = frozenset(
    {
        "44b6_d29c9ab2",
        "44b6_3a861e03",
        "44b6_d5e7d891",
        "44b6_ddf577ad",
        "6bba_09961292",
        "6bba_bb9f20c3",
        "6bba_784a78c9",
        "6bba_57b7cc1e",
        "44b6_12dfb391",
        "44b6_267148e4",
        "6bba_062c8d37",
        "6bba_07e24132",
    }
)
EXPECTED_SPOTIFLOW_PARAMETERS = 35_489_892
INPUT_SHAPE = (32, 64, 64)
BIOHUB_INPUT_VOXEL_UM = (1.625, 1.625, 1.625)


@dataclass(frozen=True)
class MovieRecord:
    stem: str
    image_path: Path
    frame_count: int
    q_low: float
    q_high: float
    annotations: dict[int, np.ndarray]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def select_frame_pairs(
    frame_counts: dict[str, int], *, pairs_per_movie: int, seed: int
) -> list[tuple[str, int]]:
    """Build a deterministic, prefix-interleaved set of consecutive pairs."""

    if pairs_per_movie <= 0:
        raise ValueError("pairs_per_movie must be positive")
    selected: list[tuple[str, int]] = []
    for stem, frame_count in sorted(frame_counts.items()):
        if stem in VALIDATION_STEMS:
            continue
        if frame_count < 2:
            raise ValueError(f"movie {stem} has fewer than two frames")
        candidates = np.linspace(
            0, frame_count - 2, num=min(pairs_per_movie, frame_count - 1), dtype=int
        )
        selected.extend((stem, int(frame)) for frame in np.unique(candidates))
    if not selected:
        raise RuntimeError("no non-validation frame pairs were selected")
    random.Random(seed).shuffle(selected)
    return selected


def annotations_to_output_grid(
    annotations_original: np.ndarray,
    *,
    z_start: int,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
) -> np.ndarray:
    """Map original Biohub z/y/x voxels into a cropped Spotiflow output grid."""

    annotations = np.asarray(annotations_original, dtype=np.float32).reshape(-1, 3)
    input_size = np.asarray(tuple(input_shape), dtype=np.float32)
    output_size = np.asarray(tuple(output_shape), dtype=np.float32)
    if input_size.shape != (3,) or output_size.shape != (3,):
        raise ValueError("input_shape and output_shape must be three-dimensional")
    pooled = annotations / np.asarray((1.0, 4.0, 4.0), dtype=np.float32)
    pooled[:, 0] -= float(z_start)
    inside = np.all((pooled >= 0) & (pooled < input_size), axis=1)
    return (pooled[inside] * (output_size / input_size)).astype(np.float32)


def _graph_points_by_frame(path: Path) -> dict[int, np.ndarray]:
    import tracksdata as td

    graph = td.graph.IndexedRXGraph.from_geff(path)
    graph = graph[0] if isinstance(graph, tuple) else graph
    points: dict[int, list[tuple[float, float, float]]] = {}
    for row in graph.node_attrs().iter_rows(named=True):
        points.setdefault(int(row["t"]), []).append(
            (float(row["z"]), float(row["y"]), float(row["x"]))
        )
    return {
        frame: np.asarray(coords, dtype=np.float32).reshape(-1, 3)
        for frame, coords in points.items()
    }


def discover_movies(train_dir: Path) -> list[MovieRecord]:
    from biohub_tracking.io import open_dataset

    records: list[MovieRecord] = []
    for image_path in sorted(train_dir.glob("*.zarr")):
        stem = image_path.stem
        if stem in VALIDATION_STEMS:
            continue
        truth_path = train_dir / f"{stem}.geff"
        if not truth_path.is_dir():
            raise FileNotFoundError(truth_path)
        dataset = open_dataset(
            image_path, normalize=False, load_image=False, require_tracks=False
        )
        if "0.001" not in dataset.quantiles or "0.999" not in dataset.quantiles:
            raise ValueError(f"missing image quantiles for {stem}")
        records.append(
            MovieRecord(
                stem=stem,
                image_path=image_path,
                frame_count=int(dataset.image_shape[0]),
                q_low=float(dataset.quantiles["0.001"]),
                q_high=float(dataset.quantiles["0.999"]),
                annotations=_graph_points_by_frame(truth_path),
            )
        )
    if not records or any(record.stem in VALIDATION_STEMS for record in records):
        raise RuntimeError("training discovery violated the frozen validation split")
    return records


@lru_cache(maxsize=12)
def _zarr_array(path: str):
    import zarr

    return zarr.open_group(path, mode="r")["0"]


def load_pair(record: MovieRecord, frame: int, z_start: int) -> np.ndarray:
    array = _zarr_array(str(record.image_path))
    values = array[frame : frame + 2, z_start : z_start + 32, ::4, ::4].astype(
        np.float32
    )
    if values.shape != (2, *INPUT_SHAPE):
        raise ValueError(
            f"unexpected crop shape for {record.stem} frame {frame}: {values.shape}"
        )
    return values


def spotiflow_heatmap_logits(model, images):
    """Forward only the heatmap branch, skipping the unused flow head."""

    values = model._bg_remover(images)
    if model._downsampler is not None:
        values = model._downsampler(values)
    features = model._backbone(values)
    return tuple(model._post(features))[0]


def candidate_parameters(model) -> list[tuple[str, Any]]:
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith(("_post.", "_backbone.up_blocks."))
    ]


def set_training_phase(model, *, deep_decoder: bool) -> int:
    """Freeze the encoder; warm the final decoder block before all decoder blocks."""

    model.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if name.startswith("_post.") or name.startswith("_backbone.up_blocks.0."):
            parameter.requires_grad_(True)
        elif deep_decoder and name.startswith("_backbone.up_blocks."):
            parameter.requires_grad_(True)
    model.eval()
    model._post.train()
    model._backbone.up_blocks[0].train()
    if deep_decoder:
        model._backbone.up_blocks.train()
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def copy_model_bundle(base_model: Path, output_dir: Path) -> None:
    for name in ("config.yaml", "thresholds.yaml", "train_config.yaml"):
        source = base_model / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, output_dir / name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-dir", type=Path, required=True)
    parser.add_argument("--spotiflow-model", type=Path, required=True)
    parser.add_argument("--primary-teacher", type=Path, required=True)
    parser.add_argument("--secondary-teacher", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--warmup-steps", type=int, default=256)
    parser.add_argument("--pairs-per-movie", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--max-wall-seconds", type=int, default=6300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or not 0 <= args.warmup_steps < args.steps:
        raise ValueError("steps must be positive and warmup-steps must be smaller")
    if not 0 < args.learning_rate <= 3e-5:
        raise ValueError("learning rate is outside the conservative PU range")
    started = time.monotonic()

    import torch
    import torch.nn.functional as torch_functional
    from spotiflow.model import Spotiflow
    from spotiflow.utils import normalize

    if not torch.cuda.is_available():
        raise RuntimeError("positive-unlabeled adaptation requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = discover_movies(args.competition_dir / "train")
    by_stem = {record.stem: record for record in records}
    frame_pairs = select_frame_pairs(
        {record.stem: record.frame_count for record in records},
        pairs_per_movie=args.pairs_per_movie,
        seed=args.seed,
    )
    if set(by_stem) & VALIDATION_STEMS:
        raise RuntimeError("validation movie entered the training inventory")

    primary = load_public_teacher(args.primary_teacher, device=device)
    secondary = load_public_teacher(args.secondary_teacher, device=device)
    student = Spotiflow.from_folder(
        str(args.spotiflow_model), inference_mode=False, map_location="cuda"
    )
    parameter_count = sum(parameter.numel() for parameter in student.parameters())
    if parameter_count != EXPECTED_SPOTIFLOW_PARAMETERS:
        raise RuntimeError(f"unexpected Spotiflow parameter count: {parameter_count}")

    all_candidates = candidate_parameters(student)
    if not all_candidates:
        raise RuntimeError("no decoder/head parameters selected for adaptation")
    warm_trainable = set_training_phase(student, deep_decoder=False)
    deep_names = {name for name, _ in all_candidates}
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in all_candidates],
        lr=args.learning_rate,
        weight_decay=1e-5,
    )
    scaler = torch.amp.GradScaler("cuda")

    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "training_stems": sorted(by_stem),
        "training_movies": len(by_stem),
        "validation_stems_excluded": sorted(VALIDATION_STEMS),
        "validation_overlap": sorted(set(by_stem) & VALIDATION_STEMS),
        "frame_pairs_per_cycle": len(frame_pairs),
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "pairs_per_movie": args.pairs_per_movie,
        "learning_rate": args.learning_rate,
        "max_wall_seconds": args.max_wall_seconds,
        "student": {
            "base_best_sha256": sha256_file(args.spotiflow_model / "best.pt"),
            "parameter_count": parameter_count,
            "warm_trainable_parameters": warm_trainable,
            "candidate_parameter_names_sha256": hashlib.sha256(
                "\n".join(sorted(deep_names)).encode("utf-8")
            ).hexdigest(),
        },
        "teachers": {
            "primary_sha256": sha256_file(args.primary_teacher),
            "secondary_sha256": sha256_file(args.secondary_teacher),
            "frozen": True,
            "yx_flip_tta": True,
        },
        "targets": {
            "teacher_high_threshold": 0.96875,
            "teacher_low_support_threshold": 0.10,
            "teacher_support_dilation_output_voxels": 2,
            "consensus_radius_um": 5.0,
            "annotation_merge_radius_um": 5.0,
            "background_weight": 0.01,
            "positive_sigma_output_voxels": 0.75,
            "unknown_voxels_have_zero_loss": True,
        },
        "selection_labels_read_during_training": False,
        "acceptance_labels_read_during_training": False,
        "competition_submission_performed": False,
    }
    atomic_json(args.output_dir / "run_manifest.json", manifest)
    copy_model_bundle(args.spotiflow_model, args.output_dir)

    rng = np.random.default_rng(args.seed)
    metrics: list[dict[str, Any]] = []
    deep_unfrozen = False
    deep_trainable = warm_trainable
    optimizer.zero_grad(set_to_none=True)

    for step in range(args.steps):
        elapsed = time.monotonic() - started
        if elapsed >= args.max_wall_seconds:
            raise TimeoutError(
                f"PU training reached wall guard at step {step}/{args.steps}"
            )
        if step == args.warmup_steps:
            deep_trainable = set_training_phase(student, deep_decoder=True)
            deep_unfrozen = True
            print(
                "PU PHASE",
                json.dumps(
                    {"step": step, "phase": "all_decoder", "trainable": deep_trainable}
                ),
                flush=True,
            )

        stem, frame = frame_pairs[step % len(frame_pairs)]
        record = by_stem[stem]
        z_start = int(rng.integers(0, 33))
        raw_pair = load_pair(record, frame, z_start)
        teacher_pair = np.clip(
            (raw_pair - record.q_low) / (record.q_high - record.q_low + 1e-6),
            0.0,
            None,
        ).astype(np.float32)
        teacher_tensor = torch.from_numpy(teacher_pair).unsqueeze(0).to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            primary_probability = teacher_probabilities(primary, teacher_tensor, yx_tta=True)
            secondary_probability = teacher_probabilities(
                secondary, teacher_tensor, yx_tta=True
            )

        frame_offset = step % 2
        raw_image = raw_pair[frame_offset]
        student_image = normalize(raw_image).astype(np.float32)
        student_tensor = torch.from_numpy(student_image).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            probe_logits = spotiflow_heatmap_logits(student, student_tensor)
        output_shape = tuple(int(value) for value in probe_logits.shape[-3:])
        primary_output = torch_functional.adaptive_max_pool3d(
            primary_probability[:, frame_offset].unsqueeze(1).float(), output_shape
        )[0, 0].cpu().numpy()
        secondary_output = torch_functional.adaptive_max_pool3d(
            secondary_probability[:, frame_offset].unsqueeze(1).float(), output_shape
        )[0, 0].cpu().numpy()
        annotations = annotations_to_output_grid(
            record.annotations.get(frame + frame_offset, np.empty((0, 3))),
            z_start=z_start,
            input_shape=INPUT_SHAPE,
            output_shape=output_shape,
        )
        output_voxel = tuple(
            BIOHUB_INPUT_VOXEL_UM[index] * INPUT_SHAPE[index] / output_shape[index]
            for index in range(3)
        )
        targets = build_pu_targets(
            primary_output,
            secondary_output,
            annotations,
            high_threshold=0.96875,
            low_support_threshold=0.10,
            consensus_radius=5.0,
            annotation_merge_radius=5.0,
            positive_sigma=0.75,
            support_dilation_voxels=2,
            background_weight=0.01,
            voxel_size=output_voxel,
        )

        transform = YXTransform(
            flip_y=bool(rng.integers(0, 2)),
            flip_x=bool(rng.integers(0, 2)),
            rotate_k=int(rng.integers(0, 4)),
        )
        weak_image = transform.apply_array(student_image).astype(np.float32)
        gamma = float(rng.uniform(0.85, 1.15))
        strong_image = np.power(np.clip(weak_image, 0.0, None), gamma)
        strong_image = strong_image * float(rng.uniform(0.9, 1.1))
        strong_image += rng.normal(0.0, 0.025, size=strong_image.shape).astype(np.float32)
        weak = torch.from_numpy(weak_image).unsqueeze(0).unsqueeze(0).to(device)
        strong = torch.from_numpy(strong_image).unsqueeze(0).unsqueeze(0).to(device)
        target_tensor = torch.from_numpy(transform.apply_array(targets.heatmap)).to(device)
        weight_tensor = torch.from_numpy(transform.apply_array(targets.weights)).to(device)
        unknown_tensor = torch.from_numpy(transform.apply_array(targets.unknown_mask)).to(device)
        target_tensor = target_tensor.unsqueeze(0).unsqueeze(0)
        weight_tensor = weight_tensor.unsqueeze(0).unsqueeze(0)
        unknown_tensor = unknown_tensor.unsqueeze(0).unsqueeze(0)

        with torch.autocast("cuda", dtype=torch.float16):
            weak_logits = spotiflow_heatmap_logits(student, weak)
            strong_logits = spotiflow_heatmap_logits(student, strong)
            weak_loss = weighted_pu_bce_with_logits(
                weak_logits, target_tensor, weight_tensor
            )
            strong_loss = weighted_pu_bce_with_logits(
                strong_logits, target_tensor, weight_tensor
            )
            if torch.any(unknown_tensor):
                consistency = torch_functional.mse_loss(
                    torch.sigmoid(weak_logits[unknown_tensor]),
                    torch.sigmoid(strong_logits[unknown_tensor]),
                )
            else:
                consistency = weak_logits.sum() * 0.0
            loss = 0.5 * (weak_loss + strong_loss) + 0.05 * consistency
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in all_candidates if parameter.requires_grad], 1.0
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if (step + 1) % 16 == 0 or step == 0:
            row = {
                "step": step + 1,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "stem": stem,
                "frame": frame + frame_offset,
                "z_start": z_start,
                "loss": float(loss.detach().cpu()),
                "weak_pu_loss": float(weak_loss.detach().cpu()),
                "strong_pu_loss": float(strong_loss.detach().cpu()),
                "consistency_loss": float(consistency.detach().cpu()),
                "consensus_positives": targets.consensus_count,
                "forced_annotations": targets.forced_annotation_count,
                "positive_voxels": int(targets.positive_mask.sum()),
                "unknown_fraction": float(targets.unknown_mask.mean()),
                "deep_decoder_unfrozen": deep_unfrozen,
            }
            metrics.append(row)
            atomic_json(
                args.output_dir / "training_progress.json",
                {
                    "last": row,
                    "records": metrics,
                    "target_steps": args.steps,
                    "completed": False,
                },
            )
            print("PU TRAIN", json.dumps(row, sort_keys=True), flush=True)

    torch.save({"state_dict": student.state_dict()}, args.output_dir / "best.pt")
    best_hash = sha256_file(args.output_dir / "best.pt")
    result = {
        "schema_version": 1,
        "status": "completed",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "steps": args.steps,
        "parameter_count": parameter_count,
        "warm_trainable_parameters": warm_trainable,
        "deep_trainable_parameters": deep_trainable,
        "base_weight_sha256": manifest["student"]["base_best_sha256"],
        "best_weight_sha256": best_hash,
        "weights_changed": best_hash != manifest["student"]["base_best_sha256"],
        "training_movies": len(by_stem),
        "validation_overlap": [],
        "selection_labels_read_during_training": False,
        "acceptance_labels_read_during_training": False,
        "competition_submission_performed": False,
        "last_metric": metrics[-1],
    }
    atomic_json(args.output_dir / "pu_training_result.json", result)
    atomic_json(
        args.output_dir / "training_progress.json",
        {
            "last": metrics[-1],
            "records": metrics,
            "target_steps": args.steps,
            "completed": True,
        },
    )
    print("PU TERMINAL", json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
