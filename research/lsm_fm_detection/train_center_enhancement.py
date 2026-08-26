#!/usr/bin/env python
"""Train a sparse-label-safe center enhancer on frozen feature-36 peaks."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from center_enhancement import (
        build_center_enhancement_model,
        center_enhancement_loss,
        extract_center_patches,
        match_annotated_peaks,
    )
    from data import VALIDATION_STEMS, atomic_json, discover_movies
    from inference import predict_probability_batch
    from localization_refinement import refine_peaks_weighted
    from pu_targets import extract_local_peaks
    from train_spatialdino_pu_detector import normalize_spatialdino_frame
except ModuleNotFoundError:
    from research.lsm_fm_detection.center_enhancement import (
        build_center_enhancement_model,
        center_enhancement_loss,
        extract_center_patches,
        match_annotated_peaks,
    )
    from research.spatialdino_detection.data import (
        VALIDATION_STEMS,
        atomic_json,
        discover_movies,
    )
    from research.spatialdino_detection.inference import predict_probability_batch
    from research.spatialdino_detection.train_pu_detector import (
        normalize_spatialdino_frame,
    )
    from research.spotiflow_biohub.pu_targets import extract_local_peaks
    from research.lsm_fm_detection.localization_refinement import refine_peaks_weighted


PATCH_SHAPE = (7, 7, 7)
LOW_PROBABILITY_THRESHOLD = 0.02
MAXIMUM_LABEL_OFFSET_VOXELS = 3.0
EXPECTED_DETECTOR_PARAMETERS = 35_072_515
CONTROL_REFINEMENT_RADIUS = 2
CONTROL_PROBABILITY_POWER = 2.0


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_annotated_frames(
    annotations: dict[int, np.ndarray], *, maximum_frames: int
) -> list[int]:
    """Choose a fixed temporal spread without consulting image predictions."""

    if maximum_frames <= 0:
        raise ValueError("maximum_frames must be positive")
    frames = sorted(int(frame) for frame, points in annotations.items() if len(points))
    if len(frames) <= maximum_frames:
        return frames
    positions = np.linspace(0, len(frames) - 1, num=maximum_frames, dtype=np.int64)
    return [frames[index] for index in np.unique(positions)]


def augment_patches(patches, offsets, *, generator):
    """Apply geometry-consistent axis flips to raw/probability patches."""

    import torch

    result_patches = patches.clone()
    result_offsets = offsets.clone()
    for spatial_axis in range(3):
        mask = torch.rand(len(patches), generator=generator, device="cpu") < 0.5
        mask = mask.to(patches.device)
        if bool(mask.any()):
            result_patches[mask] = torch.flip(
                result_patches[mask], dims=(spatial_axis + 2,)
            )
            result_offsets[mask, spatial_axis] *= -1.0
    return result_patches, result_offsets


def split_examples(count: int, *, seed: int, validation_fraction: float = 0.15):
    if count < 8:
        raise RuntimeError("too few matched sparse-label examples for center enhancement")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must lie in (0, 0.5)")
    order = np.random.default_rng(seed).permutation(count)
    validation_count = max(1, int(round(count * validation_fraction)))
    return order[validation_count:], order[:validation_count]


def load_frozen_detector(args, device):
    import torch

    try:
        from lsm_fm_image_text_model import build_lsm_fm_detector
    except ModuleNotFoundError:
        from research.lsm_fm_detection.image_text_model import build_lsm_fm_detector

    result = json.loads(args.detector_training_result.read_text(encoding="utf-8"))
    if result.get("status") != "completed" or result.get("validation_overlap") != []:
        raise ValueError("frozen detector training evidence is invalid")
    if result.get("public_predictions_copied") is not False:
        raise ValueError("detector evidence does not prove independent predictions")
    if sha256_file(args.detector_model) != result.get("best_weight_sha256"):
        raise ValueError("frozen detector checkpoint hash mismatch")
    model = build_lsm_fm_detector(
        args.lsm_fm_checkpoint,
        expected_sha256=args.lsm_fm_checkpoint_sha256,
    )
    state = torch.load(args.detector_model, map_location="cpu", weights_only=True)
    model.load_state_dict(state["state_dict"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_DETECTOR_PARAMETERS:
        raise RuntimeError("unexpected frozen detector parameter count")
    return model.requires_grad_(False).eval().to(device), result


def build_training_examples(
    detector,
    records,
    *,
    device,
    frames_per_movie: int,
    detector_batch_size: int,
    partial_path: Path,
    started: float,
    max_wall_seconds: float,
):
    import torch
    import zarr

    patch_chunks: list[np.ndarray] = []
    offset_chunks: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for record in records:
        frames = select_annotated_frames(
            record.annotations, maximum_frames=frames_per_movie
        )
        array = zarr.open_group(str(record.image_path), mode="r")["0"]
        movie_patches: list[np.ndarray] = []
        movie_offsets: list[np.ndarray] = []
        candidates = annotations = matches = rejected_large_offset = 0
        matched_distances: list[float] = []
        for start in range(0, len(frames), detector_batch_size):
            batch_frames = frames[start : start + detector_batch_size]
            loaded = [
                normalize_spatialdino_frame(
                    array[frame, :, ::4, ::4].astype(np.float32)
                )
                for frame in batch_frames
            ]
            images = torch.from_numpy(np.stack(loaded)[:, None]).to(device)
            probabilities = predict_probability_batch(
                detector, images, yx_tta=True
            ).cpu().numpy()
            for local_index, frame in enumerate(batch_frames):
                probability = probabilities[local_index, 0]
                peak_set = extract_local_peaks(
                    probability,
                    threshold=LOW_PROBABILITY_THRESHOLD,
                    min_distance_voxels=1,
                )
                control_points = refine_peaks_weighted(
                    probability,
                    peak_set.coords,
                    radius=CONTROL_REFINEMENT_RADIUS,
                    probability_power=CONTROL_PROBABILITY_POWER,
                )
                matched = match_annotated_peaks(
                    control_points, record.annotations[frame]
                )
                valid = np.max(np.abs(matched.offsets_input), axis=1) <= MAXIMUM_LABEL_OFFSET_VOXELS
                rejected_large_offset += int(np.count_nonzero(~valid))
                chosen_indices = matched.peak_indices[valid]
                if len(chosen_indices):
                    movie_patches.append(
                        extract_center_patches(
                            loaded[local_index],
                            probability,
                            control_points[chosen_indices],
                            patch_shape=PATCH_SHAPE,
                        )
                    )
                    movie_offsets.append(matched.offsets_input[valid])
                    matched_distances.extend(matched.distances_um[valid].tolist())
                candidates += len(peak_set.coords)
                annotations += len(record.annotations[frame])
                matches += int(np.count_nonzero(valid))
            del images, probabilities
        if movie_patches:
            patch_chunks.append(np.concatenate(movie_patches))
            offset_chunks.append(np.concatenate(movie_offsets))
        row = {
            "stem": record.stem,
            "frames": frames,
            "candidate_peaks": candidates,
            "sparse_annotations": annotations,
            "matched_examples": matches,
            "rejected_large_offset": rejected_large_offset,
            "mean_initial_match_distance_um": (
                float(np.mean(matched_distances)) if matched_distances else None
            ),
        }
        rows.append(row)
        atomic_json(partial_path, {"completed": False, "movies": rows})
        print("CENTER ENHANCEMENT EXAMPLES", json.dumps(row, sort_keys=True), flush=True)
        if time.monotonic() - started > max_wall_seconds:
            raise TimeoutError("center-enhancement example generation reached wall guard")
    if not patch_chunks:
        raise RuntimeError("no matched sparse-label center-enhancement examples")
    return np.concatenate(patch_chunks), np.concatenate(offset_chunks), rows


def validation_metrics(model, patches, offsets, *, batch_size: int, device):
    import torch

    predictions = []
    with torch.inference_mode():
        for start in range(0, len(patches), batch_size):
            values = torch.from_numpy(patches[start : start + batch_size]).to(device)
            predictions.append(model(values)["offsets"].cpu().numpy())
    predicted = np.concatenate(predictions)
    initial_um = np.linalg.norm(offsets * 1.625, axis=1)
    residual_um = np.linalg.norm((offsets - predicted) * 1.625, axis=1)
    return {
        "examples": len(offsets),
        "initial_mean_distance_um": float(np.mean(initial_um)),
        "refined_mean_distance_um": float(np.mean(residual_um)),
        "initial_within_5um": float(np.mean(initial_um <= 5.0)),
        "refined_within_5um": float(np.mean(residual_um <= 5.0)),
        "mean_predicted_offset_voxels": np.mean(predicted, axis=0).tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-dir", type=Path, required=True)
    parser.add_argument("--lsm-fm-checkpoint", type=Path, required=True)
    parser.add_argument("--lsm-fm-checkpoint-sha256", required=True)
    parser.add_argument("--detector-model", type=Path, required=True)
    parser.add_argument("--detector-training-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames-per-movie", type=int, default=12)
    parser.add_argument("--detector-batch-size", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--max-wall-seconds", type=float, default=3000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.frames_per_movie, args.detector_batch_size, args.train_batch_size, args.steps) <= 0:
        raise ValueError("frame, batch, and step counts must be positive")
    if not 0.0 < args.learning_rate <= 1e-3 or args.max_wall_seconds <= 0:
        raise ValueError("learning rate or wall budget is outside the safe range")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("center-enhancement training requires CUDA")
    started = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detector, detector_result = load_frozen_detector(args, device)
    records = discover_movies(args.competition_dir / "train")
    training_stems = {record.stem for record in records}
    if training_stems & VALIDATION_STEMS:
        raise RuntimeError("frozen validation movie entered center-enhancement training")
    patches, offsets, example_rows = build_training_examples(
        detector,
        records,
        device=device,
        frames_per_movie=args.frames_per_movie,
        detector_batch_size=args.detector_batch_size,
        partial_path=args.output_dir / "example_generation.json",
        started=started,
        max_wall_seconds=args.max_wall_seconds,
    )
    del detector
    torch.cuda.empty_cache()

    train_indices, validation_indices = split_examples(len(patches), seed=args.seed)
    model = build_center_enhancement_model(channels=32).to(device)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    cpu_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    metrics: list[dict[str, Any]] = []
    model.train()
    for step in range(args.steps):
        chosen = rng.choice(
            train_indices,
            size=min(args.train_batch_size, len(train_indices)),
            replace=len(train_indices) < args.train_batch_size,
        )
        batch_patches = torch.from_numpy(patches[chosen]).to(device)
        batch_offsets = torch.from_numpy(offsets[chosen]).to(device)
        batch_patches, batch_offsets = augment_patches(
            batch_patches, batch_offsets, generator=cpu_generator
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            output = model(batch_patches)
            losses = center_enhancement_loss(output["logits"], batch_offsets)
        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        decay = min(0.995, float(step + 1) / float(step + 10))
        with torch.no_grad():
            for ema_value, model_value in zip(ema.state_dict().values(), model.state_dict().values()):
                if ema_value.is_floating_point():
                    ema_value.mul_(decay).add_(model_value.detach(), alpha=1.0 - decay)
                else:
                    ema_value.copy_(model_value)
        if step == 0 or (step + 1) % 32 == 0 or step + 1 == args.steps:
            row = {
                "step": step + 1,
                "loss": float(losses["loss"].detach().cpu()),
                "dense_loss": float(losses["dense_loss"].cpu()),
                "offset_loss": float(losses["offset_loss"].cpu()),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            metrics.append(row)
            atomic_json(
                args.output_dir / "training_progress.json",
                {"completed": False, "records": metrics},
            )
            print("CENTER ENHANCEMENT TRAIN", json.dumps(row, sort_keys=True), flush=True)
        if time.monotonic() - started > args.max_wall_seconds:
            raise TimeoutError("center-enhancement training reached wall guard")

    validation = validation_metrics(
        ema,
        patches[validation_indices],
        offsets[validation_indices],
        batch_size=args.train_batch_size,
        device=device,
    )
    checkpoint_path = args.output_dir / "center_enhancement.pt"
    torch.save({"state_dict": ema.state_dict(), "channels": 32}, checkpoint_path)
    result = {
        "schema_version": 1,
        "status": "completed",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "training_stems": sorted(training_stems),
        "training_movies": len(training_stems),
        "validation_stems_excluded": sorted(VALIDATION_STEMS),
        "validation_overlap": sorted(training_stems & VALIDATION_STEMS),
        "matched_examples": len(patches),
        "fit_examples": len(train_indices),
        "internal_validation_examples": len(validation_indices),
        "example_rows": example_rows,
        "steps": args.steps,
        "parameter_count": sum(parameter.numel() for parameter in ema.parameters()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "detector_checkpoint_sha256": sha256_file(args.detector_model),
        "detector_training_result_sha256": sha256_file(args.detector_training_result),
        "detector_validation_overlap": detector_result["validation_overlap"],
        "validation": validation,
        "targets": {
            "one_to_one_sparse_annotation_matches_only": True,
            "unmatched_peaks_used_as_negatives": False,
            "maximum_match_distance_um": 5.0,
            "maximum_label_offset_voxels": MAXIMUM_LABEL_OFFSET_VOXELS,
            "residual_control": "probability_r2_p2",
        },
        "selection_labels_read_during_training": False,
        "acceptance_labels_read_during_training": False,
        "public_predictions_copied": False,
        "competition_submission_performed": False,
    }
    atomic_json(args.output_dir / "center_enhancement_training.json", result)
    atomic_json(
        args.output_dir / "training_progress.json",
        {"completed": True, "records": metrics},
    )
    print("CENTER ENHANCEMENT COMPLETE", json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
