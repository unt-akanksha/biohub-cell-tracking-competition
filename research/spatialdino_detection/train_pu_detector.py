#!/usr/bin/env python
"""Train an independent SpatialDINO hybrid detector with conservative PU labels.

The 21.5M-parameter microscopy-pretrained ViT is fused with a learned raw-image
3D pyramid and high-resolution decoder.  Public TemporalUNets are immutable
pseudo-label teachers only; their predictions are cached, disagreements remain
unknown, and all clean selection/acceptance movies are excluded from training.
"""

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
    from data import VALIDATION_STEMS, atomic_json, discover_movies, select_frame_pairs
    from encoder import load_spatialdino_vits8, sha256_file
    from model import HybridSpatialDinoDetector, set_detector_training_phase
    from pu_targets import YXTransform, build_pu_targets, weighted_pu_bce_with_logits
    from public_teacher import load_public_teacher, teacher_probabilities
except ModuleNotFoundError:
    from research.spatialdino_detection.data import (
        VALIDATION_STEMS,
        atomic_json,
        discover_movies,
        select_frame_pairs,
    )
    from research.spatialdino_association.encoder import (
        load_spatialdino_vits8,
        sha256_file,
    )
    from research.spatialdino_detection.model import (
        HybridSpatialDinoDetector,
        set_detector_training_phase,
    )
    from research.spotiflow_biohub.pu_targets import (
        YXTransform,
        build_pu_targets,
        weighted_pu_bce_with_logits,
    )
    from research.spotiflow_biohub.public_teacher import (
        load_public_teacher,
        teacher_probabilities,
    )


INPUT_SHAPE = (64, 64, 64)
INPUT_VOXEL_UM = (1.625, 1.625, 1.625)
EXPECTED_ENCODER_PARAMETERS = 21_501_312
EXPECTED_MODEL_PARAMETERS = 29_521_225
SPATIALDINO_SHA256 = "47f199d2e8644ca11be2d5679494bd9607f9e9a7a0b448e35b85391d70c94ed8"


def normalize_spatialdino_frame(values: np.ndarray) -> np.ndarray:
    image = np.asarray(values, dtype=np.float32)
    if image.shape != INPUT_SHAPE or not np.isfinite(image).all():
        raise ValueError(f"expected a finite {INPUT_SHAPE} frame")
    low = float(image.min())
    high = float(image.max())
    if high <= low:
        raise ValueError("image frame has no intensity range")
    return ((image - low) / (high - low)).astype(np.float32)


def annotations_to_isotropic_grid(annotations: np.ndarray) -> np.ndarray:
    points = np.asarray(annotations, dtype=np.float32).reshape(-1, 3).copy()
    points[:, 1:] /= 4.0
    inside = np.all((points >= 0) & (points < np.asarray(INPUT_SHAPE)), axis=1)
    return points[inside]


def update_ema(ema, student, *, decay: float) -> None:
    if not 0.0 <= decay < 1.0:
        raise ValueError("EMA decay must lie in [0, 1)")
    with __import__("torch").no_grad():
        ema_state = ema.state_dict()
        student_state = student.state_dict()
        if ema_state.keys() != student_state.keys():
            raise RuntimeError("EMA and student state dictionaries differ")
        for name, ema_value in ema_state.items():
            source = student_state[name].detach()
            if ema_value.is_floating_point():
                ema_value.mul_(decay).add_(source, alpha=1.0 - decay)
            else:
                ema_value.copy_(source)


def ema_decay_for_step(maximum_decay: float, step: int) -> float:
    """Warm EMA quickly so short guarded runs do not retain random weights."""

    if not 0.0 <= maximum_decay < 1.0:
        raise ValueError("EMA decay must lie in [0, 1)")
    if step <= 0:
        raise ValueError("EMA step must be positive")
    return min(float(maximum_decay), float(step + 1) / float(step + 10))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-dir", type=Path, required=True)
    parser.add_argument("--spatialdino-checkpoint", type=Path, required=True)
    parser.add_argument("--primary-teacher", type=Path, required=True)
    parser.add_argument("--secondary-teacher", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--min-steps", type=int, default=256)
    parser.add_argument("--encoder-unfreeze-step", type=int, default=256)
    parser.add_argument("--encoder-blocks", type=int, default=4)
    parser.add_argument("--pairs-per-movie", type=int, default=1)
    parser.add_argument("--decoder-learning-rate", type=float, default=2e-4)
    parser.add_argument("--encoder-learning-rate", type=float, default=2e-6)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--max-wall-seconds", type=float, default=5200.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or not 0 < args.min_steps <= args.steps:
        raise ValueError("min-steps must lie in [1, steps]")
    if not 0 <= args.encoder_unfreeze_step < args.steps:
        raise ValueError("encoder-unfreeze-step must lie in [0, steps)")
    if not 0 <= args.encoder_blocks <= 4:
        raise ValueError("encoder-blocks must lie in [0, 4]")
    if not 0 < args.decoder_learning_rate <= 5e-4:
        raise ValueError("decoder learning rate is outside the safe range")
    if not 0 < args.encoder_learning_rate <= 5e-6:
        raise ValueError("encoder learning rate is outside the safe range")
    if not 0.9 <= args.ema_decay < 1.0:
        raise ValueError("EMA decay is outside the safe range")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("SpatialDINO PU adaptation requires CUDA")
    started = time.monotonic()
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

    encoder = load_spatialdino_vits8(
        args.spatialdino_checkpoint,
        expected_sha256=SPATIALDINO_SHA256,
        map_location="cpu",
    )
    if sum(parameter.numel() for parameter in encoder.parameters()) != EXPECTED_ENCODER_PARAMETERS:
        raise RuntimeError("unexpected SpatialDINO parameter count")
    student = HybridSpatialDinoDetector(encoder).to(device)
    if sum(parameter.numel() for parameter in student.parameters()) != EXPECTED_MODEL_PARAMETERS:
        raise RuntimeError("unexpected hybrid detector parameter count")
    initial_phase = set_detector_training_phase(student, unfreeze_last_encoder_blocks=0)
    student.train()
    student.encoder.eval()
    ema = copy.deepcopy(student).requires_grad_(False).eval()

    decoder_parameters = [
        parameter
        for name, parameter in student.named_parameters()
        if not name.startswith("encoder.")
    ]
    encoder_parameters = list(student.encoder.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": decoder_parameters, "lr": args.decoder_learning_rate},
            {"params": encoder_parameters, "lr": args.encoder_learning_rate},
        ],
        weight_decay=1e-5,
    )
    scaler = torch.amp.GradScaler("cuda")
    primary = load_public_teacher(args.primary_teacher, device=device)
    secondary = load_public_teacher(args.secondary_teacher, device=device)

    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "training_stems": sorted(by_stem),
        "training_movies": len(by_stem),
        "validation_stems_excluded": sorted(VALIDATION_STEMS),
        "validation_overlap": sorted(set(by_stem) & VALIDATION_STEMS),
        "frame_pairs_per_cycle": len(frame_pairs),
        "steps": args.steps,
        "minimum_steps": args.min_steps,
        "encoder_unfreeze_step": args.encoder_unfreeze_step,
        "encoder_blocks_unfrozen": args.encoder_blocks,
        "pairs_per_movie": args.pairs_per_movie,
        "max_wall_seconds": args.max_wall_seconds,
        "student": {
            "architecture": "SpatialDINO ViT-S/8 + raw 3D pyramid + UNETR decoder",
            "parameter_count": EXPECTED_MODEL_PARAMETERS,
            "encoder_parameter_count": EXPECTED_ENCODER_PARAMETERS,
            "base_encoder_sha256": sha256_file(args.spatialdino_checkpoint),
            "initial_trainable": initial_phase,
            "decoder_learning_rate": args.decoder_learning_rate,
            "encoder_learning_rate": args.encoder_learning_rate,
            "ema_decay": args.ema_decay,
            "ema_warmup": "min(maximum_decay, (step + 1) / (step + 10))",
            "full_resolution_heatmap": True,
        },
        "teachers": {
            "primary_sha256": sha256_file(args.primary_teacher),
            "secondary_sha256": sha256_file(args.secondary_teacher),
            "frozen": True,
            "pseudo_labels_only": True,
            "yx_flip_tta": True,
            "probability_cache": True,
            "both_pair_frames_cached_per_teacher_forward": True,
        },
        "targets": {
            "teacher_high_threshold": 0.96875,
            "teacher_low_support_threshold": 0.10,
            "teacher_support_dilation_voxels": 2,
            "consensus_radius_um": 5.0,
            "annotation_merge_radius_um": 5.0,
            "background_weight": 0.01,
            "positive_sigma_voxels": 1.0,
            "unknown_voxels_have_zero_loss": True,
        },
        "selection_labels_read_during_training": False,
        "acceptance_labels_read_during_training": False,
        "public_predictions_copied": False,
        "competition_submission_performed": False,
    }
    atomic_json(args.output_dir / "run_manifest.json", manifest)

    rng = np.random.default_rng(args.seed)
    pair_cache: dict[tuple[str, int], np.ndarray] = {}
    target_cache: dict[tuple[str, int, int], Any] = {}
    metrics: list[dict[str, Any]] = []
    unfreeze_phase: dict[str, int] | None = None
    actual_steps = 0
    budget_stop_requested = False
    optimizer.zero_grad(set_to_none=True)

    def load_pair(stem: str, frame: int) -> np.ndarray:
        key = (stem, frame)
        if key not in pair_cache:
            import zarr

            array = zarr.open_group(str(by_stem[stem].image_path), mode="r")["0"]
            pair = array[frame : frame + 2, :, ::4, ::4].astype(np.float32)
            if pair.shape != (2, *INPUT_SHAPE):
                raise ValueError(f"unexpected frame-pair shape for {stem}: {pair.shape}")
            pair_cache[key] = pair
        return pair_cache[key]

    for step in range(args.steps):
        elapsed = time.monotonic() - started
        if elapsed >= args.max_wall_seconds and step >= args.min_steps:
            budget_stop_requested = True
            print("DINO PU BUDGET STOP", json.dumps({"step": step, "elapsed": elapsed}), flush=True)
            break
        if step == args.encoder_unfreeze_step and args.encoder_blocks:
            unfreeze_phase = set_detector_training_phase(
                student, unfreeze_last_encoder_blocks=args.encoder_blocks
            )
            student.train()
            student.encoder.eval()
            print("DINO PU PHASE", json.dumps(unfreeze_phase, sort_keys=True), flush=True)

        stem, pair_frame = frame_pairs[step % len(frame_pairs)]
        frame_offset = step % 2
        record = by_stem[stem]
        raw_pair = load_pair(stem, pair_frame)
        target_key = (stem, pair_frame, frame_offset)
        target_cache_hit = target_key in target_cache
        if not target_cache_hit:
            teacher_pair = np.clip(
                (raw_pair - record.q_low) / (record.q_high - record.q_low + 1e-6),
                0.0,
                None,
            ).astype(np.float32)
            teacher_tensor = torch.from_numpy(teacher_pair).unsqueeze(0).to(device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                primary_probability = teacher_probabilities(primary, teacher_tensor, yx_tta=True)
                secondary_probability = teacher_probabilities(secondary, teacher_tensor, yx_tta=True)
            # One TemporalUNet call already predicts both frames. Cache both PU
            # targets now so the opposite frame never repeats eight TTA teacher
            # forwards on the next training cycle.
            for cached_offset in (0, 1):
                annotations = annotations_to_isotropic_grid(
                    record.annotations.get(pair_frame + cached_offset, np.empty((0, 3)))
                )
                target_cache[(stem, pair_frame, cached_offset)] = build_pu_targets(
                    primary_probability[0, cached_offset].float().cpu().numpy(),
                    secondary_probability[0, cached_offset].float().cpu().numpy(),
                    annotations,
                    high_threshold=0.96875,
                    low_support_threshold=0.10,
                    consensus_radius=5.0,
                    annotation_merge_radius=5.0,
                    positive_sigma=1.0,
                    support_dilation_voxels=2,
                    background_weight=0.01,
                    voxel_size=INPUT_VOXEL_UM,
                )
            del teacher_tensor, primary_probability, secondary_probability
        targets = target_cache[target_key]

        image = normalize_spatialdino_frame(raw_pair[frame_offset])
        transform = YXTransform(
            flip_y=bool(rng.integers(0, 2)),
            flip_x=bool(rng.integers(0, 2)),
            rotate_k=int(rng.integers(0, 4)),
        )
        weak_image = transform.apply_array(image).astype(np.float32)
        gamma = float(rng.uniform(0.80, 1.20))
        strong_image = np.power(np.clip(weak_image, 0.0, 1.0), gamma)
        strong_image = strong_image * float(rng.uniform(0.85, 1.15))
        strong_image += rng.normal(0.0, 0.025, strong_image.shape).astype(np.float32)
        weak = torch.from_numpy(weak_image).unsqueeze(0).unsqueeze(0).to(device)
        strong = torch.from_numpy(strong_image).unsqueeze(0).unsqueeze(0).to(device)
        target = torch.from_numpy(transform.apply_array(targets.heatmap)).unsqueeze(0).unsqueeze(0).to(device)
        weights = torch.from_numpy(transform.apply_array(targets.weights)).unsqueeze(0).unsqueeze(0).to(device)
        unknown = torch.from_numpy(transform.apply_array(targets.unknown_mask)).unsqueeze(0).unsqueeze(0).to(device)

        with torch.autocast("cuda", dtype=torch.float16):
            weak_logits = student(weak)
            strong_logits = student(strong)
            weak_loss = weighted_pu_bce_with_logits(weak_logits, target, weights)
            strong_loss = weighted_pu_bce_with_logits(strong_logits, target, weights)
            consistency = (
                torch.nn.functional.mse_loss(
                    torch.sigmoid(weak_logits[unknown]),
                    torch.sigmoid(strong_logits[unknown]),
                )
                if torch.any(unknown)
                else weak_logits.sum() * 0.0
            )
            loss = 0.5 * (weak_loss + strong_loss) + 0.05 * consistency
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in student.parameters() if parameter.requires_grad], 1.0
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        current_ema_decay = ema_decay_for_step(args.ema_decay, step + 1)
        update_ema(ema, student, decay=current_ema_decay)
        actual_steps = step + 1

        if actual_steps % 16 == 0 or step == 0:
            row = {
                "step": actual_steps,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "stem": stem,
                "frame": pair_frame + frame_offset,
                "loss": float(loss.detach().cpu()),
                "weak_pu_loss": float(weak_loss.detach().cpu()),
                "strong_pu_loss": float(strong_loss.detach().cpu()),
                "consistency_loss": float(consistency.detach().cpu()),
                "consensus_positives": targets.consensus_count,
                "forced_annotations": targets.forced_annotation_count,
                "positive_voxels": int(targets.positive_mask.sum()),
                "unknown_fraction": float(targets.unknown_mask.mean()),
                "target_cache_hit": target_cache_hit,
                "encoder_blocks_unfrozen": args.encoder_blocks if unfreeze_phase else 0,
                "ema_decay": current_ema_decay,
            }
            metrics.append(row)
            atomic_json(
                args.output_dir / "training_progress.json",
                {"completed": False, "last": row, "records": metrics},
            )
            print("DINO PU TRAIN", json.dumps(row, sort_keys=True), flush=True)

    if actual_steps < args.min_steps or not metrics:
        raise RuntimeError(f"training stopped before minimum evidence: {actual_steps}/{args.min_steps}")
    torch.save({"state_dict": ema.state_dict()}, args.output_dir / "best.pt")
    best_hash = sha256_file(args.output_dir / "best.pt")
    result = {
        "schema_version": 1,
        "status": "completed",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "target_steps": args.steps,
        "actual_steps": actual_steps,
        "budget_stop_requested": budget_stop_requested,
        "parameter_count": EXPECTED_MODEL_PARAMETERS,
        "initial_trainable": initial_phase,
        "unfrozen_trainable": unfreeze_phase,
        "base_encoder_sha256": SPATIALDINO_SHA256,
        "best_weight_sha256": best_hash,
        "ema_checkpoint": True,
        "training_movies": len(by_stem),
        "teacher_target_cache_entries": len(target_cache),
        "validation_overlap": [],
        "selection_labels_read_during_training": False,
        "acceptance_labels_read_during_training": False,
        "public_predictions_copied": False,
        "competition_submission_performed": False,
        "last_metric": metrics[-1],
    }
    atomic_json(args.output_dir / "pu_training_result.json", result)
    atomic_json(
        args.output_dir / "training_progress.json",
        {"completed": True, "last": metrics[-1], "records": metrics},
    )
    print("DINO PU COMPLETE", json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
