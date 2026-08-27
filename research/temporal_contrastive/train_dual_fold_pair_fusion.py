#!/usr/bin/env python
"""Train reciprocal Biohub pair-fusion appearance models on exactly two GPUs.

This predeclared v2 lane retains the v1 physical temporal encoder and clean
splits, but trains a candidate-limited pair MLP as the primary association
objective.  It creates model evidence only and has no submission path.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import train_dual_fold_patch as base
    from pair_fusion import (
        DEFAULT_PAIR_CHUNK_SIZE,
        EXPECTED_PARAMETER_COUNT,
        PAIR_FEATURE_WIDTH,
        PAIR_FUSION_FAMILY,
        PAIR_FUSION_POLICY,
        PAIR_HIDDEN_WIDTHS,
        PAIR_LOSS_POLICY,
        PAIR_PROJECTION_WIDTH,
        PhysicalPairFusionAssociationModel,
        masked_multi_positive_pair_nll,
        pair_logit_metrics,
    )
    from model import masked_multi_positive_info_nce
except ModuleNotFoundError:
    from research.temporal_contrastive import train_dual_fold_patch as base
    from research.temporal_contrastive.model import masked_multi_positive_info_nce
    from research.temporal_contrastive.pair_fusion import (
        DEFAULT_PAIR_CHUNK_SIZE,
        EXPECTED_PARAMETER_COUNT,
        PAIR_FEATURE_WIDTH,
        PAIR_FUSION_FAMILY,
        PAIR_FUSION_POLICY,
        PAIR_HIDDEN_WIDTHS,
        PAIR_LOSS_POLICY,
        PAIR_PROJECTION_WIDTH,
        PhysicalPairFusionAssociationModel,
        masked_multi_positive_pair_nll,
        pair_logit_metrics,
    )


RUN_ID = "temporal-patch-pair-fusion-v2"
EMBEDDING_AUXILIARY_LOSS_WEIGHT = 0.25


def physical_coordinates(
    values: np.ndarray,
    voxel_size_zyx_um: tuple[float, float, float],
    device: torch.device,
) -> torch.Tensor:
    coords = torch.as_tensor(values, dtype=torch.float32, device=device)
    spacing = torch.as_tensor(
        voxel_size_zyx_um, dtype=torch.float32, device=device
    )
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("node coordinates must have shape (N, 3)")
    if spacing.shape != (3,) or not torch.isfinite(spacing).all() or torch.any(
        spacing <= 0
    ):
        raise ValueError("voxel size must contain three positive finite values")
    return coords * spacing[None]


def pair_logits_for_transition(
    model: PhysicalPairFusionAssociationModel,
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    source_division_logits: torch.Tensor,
    batch: base.TransitionBatch,
    voxel_size_zyx_um: tuple[float, float, float],
    device: torch.device,
    *,
    candidate_radius_um: float,
    pair_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidates = torch.as_tensor(
        batch.candidate_mask, dtype=torch.bool, device=device
    )
    positives = torch.as_tensor(
        batch.positive_mask, dtype=torch.bool, device=device
    )
    logits = model.candidate_pair_logits(
        source_embeddings,
        target_embeddings,
        physical_coordinates(batch.source_coords, voxel_size_zyx_um, device),
        physical_coordinates(batch.target_coords, voxel_size_zyx_um, device),
        source_division_logits,
        candidates,
        candidate_radius_um=candidate_radius_um,
        chunk_size=pair_chunk_size,
    )
    return logits, candidates, positives


@torch.no_grad()
def validate_pair_model(
    model: PhysicalPairFusionAssociationModel,
    examples: list[
        tuple[
            np.ndarray,
            np.ndarray,
            base.TransitionBatch,
            tuple[float, float, float],
        ]
    ],
    device: torch.device,
    *,
    candidate_radius_um: float,
    pair_chunk_size: int,
) -> dict[str, float | int]:
    model.eval()
    rows = []
    for source_volume, target_volume, batch, voxel_size in examples:
        source, target, divisions = base.encode_transition(
            model,
            source_volume,
            target_volume,
            batch,
            voxel_size,
            device,
            augment=False,
        )
        logits, candidates, positives = pair_logits_for_transition(
            model,
            source,
            target,
            divisions,
            batch,
            voxel_size,
            device,
            candidate_radius_um=candidate_radius_um,
            pair_chunk_size=pair_chunk_size,
        )
        rows.append(pair_logit_metrics(logits, candidates, positives))
    model.train()
    return base.aggregate_metrics(rows)


def train_worker(args: argparse.Namespace) -> None:
    started = time.monotonic()
    if args.fold not in base.FOLD_SPECS:
        raise ValueError(f"unknown fold: {args.fold}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "isolated pair-fusion worker requires one GPU, "
            f"saw {torch.cuda.device_count()}"
        )
    if args.embedding_loss_weight != EMBEDDING_AUXILIARY_LOSS_WEIGHT:
        raise ValueError("the frozen embedding auxiliary loss weight changed")
    if args.pair_chunk_size != DEFAULT_PAIR_CHUNK_SIZE:
        raise ValueError("the frozen pair chunk size changed")
    device = torch.device("cuda:0")
    spec = base.FOLD_SPECS[args.fold]
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

    manifest, synthetic_train, synthetic_validation = base.synthetic_split(
        args.synthetic_root,
        validation_count=args.synthetic_validation_movies,
        training_count=args.synthetic_train_movies,
    )
    real_paths = list((args.competition_dir / "train").glob("*.geff"))
    _source_validation, _source_calibration, real_train = (
        base.real_prefix_partition(
            real_paths,
            prefix=str(spec["source_prefix"]),
            validation_count=args.real_validation_movies,
            calibration_count=args.real_calibration_movies,
            training_count=args.real_train_movies,
        )
    )
    real_validation, real_calibration, _target_training = (
        base.real_prefix_partition(
            real_paths,
            prefix=str(spec["target_prefix"]),
            validation_count=args.real_validation_movies,
            calibration_count=args.real_calibration_movies,
            training_count=args.real_train_movies,
        )
    )
    if len(real_calibration) != args.real_calibration_movies:
        raise RuntimeError("reciprocal target pool cannot fill calibration")
    used_stems = {
        path.stem for path in real_train + real_validation + real_calibration
    }
    if used_stems & base.OPENED_ACCEPTANCE_STEMS:
        raise RuntimeError("opened acceptance labels entered pair-fusion training")
    store = base.MovieStore(args.competition_dir)
    real_fixed = base.fixed_examples(
        store,
        real_validation,
        synthetic=False,
        count=args.real_validation_transitions,
        seed=seed + 101,
        args=args,
    )
    synthetic_fixed = base.fixed_examples(
        store,
        synthetic_validation,
        synthetic=True,
        count=args.synthetic_validation_transitions,
        seed=seed + 202,
        args=args,
    )

    model = PhysicalPairFusionAssociationModel(
        base_channels=args.base_channels,
        embedding_channels=args.embedding_channels,
    ).to(device)
    ema_model = PhysicalPairFusionAssociationModel(
        base_channels=args.base_channels,
        embedding_channels=args.embedding_channels,
    ).to(device)
    ema_model.load_state_dict(model.state_dict(), strict=True)
    ema_model.requires_grad_(False)
    ema_model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"unexpected pair-fusion parameter count: {parameter_count}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda")
    validation_options = {
        "candidate_radius_um": args.candidate_radius_um,
        "pair_chunk_size": args.pair_chunk_size,
    }
    initial_real = validate_pair_model(
        ema_model, real_fixed, device, **validation_options
    )
    initial_synthetic = validate_pair_model(
        ema_model, synthetic_fixed, device, **validation_options
    )
    ema_model.eval()
    best_state = base.state_dict_cpu(ema_model)
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
    base.atomic_json(
        output_dir / "training_config.json",
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "appearance_family": PAIR_FUSION_FAMILY,
            "fold": args.fold,
            "seed": seed,
            "parameter_count": parameter_count,
            "base_channels": args.base_channels,
            "embedding_channels": args.embedding_channels,
            "pair_feature_width": PAIR_FEATURE_WIDTH,
            "pair_projection_width": PAIR_PROJECTION_WIDTH,
            "pair_hidden_widths": list(PAIR_HIDDEN_WIDTHS),
            "pair_fusion_policy": PAIR_FUSION_POLICY,
            "pair_loss_policy": PAIR_LOSS_POLICY,
            "embedding_auxiliary_loss_weight": args.embedding_loss_weight,
            "pair_chunk_size": args.pair_chunk_size,
            "checkpoint_weight_source": (
                "optimizer-step exponential moving average"
            ),
            "ema_decay": args.ema_decay,
            "division_prior_correction": (
                "class-conditional importance weighting"
            ),
            "link_loss_policy": base.LINK_LOSS_POLICY,
            "real_division_rate": base.REAL_DIVISION_RATE,
            "synthetic_division_rate": base.SYNTHETIC_DIVISION_RATE,
            "source_prefix": spec["source_prefix"],
            "target_prefix": spec["target_prefix"],
            "opened_acceptance_stems_excluded": sorted(
                base.OPENED_ACCEPTANCE_STEMS
            ),
            "synthetic_manifest_sha256": base.sha256_file(manifest),
            "synthetic_train_names": [path.name for path in synthetic_train],
            "synthetic_validation_names": [
                path.name for path in synthetic_validation
            ],
            "real_train_stems": [path.stem for path in real_train],
            "real_validation_stems": [path.stem for path in real_validation],
            "real_calibration_stems_reserved": [
                path.stem for path in real_calibration
            ],
            "real_split_policy": (
                "global deterministic disjoint partition per embryo prefix"
            ),
            "calibration_ground_truth_read": False,
            "candidate_radius_um": args.candidate_radius_um,
            "patch_shape": [17, 17, 17],
            "patch_half_extent_um": [8.0, 8.0, 8.0],
            "input_channels": 3,
            "temporal_frame_offsets": [-1, 0, 1],
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
        },
    )

    optimizer.zero_grad(set_to_none=True)
    attempted_batches = 0
    completed_step = 0
    accumulated_batches = 0
    optimizer_steps = 0
    skipped_batches = 0
    rolling: list[float] = []
    maximum_attempts = max(args.steps * 2, args.steps + 100)
    while completed_step < args.steps and attempted_batches < maximum_attempts:
        if (
            time.monotonic() - started
            >= args.max_wall_seconds - args.finalization_reserve_seconds
        ):
            break
        attempted_batches += 1
        use_real = bool(rng.random() < args.real_replay_probability)
        paths = real_train if use_real else synthetic_train
        try:
            example = base.load_random_transition(
                store,
                paths[int(rng.integers(0, len(paths)))],
                synthetic=not use_real,
                args=args,
                rng=rng,
            )
        except ValueError:
            skipped_batches += 1
            continue
        completed_step += 1
        step = completed_step
        source_volume, target_volume, batch, voxel_size = example
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            source, target, division_logits = base.encode_transition(
                model,
                source_volume,
                target_volume,
                batch,
                voxel_size,
                device,
                augment=True,
            )
            pair_logits, candidates, positives = pair_logits_for_transition(
                model,
                source,
                target,
                division_logits,
                batch,
                voxel_size,
                device,
                candidate_radius_um=args.candidate_radius_um,
                pair_chunk_size=args.pair_chunk_size,
            )
            pair_loss = masked_multi_positive_pair_nll(
                pair_logits, positives, candidates
            )
            embedding_loss = masked_multi_positive_info_nce(
                source,
                target,
                positives,
                candidates,
                temperature=args.temperature,
            )
            division_loss = base.division_prior_corrected_bce(
                division_logits,
                torch.as_tensor(batch.division_target, device=device),
                synthetic=not use_real,
            )
            loss = (
                pair_loss
                + args.embedding_loss_weight * embedding_loss
                + args.division_loss_weight * division_loss
            )
            scaled_loss = loss / args.gradient_accumulation
        scaler.scale(scaled_loss).backward()
        accumulated_batches += 1
        if accumulated_batches == args.gradient_accumulation:
            base.finish_optimizer_step(
                model,
                ema_model,
                optimizer,
                scaler,
                accumulated_batches=accumulated_batches,
                gradient_accumulation=args.gradient_accumulation,
                ema_decay=args.ema_decay,
            )
            accumulated_batches = 0
            optimizer_steps += 1
        rolling.append(float(loss.detach()))
        progress = min(step / args.steps, 1.0)
        learning_rate = args.minimum_learning_rate + 0.5 * (
            args.learning_rate - args.minimum_learning_rate
        ) * (1.0 + np.cos(np.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = float(learning_rate)
        if step % args.log_every == 0:
            print(
                f"{args.fold} step={step} "
                f"loss={np.mean(rolling[-args.log_every:]):.6f} "
                f"skipped={skipped_batches}",
                flush=True,
            )
        should_validate = bool(
            step in {500, 1500, 3000}
            or step % args.validation_every == 0
            or step == args.steps
        )
        if should_validate:
            real_metrics = validate_pair_model(
                ema_model, real_fixed, device, **validation_options
            )
            synthetic_metrics = validate_pair_model(
                ema_model, synthetic_fixed, device, **validation_options
            )
            ema_model.eval()
            score = 0.85 * float(real_metrics["composite"]) + 0.15 * float(
                synthetic_metrics["composite"]
            )
            eligible = bool(
                float(real_metrics["top1"]) >= args.minimum_real_top1
                and float(synthetic_metrics["top1"])
                >= args.minimum_synthetic_top1
            )
            row = {
                "step": step,
                "score": score,
                "eligible": eligible,
                "real": real_metrics,
                "synthetic": synthetic_metrics,
            }
            history.append(row)
            base.atomic_json(output_dir / "validation_latest.json", row)
            if eligible and score > best_score:
                best_step = step
                best_score = score
                best_real = real_metrics
                best_synthetic = synthetic_metrics
                best_state = base.state_dict_cpu(ema_model)
            model.train()

    if accumulated_batches:
        base.finish_optimizer_step(
            model,
            ema_model,
            optimizer,
            scaler,
            accumulated_batches=accumulated_batches,
            gradient_accumulation=args.gradient_accumulation,
            ema_decay=args.ema_decay,
        )
        optimizer_steps += 1
    model.load_state_dict(best_state)
    model_path = output_dir / "appearance_model.pt"
    torch.save(model.state_dict(), model_path)
    base.atomic_json(output_dir / "validation_history.json", {"rows": history})
    terminal = {
        "schema_version": 1,
        "status": "completed",
        "run_id": RUN_ID,
        "appearance_family": PAIR_FUSION_FAMILY,
        "fold": args.fold,
        "elapsed_seconds": time.monotonic() - started,
        "attempted_batches": attempted_batches,
        "completed_step": completed_step,
        "optimizer_steps": optimizer_steps,
        "skipped_batches": skipped_batches,
        "best_step": best_step,
        "best_score": best_score,
        "initial_real": initial_real,
        "initial_synthetic": initial_synthetic,
        "best_real": best_real,
        "best_synthetic": best_synthetic,
        "model_sha256": base.sha256_file(model_path),
        "parameter_count": parameter_count,
        "input_channels": 3,
        "temporal_frame_offsets": [-1, 0, 1],
        "pair_feature_width": PAIR_FEATURE_WIDTH,
        "pair_projection_width": PAIR_PROJECTION_WIDTH,
        "pair_hidden_widths": list(PAIR_HIDDEN_WIDTHS),
        "pair_fusion_policy": PAIR_FUSION_POLICY,
        "pair_loss_policy": PAIR_LOSS_POLICY,
        "embedding_auxiliary_loss_weight": args.embedding_loss_weight,
        "pair_chunk_size": args.pair_chunk_size,
        "checkpoint_weight_source": "optimizer-step exponential moving average",
        "ema_decay": args.ema_decay,
        "division_prior_correction": "class-conditional importance weighting",
        "link_loss_policy": base.LINK_LOSS_POLICY,
        "real_division_rate": base.REAL_DIVISION_RATE,
        "synthetic_division_rate": base.SYNTHETIC_DIVISION_RATE,
        "real_split_policy": (
            "global deterministic disjoint partition per embryo prefix"
        ),
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    base.atomic_json(output_dir / "worker_terminal.json", terminal)
    print(json.dumps(base._plain(terminal), indent=2), flush=True)


def orchestrate(args: argparse.Namespace) -> None:
    started = time.monotonic()
    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"pair-fusion training requires exactly two GPUs, "
            f"saw {torch.cuda.device_count()}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for gpu_index, fold in enumerate(base.FOLD_SPECS):
        log_handle = (args.output_dir / f"{fold}.log").open(
            "w", encoding="utf-8"
        )
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
                raise TimeoutError("pair-fusion orchestrator exceeded hard stop")
            if len(return_codes) < len(processes):
                time.sleep(5)
    finally:
        base.terminate_and_reap_processes(
            [process for _fold, process, _handle in processes]
        )
        for _fold, _process, handle in processes:
            handle.close()
    failures = {fold: code for fold, code in return_codes.items() if code != 0}
    if failures:
        raise RuntimeError(f"pair-fusion workers failed: {failures}")
    terminals = {
        fold: json.loads(
            (args.output_dir / fold / "worker_terminal.json").read_text(
                encoding="utf-8"
            )
        )
        for fold in base.FOLD_SPECS
    }
    terminal = {
        "schema_version": 1,
        "status": "completed",
        "run_id": RUN_ID,
        "appearance_family": PAIR_FUSION_FAMILY,
        "elapsed_seconds": time.monotonic() - started,
        "gpu_count": 2,
        "folds": terminals,
        "both_folds_trained": all(
            int(row["best_step"]) > 0 for row in terminals.values()
        ),
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    base.atomic_json(args.output_dir / "training_terminal.json", terminal)
    print(json.dumps(base._plain(terminal), indent=2), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    role = result.add_mutually_exclusive_group(required=True)
    role.add_argument("--orchestrate", action="store_true")
    role.add_argument("--worker", action="store_true")
    result.add_argument("--fold", choices=sorted(base.FOLD_SPECS))
    result.add_argument("--competition-dir", type=Path, required=True)
    result.add_argument("--synthetic-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--seed", type=int, default=41027)
    result.add_argument("--steps", type=int, default=30000)
    result.add_argument("--max-wall-seconds", type=int, default=36000)
    result.add_argument(
        "--orchestrator-hard-stop-seconds", type=int, default=37800
    )
    result.add_argument("--finalization-reserve-seconds", type=int, default=1200)
    result.add_argument("--synthetic-train-movies", type=int, default=1900)
    result.add_argument("--synthetic-validation-movies", type=int, default=128)
    result.add_argument("--real-train-movies", type=int, default=96)
    result.add_argument("--real-validation-movies", type=int, default=12)
    result.add_argument("--real-calibration-movies", type=int, default=12)
    result.add_argument("--real-validation-transitions", type=int, default=48)
    result.add_argument(
        "--synthetic-validation-transitions", type=int, default=32
    )
    result.add_argument("--real-replay-probability", type=float, default=0.20)
    result.add_argument("--base-channels", type=int, default=64)
    result.add_argument("--embedding-channels", type=int, default=256)
    result.add_argument("--candidate-radius-um", type=float, default=32.0)
    result.add_argument("--max-sources", type=int, default=48)
    result.add_argument("--max-targets", type=int, default=128)
    result.add_argument("--temperature", type=float, default=0.10)
    result.add_argument(
        "--embedding-loss-weight",
        type=float,
        default=EMBEDDING_AUXILIARY_LOSS_WEIGHT,
    )
    result.add_argument("--division-loss-weight", type=float, default=0.20)
    result.add_argument(
        "--pair-chunk-size", type=int, default=DEFAULT_PAIR_CHUNK_SIZE
    )
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
