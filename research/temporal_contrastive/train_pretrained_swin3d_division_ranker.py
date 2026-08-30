#!/usr/bin/env python
"""Train and one-shot audit a pretrained Swin3D division ranker.

The official optimization movies are split again into train, tuning, and audit
roles by movie. Checkpoint selection uses only the nested tuning split. The
nested audit is opened once after the checkpoint is frozen and can only accept
or reject this model as an additional scale-invariant rank voter. The upstream
selection movies opened by v1 are not read; no absolute threshold is fitted.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.temporal_contrastive.train_focused_division_gate import augment_patches
from research.temporal_contrastive.train_real_division_gate import (
    atomic_json,
    balanced_rows,
    load_role,
    sha256_file,
    threshold_metrics,
    validate_manifest,
)


RUN_ID = "competition-pretrained-swin3d-division-ranker-v2"
FAMILY = "kinetics400_swin3d_b_last_stage_pairwise_division_ranker_v2"
EXPECTED_PARAMETER_COUNT = 87_640_009
EXPECTED_TRAINABLE_PARAMETERS = 25_357_761
PRETRAINED_WEIGHTS = "Swin3D_B_Weights.KINETICS400_V1"
TRAINABLE_PREFIXES = ("features.6.", "norm.", "head.")


def state_dict_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def nested_tuning_and_audit_stems(
    inventory: list[dict[str, Any]], targets: torch.Tensor, *, seed: int
) -> tuple[set[str], set[str]]:
    labels = targets.detach().float().cpu().numpy() > 0.5
    by_stem: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(inventory):
        record = by_stem.setdefault(
            str(row["stem"]), {"embryo": str(row["embryo"]), "positives": 0, "rows": 0}
        )
        record["positives"] += int(labels[index])
        record["rows"] += 1
    tuning: set[str] = set()
    audit: set[str] = set()
    for embryo in ("44b6", "6bba"):
        rows = [
            (stem, record)
            for stem, record in by_stem.items()
            if record["embryo"] == embryo and record["positives"] > 0
        ]
        rows.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{embryo}:{row[0]}".encode("utf-8")
            ).hexdigest()
        )
        total_positives = sum(int(record["positives"]) for _, record in rows)
        role_target = max(2, int(math.ceil(0.20 * total_positives)))
        accumulated = 0
        role = tuning
        for stem, record in rows:
            role.add(stem)
            accumulated += int(record["positives"])
            if accumulated >= role_target:
                if role is tuning:
                    role = audit
                    accumulated = 0
                else:
                    break
    if tuning & audit:
        raise RuntimeError("nested Swin3D tuning and audit movies overlap")
    labels = targets.detach().cpu()
    tuning_mask = torch.as_tensor(
        [str(row["stem"]) in tuning for row in inventory], dtype=torch.bool
    )
    audit_mask = torch.as_tensor(
        [str(row["stem"]) in audit for row in inventory], dtype=torch.bool
    )
    train_mask = ~(tuning_mask | audit_mask)
    for role, mask in (
        ("train", train_mask),
        ("tuning", tuning_mask),
        ("audit", audit_mask),
    ):
        if not (
            torch.any(labels[mask] > 0.5) and torch.any(labels[mask] < 0.5)
        ):
            raise RuntimeError(f"nested Swin3D {role} split lost a class")
    return tuning, audit


def configure_model(device: torch.device) -> tuple[torch.nn.Module, dict[str, int]]:
    from torchvision.models.video import Swin3D_B_Weights, swin3d_b

    model = swin3d_b(weights=Swin3D_B_Weights.KINETICS400_V1, progress=True)
    model.head = torch.nn.Linear(model.head.in_features, 1)
    model.to(device)
    trainable = 0
    frozen = 0
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(TRAINABLE_PREFIXES))
        if parameter.requires_grad:
            trainable += parameter.numel()
        else:
            frozen += parameter.numel()
    if (
        trainable != EXPECTED_TRAINABLE_PARAMETERS
        or trainable + frozen != EXPECTED_PARAMETER_COUNT
    ):
        raise RuntimeError(
            f"Swin3D parameter inventory changed: {trainable}+{frozen}"
        )
    return model, {
        "parameter_count": trainable + frozen,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
    }


def rank_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = logits.float().flatten()
    targets = targets.float().flatten()
    positives = logits[targets > 0.5]
    negatives = logits[targets < 0.5]
    if not len(positives) or not len(negatives):
        raise RuntimeError("pairwise division batch lost a class")
    pairwise = F.softplus(0.5 - positives[:, None] + negatives[None, :]).mean()
    auxiliary = F.binary_cross_entropy_with_logits(logits, targets)
    return pairwise + 0.20 * auxiliary


def update_ema(model: torch.nn.Module, ema: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_value, value in zip(
            ema.state_dict().values(), model.state_dict().values(), strict=True
        ):
            if ema_value.is_floating_point():
                ema_value.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                ema_value.copy_(value)


@torch.inference_mode()
def predict_tta(
    model: torch.nn.Module, patches: torch.Tensor, *, batch_size: int
) -> torch.Tensor:
    model.eval()
    pieces: list[torch.Tensor] = []
    for start in range(0, len(patches), batch_size):
        batch = patches[start : start + batch_size]
        views = (batch, batch.flip(3), batch.flip(4), batch.flip((3, 4)))
        scores = []
        for view in views:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                scores.append(model(view).float().flatten())
        pieces.append(torch.stack(scores).mean(dim=0))
    return torch.cat(pieces)


def acceptance_metrics(targets: torch.Tensor, scores: torch.Tensor) -> dict[str, Any]:
    return threshold_metrics(targets, scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=732_451)
    parser.add_argument("--steps", type=int, default=1_500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=24)
    parser.add_argument("--validation-every", type=int, default=150)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--minimum-learning-rate", type=float, default=2e-7)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--required-gpu-name", default="A10G")
    args = parser.parse_args()
    if min(
        args.steps,
        args.batch_size,
        args.validation_batch_size,
        args.validation_every,
        args.log_every,
    ) <= 0:
        raise ValueError("Swin3D training counts must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Swin3D rank training requires one Antelume GPU")
    gpu_name = torch.cuda.get_device_name(0)
    if args.required_gpu_name.lower() not in gpu_name.lower():
        raise RuntimeError(f"required GPU {args.required_gpu_name!r}, saw {gpu_name!r}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    manifest_path = args.data_root / "real_division_patch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    args.output_root.mkdir(parents=True, exist_ok=False)
    optimization_x, optimization_y, _, optimization_inventory = load_role(
        args.data_root, manifest, "optimization", device
    )
    tuning_stems, audit_stems = nested_tuning_and_audit_stems(
        optimization_inventory, optimization_y, seed=args.seed
    )
    tuning_mask = torch.as_tensor(
        [str(row["stem"]) in tuning_stems for row in optimization_inventory],
        dtype=torch.bool,
        device=device,
    )
    audit_mask = torch.as_tensor(
        [str(row["stem"]) in audit_stems for row in optimization_inventory],
        dtype=torch.bool,
        device=device,
    )
    train_mask = ~(tuning_mask | audit_mask)
    train_x = optimization_x[train_mask]
    train_y = optimization_y[train_mask]
    tuning_x = optimization_x[tuning_mask]
    tuning_y = optimization_y[tuning_mask]
    audit_x = optimization_x[audit_mask]
    audit_y = optimization_y[audit_mask]
    train_stems = {
        str(row["stem"])
        for row in optimization_inventory
        if str(row["stem"]) not in tuning_stems | audit_stems
    }
    audit_inventory = [
        row
        for row in optimization_inventory
        if str(row["stem"]) in audit_stems
    ]
    if (
        train_stems & tuning_stems
        or train_stems & audit_stems
        or tuning_stems & audit_stems
    ):
        raise RuntimeError("Swin3D train, tuning, and one-shot audit movies overlap")

    model, inventory = configure_model(device)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed + 91_337)
    initial_scores = predict_tta(
        ema, tuning_x, batch_size=args.validation_batch_size
    )
    initial_metrics = acceptance_metrics(tuning_y, initial_scores)
    best_metrics = initial_metrics
    best_step = 0
    best_state = state_dict_cpu(ema)
    history: list[dict[str, Any]] = []
    atomic_json(
        args.output_root / "training_config.json",
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "family": FAMILY,
            "pretrained_weights": PRETRAINED_WEIGHTS,
            "seed": args.seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "validation_batch_size": args.validation_batch_size,
            "validation_every": args.validation_every,
            "learning_rate": args.learning_rate,
            "minimum_learning_rate": args.minimum_learning_rate,
            "weight_decay": args.weight_decay,
            "ema_decay": args.ema_decay,
            "selection_metric": "nested movie-disjoint average precision",
            "absolute_threshold_used": False,
            "tuning_stems": sorted(tuning_stems),
            "audit_stems": sorted(audit_stems),
            "audit_opened": False,
            "upstream_selection_movies_read": False,
            **inventory,
            "competition_train_data_read": True,
            "competition_test_data_read": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
        },
    )

    started = time.monotonic()
    for step in range(1, args.steps + 1):
        model.train()
        rows = balanced_rows(train_y, args.batch_size, generator)
        augmented = augment_patches(train_x[rows], generator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(augmented).flatten()
            loss = rank_loss(logits, train_y[rows])
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            1.0,
        )
        scaler.step(optimizer)
        scaler.update()
        update_ema(model, ema, args.ema_decay)
        progress = step / args.steps
        learning_rate = args.minimum_learning_rate + 0.5 * (
            args.learning_rate - args.minimum_learning_rate
        ) * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        if step == 1 or step % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        "learning_rate": learning_rate,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if step % args.validation_every == 0 or step == args.steps:
            tuning_scores = predict_tta(
                ema, tuning_x, batch_size=args.validation_batch_size
            )
            metrics = acceptance_metrics(tuning_y, tuning_scores)
            improved = (
                float(metrics["average_precision"]),
                float(metrics["recall_at_zero_false_positives"]),
            ) > (
                float(best_metrics["average_precision"]),
                float(best_metrics["recall_at_zero_false_positives"]),
            )
            history.append({"step": step, "metrics": metrics, "improved": improved})
            atomic_json(args.output_root / "tuning_latest.json", history[-1])
            if improved:
                best_step = step
                best_metrics = metrics
                best_state = state_dict_cpu(ema)

    model.load_state_dict(best_state, strict=True)
    model.requires_grad_(False).eval()
    checkpoint = args.output_root / "swin3d_division_ranker.pt"
    torch.save(model.state_dict(), checkpoint)
    atomic_json(args.output_root / "tuning_history.json", {"rows": history})

    # The nested one-shot audit is deliberately after checkpoint persistence.
    audit_scores = predict_tta(model, audit_x, batch_size=args.validation_batch_size)
    audit_metrics = acceptance_metrics(audit_y, audit_scores)
    audit_by_embryo = {}
    for embryo in ("44b6", "6bba"):
        indices = torch.as_tensor(
            [
                index
                for index, row in enumerate(audit_inventory)
                if row["embryo"] == embryo
            ],
            dtype=torch.long,
            device=device,
        )
        audit_by_embryo[embryo] = acceptance_metrics(
            audit_y[indices], audit_scores[indices]
        )
    accepted = (
        float(best_metrics["average_precision"]) >= 0.60
        and float(audit_metrics["average_precision"]) >= 0.55
        and all(
            float(metrics["average_precision"]) >= 0.40
            for metrics in audit_by_embryo.values()
        )
    )
    terminal = {
        "schema_version": 1,
        "status": "accepted_as_rank_voter" if accepted else "rejected",
        "run_id": RUN_ID,
        "family": FAMILY,
        "pretrained_weights": PRETRAINED_WEIGHTS,
        "elapsed_seconds": time.monotonic() - started,
        "gpu_name": gpu_name,
        "best_step": best_step,
        "initial_tuning": initial_metrics,
        "best_tuning": best_metrics,
        "audit": audit_metrics,
        "audit_by_embryo": audit_by_embryo,
        "audit_opened_once_after_checkpoint_freeze": True,
        "upstream_selection_movies_read": False,
        "checkpoint_sha256": sha256_file(checkpoint),
        "manifest_sha256": sha256_file(manifest_path),
        "absolute_threshold_used": False,
        **inventory,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_ensemble_evaluation": accepted,
        "authorized_for_submission": False,
    }
    atomic_json(args.output_root / "swin3d_division_ranker_terminal.json", terminal)
    print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
