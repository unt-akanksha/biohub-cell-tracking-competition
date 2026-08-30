#!/usr/bin/env python
"""Fine-tune two independent division gates on official Biohub train patches.

The four complete-movie probes are excluded by the upstream extractor.  Both
models optimize on the pooled optimization movies, select checkpoints and a
single ensemble threshold on the pooled movie-disjoint selection set, and
leave the final probes unopened for a later one-shot acceptance evaluation.
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

from research.temporal_contrastive.multiscale_contextual_pair_fusion import (
    EXPECTED_PARAMETER_COUNT,
    MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
    MultiscaleContextualPairFusionAssociationModel,
)
from research.temporal_contrastive.train_focused_division_gate import (
    augment_patches,
    division_logits,
)


RUN_ID = "competition-real-division-gate-v1"
FAMILY = "competition_real_temporal_multiscale_division_gate_v1"
FOLDS = ("target_44b6", "target_6bba")
SEED_OFFSETS = {"target_44b6": 0, "target_6bba": 10_003}
TRAINABLE_PREFIXES = {
    "head_only": ("division.", "division_adapter."),
    "focused": (
        "division.",
        "axial_projection.",
        "projection_stem.",
        "projection_encoder.",
        "division_adapter.",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def state_dict_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def validate_manifest(manifest: dict[str, Any]) -> None:
    summary = manifest.get("summary", {})
    split = summary.get("by_embryo_role", {})
    expected_positive_counts = {
        "44b6": {"optimization": 19, "selection": 5},
        "6bba": {"optimization": 96, "selection": 26},
    }
    observed = {
        embryo: {
            role: split.get(embryo, {}).get(role, {}).get("division_positives")
            for role in ("optimization", "selection")
        }
        for embryo in ("44b6", "6bba")
    }
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("status") == "complete"
        and manifest.get("run_id") == "competition-real-division-patches-v1"
        and manifest.get("competition_train_data_read") is True
        and manifest.get("competition_test_data_read") is False
        and manifest.get("public_leaderboard_used_for_selection") is False
        and manifest.get("submission_created") is False
        and manifest.get("authorized_for_submission") is False
        and summary.get("movies") == 199
        and summary.get("excluded_final_probe_movies") == 4
        and summary.get("division_positives") == 146
        and summary.get("ordinary_unlabeled_controls") == 1247
        and observed == expected_positive_counts
        and len(manifest.get("final_probe_stems", [])) == 4
    ):
        raise ValueError("real division patch manifest is ineligible")


def load_role(
    root: Path,
    manifest: dict[str, Any],
    role: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    patches: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    inventory: list[dict[str, Any]] = []
    records = [record for record in manifest["shards"] if record["role"] == role]
    if not records:
        raise ValueError(f"real division {role} inventory is empty")
    for record in records:
        path = root / str(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"real division shard verification failed: {path}")
        with np.load(path, allow_pickle=False) as data:
            source = np.asarray(data["source_patches"], dtype=np.float32)
            target = np.asarray(data["division_target"], dtype=np.float32)
            weight = np.asarray(data["label_weight"], dtype=np.float32)
            metadata = json.loads(str(data["metadata_json"].item()))
        if not (
            source.ndim == 5
            and source.shape[1:] == (3, 17, 17, 17)
            and target.shape == weight.shape == (len(source),)
            and np.isin(target, (0.0, 1.0)).all()
            and np.all(weight[target > 0.5] == 1.0)
            and np.all(weight[target < 0.5] == 0.25)
            and metadata.get("stem") == record["stem"]
            and metadata.get("role") == role
            and metadata.get("competition_test_data_read") is False
        ):
            raise ValueError(f"real division shard contract changed: {path}")
        patches.append(torch.as_tensor(source, device=device))
        targets.append(torch.as_tensor(target, device=device))
        weights.append(torch.as_tensor(weight, device=device))
        inventory.extend(
            {
                "stem": record["stem"],
                "embryo": record["embryo"],
                "timepoint": int(record["timepoint"]),
                "source_row": index,
            }
            for index in range(len(source))
        )
    result_patches = torch.cat(patches)
    result_targets = torch.cat(targets)
    result_weights = torch.cat(weights)
    if not torch.any(result_targets > 0.5) or not torch.any(result_targets < 0.5):
        raise RuntimeError(f"real division {role} inventory lost a class")
    return result_patches, result_targets, result_weights, inventory


def average_precision(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.detach().float().cpu() > 0.5
    scores = scores.detach().float().cpu()
    positives = int(labels.sum())
    if positives <= 0 or positives >= len(labels):
        raise ValueError("average precision requires both classes")
    order = torch.argsort(scores, descending=True, stable=True)
    ranked = labels[order]
    cumulative = torch.cumsum(ranked.to(torch.float64), dim=0)
    ranks = torch.arange(1, len(ranked) + 1, dtype=torch.float64)
    return float((cumulative[ranked] / ranks[ranked]).mean())


def threshold_metrics(labels: torch.Tensor, scores: torch.Tensor) -> dict[str, Any]:
    labels = labels.detach().float().cpu() > 0.5
    scores = scores.detach().float().cpu()
    order = torch.argsort(scores, descending=True, stable=True)
    ranked_labels = labels[order]
    ranked_scores = scores[order]
    tp = torch.cumsum(ranked_labels.to(torch.int64), dim=0)
    fp = torch.cumsum((~ranked_labels).to(torch.int64), dim=0)
    positives = int(labels.sum())
    fn = positives - tp
    precision = tp.float() / (tp + fp).clamp_min(1).float()
    recall = tp.float() / max(positives, 1)
    jaccard = tp.float() / (tp + fp + fn).clamp_min(1).float()
    zero_fp = torch.nonzero((fp == 0) & (tp > 0), as_tuple=False).flatten()
    precise = torch.nonzero((precision >= 0.95) & (tp > 0), as_tuple=False).flatten()
    best_index = int(torch.argmax(jaccard))
    return {
        "rows": len(labels),
        "positives": positives,
        "average_precision": average_precision(labels.float(), scores),
        "best_jaccard": float(jaccard[best_index]),
        "best_jaccard_threshold": float(ranked_scores[best_index]),
        "recall_at_zero_false_positives": (
            float(recall[zero_fp[-1]]) if len(zero_fp) else 0.0
        ),
        "recall_at_precision_0_95": (
            float(recall[precise].max()) if len(precise) else 0.0
        ),
        "binary_cross_entropy": float(
            F.binary_cross_entropy_with_logits(scores, labels.float())
        ),
    }


def selection_utility(metrics: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(metrics["recall_at_zero_false_positives"]),
        float(metrics["recall_at_precision_0_95"]),
        float(metrics["best_jaccard"]),
        float(metrics["average_precision"]),
        -float(metrics["binary_cross_entropy"]),
    )


def select_frozen_threshold(labels: torch.Tensor, scores: torch.Tensor) -> dict[str, Any]:
    labels = labels.detach().float().cpu() > 0.5
    scores = scores.detach().float().cpu()
    order = torch.argsort(scores, descending=True, stable=True)
    ranked_labels = labels[order]
    ranked_scores = scores[order]
    tp = torch.cumsum(ranked_labels.to(torch.int64), dim=0)
    fp = torch.cumsum((~ranked_labels).to(torch.int64), dim=0)
    fn = int(labels.sum()) - tp
    precision = tp.float() / (tp + fp).clamp_min(1).float()
    recall = tp.float() / max(int(labels.sum()), 1)
    jaccard = tp.float() / (tp + fp + fn).clamp_min(1).float()
    eligible = torch.nonzero((fp == 0) & (tp >= 2), as_tuple=False).flatten()
    if len(eligible):
        index = int(eligible[-1])
        policy = "maximum recall with zero false positives and at least two positives"
    else:
        eligible = torch.nonzero((precision >= 0.95) & (tp > 0), as_tuple=False).flatten()
        if not len(eligible):
            raise RuntimeError("selection cannot provide a high-precision threshold")
        index = int(eligible[torch.argmax(jaccard[eligible])])
        policy = "maximum Jaccard among thresholds with at least 0.95 precision"
    threshold = float(ranked_scores[index])
    return {
        "threshold": threshold,
        "policy": policy,
        "tp": int(tp[index]),
        "fp": int(fp[index]),
        "fn": int(fn[index]),
        "precision": float(precision[index]),
        "recall": float(recall[index]),
        "jaccard": float(jaccard[index]),
    }


def balanced_rows(
    targets: torch.Tensor, batch_size: int, generator: torch.Generator
) -> torch.Tensor:
    positive = torch.nonzero(targets > 0.5, as_tuple=False).flatten()
    negative = torch.nonzero(targets < 0.5, as_tuple=False).flatten()
    positive_count = max(1, batch_size // 5)
    negative_count = batch_size - positive_count
    chosen = torch.cat(
        (
            positive[
                torch.randint(
                    len(positive),
                    (positive_count,),
                    generator=generator,
                    device=targets.device,
                )
            ],
            negative[
                torch.randint(
                    len(negative),
                    (negative_count,),
                    generator=generator,
                    device=targets.device,
                )
            ],
        )
    )
    return chosen[
        torch.randperm(len(chosen), generator=generator, device=targets.device)
    ]


def weighted_focal_loss(
    logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    logits = logits.float()
    targets = targets.float()
    weights = weights.float()
    base = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = torch.sigmoid(logits)
    target_probability = torch.where(targets > 0.5, probability, 1.0 - probability)
    losses = ((1.0 - target_probability) ** 2.0) * base * weights
    return losses.sum() / weights.sum().clamp_min(1e-6)


@torch.inference_mode()
def predict(
    model: torch.nn.Module, patches: torch.Tensor, *, batch_size: int
) -> torch.Tensor:
    model.eval()
    pieces = []
    for start in range(0, len(patches), batch_size):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pieces.append(division_logits(model, patches[start : start + batch_size]).float())
    return torch.cat(pieces)


def configure_parameters(model: torch.nn.Module, mode: str) -> dict[str, int]:
    prefixes = TRAINABLE_PREFIXES[mode]
    trainable = 0
    frozen = 0
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
        if parameter.requires_grad:
            trainable += parameter.numel()
        else:
            frozen += parameter.numel()
    if trainable <= 0 or trainable + frozen != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("real division parameter inventory changed")
    return {"trainable_parameters": trainable, "frozen_parameters": frozen}


def update_ema(model: torch.nn.Module, ema: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_value, value in zip(
            ema.state_dict().values(), model.state_dict().values(), strict=True
        ):
            if ema_value.is_floating_point():
                ema_value.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                ema_value.copy_(value)


def train_fold(
    *,
    fold: str,
    initial_model_path: Path,
    train_patches: torch.Tensor,
    train_targets: torch.Tensor,
    train_weights: torch.Tensor,
    selection_patches: torch.Tensor,
    selection_targets: torch.Tensor,
    output_root: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    started = time.monotonic()
    seed = args.seed + SEED_OFFSETS[fold]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device).manual_seed(seed + 91_337)
    model = MultiscaleContextualPairFusionAssociationModel().to(device)
    model.load_state_dict(
        torch.load(initial_model_path, map_location=device, weights_only=True),
        strict=True,
    )
    initialization_sha256 = sha256_file(initial_model_path)
    inventory = configure_parameters(model, args.train_mode)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda")
    initial_logits = predict(ema, selection_patches, batch_size=args.validation_batch_size)
    initial_metrics = threshold_metrics(selection_targets, initial_logits)
    best_metrics = initial_metrics
    best_state = state_dict_cpu(ema)
    best_step = 0
    history: list[dict[str, Any]] = []
    fold_root = output_root / fold
    fold_root.mkdir(parents=True, exist_ok=False)
    atomic_json(
        fold_root / "training_config.json",
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "family": FAMILY,
            "fold": fold,
            "seed": seed,
            "train_mode": args.train_mode,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "minimum_learning_rate": args.minimum_learning_rate,
            "weight_decay": args.weight_decay,
            "ema_decay": args.ema_decay,
            "initial_model_sha256": initialization_sha256,
            **inventory,
            "competition_train_data_read": True,
            "competition_test_data_read": False,
            "final_probe_opened": False,
            "submission_created": False,
        },
    )
    for step in range(1, args.steps + 1):
        model.train()
        rows = balanced_rows(train_targets, args.batch_size, generator)
        augmented = augment_patches(train_patches[rows], generator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = division_logits(model, augmented)
            loss = weighted_focal_loss(logits, train_targets[rows], train_weights[rows])
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            2.0,
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
                        "fold": fold,
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        "learning_rate": learning_rate,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if step % args.validation_every == 0 or step == args.steps:
            candidate_logits = predict(
                ema, selection_patches, batch_size=args.validation_batch_size
            )
            metrics = threshold_metrics(selection_targets, candidate_logits)
            improved = selection_utility(metrics) > selection_utility(best_metrics)
            history.append({"step": step, "metrics": metrics, "improved": improved})
            atomic_json(fold_root / "selection_latest.json", history[-1])
            if improved:
                best_step = step
                best_metrics = metrics
                best_state = state_dict_cpu(ema)
    model.load_state_dict(best_state, strict=True)
    checkpoint = fold_root / "division_model.pt"
    torch.save(model.state_dict(), checkpoint)
    atomic_json(fold_root / "selection_history.json", {"rows": history})
    terminal = {
        "schema_version": 1,
        "status": "completed",
        "run_id": RUN_ID,
        "family": FAMILY,
        "fold": fold,
        "elapsed_seconds": time.monotonic() - started,
        "completed_step": args.steps,
        "best_step": best_step,
        "initial_selection": initial_metrics,
        "best_selection": best_metrics,
        "selection_improved": selection_utility(best_metrics)
        > selection_utility(initial_metrics),
        "model_sha256": sha256_file(checkpoint),
        "initial_model_sha256": initialization_sha256,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        **inventory,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "final_probe_opened": False,
        "checkpoint_frozen_before_final_probe": True,
        "submission_created": False,
    }
    atomic_json(fold_root / "worker_terminal.json", terminal)
    del model, ema, optimizer
    torch.cuda.empty_cache()
    return terminal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--target-44b6-initial-model", type=Path, required=True)
    parser.add_argument("--target-6bba-initial-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-mode", choices=tuple(TRAINABLE_PREFIXES), default="head_only")
    parser.add_argument("--seed", type=int, default=105_041)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-batch-size", type=int, default=96)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--required-gpu-name", default="A10G")
    args = parser.parse_args()
    counts = (
        args.steps,
        args.batch_size,
        args.validation_batch_size,
        args.validation_every,
        args.log_every,
    )
    if min(counts) <= 0:
        raise ValueError("real division training counts must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("real division training requires one Antelume GPU")
    gpu_name = torch.cuda.get_device_name(0)
    if args.required_gpu_name.lower() not in gpu_name.lower():
        raise RuntimeError(f"required Antelume GPU {args.required_gpu_name!r}, saw {gpu_name!r}")
    manifest_path = args.data_root / "real_division_patch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    args.output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    train_patches, train_targets, train_weights, train_inventory = load_role(
        args.data_root, manifest, "optimization", device
    )
    selection_patches, selection_targets, _, selection_inventory = load_role(
        args.data_root, manifest, "selection", device
    )
    train_stems = {row["stem"] for row in train_inventory}
    selection_stems = {row["stem"] for row in selection_inventory}
    if train_stems & selection_stems:
        raise RuntimeError("real division optimization and selection movies overlap")
    initial_models = {
        "target_44b6": args.target_44b6_initial_model,
        "target_6bba": args.target_6bba_initial_model,
    }
    initial_hashes = {fold: sha256_file(path) for fold, path in initial_models.items()}
    if len(set(initial_hashes.values())) != len(initial_hashes):
        raise ValueError("real division models require independent initial checkpoints")
    started = time.monotonic()
    terminals = {
        fold: train_fold(
            fold=fold,
            initial_model_path=initial_models[fold],
            train_patches=train_patches,
            train_targets=train_targets,
            train_weights=train_weights,
            selection_patches=selection_patches,
            selection_targets=selection_targets,
            output_root=args.output_root,
            args=args,
            device=device,
        )
        for fold in FOLDS
    }
    models = []
    for fold in FOLDS:
        model = MultiscaleContextualPairFusionAssociationModel().to(device)
        model.load_state_dict(
            torch.load(
                args.output_root / fold / "division_model.pt",
                map_location=device,
                weights_only=True,
            ),
            strict=True,
        )
        models.append(model.requires_grad_(False).eval())
    ensemble_logits = torch.stack(
        [
            predict(model, selection_patches, batch_size=args.validation_batch_size)
            for model in models
        ]
    ).mean(dim=0)
    frozen = select_frozen_threshold(selection_targets, ensemble_logits)
    ensemble_metrics = threshold_metrics(selection_targets, ensemble_logits)
    selection_by_embryo = {}
    for embryo in ("44b6", "6bba"):
        indices = torch.as_tensor(
            [
                index
                for index, row in enumerate(selection_inventory)
                if row["embryo"] == embryo
            ],
            device=device,
        )
        selection_by_embryo[embryo] = threshold_metrics(
            selection_targets[indices], ensemble_logits[indices]
        )
    accepted = bool(
        frozen["fp"] == 0
        and frozen["tp"] >= 2
        and ensemble_metrics["average_precision"] >= 0.50
        and all(
            metrics["average_precision"] >= 0.40
            for metrics in selection_by_embryo.values()
        )
    )
    terminal = {
        "schema_version": 1,
        "status": "accepted_at_selection" if accepted else "rejected_at_selection",
        "run_id": RUN_ID,
        "family": FAMILY,
        "appearance_family": MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        "gpu_name": gpu_name,
        "execution_gpu_count": 1,
        "execution_policy": "two independent models sequentially on Antelume A10G",
        "elapsed_seconds": time.monotonic() - started,
        "train_mode": args.train_mode,
        "train_rows": len(train_targets),
        "train_positives": int((train_targets > 0.5).sum()),
        "selection_rows": len(selection_targets),
        "selection_positives": int((selection_targets > 0.5).sum()),
        "train_movie_count": len(train_stems),
        "selection_movie_count": len(selection_stems),
        "manifest_sha256": sha256_file(manifest_path),
        "folds": terminals,
        "ensemble_selection": ensemble_metrics,
        "selection_by_embryo": selection_by_embryo,
        "frozen_division_logit_threshold": frozen["threshold"],
        "threshold_selection": frozen,
        "selection_gate_passed": accepted,
        "final_probe_opened": False,
        "checkpoint_frozen_before_final_probe": True,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_final_probe": accepted,
        "authorized_for_submission": False,
    }
    atomic_json(args.output_root / "real_division_gate_terminal.json", terminal)
    print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
