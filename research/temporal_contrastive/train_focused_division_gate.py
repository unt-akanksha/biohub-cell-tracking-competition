#!/usr/bin/env python
"""Train a focused external division gate and freeze its recovery policy.

The two folds run sequentially on one Antelume GPU.  They warm-start from the
prediction-preserving step-zero v4 checkpoints rejected by the association
gate, optimize only division-relevant morphology paths on ZSNS004, select on
new ZSNS005 windows, then open a disjoint untouched audit once.
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

try:
    from calibrate_division_recovery_policy import (
        recovery_metrics,
        score_records,
        select_threshold,
        sha256_file,
    )
    from multiscale_contextual_pair_fusion import (
        EXPECTED_PARAMETER_COUNT,
        MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        MultiscaleContextualPairFusionAssociationModel,
    )
    from train_dual_fold_division_localization import (
        AUDIT_TIMEPOINTS,
        SELECTION_TIMEPOINTS,
        discover_shards as discover_localization_shards,
    )
    from train_zebrahub_contextual_pretrain import (
        discover_shards as discover_contextual_shards,
    )
except ModuleNotFoundError:
    from research.temporal_contrastive.calibrate_division_recovery_policy import (
        recovery_metrics,
        score_records,
        select_threshold,
        sha256_file,
    )
    from research.temporal_contrastive.multiscale_contextual_pair_fusion import (
        EXPECTED_PARAMETER_COUNT,
        MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        MultiscaleContextualPairFusionAssociationModel,
    )
    from research.temporal_contrastive.train_dual_fold_division_localization import (
        AUDIT_TIMEPOINTS,
        SELECTION_TIMEPOINTS,
        discover_shards as discover_localization_shards,
    )
    from research.temporal_contrastive.train_zebrahub_contextual_pretrain import (
        discover_shards as discover_contextual_shards,
    )


RUN_ID = "focused-division-gate-v1"
POLICY_RUN_ID = "external-division-recovery-policy-v1"
FAMILY = "temporal_multiscale_focused_division_gate_v1"
PARENT_RUN_ID = "zebrahub-multiscale-contextual-pretrain-v1"
FOLDS = ("target_44b6", "target_6bba")
SEED_OFFSETS = {"target_44b6": 0, "target_6bba": 10_003}
TRAINABLE_PREFIXES = (
    "division.",
    "axial_projection.",
    "projection_stem.",
    "projection_encoder.",
    "division_adapter.",
)


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


def load_rejected_zero_residual_parent(
    model: MultiscaleContextualPairFusionAssociationModel,
    root: Path,
    fold: str,
) -> dict[str, Any]:
    aggregate_path = root / "pretraining_terminal.json"
    worker_path = root / fold / "worker_terminal.json"
    model_path = root / fold / "pretrained_model.pt"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    worker = json.loads(worker_path.read_text(encoding="utf-8"))
    digest = sha256_file(model_path)
    initialization = worker.get("initialization", {})
    if not (
        aggregate.get("schema_version") == 1
        and aggregate.get("status") == "completed"
        and aggregate.get("run_id") == PARENT_RUN_ID
        and aggregate.get("appearance_family")
        == MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY
        and aggregate.get("gpu_count") == 2
        and aggregate.get("both_folds_improved") is False
        and aggregate.get("competition_data_read") is False
        and aggregate.get("public_code_copied") is False
        and aggregate.get("public_predictions_copied") is False
        and aggregate.get("public_leaderboard_used_for_selection") is False
        and aggregate.get("submission_created") is False
        and worker == aggregate.get("folds", {}).get(fold)
        and worker.get("status") == "completed"
        and worker.get("fold") == fold
        and worker.get("parameter_count") == EXPECTED_PARAMETER_COUNT
        and int(worker.get("best_step", -1)) == 0
        and worker.get("selection_gate_passed") is False
        and worker.get("audit_gate_passed") is False
        and worker.get("best_selection") == worker.get("initial_selection")
        and worker.get("final_audit") == worker.get("initial_audit")
        and worker.get("model_sha256") == digest
        and initialization.get("source_family")
        == "temporal_contextual_pair_fusion_v3"
        and initialization.get("target_family")
        == MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY
        and initialization.get("initial_predictions_numerically_preserved") is True
        and worker.get("competition_data_read") is False
        and worker.get("public_code_copied") is False
        and worker.get("public_predictions_copied") is False
        and worker.get("public_leaderboard_used_for_selection") is False
        and worker.get("submission_created") is False
    ):
        raise ValueError(f"ineligible zero-residual v4 bootstrap: {fold}")
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True), strict=True
    )
    return {
        "run_id": PARENT_RUN_ID,
        "appearance_family": MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        "fold": fold,
        "model_sha256": digest,
        "worker_terminal_sha256": sha256_file(worker_path),
        "parent_association_gate_passed": False,
        "parent_stage": "rejected_zero_residual_v3_bootstrap",
        "initial_predictions_numerically_preserved": True,
        "authorized_for_focused_division_initialization_only": True,
    }


def configure_focused_parameters(model: torch.nn.Module) -> dict[str, int]:
    trainable = 0
    frozen = 0
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(TRAINABLE_PREFIXES))
        if parameter.requires_grad:
            trainable += parameter.numel()
        else:
            frozen += parameter.numel()
    if trainable <= 0 or trainable + frozen != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("focused division parameter inventory changed")
    return {"trainable_parameters": trainable, "frozen_parameters": frozen}


def division_logits(
    model: MultiscaleContextualPairFusionAssociationModel,
    patches: torch.Tensor,
) -> torch.Tensor:
    hidden = model.encoder(model.stem(patches)).mean(dim=(2, 3, 4))
    physical = model.division(hidden).squeeze(1)
    axial = model.axial_projection(patches)
    projected = model.projection_encoder(model.projection_stem(axial)).mean(
        dim=(2, 3)
    )
    return physical + model.division_adapter(projected).squeeze(1)


def load_division_inventory(
    records: list[Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, int]:
    patches: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    tensor_bytes = 0
    for record in records:
        with np.load(record.path) as data:
            source = np.asarray(data["source_patches"], dtype=np.float32)
            division = np.asarray(data["division_target"], dtype=np.float32)
        if source.ndim != 5 or source.shape[1:] != (3, 17, 17, 17):
            raise ValueError(f"division source patch shape changed: {record.path.name}")
        if division.shape != (len(source),) or not np.isin(division, (0.0, 1.0)).all():
            raise ValueError(f"division targets changed: {record.path.name}")
        source_tensor = torch.as_tensor(source, device=device)
        target_tensor = torch.as_tensor(division, device=device)
        patches.append(source_tensor)
        targets.append(target_tensor)
        tensor_bytes += source_tensor.numel() * source_tensor.element_size()
        tensor_bytes += target_tensor.numel() * target_tensor.element_size()
    result_patches = torch.cat(patches)
    result_targets = torch.cat(targets)
    if not torch.any(result_targets > 0.5) or not torch.any(result_targets < 0.5):
        raise RuntimeError("focused division inventory lost a class")
    return result_patches, result_targets, tensor_bytes


def average_precision(logits: torch.Tensor, targets: torch.Tensor) -> float:
    labels = targets.detach().float().cpu()
    scores = logits.detach().float().cpu()
    positive_count = int((labels > 0.5).sum())
    if positive_count <= 0 or positive_count >= len(labels):
        raise ValueError("average precision requires both classes")
    order = torch.argsort(scores, descending=True, stable=True)
    ranked = labels[order] > 0.5
    cumulative = torch.cumsum(ranked.to(torch.float64), dim=0)
    ranks = torch.arange(1, len(ranked) + 1, dtype=torch.float64)
    return float((cumulative[ranked] / ranks[ranked]).mean())


def classification_metrics(
    logits: torch.Tensor, targets: torch.Tensor
) -> dict[str, float | int]:
    labels = targets.detach().float().cpu() > 0.5
    scores = logits.detach().float().cpu()
    order = torch.argsort(scores, descending=True, stable=True)
    ranked = labels[order]
    tp = torch.cumsum(ranked.to(torch.float64), dim=0)
    fp = torch.cumsum((~ranked).to(torch.float64), dim=0)
    positives = int(labels.sum())
    fn = positives - tp
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / max(positives, 1)
    jaccard = tp / (tp + fp + fn).clamp_min(1.0)
    high_precision = recall[precision >= 0.80]
    return {
        "rows": len(labels),
        "positives": positives,
        "average_precision": average_precision(scores, labels.float()),
        "best_jaccard": float(jaccard.max()),
        "recall_at_precision_0_80": (
            float(high_precision.max()) if len(high_precision) else 0.0
        ),
        "binary_cross_entropy": float(
            F.binary_cross_entropy_with_logits(scores, labels.float())
        ),
    }


def focused_selection_gate(
    baseline: dict[str, float | int], candidate: dict[str, float | int]
) -> dict[str, Any]:
    inventory_unchanged = bool(
        int(candidate["rows"]) == int(baseline["rows"])
        and int(candidate["positives"]) == int(baseline["positives"])
    )
    gains = {
        "average_precision": float(candidate["average_precision"])
        - float(baseline["average_precision"]),
        "best_jaccard": float(candidate["best_jaccard"])
        - float(baseline["best_jaccard"]),
        "recall_at_precision_0_80": float(candidate["recall_at_precision_0_80"])
        - float(baseline["recall_at_precision_0_80"]),
    }
    passed = bool(
        inventory_unchanged
        and gains["average_precision"] >= 0.02
        and gains["best_jaccard"] > 0.0
        and gains["recall_at_precision_0_80"] >= 0.0
    )
    return {"passed": passed, "inventory_unchanged": inventory_unchanged, "gains": gains}


def augment_patches(
    patches: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    spatial_code = int(
        torch.randint(0, 32, (1,), generator=generator, device=patches.device).item()
    )
    result = torch.rot90(patches, spatial_code % 4, dims=(3, 4))
    if spatial_code & 4:
        result = torch.flip(result, dims=(2,))
    if spatial_code & 8:
        result = torch.flip(result, dims=(3,))
    if spatial_code & 16:
        result = torch.flip(result, dims=(4,))
    gain = 0.90 + 0.20 * torch.rand(
        (len(result), result.shape[1], 1, 1, 1),
        generator=generator,
        device=result.device,
    )
    noise = 0.02 * torch.randn(
        result.shape, generator=generator, device=result.device, dtype=result.dtype
    )
    return (result * gain + noise).clamp(-6.0, 6.0)


def balanced_rows(
    targets: torch.Tensor, batch_size: int, generator: torch.Generator
) -> torch.Tensor:
    positive = torch.nonzero(targets > 0.5, as_tuple=False).flatten()
    negative = torch.nonzero(targets < 0.5, as_tuple=False).flatten()
    positive_count = max(1, batch_size // 3)
    negative_count = batch_size - positive_count
    selected_positive = positive[
        torch.randint(
            0, len(positive), (positive_count,), generator=generator, device=targets.device
        )
    ]
    selected_negative = negative[
        torch.randint(
            0, len(negative), (negative_count,), generator=generator, device=targets.device
        )
    ]
    selected = torch.cat((selected_positive, selected_negative))
    return selected[
        torch.randperm(len(selected), generator=generator, device=targets.device)
    ]


def focal_division_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    base = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
    probability = torch.sigmoid(logits.float())
    target_probability = torch.where(targets > 0.5, probability, 1.0 - probability)
    return (((1.0 - target_probability) ** 2.0) * base).mean()


@torch.inference_mode()
def evaluate_classifier(
    model: MultiscaleContextualPairFusionAssociationModel,
    patches: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int,
) -> dict[str, float | int]:
    model.eval()
    logits = []
    for start in range(0, len(patches), batch_size):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits.append(division_logits(model, patches[start : start + batch_size]).float())
    return classification_metrics(torch.cat(logits), targets)


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
    data_root: Path,
    localization_root: Path,
    parent_root: Path,
    output_root: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.monotonic()
    seed = args.seed + SEED_OFFSETS[fold]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device).manual_seed(seed + 91_337)

    train_records = discover_contextual_shards(
        data_root / "train",
        expected_source="ZSNS004",
        expected_role="external_pretraining",
    )
    if len(train_records) != 64:
        raise RuntimeError("focused division training requires all 64 ZSNS004 shards")
    selection_records = discover_localization_shards(
        localization_root / "selection",
        expected_source="ZSNS005",
        expected_role="external_validation",
        expected_timepoints=SELECTION_TIMEPOINTS,
    )
    train_patches, train_targets, train_cache_bytes = load_division_inventory(
        train_records, device
    )
    selection_patches, selection_targets, selection_cache_bytes = load_division_inventory(
        selection_records, device
    )

    model = MultiscaleContextualPairFusionAssociationModel()
    initialization = load_rejected_zero_residual_parent(model, parent_root, fold)
    model.to(device)
    parameter_inventory = configure_focused_parameters(model)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda")
    initial_selection = evaluate_classifier(
        ema,
        selection_patches,
        selection_targets,
        batch_size=args.validation_batch_size,
    )
    best_step = 0
    best_selection = initial_selection
    best_state: dict[str, torch.Tensor] | None = None
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
            "steps": args.steps,
            "batch_size": args.batch_size,
            "selection_timepoints": list(SELECTION_TIMEPOINTS),
            "audit_timepoints": list(AUDIT_TIMEPOINTS),
            "audit_arrays_read": False,
            "initialization": initialization,
            **parameter_inventory,
            "competition_data_read": False,
            "public_code_copied": False,
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
        },
    )

    model.train()
    for step in range(1, args.steps + 1):
        rows = balanced_rows(train_targets, args.batch_size, generator)
        patches = augment_patches(train_patches[rows], generator)
        targets = train_targets[rows]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = division_logits(model, patches)
            loss = focal_division_loss(logits, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], 2.0
        )
        scaler.step(optimizer)
        scaler.update()
        update_ema(model, ema, args.ema_decay)
        progress = step / max(args.steps, 1)
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
            metrics = evaluate_classifier(
                ema,
                selection_patches,
                selection_targets,
                batch_size=args.validation_batch_size,
            )
            gate = focused_selection_gate(initial_selection, metrics)
            row = {"step": step, "metrics": metrics, "gate": gate}
            history.append(row)
            atomic_json(fold_root / "selection_latest.json", row)
            if gate["passed"] and float(metrics["average_precision"]) > float(
                best_selection["average_precision"]
            ):
                best_step = step
                best_selection = metrics
                best_state = state_dict_cpu(ema)
            model.train()

    atomic_json(fold_root / "selection_history.json", {"rows": history})
    selection_gate = focused_selection_gate(initial_selection, best_selection)
    terminal: dict[str, Any] = {
        "schema_version": 1,
        "status": "accepted_at_selection" if selection_gate["passed"] else "rejected_at_selection",
        "run_id": RUN_ID,
        "family": FAMILY,
        "appearance_family": MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        "fold": fold,
        "elapsed_seconds": time.monotonic() - started,
        "completed_step": args.steps,
        "best_step": best_step,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        **parameter_inventory,
        "initial_selection": initial_selection,
        "best_selection": best_selection,
        "selection_gate": selection_gate,
        "selection_gate_passed": bool(selection_gate["passed"]),
        "audit_opened": False,
        "checkpoint_frozen_before_audit": False,
        "train_rows": len(train_targets),
        "selection_rows": len(selection_targets),
        "train_cache_bytes": train_cache_bytes,
        "selection_cache_bytes": selection_cache_bytes,
        "initialization": initialization,
        "competition_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    if best_state is not None and selection_gate["passed"]:
        model.load_state_dict(best_state, strict=True)
        checkpoint = fold_root / "division_model.pt"
        torch.save(model.state_dict(), checkpoint)
        terminal.update(
            {
                "model_sha256": sha256_file(checkpoint),
                "checkpoint_frozen_before_audit": True,
            }
        )
    atomic_json(fold_root / "worker_terminal.json", terminal)
    del model, ema, optimizer, train_patches, train_targets, selection_patches, selection_targets
    torch.cuda.empty_cache()
    return terminal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--localization-root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=92_117)
    parser.add_argument("--steps", type=int, default=3_000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--validation-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--patch-batch-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("focused division training requires one Antelume GPU")
    counts = (
        args.steps,
        args.batch_size,
        args.validation_batch_size,
        args.validation_every,
        args.log_every,
        args.patch_batch_size,
    )
    if min(counts) <= 0:
        raise ValueError("focused division counts must be positive")
    args.output_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    device = torch.device("cuda:0")

    terminals = {
        fold: train_fold(
            fold=fold,
            data_root=args.data_root,
            localization_root=args.localization_root,
            parent_root=args.parent_root,
            output_root=args.output_root,
            device=device,
            args=args,
        )
        for fold in FOLDS
    }
    both_selected = all(
        row.get("status") == "accepted_at_selection"
        and row.get("selection_gate_passed") is True
        and row.get("checkpoint_frozen_before_audit") is True
        for row in terminals.values()
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "rejected_at_selection",
        "run_id": RUN_ID,
        "family": FAMILY,
        "appearance_family": MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        "execution_gpu_count": 1,
        "execution_policy": "two independent folds sequentially on Antelume A10G",
        "folds": terminals,
        "both_folds_selected": both_selected,
        "audit_opened": False,
        "competition_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    if both_selected:
        models = []
        hashes: dict[str, str] = {}
        for fold in FOLDS:
            checkpoint = args.output_root / fold / "division_model.pt"
            model = MultiscaleContextualPairFusionAssociationModel().to(device)
            model.load_state_dict(
                torch.load(checkpoint, map_location=device, weights_only=True), strict=True
            )
            model.requires_grad_(False).eval()
            models.append(model)
            hashes[fold] = sha256_file(checkpoint)

        selection_records = discover_localization_shards(
            args.localization_root / "selection",
            expected_source="ZSNS005",
            expected_role="external_validation",
            expected_timepoints=SELECTION_TIMEPOINTS,
        )
        selection_rows = score_records(
            models,
            selection_records,
            device,
            patch_batch_size=args.patch_batch_size,
        )
        threshold, selection = select_threshold(selection_rows)
        selection_inventory = {
            (row["shard"], row["source_row"]) for row in selection_rows
        }
        del selection_rows
        audit_records = discover_localization_shards(
            args.localization_root / "audit",
            expected_source="ZSNS005",
            expected_role="external_validation",
            expected_timepoints=AUDIT_TIMEPOINTS,
        )
        audit_rows = score_records(
            models,
            audit_records,
            device,
            patch_batch_size=args.patch_batch_size,
        )
        audit_inventory = {(row["shard"], row["source_row"]) for row in audit_rows}
        if selection_inventory & audit_inventory:
            raise RuntimeError("focused division selection and audit rows overlap")
        audit = recovery_metrics(audit_rows, threshold)
        accepted = bool(
            int(audit["division_tp"]) > 0
            and float(audit["edge_precision"]) >= 0.75
            and float(audit["recovery_composite"]) > 0.0
        )
        policy = {
            "schema_version": 1,
            "status": "accepted" if accepted else "rejected",
            "run_id": POLICY_RUN_ID,
            "model_training_run_id": RUN_ID,
            "appearance_family": MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
            "focused_division_family": FAMILY,
            "model_sha256": hashes,
            "ensemble": "mean of two independently trained focused division folds",
            "selection_policy": (
                "maximize external recovery composite with >=0.80 second-edge "
                "precision; tie-break by division Jaccard, fewer decisions, threshold"
            ),
            "frozen_division_logit_threshold": threshold,
            "selection": selection,
            "audit": audit,
            "selection_timepoints": list(SELECTION_TIMEPOINTS),
            "audit_timepoints": list(AUDIT_TIMEPOINTS),
            "audit_opened_after_threshold_freeze": True,
            "competition_data_read": False,
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
            "authorized_for_competition_graph_evaluation": accepted,
            "authorized_for_submission": False,
        }
        atomic_json(args.output_root / "division-recovery-policy.json", policy)
        result.update(
            {
                "status": "accepted" if accepted else "rejected_at_audit",
                "audit_opened": True,
                "audit_opened_after_threshold_freeze": True,
                "frozen_division_logit_threshold": threshold,
                "selection": selection,
                "audit": audit,
                "model_sha256": hashes,
                "authorized_for_competition_graph_evaluation": accepted,
                "authorized_for_submission": False,
            }
        )
        for fold in FOLDS:
            terminals[fold]["audit_opened_by_aggregate_after_both_checkpoints_frozen"] = True
    result["elapsed_seconds"] = time.monotonic() - started
    atomic_json(args.output_root / "focused_division_gate_terminal.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if result["status"] != "accepted":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
