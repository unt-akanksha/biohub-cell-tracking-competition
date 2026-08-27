#!/usr/bin/env python
"""Pretrain two independent contextual v3 folds on frozen ZebraHub shards.

ZSNS004 is the only optimization source and ZSNS005 is the only checkpoint
selection source.  The script requires exactly two GPUs, creates weights and
evidence only, and has no competition-data or submission path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import train_dual_fold_patch as base
    from contextual_pair_fusion import (
        CONTEXTUAL_PAIR_FUSION_FAMILY,
        EXPECTED_PARAMETER_COUNT,
        RECIPROCAL_PARENT_LOSS_WEIGHT,
        ContextualPairFusionAssociationModel,
        masked_reciprocal_parent_nll,
    )
    from model import masked_multi_positive_info_nce
    from pair_fusion import masked_multi_positive_pair_nll, pair_logit_metrics
    from verify_zebrahub_contextual_dataset import verify_dataset
except ModuleNotFoundError:
    from research.temporal_contrastive import train_dual_fold_patch as base
    from research.temporal_contrastive.contextual_pair_fusion import (
        CONTEXTUAL_PAIR_FUSION_FAMILY,
        EXPECTED_PARAMETER_COUNT,
        RECIPROCAL_PARENT_LOSS_WEIGHT,
        ContextualPairFusionAssociationModel,
        masked_reciprocal_parent_nll,
    )
    from research.temporal_contrastive.model import (
        masked_multi_positive_info_nce,
    )
    from research.temporal_contrastive.pair_fusion import (
        masked_multi_positive_pair_nll,
        pair_logit_metrics,
    )
    from research.temporal_contrastive.verify_zebrahub_contextual_dataset import (
        verify_dataset,
    )


RUN_ID = "zebrahub-contextual-pretrain-v1"
TRAIN_SOURCE = "ZSNS004"
VALIDATION_SOURCE = "ZSNS005"
FOLDS = ("target_44b6", "target_6bba")
SEED_OFFSETS = {"target_44b6": 0, "target_6bba": 10_003}
AUGMENTATION_MODES = ("microscopy_v1", "none")
AUGMENTATION_POLICY = (
    "fixed-physical-scale XY quarter-rotations, independent axis flips, "
    "per-patch/channel gain, and mild Gaussian noise; validation unaugmented"
)
VALIDATION_SELECTION_TIMEPOINTS = frozenset(
    (*range(96, 100), *range(376, 380))
)
VALIDATION_AUDIT_TIMEPOINTS = frozenset(
    (*range(236, 240), *range(516, 520))
)
VALIDATION_PARTITION_POLICY = (
    "ZSNS005 disjoint developmental windows: t0096-0099/t0376-0379 "
    "checkpoint selection; t0236-0239/t0516-0519 one-shot audit"
)
MINIMUM_VALIDATION_COMPOSITE_GAIN = 0.01


@dataclass(frozen=True)
class ShardRecord:
    path: Path
    manifest_path: Path
    sha256: str
    manifest_sha256: str
    csv_timepoint: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_shards(
    root: Path,
    *,
    expected_source: str,
    expected_role: str,
) -> list[ShardRecord]:
    records: list[ShardRecord] = []
    for path in sorted(root.glob("*.npz")):
        manifest_path = path.with_suffix(".manifest.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing ZebraHub shard manifest: {path.name}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shard_hash = sha256_file(path)
        valid = bool(
            manifest.get("schema_version") == 1
            and manifest.get("source") == expected_source
            and manifest.get("source_role") == expected_role
            and manifest.get("organizer_declared_test_overlap") is False
            and manifest.get("competition_test_data_read") is False
            and manifest.get("public_competition_predictions_read") is False
            and manifest.get("leaderboard_used") is False
            and manifest.get("submission_created") is False
            and manifest.get("candidate_context_width") == 18
            and manifest.get("shard", {}).get("path") == path.name
            and manifest.get("shard", {}).get("bytes") == path.stat().st_size
            and manifest.get("shard", {}).get("sha256") == shard_hash
        )
        if not valid:
            raise ValueError(f"invalid ZebraHub shard evidence: {path.name}")
        records.append(
            ShardRecord(
                path=path,
                manifest_path=manifest_path,
                sha256=shard_hash,
                manifest_sha256=sha256_file(manifest_path),
                csv_timepoint=int(manifest["csv_timepoint"]),
            )
        )
    if not records:
        raise ValueError(f"no verified {expected_source} shards below {root}")
    if len({row.csv_timepoint for row in records}) != len(records):
        raise ValueError(f"duplicate {expected_source} timepoints entered the dataset")
    return records


def inventory_sha256(records: list[ShardRecord]) -> str:
    payload = [
        {
            "path": row.path.name,
            "sha256": row.sha256,
            "manifest_sha256": row.manifest_sha256,
            "csv_timepoint": row.csv_timepoint,
        }
        for row in records
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def partition_validation_records(
    records: list[ShardRecord],
) -> tuple[list[ShardRecord], list[ShardRecord]]:
    """Create a fixed, non-overlapping ZSNS005 selection/audit partition."""

    by_timepoint = {row.csv_timepoint: row for row in records}
    expected = VALIDATION_SELECTION_TIMEPOINTS | VALIDATION_AUDIT_TIMEPOINTS
    if set(by_timepoint) != expected or len(by_timepoint) != len(records):
        raise ValueError("ZSNS005 validation timepoint inventory changed")
    selection = [by_timepoint[value] for value in sorted(VALIDATION_SELECTION_TIMEPOINTS)]
    audit = [by_timepoint[value] for value in sorted(VALIDATION_AUDIT_TIMEPOINTS)]
    if {row.sha256 for row in selection} & {row.sha256 for row in audit}:
        raise RuntimeError("ZSNS005 selection and audit shards overlap")
    return selection, audit


def validation_improvement_gate(
    initial: dict[str, float | int],
    candidate: dict[str, float | int],
    *,
    minimum_composite_gain: float = MINIMUM_VALIDATION_COMPOSITE_GAIN,
) -> dict[str, object]:
    """Require broad association improvement, not a noisy composite uptick."""

    if minimum_composite_gain <= 0:
        raise ValueError("minimum validation composite gain must be positive")
    metric_names = ("composite", "top1", "mrr", "division_top2")
    gains = {
        name: float(candidate[name]) - float(initial[name]) for name in metric_names
    }
    inventory_unchanged = bool(
        int(candidate["rows"]) == int(initial["rows"])
        and int(candidate["division_rows"]) == int(initial["division_rows"])
        and int(candidate["transitions"]) == int(initial["transitions"])
    )
    passed = bool(
        inventory_unchanged
        and gains["composite"] >= float(minimum_composite_gain)
        and gains["top1"] > 0.0
        and gains["mrr"] > 0.0
        and gains["division_top2"] >= 0.0
    )
    return {
        "passed": passed,
        "minimum_composite_gain": float(minimum_composite_gain),
        "inventory_unchanged": inventory_unchanged,
        "gains": gains,
    }


def load_shard(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    with np.load(path) as data:
        result = {
            "source_patches": torch.as_tensor(
                data["source_patches"].astype(np.float32), device=device
            ),
            "target_patches": torch.as_tensor(
                data["target_patches"].astype(np.float32), device=device
            ),
            "source_coords_um": torch.as_tensor(
                data["source_coords_um"], dtype=torch.float32, device=device
            ),
            "target_coords_um": torch.as_tensor(
                data["target_coords_um"], dtype=torch.float32, device=device
            ),
            "candidate_mask": torch.as_tensor(
                data["candidate_mask"], dtype=torch.bool, device=device
            ),
            "positive_mask": torch.as_tensor(
                data["positive_mask"], dtype=torch.bool, device=device
            ),
            "division_target": torch.as_tensor(
                data["division_target"], dtype=torch.float32, device=device
            ),
            "candidate_context": torch.as_tensor(
                data["candidate_context"], dtype=torch.float32, device=device
            ),
        }
    source_count = len(result["source_patches"])
    target_count = len(result["target_patches"])
    if result["candidate_mask"].shape != (source_count, target_count):
        raise ValueError(f"shard candidate inventory changed: {path.name}")
    if torch.any(result["positive_mask"] & ~result["candidate_mask"]):
        raise ValueError(f"shard positives escaped candidates: {path.name}")
    if result["candidate_context"].shape != (source_count, target_count, 18):
        raise ValueError(f"shard context width changed: {path.name}")
    return result


def preload_shards(
    records: list[ShardRecord], device: torch.device
) -> tuple[dict[Path, dict[str, torch.Tensor]], int]:
    """Materialize each verified immutable shard once on the worker device."""

    cache: dict[Path, dict[str, torch.Tensor]] = {}
    tensor_bytes = 0
    for record in records:
        batch = load_shard(record.path, device)
        cache[record.path] = batch
        tensor_bytes += sum(
            value.numel() * value.element_size() for value in batch.values()
        )
    if len(cache) != len(records):
        raise RuntimeError("preloaded shard cache changed the verified inventory")
    return cache, tensor_bytes


def cached_batch(
    cache: dict[Path, dict[str, torch.Tensor]], path: Path
) -> dict[str, torch.Tensor]:
    """Return an assignment-isolated view over immutable cached tensors."""

    if path not in cache:
        raise KeyError(f"shard is absent from the device cache: {path.name}")
    return dict(cache[path])


def augment_normalized_patches(
    patches: torch.Tensor,
    *,
    generator: torch.Generator,
    spatial_code: int,
) -> torch.Tensor:
    """Apply microscopy-safe invariances without changing physical scale."""

    if patches.ndim != 5 or patches.shape[1:] != (3, 17, 17, 17):
        raise ValueError("normalized patches must have shape (N, 3, 17, 17, 17)")
    if not torch.is_floating_point(patches):
        raise ValueError("normalized patches must be floating point")
    if not 0 <= int(spatial_code) < 32:
        raise ValueError("spatial augmentation code must be in [0, 32)")
    if len(patches) == 0:
        return patches.clone()
    result = torch.rot90(
        patches,
        int(spatial_code) & 3,
        dims=(-2, -1),
    )
    flip_axes = [
        axis
        for bit, axis in enumerate((-3, -2, -1), start=2)
        if int(spatial_code) & (1 << bit)
    ]
    if flip_axes:
        result = torch.flip(result, dims=flip_axes)
    gain = 0.90 + 0.20 * torch.rand(
        (len(result), result.shape[1], 1, 1, 1),
        generator=generator,
        device=result.device,
        dtype=result.dtype,
    )
    noise_scale = 0.01 + 0.04 * torch.rand(
        (len(result), result.shape[1], 1, 1, 1),
        generator=generator,
        device=result.device,
        dtype=result.dtype,
    )
    noise = torch.randn(
        result.shape,
        generator=generator,
        device=result.device,
        dtype=result.dtype,
    )
    return (result * gain + noise * noise_scale).clamp(-6.0, 6.0)


def encode_patches(
    model: ContextualPairFusionAssociationModel,
    patches: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings: list[torch.Tensor] = []
    divisions: list[torch.Tensor] = []
    for start in range(0, len(patches), batch_size):
        embedding, division = model(patches[start : start + batch_size])
        embeddings.append(embedding)
        divisions.append(division)
    return torch.cat(embeddings), torch.cat(divisions)


def shard_forward(
    model: ContextualPairFusionAssociationModel,
    batch: dict[str, torch.Tensor],
    *,
    patch_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source, divisions = encode_patches(
        model, batch["source_patches"], batch_size=patch_batch_size
    )
    target, _ = encode_patches(
        model, batch["target_patches"], batch_size=patch_batch_size
    )
    logits = model.candidate_pair_logits(
        source,
        target,
        batch["source_coords_um"],
        batch["target_coords_um"],
        divisions,
        batch["candidate_mask"],
        batch["candidate_context"],
    )
    pair_loss = masked_multi_positive_pair_nll(
        logits, batch["positive_mask"], batch["candidate_mask"]
    )
    reciprocal_parent_loss = masked_reciprocal_parent_nll(
        logits, batch["positive_mask"], batch["candidate_mask"]
    )
    embedding_loss = masked_multi_positive_info_nce(
        source,
        target,
        batch["positive_mask"],
        batch["candidate_mask"],
        temperature=0.10,
    )
    division_loss = F.binary_cross_entropy_with_logits(
        divisions.float(), batch["division_target"]
    )
    return logits, pair_loss, reciprocal_parent_loss, embedding_loss, division_loss


@torch.inference_mode()
def validate(
    model: ContextualPairFusionAssociationModel,
    records: list[ShardRecord],
    device: torch.device,
    *,
    patch_batch_size: int,
    cache: dict[Path, dict[str, torch.Tensor]] | None = None,
) -> dict[str, float | int]:
    model.eval()
    rows = []
    for record in records:
        batch = (
            cached_batch(cache, record.path)
            if cache is not None
            else load_shard(record.path, device)
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits, _pair, _parent, _embedding, _division = shard_forward(
                model, batch, patch_batch_size=patch_batch_size
            )
        rows.append(
            pair_logit_metrics(
                logits, batch["candidate_mask"], batch["positive_mask"]
            )
        )
    model.train()
    return base.aggregate_metrics(rows)


def train_worker(args: argparse.Namespace) -> None:
    started = time.monotonic()
    if args.fold not in FOLDS:
        raise ValueError(f"unknown pretraining fold: {args.fold}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"isolated ZebraHub worker requires one GPU, saw {torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    seed = args.seed + SEED_OFFSETS[args.fold]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    augmentation_generator = torch.Generator(device=device)
    augmentation_generator.manual_seed(seed + 77_777)
    dataset_verification = verify_dataset(args.data_root)
    train_records = discover_shards(
        args.data_root / "train",
        expected_source=TRAIN_SOURCE,
        expected_role="external_pretraining",
    )
    validation_records = discover_shards(
        args.data_root / "validation",
        expected_source=VALIDATION_SOURCE,
        expected_role="external_validation",
    )
    if len(train_records) < args.minimum_train_shards:
        raise RuntimeError("ZSNS004 pretraining coverage is incomplete")
    if len(validation_records) < args.minimum_validation_shards:
        raise RuntimeError("ZSNS005 validation coverage is incomplete")
    if {row.sha256 for row in train_records} & {
        row.sha256 for row in validation_records
    }:
        raise RuntimeError("external train and validation shards overlap")
    selection_records, audit_records = partition_validation_records(
        validation_records
    )

    train_cache, train_cache_bytes = preload_shards(train_records, device)
    validation_cache, validation_cache_bytes = preload_shards(
        validation_records, device
    )

    output_dir = args.output_dir / args.fold
    output_dir.mkdir(parents=True, exist_ok=True)
    model = ContextualPairFusionAssociationModel().to(device)
    ema_model = ContextualPairFusionAssociationModel().to(device)
    ema_model.load_state_dict(model.state_dict(), strict=True)
    ema_model.requires_grad_(False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"unexpected contextual parameter count: {parameter_count}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda")
    initial_selection = validate(
        ema_model,
        selection_records,
        device,
        patch_batch_size=args.patch_batch_size,
        cache=validation_cache,
    )
    initial_audit = validate(
        ema_model,
        audit_records,
        device,
        patch_batch_size=args.patch_batch_size,
        cache=validation_cache,
    )
    best_score = float(initial_selection["composite"])
    best_step = 0
    best_metrics = initial_selection
    best_state = base.state_dict_cpu(ema_model)
    history = [
        {
            "step": 0,
            "metrics": initial_selection,
            "score": best_score,
            "selection_gate": validation_improvement_gate(
                initial_selection, initial_selection
            ),
        }
    ]
    base.atomic_json(
        output_dir / "training_config.json",
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "appearance_family": CONTEXTUAL_PAIR_FUSION_FAMILY,
            "fold": args.fold,
            "seed": seed,
            "parameter_count": parameter_count,
            "external_training_source": TRAIN_SOURCE,
            "external_validation_source": VALIDATION_SOURCE,
            "train_shards": len(train_records),
            "validation_shards": len(validation_records),
            "train_inventory_sha256": inventory_sha256(train_records),
            "validation_inventory_sha256": inventory_sha256(validation_records),
            "dataset_manifest_sha256": dataset_verification["manifest_sha256"],
            "selection_policy": (
                "ZSNS005 fixed selection windows; composite ranking among "
                "checkpoints that improve top1, MRR, and division recall"
            ),
            "validation_partition_policy": VALIDATION_PARTITION_POLICY,
            "selection_shards": len(selection_records),
            "audit_shards": len(audit_records),
            "selection_inventory_sha256": inventory_sha256(selection_records),
            "audit_inventory_sha256": inventory_sha256(audit_records),
            "minimum_validation_composite_gain": (
                MINIMUM_VALIDATION_COMPOSITE_GAIN
            ),
            "pair_loss_policy": (
                "outgoing all-positive child ranking plus eligible incoming "
                "one-parent ranking"
            ),
            "reciprocal_parent_loss_weight": RECIPROCAL_PARENT_LOSS_WEIGHT,
            "shard_cache_policy": "verified immutable tensors preloaded once per GPU",
            "train_cache_bytes": train_cache_bytes,
            "validation_cache_bytes": validation_cache_bytes,
            "validation_precision": "CUDA float16 autocast",
            "augmentation_mode": args.augmentation_mode,
            "augmentation_policy": (
                AUGMENTATION_POLICY if args.augmentation_mode == "microscopy_v1" else "none"
            ),
            "validation_augmentation": "none",
            "competition_data_read": False,
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
        },
    )

    optimizer.zero_grad(set_to_none=True)
    completed_step = 0
    optimizer_steps = 0
    accumulated = 0
    while completed_step < args.steps:
        if time.monotonic() - started >= args.max_wall_seconds - args.finalization_reserve_seconds:
            break
        completed_step += 1
        record = train_records[int(rng.integers(0, len(train_records)))]
        batch = cached_batch(train_cache, record.path)
        if args.augmentation_mode == "microscopy_v1":
            batch["source_patches"] = augment_normalized_patches(
                batch["source_patches"],
                generator=augmentation_generator,
                spatial_code=int(rng.integers(0, 32)),
            )
            batch["target_patches"] = augment_normalized_patches(
                batch["target_patches"],
                generator=augmentation_generator,
                spatial_code=int(rng.integers(0, 32)),
            )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            (
                logits,
                pair_loss,
                reciprocal_parent_loss,
                embedding_loss,
                division_loss,
            ) = shard_forward(
                model, batch, patch_batch_size=args.patch_batch_size
            )
            loss = (
                pair_loss
                + RECIPROCAL_PARENT_LOSS_WEIGHT * reciprocal_parent_loss
                + 0.25 * embedding_loss
                + 0.20 * division_loss
            )
            scaled_loss = loss / args.gradient_accumulation
        scaler.scale(scaled_loss).backward()
        accumulated += 1
        if accumulated == args.gradient_accumulation:
            base.finish_optimizer_step(
                model,
                ema_model,
                optimizer,
                scaler,
                accumulated_batches=accumulated,
                gradient_accumulation=args.gradient_accumulation,
                ema_decay=args.ema_decay,
            )
            accumulated = 0
            optimizer_steps += 1
            progress = completed_step / max(args.steps, 1)
            learning_rate = args.minimum_learning_rate + 0.5 * (
                args.learning_rate - args.minimum_learning_rate
            ) * (1.0 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
        if completed_step % args.validation_every == 0:
            metrics = validate(
                ema_model,
                selection_records,
                device,
                patch_batch_size=args.patch_batch_size,
                cache=validation_cache,
            )
            score = float(metrics["composite"])
            selection_gate = validation_improvement_gate(
                initial_selection, metrics
            )
            row = {
                "step": completed_step,
                "metrics": metrics,
                "score": score,
                "selection_gate": selection_gate,
            }
            history.append(row)
            base.atomic_json(output_dir / "validation_latest.json", row)
            if bool(selection_gate["passed"]) and score > best_score:
                best_score = score
                best_step = completed_step
                best_metrics = metrics
                best_state = base.state_dict_cpu(ema_model)
    if accumulated:
        base.finish_optimizer_step(
            model,
            ema_model,
            optimizer,
            scaler,
            accumulated_batches=accumulated,
            gradient_accumulation=args.gradient_accumulation,
            ema_decay=args.ema_decay,
        )
        optimizer_steps += 1
    model.load_state_dict(best_state, strict=True)
    final_audit = validate(
        model,
        audit_records,
        device,
        patch_batch_size=args.patch_batch_size,
        cache=validation_cache,
    )
    selection_gate = validation_improvement_gate(
        initial_selection, best_metrics
    )
    audit_gate = validation_improvement_gate(initial_audit, final_audit)
    model_path = output_dir / "pretrained_model.pt"
    torch.save(model.state_dict(), model_path)
    base.atomic_json(output_dir / "validation_history.json", {"rows": history})
    terminal = {
        "schema_version": 1,
        "status": "completed",
        "run_id": RUN_ID,
        "appearance_family": CONTEXTUAL_PAIR_FUSION_FAMILY,
        "fold": args.fold,
        "elapsed_seconds": time.monotonic() - started,
        "completed_step": completed_step,
        "optimizer_steps": optimizer_steps,
        "best_step": best_step,
        "initial_validation": initial_selection,
        "best_validation": best_metrics,
        "initial_selection": initial_selection,
        "best_selection": best_metrics,
        "initial_audit": initial_audit,
        "final_audit": final_audit,
        "selection_gate": selection_gate,
        "audit_gate": audit_gate,
        "selection_gate_passed": bool(selection_gate["passed"]),
        "audit_gate_passed": bool(audit_gate["passed"]),
        "validation_partition_policy": VALIDATION_PARTITION_POLICY,
        "selection_inventory_sha256": inventory_sha256(selection_records),
        "audit_inventory_sha256": inventory_sha256(audit_records),
        "best_score": best_score,
        "model_sha256": base.sha256_file(model_path),
        "parameter_count": parameter_count,
        "external_training_source": TRAIN_SOURCE,
        "external_validation_source": VALIDATION_SOURCE,
        "augmentation_mode": args.augmentation_mode,
        "augmentation_policy": (
            AUGMENTATION_POLICY if args.augmentation_mode == "microscopy_v1" else "none"
        ),
        "validation_augmentation": "none",
        "pair_loss_policy": (
            "outgoing all-positive child ranking plus eligible incoming "
            "one-parent ranking"
        ),
        "reciprocal_parent_loss_weight": RECIPROCAL_PARENT_LOSS_WEIGHT,
        "shard_cache_policy": "verified immutable tensors preloaded once per GPU",
        "train_cache_bytes": train_cache_bytes,
        "validation_cache_bytes": validation_cache_bytes,
        "validation_precision": "CUDA float16 autocast",
        "dataset_manifest_sha256": dataset_verification["manifest_sha256"],
        "competition_data_read": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    base.atomic_json(output_dir / "worker_terminal.json", terminal)
    print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)


def orchestrate(args: argparse.Namespace) -> None:
    started = time.monotonic()
    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"ZebraHub contextual pretraining requires exactly two GPUs, saw {torch.cuda.device_count()}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for gpu_index, fold in enumerate(FOLDS):
        log_handle = (args.output_dir / f"{fold}.log").open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *[value for value in sys.argv[1:] if value != "--orchestrate"],
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
                raise TimeoutError("ZebraHub pretraining orchestrator exceeded hard stop")
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
        raise RuntimeError(f"ZebraHub pretraining workers failed: {failures}")
    terminals = {
        fold: json.loads(
            (args.output_dir / fold / "worker_terminal.json").read_text(
                encoding="utf-8"
            )
        )
        for fold in FOLDS
    }
    terminal = {
        "schema_version": 1,
        "status": "completed",
        "run_id": RUN_ID,
        "appearance_family": CONTEXTUAL_PAIR_FUSION_FAMILY,
        "elapsed_seconds": time.monotonic() - started,
        "gpu_count": 2,
        "folds": terminals,
        "both_folds_improved": all(
            int(row["best_step"]) > 0
            and row.get("selection_gate_passed") is True
            and row.get("audit_gate_passed") is True
            for row in terminals.values()
        ),
        "validation_partition_policy": VALIDATION_PARTITION_POLICY,
        "minimum_validation_composite_gain": MINIMUM_VALIDATION_COMPOSITE_GAIN,
        "augmentation_mode": args.augmentation_mode,
        "augmentation_policy": (
            AUGMENTATION_POLICY if args.augmentation_mode == "microscopy_v1" else "none"
        ),
        "validation_augmentation": "none",
        "competition_data_read": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    base.atomic_json(args.output_dir / "pretraining_terminal.json", terminal)
    print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    role = result.add_mutually_exclusive_group(required=True)
    role.add_argument("--orchestrate", action="store_true")
    role.add_argument("--worker", action="store_true")
    result.add_argument("--fold", choices=FOLDS)
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--seed", type=int, default=51_004)
    result.add_argument("--steps", type=int, default=12_000)
    result.add_argument("--minimum-train-shards", type=int, default=64)
    result.add_argument("--minimum-validation-shards", type=int, default=16)
    result.add_argument("--patch-batch-size", type=int, default=48)
    result.add_argument("--gradient-accumulation", type=int, default=2)
    result.add_argument("--validation-every", type=int, default=500)
    result.add_argument("--learning-rate", type=float, default=2e-4)
    result.add_argument("--minimum-learning-rate", type=float, default=2e-6)
    result.add_argument("--weight-decay", type=float, default=1e-5)
    result.add_argument("--ema-decay", type=float, default=0.997)
    result.add_argument(
        "--augmentation-mode", choices=AUGMENTATION_MODES, default="microscopy_v1"
    )
    result.add_argument("--max-wall-seconds", type=int, default=21_600)
    result.add_argument("--orchestrator-hard-stop-seconds", type=int, default=22_800)
    result.add_argument("--finalization-reserve-seconds", type=int, default=1_200)
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
