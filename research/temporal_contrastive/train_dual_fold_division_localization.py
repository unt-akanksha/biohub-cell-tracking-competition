#!/usr/bin/env python
"""Train two externally pretrained v4 localizers on two isolated GPUs."""

from __future__ import annotations

import argparse
import copy
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
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from multiscale_division_localization import (
        DIVISION_LOCALIZATION_FAMILY,
        EXPECTED_PARAMETER_COUNT,
        MultiscaleDivisionLocalizationModel,
        integer_jitter_crops,
        load_multiscale_v4_warm_start,
        localization_loss,
    )
except ModuleNotFoundError:
    from research.temporal_contrastive.multiscale_division_localization import (
        DIVISION_LOCALIZATION_FAMILY,
        EXPECTED_PARAMETER_COUNT,
        MultiscaleDivisionLocalizationModel,
        integer_jitter_crops,
        load_multiscale_v4_warm_start,
        localization_loss,
    )


RUN_ID = "multiscale-division-localization-v1"
PARENT_RUN_ID = "zebrahub-multiscale-contextual-pretrain-v1"
PARENT_FAMILY = "temporal_multiscale_contextual_pair_fusion_v4"
PARENT_PARAMETER_COUNT = 46_386_607
PARENT_AGGREGATE_NAME = "pretraining_terminal.json"
PARENT_MODEL_NAME = "pretrained_model.pt"
FOLDS = ("target_44b6", "target_6bba")
SEED_OFFSETS = {"target_44b6": 0, "target_6bba": 10_003}
SELECTION_TIMEPOINTS = (156, 157, 158, 159, 456, 457, 458, 459)
AUDIT_TIMEPOINTS = (276, 277, 278, 279, 556, 557, 558, 559)
EVALUATION_SHIFTS = tuple(
    (z, y, x)
    for z in range(-2, 3)
    for y in range(-2, 3)
    for x in range(-2, 3)
    if (z, y, x) != (0, 0, 0)
)


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


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def state_dict_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def discover_shards(
    root: Path,
    *,
    expected_source: str,
    expected_role: str,
    expected_timepoints: tuple[int, ...] | None = None,
) -> list[ShardRecord]:
    records: list[ShardRecord] = []
    for path in sorted(root.glob("*.npz")):
        manifest_path = path.with_suffix(".manifest.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing shard manifest: {path.name}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shard_hash = sha256_file(path)
        if not (
            manifest.get("schema_version") == 1
            and manifest.get("source") == expected_source
            and manifest.get("source_role") == expected_role
            and manifest.get("patch_shape") == [17, 17, 17]
            and manifest.get("patch_half_extent_um") == [8.0, 8.0, 8.0]
            and manifest.get("organizer_declared_test_overlap") is False
            and manifest.get("competition_test_data_read") is False
            and manifest.get("public_competition_predictions_read") is False
            and manifest.get("leaderboard_used") is False
            and manifest.get("submission_created") is False
            and manifest.get("shard", {}).get("path") == path.name
            and manifest.get("shard", {}).get("bytes") == path.stat().st_size
            and manifest.get("shard", {}).get("sha256") == shard_hash
        ):
            raise ValueError(f"invalid localization shard evidence: {path.name}")
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
        raise ValueError(f"no verified shards below {root}")
    timepoints = tuple(row.csv_timepoint for row in records)
    if len(set(timepoints)) != len(records):
        raise ValueError("shard timepoints repeat")
    if expected_timepoints is not None and timepoints != expected_timepoints:
        raise ValueError(
            f"localization timepoints changed: expected={expected_timepoints}, saw={timepoints}"
        )
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


def load_division_event_patches(path: Path, device: torch.device) -> torch.Tensor:
    with np.load(path) as data:
        required = {
            "source_patches",
            "target_patches",
            "positive_mask",
            "division_target",
        }
        if not required <= set(data.files):
            raise ValueError(f"localization shard arrays changed: {path.name}")
        sources = np.asarray(data["source_patches"], dtype=np.float32)
        targets = np.asarray(data["target_patches"], dtype=np.float32)
        positives = np.asarray(data["positive_mask"], dtype=bool)
        division = np.asarray(data["division_target"], dtype=np.float32) > 0.5
    parent_rows = np.flatnonzero(division)
    daughter_rows = np.flatnonzero(positives[parent_rows].any(axis=0))
    if not len(parent_rows) or len(daughter_rows) < 2:
        raise ValueError(f"localization shard has no complete division event: {path.name}")
    patches = np.concatenate((sources[parent_rows], targets[daughter_rows]), axis=0)
    if patches.ndim != 5 or patches.shape[1:] != (3, 17, 17, 17):
        raise ValueError(f"localization event patch shape changed: {path.name}")
    if not np.isfinite(patches).all():
        raise ValueError(f"localization event patches are nonfinite: {path.name}")
    return torch.as_tensor(patches, dtype=torch.float32, device=device)


def preload_event_patches(
    records: list[ShardRecord], device: torch.device
) -> tuple[dict[Path, torch.Tensor], int, int]:
    cache = {record.path: load_division_event_patches(record.path, device) for record in records}
    count = sum(len(value) for value in cache.values())
    tensor_bytes = sum(value.numel() * value.element_size() for value in cache.values())
    if count < len(records) * 3:
        raise RuntimeError("division localization cache lost event triplets")
    return cache, count, tensor_bytes


def load_v4_parent(
    model: MultiscaleDivisionLocalizationModel, root: Path, fold: str
) -> dict[str, Any]:
    aggregate_path = root / PARENT_AGGREGATE_NAME
    worker_path = root / fold / "worker_terminal.json"
    model_path = root / fold / PARENT_MODEL_NAME
    if not all(path.is_file() for path in (aggregate_path, worker_path, model_path)):
        raise FileNotFoundError(
            f"external v4 pretraining evidence is incomplete for {fold}"
        )
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    worker = json.loads(worker_path.read_text(encoding="utf-8"))
    model_hash = sha256_file(model_path)
    if not (
        aggregate.get("schema_version") == 1
        and aggregate.get("status") == "completed"
        and aggregate.get("run_id") == PARENT_RUN_ID
        and aggregate.get("appearance_family") == PARENT_FAMILY
        and aggregate.get("gpu_count") == 2
        and aggregate.get("both_folds_improved") is True
        and aggregate.get("competition_data_read") is False
        and aggregate.get("public_code_copied") is False
        and aggregate.get("public_predictions_copied") is False
        and aggregate.get("public_leaderboard_used_for_selection") is False
        and aggregate.get("submission_created") is False
        and worker == aggregate.get("folds", {}).get(fold)
        and worker.get("status") == "completed"
        and worker.get("run_id") == PARENT_RUN_ID
        and worker.get("appearance_family") == PARENT_FAMILY
        and worker.get("fold") == fold
        and worker.get("parameter_count") == PARENT_PARAMETER_COUNT
        and int(worker.get("best_step", 0)) > 0
        and worker.get("selection_gate_passed") is True
        and worker.get("audit_gate_passed") is True
        and worker.get("model_sha256") == model_hash
        and worker.get("competition_data_read") is False
        and worker.get("public_code_copied") is False
        and worker.get("public_predictions_copied") is False
        and worker.get("public_leaderboard_used_for_selection") is False
        and worker.get("submission_created") is False
    ):
        raise ValueError(f"external v4 pretraining evidence is ineligible for {fold}")
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    missing = load_multiscale_v4_warm_start(model, state)
    return {
        "run_id": PARENT_RUN_ID,
        "appearance_family": PARENT_FAMILY,
        "fold": fold,
        "model_sha256": model_hash,
        "worker_terminal_sha256": sha256_file(worker_path),
        "new_parameter_keys": len(missing),
        "parent_stage": "external_pretraining",
        "association_predictions_numerically_preserved": True,
    }


def evaluation_shift_tensor(device: torch.device) -> torch.Tensor:
    return torch.tensor(EVALUATION_SHIFTS, dtype=torch.int64, device=device)


def localization_gate(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    axis_nonregressive = all(
        float(right) <= float(left) + 1e-12
        for left, right in zip(
            baseline["axis_mae_um"], candidate["axis_mae_um"], strict=True
        )
    )
    passed = bool(
        int(candidate["examples"]) == int(baseline["examples"])
        and float(candidate["mean_residual_um"])
        < float(baseline["mean_residual_um"])
        and float(candidate["p90_residual_um"])
        < float(baseline["p90_residual_um"])
        and axis_nonregressive
    )
    return {
        "passed": passed,
        "inventory_unchanged": int(candidate["examples"]) == int(baseline["examples"]),
        "mean_residual_gain_um": float(baseline["mean_residual_um"])
        - float(candidate["mean_residual_um"]),
        "p90_residual_gain_um": float(baseline["p90_residual_um"])
        - float(candidate["p90_residual_um"]),
        "axis_mae_nonregressive": axis_nonregressive,
    }


@torch.inference_mode()
def validate_localization(
    model: MultiscaleDivisionLocalizationModel,
    cache: dict[Path, torch.Tensor],
    device: torch.device,
    *,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    patches = torch.cat(list(cache.values()), dim=0)
    shifts = evaluation_shift_tensor(device)
    shifts_per_patch = len(shifts)
    total = len(patches) * shifts_per_patch
    residual_chunks: list[torch.Tensor] = []
    for start in range(0, total, batch_size):
        indices = torch.arange(start, min(start + batch_size, total), device=device)
        patch_rows = torch.div(indices, shifts_per_patch, rounding_mode="floor")
        shift_rows = indices.remainder(shifts_per_patch)
        selected_shifts = shifts[shift_rows]
        crops, targets = integer_jitter_crops(patches[patch_rows], selected_shifts)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            predicted = model.localization_offsets_um(crops)
        residual_chunks.append((predicted.float() - targets.float()).cpu())
    residual = torch.cat(residual_chunks)
    distances = torch.linalg.vector_norm(residual, dim=1)
    return {
        "event_nodes": len(patches),
        "shifts_per_node": shifts_per_patch,
        "examples": len(residual),
        "mean_residual_um": float(distances.mean()),
        "p90_residual_um": float(torch.quantile(distances, 0.9)),
        "axis_mae_um": residual.abs().mean(dim=0).tolist(),
    }


def baseline_metrics(event_nodes: int) -> dict[str, Any]:
    shifts = torch.tensor(EVALUATION_SHIFTS, dtype=torch.float32)
    residual = shifts.repeat(event_nodes, 1)
    distances = torch.linalg.vector_norm(residual, dim=1)
    return {
        "event_nodes": event_nodes,
        "shifts_per_node": len(EVALUATION_SHIFTS),
        "examples": len(residual),
        "mean_residual_um": float(distances.mean()),
        "p90_residual_um": float(torch.quantile(distances, 0.9)),
        "axis_mae_um": residual.abs().mean(dim=0).tolist(),
    }


def update_ema(
    model: torch.nn.Module, ema: torch.nn.Module, *, decay: float
) -> None:
    with torch.no_grad():
        for ema_value, value in zip(
            ema.state_dict().values(), model.state_dict().values(), strict=True
        ):
            if ema_value.is_floating_point():
                ema_value.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                ema_value.copy_(value)


def train_worker(args: argparse.Namespace) -> None:
    started = time.monotonic()
    if args.fold not in FOLDS:
        raise ValueError(f"unknown localization fold: {args.fold}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"isolated localization worker requires one GPU, saw {torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    seed = args.seed + SEED_OFFSETS[args.fold]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device).manual_seed(seed + 91_337)
    rng = np.random.default_rng(seed)

    train_records = discover_shards(
        args.train_root,
        expected_source="ZSNS004",
        expected_role="external_pretraining",
    )
    if len(train_records) != 64:
        raise RuntimeError("localization optimization requires all 64 ZSNS004 shards")
    selection_records = discover_shards(
        args.localization_root / "selection",
        expected_source="ZSNS005",
        expected_role="external_validation",
        expected_timepoints=SELECTION_TIMEPOINTS,
    )
    if {row.sha256 for row in train_records} & {row.sha256 for row in selection_records}:
        raise RuntimeError("localization optimization and selection overlap")
    train_cache, train_nodes, train_cache_bytes = preload_event_patches(
        train_records, device
    )
    selection_cache, selection_nodes, selection_cache_bytes = preload_event_patches(
        selection_records, device
    )

    output_dir = args.output_dir / args.fold
    output_dir.mkdir(parents=True, exist_ok=True)
    model = MultiscaleDivisionLocalizationModel().to(device)
    initialization = load_v4_parent(model, args.v4_root, args.fold)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"unexpected localization parameter count: {parameter_count}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda")
    baseline_selection = baseline_metrics(selection_nodes)
    initial_selection = validate_localization(
        ema, selection_cache, device, batch_size=args.validation_batch_size
    )
    if initial_selection != baseline_selection:
        raise RuntimeError("zero localization adapter does not reproduce the control")
    history: list[dict[str, Any]] = []
    best_step = 0
    best_metrics = initial_selection
    best_state: dict[str, torch.Tensor] | None = None
    train_paths = tuple(train_cache)

    atomic_json(
        output_dir / "training_config.json",
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "family": DIVISION_LOCALIZATION_FAMILY,
            "fold": args.fold,
            "seed": seed,
            "parameter_count": parameter_count,
            "train_shards": len(train_records),
            "selection_shards": len(selection_records),
            "train_event_nodes": train_nodes,
            "selection_event_nodes": selection_nodes,
            "train_inventory_sha256": inventory_sha256(train_records),
            "selection_inventory_sha256": inventory_sha256(selection_records),
            "audit_inventory_read": False,
            "evaluation_shifts": [list(value) for value in EVALUATION_SHIFTS],
            "selection_policy": "minimum mean residual among checkpoints passing fixed mean, p90, and axis gates",
            "initialization": initialization,
            "competition_data_read": False,
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
        },
    )

    completed_step = 0
    model.train()
    for step in range(1, args.steps + 1):
        if time.monotonic() - started >= args.max_wall_seconds - args.finalization_reserve_seconds:
            break
        completed_step = step
        shard = train_cache[train_paths[int(rng.integers(0, len(train_paths)))]]
        indices = torch.randint(
            0, len(shard), (args.batch_size,), generator=generator, device=device
        )
        shifts = torch.randint(
            -2, 3, (args.batch_size, 3), generator=generator, device=device
        )
        zero_rows = torch.all(shifts == 0, dim=1)
        shifts[zero_rows, 0] = 1
        crops, targets = integer_jitter_crops(shard[indices], shifts)
        gain = 0.9 + 0.2 * torch.rand(
            (len(crops), 3, 1, 1, 1), generator=generator, device=device
        )
        noise = 0.02 * torch.randn(
            crops.shape, generator=generator, device=device, dtype=crops.dtype
        )
        crops = (crops * gain + noise).clamp(-6.0, 6.0)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            predicted = model.localization_offsets_um(crops)
            loss = localization_loss(predicted.float(), targets.float())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        update_ema(model, ema, decay=args.ema_decay)
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
                        "fold": args.fold,
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        "learning_rate": learning_rate,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if step % args.validation_every == 0 or step == args.steps:
            metrics = validate_localization(
                ema,
                selection_cache,
                device,
                batch_size=args.validation_batch_size,
            )
            gate = localization_gate(baseline_selection, metrics)
            row = {"step": step, "metrics": metrics, "gate": gate}
            history.append(row)
            atomic_json(output_dir / "selection_latest.json", row)
            if gate["passed"] and float(metrics["mean_residual_um"]) < float(
                best_metrics["mean_residual_um"]
            ):
                best_step = step
                best_metrics = metrics
                best_state = state_dict_cpu(ema)
            model.train()

    atomic_json(output_dir / "selection_history.json", {"rows": history})
    selection_gate = localization_gate(baseline_selection, best_metrics)
    terminal: dict[str, Any] = {
        "schema_version": 1,
        "status": "rejected_at_selection",
        "run_id": RUN_ID,
        "family": DIVISION_LOCALIZATION_FAMILY,
        "fold": args.fold,
        "elapsed_seconds": time.monotonic() - started,
        "completed_step": completed_step,
        "best_step": best_step,
        "parameter_count": parameter_count,
        "baseline_selection": baseline_selection,
        "best_selection": best_metrics,
        "selection_gate": selection_gate,
        "selection_gate_passed": bool(selection_gate["passed"]),
        "audit_opened": False,
        "checkpoint_frozen_before_audit": False,
        "train_inventory_sha256": inventory_sha256(train_records),
        "selection_inventory_sha256": inventory_sha256(selection_records),
        "train_event_nodes": train_nodes,
        "selection_event_nodes": selection_nodes,
        "train_cache_bytes": train_cache_bytes,
        "selection_cache_bytes": selection_cache_bytes,
        "initialization": initialization,
        "competition_data_read": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    if best_state is not None and selection_gate["passed"]:
        model.load_state_dict(best_state, strict=True)
        checkpoint = output_dir / "localization_model.pt"
        torch.save(model.state_dict(), checkpoint)
        checkpoint_hash = sha256_file(checkpoint)
        # Audit records and arrays are intentionally first addressed only after
        # the selected checkpoint has been serialized and hash-frozen.
        audit_records = discover_shards(
            args.localization_root / "audit",
            expected_source="ZSNS005",
            expected_role="external_validation",
            expected_timepoints=AUDIT_TIMEPOINTS,
        )
        audit_cache, audit_nodes, audit_cache_bytes = preload_event_patches(
            audit_records, device
        )
        baseline_audit = baseline_metrics(audit_nodes)
        final_audit = validate_localization(
            model, audit_cache, device, batch_size=args.validation_batch_size
        )
        audit_gate = localization_gate(baseline_audit, final_audit)
        terminal.update(
            {
                "status": "completed" if audit_gate["passed"] else "rejected_at_audit",
                "model_sha256": checkpoint_hash,
                "checkpoint_frozen_before_audit": True,
                "audit_opened": True,
                "audit_inventory_sha256": inventory_sha256(audit_records),
                "audit_event_nodes": audit_nodes,
                "audit_cache_bytes": audit_cache_bytes,
                "baseline_audit": baseline_audit,
                "final_audit": final_audit,
                "audit_gate": audit_gate,
                "audit_gate_passed": bool(audit_gate["passed"]),
            }
        )
    atomic_json(output_dir / "worker_terminal.json", terminal)
    print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)


def terminate_processes(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 20.0
    for process in processes:
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.2)
        if process.poll() is None:
            process.kill()
        process.wait()


def orchestrate(args: argparse.Namespace) -> None:
    started = time.monotonic()
    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"division localization requires exactly two GPUs, saw {torch.cuda.device_count()}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[str, subprocess.Popen[str], Any]] = []
    for gpu_index, fold in enumerate(FOLDS):
        handle = (args.output_dir / f"{fold}.log").open("w", encoding="utf-8")
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
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((fold, process, handle))
    return_codes: dict[str, int] = {}
    try:
        while len(return_codes) < len(processes):
            for fold, process, _handle in processes:
                code = process.poll()
                if code is not None and fold not in return_codes:
                    return_codes[fold] = int(code)
            if time.monotonic() - started >= args.orchestrator_hard_stop_seconds:
                raise TimeoutError("division localization orchestrator exceeded hard stop")
            if len(return_codes) < len(processes):
                time.sleep(5)
    finally:
        terminate_processes([process for _fold, process, _handle in processes])
        for _fold, _process, handle in processes:
            handle.close()
    failures = {fold: code for fold, code in return_codes.items() if code != 0}
    if failures:
        raise RuntimeError(f"division localization workers failed: {failures}")
    terminals = {
        fold: json.loads(
            (args.output_dir / fold / "worker_terminal.json").read_text(encoding="utf-8")
        )
        for fold in FOLDS
    }
    terminal = {
        "schema_version": 1,
        "status": "completed",
        "run_id": RUN_ID,
        "family": DIVISION_LOCALIZATION_FAMILY,
        "elapsed_seconds": time.monotonic() - started,
        "gpu_count": 2,
        "folds": terminals,
        "both_folds_passed": all(
            row.get("status") == "completed"
            and row.get("selection_gate_passed") is True
            and row.get("audit_gate_passed") is True
            for row in terminals.values()
        ),
        "competition_data_read": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    atomic_json(args.output_dir / "localization_terminal.json", terminal)
    print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    role = result.add_mutually_exclusive_group(required=True)
    role.add_argument("--orchestrate", action="store_true")
    role.add_argument("--worker", action="store_true")
    result.add_argument("--fold", choices=FOLDS)
    result.add_argument("--train-root", type=Path, required=True)
    result.add_argument("--localization-root", type=Path, required=True)
    result.add_argument("--v4-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--seed", type=int, default=81_004)
    result.add_argument("--steps", type=int, default=6_000)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--validation-batch-size", type=int, default=48)
    result.add_argument("--validation-every", type=int, default=500)
    result.add_argument("--log-every", type=int, default=100)
    result.add_argument("--learning-rate", type=float, default=5e-5)
    result.add_argument("--minimum-learning-rate", type=float, default=5e-7)
    result.add_argument("--weight-decay", type=float, default=1e-5)
    result.add_argument("--ema-decay", type=float, default=0.997)
    result.add_argument("--max-wall-seconds", type=int, default=21_600)
    result.add_argument("--orchestrator-hard-stop-seconds", type=int, default=22_800)
    result.add_argument("--finalization-reserve-seconds", type=int, default=1_200)
    return result


def main() -> None:
    args = parser().parse_args()
    positive = (
        args.steps,
        args.batch_size,
        args.validation_batch_size,
        args.validation_every,
        args.log_every,
        args.max_wall_seconds,
        args.orchestrator_hard_stop_seconds,
        args.finalization_reserve_seconds,
    )
    if min(positive) <= 0:
        raise ValueError("localization counts and time budgets must be positive")
    if args.worker:
        if args.fold is None:
            raise ValueError("--fold is required for a worker")
        train_worker(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
