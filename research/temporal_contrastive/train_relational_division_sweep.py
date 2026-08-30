#!/usr/bin/env python
"""Train independent heavy relational missing-daughter recovery models.

Optimization and selection are movie-disjoint. Audit shards are not loaded until
every member is frozen and at least one member passes the selection gate. The
four final-probe movies remain absent from the extracted dataset.
"""

from __future__ import annotations

import argparse
import copy
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

from research.temporal_contrastive.relational_division_model import (
    RELATIONAL_DIVISION_FAMILY,
    RelationalDivisionModel,
    architecture_contract,
    load_backbone_checkpoint,
)
from research.temporal_contrastive.train_real_division_gate import (
    atomic_json,
    average_precision,
    select_frozen_threshold,
    sha256_file,
    state_dict_cpu,
    threshold_metrics,
    update_ema,
)


RUN_ID = "competition-relational-division-sweep-v1"
DATA_RUN_ID = "competition-relational-division-patches-v3"
INVENTORY_SHA256 = "94150632f5a80b2ef48a39743a425cbe1b8e57b1c131c19ef0bde3d97d1c783e"
FINAL_PROBE_STEMS = {
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
}
EXPECTED_SUMMARY = {
    "rows": 3_013,
    "positives": 134,
    "hard_negatives": 2_879,
    "inference_eligible_positives": 55,
    "inference_eligible_hard_negatives": 165,
}
ROLES = ("optimization", "selection", "audit")
SELECTION_MINIMUM_AP = 0.55
EMBRYO_MINIMUM_AP = 0.40
MINIMUM_TRUE_POSITIVES_BEFORE_FIRST_FALSE_POSITIVE = 2


def validate_manifest(manifest: dict[str, Any]) -> None:
    summary = manifest.get("summary", {})
    records = manifest.get("records", [])
    observed = {
        key: summary.get(key)
        for key in EXPECTED_SUMMARY
    }
    record_summary = {
        key: sum(int(record.get(key, -10**9)) for record in records)
        for key in EXPECTED_SUMMARY
    }
    strata = {
        (embryo, role): [
            record
            for record in records
            if record.get("embryo") == embryo and record.get("role") == role
        ]
        for embryo in ("44b6", "6bba")
        for role in ROLES
    }
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("status") == "complete"
        and manifest.get("run_id") == DATA_RUN_ID
        and manifest.get("inventory_sha256") == INVENTORY_SHA256
        and observed == EXPECTED_SUMMARY
        and set(manifest.get("final_probe_stems", [])) == FINAL_PROBE_STEMS
        and manifest.get("final_probe_movies_extracted") is False
        and manifest.get("audit_features_extracted") is True
        and manifest.get("audit_labels_scored") is False
        and manifest.get("competition_train_data_read") is True
        and manifest.get("competition_test_data_read") is False
        and manifest.get("public_code_copied") is False
        and manifest.get("public_predictions_copied") is False
        and manifest.get("public_leaderboard_used_for_selection") is False
        and manifest.get("submission_created") is False
        and manifest.get("authorized_for_submission") is False
        and records
        and record_summary == EXPECTED_SUMMARY
        and len({record.get("stem") for record in records}) == 146
        and len({record.get("path") for record in records}) == len(records)
        and all(record.get("role") in ROLES for record in records)
        and all(
            rows
            and sum(int(row.get("positives", 0)) for row in rows) > 0
            and sum(int(row.get("hard_negatives", 0)) for row in rows) > 0
            and sum(int(row.get("inference_eligible_positives", 0)) for row in rows) > 0
            and sum(int(row.get("inference_eligible_hard_negatives", 0)) for row in rows) > 0
            for rows in strata.values()
        )
        and not ({record.get("stem") for record in records} & FINAL_PROBE_STEMS)
    ):
        raise ValueError("relational division patch manifest is ineligible")


def _verify_record(root: Path, record: dict[str, Any]) -> Path:
    path = root / str(record["path"])
    if not (
        path.is_file()
        and path.stat().st_size == int(record["bytes"])
        and sha256_file(path) == record["sha256"]
    ):
        raise ValueError(f"relational shard verification failed: {path}")
    return path


def load_role(
    root: Path, manifest: dict[str, Any], role: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    if role not in ROLES:
        raise ValueError(f"unsupported relational role: {role}")
    patches: list[torch.Tensor] = []
    geometry: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    eligible: list[torch.Tensor] = []
    inventory: list[dict[str, Any]] = []
    records = [record for record in manifest["records"] if record["role"] == role]
    if not records:
        raise ValueError(f"relational {role} inventory is empty")
    for record in records:
        path = _verify_record(root, record)
        with np.load(path, allow_pickle=False) as data:
            source = np.asarray(data["relational_patches"], dtype=np.float16)
            raw_geometry = np.asarray(data["geometry_features"], dtype=np.float32)
            target = np.asarray(data["division_recovery_target"], dtype=np.float32)
            weight = np.asarray(data["label_weight"], dtype=np.float32)
            inference_eligible = np.asarray(
                data["inference_geometry_eligible"], dtype=np.bool_
            )
            metadata = json.loads(str(data["metadata_json"].item()))
        row_count = len(source)
        if not (
            source.shape == (row_count, 3, 3, 17, 17, 17)
            and raw_geometry.shape == (row_count, 9)
            and target.shape == weight.shape == inference_eligible.shape == (row_count,)
            and np.isin(target, (0.0, 1.0)).all()
            and np.all(weight[target > 0.5] == 1.0)
            and np.all(weight[(target < 0.5) & inference_eligible] == 1.0)
            and np.all(weight[(target < 0.5) & ~inference_eligible] == 0.5)
            and metadata.get("run_id") == DATA_RUN_ID
            and metadata.get("stem") == record["stem"]
            and metadata.get("embryo") == record["embryo"]
            and metadata.get("role") == role
            and metadata.get("daughter_order_invariant") is True
            and metadata.get("audit_labels_scored") is False
            and metadata.get("competition_test_data_read") is False
        ):
            raise ValueError(f"relational shard contract changed: {path}")
        patches.append(torch.from_numpy(source))
        geometry.append(torch.from_numpy(raw_geometry))
        targets.append(torch.from_numpy(target))
        weights.append(torch.from_numpy(weight))
        eligible.append(torch.from_numpy(inference_eligible))
        inventory.extend(
            {
                "stem": record["stem"],
                "embryo": record["embryo"],
                "timepoint": int(record["timepoint"]),
                "source_row": index,
            }
            for index in range(row_count)
        )
    result = (
        torch.cat(patches),
        torch.cat(geometry),
        torch.cat(targets),
        torch.cat(weights),
        torch.cat(eligible),
        inventory,
    )
    labels = result[2]
    if not torch.any(labels > 0.5) or not torch.any(labels < 0.5):
        raise RuntimeError(f"relational {role} inventory lost a class")
    return result


def eligible_metrics(
    targets: torch.Tensor,
    scores: torch.Tensor,
    eligible: torch.Tensor,
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    indices = torch.nonzero(eligible, as_tuple=False).flatten()
    labels = targets[indices]
    values = scores[indices]
    pooled = threshold_metrics(labels, values)
    order = torch.argsort(values, descending=True, stable=True)
    ranked = labels[order] > 0.5
    first_false = torch.nonzero(~ranked, as_tuple=False).flatten()
    before_first_false = int(ranked[: int(first_false[0])].sum()) if len(first_false) else int(ranked.sum())
    by_embryo: dict[str, dict[str, Any]] = {}
    for embryo in ("44b6", "6bba"):
        embryo_rows = torch.as_tensor(
            [
                row_index
                for row_index in indices.tolist()
                if inventory[row_index]["embryo"] == embryo
            ],
            dtype=torch.long,
        )
        by_embryo[embryo] = threshold_metrics(
            targets[embryo_rows], scores[embryo_rows]
        )
    return {
        **pooled,
        "true_positives_before_first_false_positive": before_first_false,
        "by_embryo": by_embryo,
    }


def selection_utility(metrics: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(metrics["true_positives_before_first_false_positive"]),
        min(float(row["average_precision"]) for row in metrics["by_embryo"].values()),
        float(metrics["average_precision"]),
        float(metrics["best_jaccard"]),
        -float(metrics["binary_cross_entropy"]),
    )


def passes_selection_gate(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["average_precision"] >= SELECTION_MINIMUM_AP
        and metrics["true_positives_before_first_false_positive"]
        >= MINIMUM_TRUE_POSITIVES_BEFORE_FIRST_FALSE_POSITIVE
        and all(
            row["average_precision"] >= EMBRYO_MINIMUM_AP
            for row in metrics["by_embryo"].values()
        )
    )


def balanced_rows(
    targets: torch.Tensor,
    eligible: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if batch_size < 4:
        raise ValueError("relational batches must contain at least four examples")
    groups = {
        "positive_eligible": torch.nonzero((targets > 0.5) & eligible).flatten(),
        "positive_broad": torch.nonzero((targets > 0.5) & ~eligible).flatten(),
        "negative_eligible": torch.nonzero((targets < 0.5) & eligible).flatten(),
        "negative_broad": torch.nonzero((targets < 0.5) & ~eligible).flatten(),
    }
    if any(not len(rows) for rows in groups.values()):
        raise ValueError("relational balanced sampler lost an eligibility/class stratum")
    counts = [batch_size // 4] * 4
    for index in range(batch_size % 4):
        counts[index] += 1
    chosen = []
    for rows, count in zip(groups.values(), counts, strict=True):
        chosen.append(rows[torch.randint(len(rows), (count,), generator=generator)])
    combined = torch.cat(chosen)
    return combined[torch.randperm(len(combined), generator=generator)]


def augment_relational_batch(
    patches: torch.Tensor,
    geometry: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    augmented = patches.float().clone()
    raw_geometry = geometry.clone()
    for dimension in (-1, -2, -3):
        if torch.rand((), generator=generator) < 0.5:
            augmented = torch.flip(augmented, dims=(dimension,))
    batch = len(augmented)
    scale = 0.90 + 0.20 * torch.rand((batch, 1, 1, 1, 1, 1), generator=generator)
    offset = -0.05 + 0.10 * torch.rand((batch, 1, 1, 1, 1, 1), generator=generator)
    noise = 0.015 * torch.randn(augmented.shape, generator=generator)
    augmented = augmented * scale + offset + noise
    swap = torch.rand((batch,), generator=generator) < 0.5
    if torch.any(swap):
        original = augmented[swap].clone()
        augmented[swap, 1] = original[:, 2]
        augmented[swap, 2] = original[:, 1]
        first_distance = raw_geometry[swap, 0].clone()
        raw_geometry[swap, 0] = raw_geometry[swap, 2]
        raw_geometry[swap, 2] = first_distance
    return augmented, raw_geometry


def focal_loss(
    logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    logits = logits.float()
    targets = targets.float()
    base = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = torch.sigmoid(logits)
    correct_probability = torch.where(targets > 0.5, probability, 1.0 - probability)
    losses = (1.0 - correct_probability).square() * base * weights.float()
    return losses.sum() / weights.sum().clamp_min(1e-6)


@torch.inference_mode()
def predict(
    model: RelationalDivisionModel,
    patches: torch.Tensor,
    geometry: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    pieces = []
    for start in range(0, len(patches), batch_size):
        batch_patches = patches[start : start + batch_size].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        batch_geometry = geometry[start : start + batch_size].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pieces.append(model(batch_patches, batch_geometry).float().cpu())
    return torch.cat(pieces)


def train_member(
    *,
    member_name: str,
    seed: int,
    initial_model_path: Path,
    train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]],
    selection_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]],
    output_root: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    started = time.monotonic()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed + 91_337)
    train_patches, train_geometry, train_targets, train_weights, train_eligible, _ = train_data
    selection_patches, selection_geometry, selection_targets, _, selection_eligible, selection_inventory = selection_data
    model = RelationalDivisionModel().to(device)
    initial_state = torch.load(initial_model_path, map_location="cpu", weights_only=True)
    load_backbone_checkpoint(model, initial_state)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(
        (
            {"params": model.backbone.parameters(), "lr": args.learning_rate * args.backbone_lr_multiplier},
            {"params": model.relational_head.parameters(), "lr": args.learning_rate},
        ),
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda")
    best_metrics: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_step = 0
    history = []
    member_root = output_root / member_name
    member_root.mkdir(parents=True, exist_ok=False)
    for step in range(1, args.steps + 1):
        model.train()
        rows = balanced_rows(train_targets, train_eligible, args.batch_size, generator)
        patch_batch, geometry_batch = augment_relational_batch(
            train_patches[rows], train_geometry[rows], generator
        )
        patch_batch = patch_batch.to(device=device, non_blocking=True)
        geometry_batch = geometry_batch.to(device=device, non_blocking=True)
        target_batch = train_targets[rows].to(device=device, non_blocking=True)
        weight_batch = train_weights[rows].to(device=device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(patch_batch, geometry_batch)
            loss = focal_loss(logits, target_batch, weight_batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        update_ema(model, ema, args.ema_decay)
        progress = step / args.steps
        head_lr = args.minimum_learning_rate + 0.5 * (
            args.learning_rate - args.minimum_learning_rate
        ) * (1.0 + math.cos(math.pi * progress))
        optimizer.param_groups[0]["lr"] = head_lr * args.backbone_lr_multiplier
        optimizer.param_groups[1]["lr"] = head_lr
        if step == 1 or step % args.log_every == 0:
            print(json.dumps({"member": member_name, "step": step, "loss": float(loss.detach().cpu()), "head_learning_rate": head_lr}, sort_keys=True), flush=True)
        if step % args.validation_every == 0 or step == args.steps:
            scores = predict(ema, selection_patches, selection_geometry, batch_size=args.validation_batch_size, device=device)
            metrics = eligible_metrics(selection_targets, scores, selection_eligible, selection_inventory)
            improved = best_metrics is None or selection_utility(metrics) > selection_utility(best_metrics)
            history.append({"step": step, "metrics": metrics, "improved": improved})
            atomic_json(member_root / "selection_latest.json", history[-1])
            if improved:
                best_metrics = metrics
                best_state = state_dict_cpu(ema)
                best_step = step
    if best_metrics is None or best_state is None:
        raise RuntimeError("relational training did not produce a selection checkpoint")
    model.load_state_dict(best_state, strict=True)
    checkpoint = member_root / "relational_model.pt"
    torch.save(model.state_dict(), checkpoint)
    selection_scores = predict(model, selection_patches, selection_geometry, batch_size=args.validation_batch_size, device=device)
    try:
        frozen = select_frozen_threshold(
            selection_targets[selection_eligible], selection_scores[selection_eligible]
        )
    except RuntimeError:
        frozen = None
    passed = bool(
        passes_selection_gate(best_metrics)
        and frozen is not None
        and frozen["fp"] == 0
        and frozen["tp"] >= 2
    )
    terminal = {
        "schema_version": 1,
        "status": "accepted_at_selection" if passed else "rejected_at_selection",
        "run_id": RUN_ID,
        "family": RELATIONAL_DIVISION_FAMILY,
        "member": member_name,
        "seed": seed,
        "elapsed_seconds": time.monotonic() - started,
        "completed_steps": args.steps,
        "best_step": best_step,
        "selection": best_metrics,
        "selection_frozen_threshold": frozen,
        "selection_gate_passed": passed,
        "model_sha256": sha256_file(checkpoint),
        "initial_backbone_sha256": sha256_file(initial_model_path),
        **architecture_contract(),
        "audit_opened": False,
        "final_probe_opened": False,
        "competition_test_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_audit": passed,
        "authorized_for_submission": False,
    }
    atomic_json(member_root / "selection_history.json", {"rows": history})
    atomic_json(member_root / "worker_terminal.json", terminal)
    del model, ema, optimizer, best_state
    torch.cuda.empty_cache()
    return terminal


def threshold_decisions(labels: torch.Tensor, scores: torch.Tensor, threshold: float) -> dict[str, Any]:
    labels = labels.float() > 0.5
    selected = scores >= float(threshold)
    tp = int((selected & labels).sum())
    fp = int((selected & ~labels).sum())
    fn = int(labels.sum()) - tp
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(int(labels.sum()), 1),
        "jaccard": tp / max(tp + fp + fn, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--initial-model", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", default="211063,311071,411083,511091")
    parser.add_argument("--steps", type=int, default=15_000)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--validation-batch-size", type=int, default=24)
    parser.add_argument("--validation-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--minimum-learning-rate", type=float, default=4e-7)
    parser.add_argument("--backbone-lr-multiplier", type=float, default=0.20)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--gradient-clip", type=float, default=2.0)
    parser.add_argument("--required-gpu-name", default="A10G")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if len(args.initial_model) != 2 or len(seeds) != 4:
        raise ValueError("overnight relational sweep requires two initial models and four seeds")
    if len({sha256_file(path) for path in args.initial_model}) != 2:
        raise ValueError("relational sweep requires distinct initial backbones")
    if min(args.steps, args.batch_size, args.validation_batch_size, args.validation_every, args.log_every) <= 0:
        raise ValueError("relational training counts must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("relational training requires one Antelume GPU")
    gpu_name = torch.cuda.get_device_name(0)
    if args.required_gpu_name.lower() not in gpu_name.lower():
        raise RuntimeError(f"required Antelume GPU {args.required_gpu_name!r}, saw {gpu_name!r}")
    manifest_path = args.data_root / "relational_division_patch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    train_data = load_role(args.data_root, manifest, "optimization")
    selection_data = load_role(args.data_root, manifest, "selection")
    train_stems = {row["stem"] for row in train_data[-1]}
    selection_stems = {row["stem"] for row in selection_data[-1]}
    audit_stems = {record["stem"] for record in manifest["records"] if record["role"] == "audit"}
    if train_stems & selection_stems or (train_stems | selection_stems) & audit_stems:
        raise RuntimeError("relational optimization, selection, and audit movies overlap")
    args.output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    started = time.monotonic()
    terminals = []
    for seed in seeds:
        for initial_index, initial_model in enumerate(args.initial_model):
            name = f"seed-{seed}-init-{initial_index + 1}"
            terminals.append(train_member(member_name=name, seed=seed + 10_003 * initial_index, initial_model_path=initial_model, train_data=train_data, selection_data=selection_data, output_root=args.output_root, args=args, device=device))
    accepted = [row for row in terminals if row["selection_gate_passed"]]
    audit_results = []
    if accepted:
        audit_data = load_role(args.data_root, manifest, "audit")
        audit_patches, audit_geometry, audit_targets, _, audit_eligible, audit_inventory = audit_data
        for row in accepted:
            model = RelationalDivisionModel().to(device)
            checkpoint = args.output_root / row["member"] / "relational_model.pt"
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
            scores = predict(model, audit_patches, audit_geometry, batch_size=args.validation_batch_size, device=device)
            metrics = eligible_metrics(audit_targets, scores, audit_eligible, audit_inventory)
            decisions = threshold_decisions(audit_targets[audit_eligible], scores[audit_eligible], row["selection_frozen_threshold"]["threshold"])
            passed = bool(passes_selection_gate(metrics) and decisions["tp"] >= 2 and decisions["precision"] >= 0.80)
            result = {"member": row["member"], "metrics": metrics, "frozen_selection_threshold_decisions": decisions, "audit_gate_passed": passed, "model_sha256": row["model_sha256"]}
            audit_results.append(result)
            atomic_json(args.output_root / row["member"] / "audit_terminal.json", result)
            del model
            torch.cuda.empty_cache()
    independently_strong = [row for row in audit_results if row["audit_gate_passed"]]
    aggregate = {
        "schema_version": 1,
        "status": "completed",
        "run_id": RUN_ID,
        "family": RELATIONAL_DIVISION_FAMILY,
        "gpu_name": gpu_name,
        "gpu_count": 1,
        "execution_policy": "eight independent 48M models sequentially on Antelume A10G",
        "elapsed_seconds": time.monotonic() - started,
        "planned_model_count": 8,
        "completed_model_count": len(terminals),
        "steps_per_model": args.steps,
        "selection_accepted_members": [row["member"] for row in accepted],
        "audit_opened": bool(accepted),
        "audit_results": audit_results,
        "independently_strong_members": [row["member"] for row in independently_strong],
        "ensemble_eligible": len(independently_strong) >= 2,
        "final_probe_opened": False,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }
    atomic_json(args.output_root / "relational_division_sweep_terminal.json", aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)
    if not independently_strong:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
