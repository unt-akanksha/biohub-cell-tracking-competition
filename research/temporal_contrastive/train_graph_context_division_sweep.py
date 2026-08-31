#!/usr/bin/env python
"""Train independently gated 74.7M image-plus-graph-context rankers."""

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

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.temporal_contrastive.graph_context_division_model import (
    CONTEXT_FEATURE_WIDTH,
    CONTEXT_TOKEN_COUNT,
    GRAPH_CONTEXT_DIVISION_FAMILY,
    GraphContextDivisionModel,
    architecture_contract,
    load_backbone_checkpoint,
)
from research.temporal_contrastive.train_relational_division_sweep import (
    calibration_free_equal_rank_ensemble,
    eligible_metrics,
    focal_loss,
    selection_utility,
    threshold_decisions,
)
from research.temporal_contrastive.train_real_division_gate import (
    atomic_json,
    select_frozen_threshold,
    sha256_file,
    state_dict_cpu,
    update_ema,
)


RUN_ID = "competition-graph-context-division-sweep-v1"
DATA_RUN_ID = "competition-graph-context-relational-patches-v1"
DATA_SOURCE_ARCHIVE_SHA256 = "66a822bce0c60d06f6a2b60ada313f0d4d55062de1f84fb60bded4ae456266c2"
INVENTORY_SHA256 = "94150632f5a80b2ef48a39743a425cbe1b8e57b1c131c19ef0bde3d97d1c783e"
GEFF_MANIFEST_SHA256 = "744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9"
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

GraphContextData = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[dict[str, Any]],
]


def validate_manifest(manifest: dict[str, Any]) -> None:
    records = manifest.get("records", [])
    summary = manifest.get("summary", {})
    record_summary = {
        key: sum(int(record.get(key, -10**9)) for record in records)
        for key in EXPECTED_SUMMARY
    }
    strata = {
        (embryo, role): [
            row
            for row in records
            if row.get("embryo") == embryo and row.get("role") == role
        ]
        for embryo in ("44b6", "6bba")
        for role in ROLES
    }
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("status") == "complete"
        and manifest.get("run_id") == DATA_RUN_ID
        and manifest.get("source_archive_sha256") == DATA_SOURCE_ARCHIVE_SHA256
        and manifest.get("inventory_sha256") == INVENTORY_SHA256
        and manifest.get("source_geff_manifest_sha256") == GEFF_MANIFEST_SHA256
        and all(summary.get(key) == value for key, value in EXPECTED_SUMMARY.items())
        and record_summary == EXPECTED_SUMMARY
        and len(records) == 2_274
        and len({record.get("path") for record in records}) == len(records)
        and set(manifest.get("final_probe_stems", [])) == FINAL_PROBE_STEMS
        and manifest.get("context_token_count") == CONTEXT_TOKEN_COUNT
        and manifest.get("context_feature_width") == CONTEXT_FEATURE_WIDTH
        and manifest.get("context_edges_read") is False
        and manifest.get("context_labels_used") is False
        and manifest.get("daughter_order_invariant") is True
        and manifest.get("audit_labels_scored") is False
        and manifest.get("final_probe_movies_extracted") is False
        and manifest.get("competition_test_data_read") is False
        and manifest.get("public_code_copied") is False
        and manifest.get("public_predictions_copied") is False
        and manifest.get("public_leaderboard_used_for_selection") is False
        and manifest.get("authorized_for_submission") is False
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
        raise ValueError("graph-context division patch manifest is ineligible")


def _verified_shard(root: Path, record: dict[str, Any]) -> Path:
    path = root / str(record["path"])
    if not (
        path.is_file()
        and path.stat().st_size == int(record["bytes"])
        and sha256_file(path) == record["sha256"]
    ):
        raise ValueError(f"graph-context shard verification failed: {path}")
    return path


def load_role(root: Path, manifest: dict[str, Any], role: str) -> GraphContextData:
    if role not in ROLES:
        raise ValueError(f"unsupported graph-context role: {role}")
    pieces: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "patches",
            "geometry",
            "context",
            "mask",
            "targets",
            "weights",
            "eligible",
        )
    }
    inventory: list[dict[str, Any]] = []
    records = [record for record in manifest["records"] if record["role"] == role]
    if not records:
        raise ValueError(f"graph-context {role} inventory is empty")
    for record in records:
        path = _verified_shard(root, record)
        with np.load(path, allow_pickle=False) as data:
            arrays = {
                "patches": np.asarray(data["relational_patches"], dtype=np.float16),
                "geometry": np.asarray(data["geometry_features"], dtype=np.float32),
                "context": np.asarray(data["graph_context_features"], dtype=np.float16),
                "mask": np.asarray(data["graph_context_mask"], dtype=np.bool_),
                "targets": np.asarray(data["division_recovery_target"], dtype=np.float32),
                "weights": np.asarray(data["label_weight"], dtype=np.float32),
                "eligible": np.asarray(data["inference_geometry_eligible"], dtype=np.bool_),
            }
            metadata = json.loads(str(data["metadata_json"].item()))
        rows = len(arrays["targets"])
        if not (
            arrays["patches"].shape == (rows, 3, 3, 17, 17, 17)
            and arrays["geometry"].shape == (rows, 9)
            and arrays["context"].shape
            == (rows, CONTEXT_TOKEN_COUNT, CONTEXT_FEATURE_WIDTH)
            and arrays["mask"].shape == (rows, CONTEXT_TOKEN_COUNT)
            and arrays["targets"].shape
            == arrays["weights"].shape
            == arrays["eligible"].shape
            == (rows,)
            and np.isfinite(arrays["context"]).all()
            and np.all(arrays["mask"][:, :3])
            and np.isin(arrays["targets"], (0.0, 1.0)).all()
            and metadata.get("run_id") == DATA_RUN_ID
            and metadata.get("source_run_id") == "competition-relational-division-patches-v3"
            and metadata.get("stem") == record["stem"]
            and metadata.get("role") == role
            and metadata.get("context_edges_read") is False
            and metadata.get("context_labels_used") is False
            and metadata.get("audit_labels_scored") is False
        ):
            raise ValueError(f"graph-context shard contract changed: {path}")
        for key, value in arrays.items():
            pieces[key].append(torch.from_numpy(value))
        inventory.extend(
            {
                "stem": record["stem"],
                "embryo": record["embryo"],
                "timepoint": int(record["timepoint"]),
                "source_row": index,
            }
            for index in range(rows)
        )
    result = tuple(torch.cat(pieces[key]) for key in pieces)
    if not torch.any(result[4] > 0.5) or not torch.any(result[4] < 0.5):
        raise RuntimeError(f"graph-context {role} inventory lost a class")
    return (*result, inventory)  # type: ignore[return-value]


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


def load_completed_member(
    *,
    member_root: Path,
    member_name: str,
    seed: int,
    initial_model_path: Path,
    steps: int,
) -> dict[str, Any]:
    """Load a complete pre-audit worker without trusting partial run state."""

    terminal_path = member_root / "worker_terminal.json"
    checkpoint = member_root / "graph_context_model.pt"
    history_path = member_root / "selection_history.json"
    if not (terminal_path.is_file() and checkpoint.is_file() and history_path.is_file()):
        raise ValueError(f"resumed graph-context member is incomplete: {member_root}")
    if (member_root / "audit_terminal.json").exists():
        raise ValueError(f"resumed graph-context member already opened audit: {member_root}")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    architecture = architecture_contract()
    selection = terminal.get("selection")
    frozen = terminal.get("selection_frozen_threshold")
    passed = terminal.get("selection_gate_passed") is True
    expected_status = "accepted_at_selection" if passed else "rejected_at_selection"
    if not (
        terminal.get("schema_version") == 1
        and terminal.get("status") == expected_status
        and terminal.get("run_id") == RUN_ID
        and terminal.get("member") == member_name
        and terminal.get("seed") == seed
        and terminal.get("completed_steps") == steps
        and terminal.get("best_step") in {
            row.get("step") for row in history.get("rows", []) if isinstance(row, dict)
        }
        and terminal.get("model_sha256") == sha256_file(checkpoint)
        and terminal.get("initial_backbone_sha256") == sha256_file(initial_model_path)
        and all(terminal.get(key) == value for key, value in architecture.items())
        and terminal.get("audit_opened") is False
        and terminal.get("final_probe_opened") is False
        and terminal.get("competition_test_data_read") is False
        and terminal.get("public_code_copied") is False
        and terminal.get("public_predictions_copied") is False
        and terminal.get("public_leaderboard_used_for_selection") is False
        and terminal.get("submission_created") is False
        and terminal.get("authorized_for_audit") is passed
        and terminal.get("authorized_for_submission") is False
        and isinstance(selection, dict)
    ):
        raise ValueError(f"resumed graph-context member failed integrity checks: {member_root}")
    selection_passed = passes_selection_gate(selection)
    frozen_passed = bool(
        isinstance(frozen, dict) and frozen.get("fp") == 0 and frozen.get("tp", 0) >= 2
    )
    if passed != (selection_passed and frozen_passed):
        raise ValueError(f"resumed graph-context gate changed: {member_root}")
    return terminal


def balanced_rows(
    targets: torch.Tensor,
    eligible: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    groups = (
        torch.nonzero((targets > 0.5) & eligible).flatten(),
        torch.nonzero((targets > 0.5) & ~eligible).flatten(),
        torch.nonzero((targets < 0.5) & eligible).flatten(),
        torch.nonzero((targets < 0.5) & ~eligible).flatten(),
    )
    if batch_size < 4 or any(not len(group) for group in groups):
        raise ValueError("graph-context balanced sampler lost a class/eligibility stratum")
    counts = [batch_size // 4] * 4
    for index in range(batch_size % 4):
        counts[index] += 1
    selected = [
        rows[torch.randint(len(rows), (count,), generator=generator)]
        for rows, count in zip(groups, counts, strict=True)
    ]
    combined = torch.cat(selected)
    return combined[torch.randperm(len(combined), generator=generator)]


def augment_batch(
    patches: torch.Tensor,
    geometry: torch.Tensor,
    context: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    augmented = patches.float().clone()
    raw_geometry = geometry.clone()
    graph_context = context.float().clone()
    for dimension, context_column in ((-1, 3), (-2, 2), (-3, 1)):
        if torch.rand((), generator=generator) < 0.5:
            augmented = torch.flip(augmented, dims=(dimension,))
            graph_context[..., context_column] *= -1.0
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
        proposed_distance = raw_geometry[swap, 0].clone()
        raw_geometry[swap, 0] = raw_geometry[swap, 2]
        raw_geometry[swap, 2] = proposed_distance
    return augmented, raw_geometry, graph_context


@torch.inference_mode()
def predict(
    model: GraphContextDivisionModel,
    data: GraphContextData,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    patches, geometry, context, mask = data[:4]
    pieces = []
    for start in range(0, len(patches), batch_size):
        stop = start + batch_size
        batch = [
            value[start:stop].to(device=device, non_blocking=True)
            for value in (patches, geometry, context, mask)
        ]
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pieces.append(model(*batch).float().cpu())
    return torch.cat(pieces)


def train_member(
    *,
    member_name: str,
    seed: int,
    initial_model_path: Path,
    train_data: GraphContextData,
    selection_data: GraphContextData,
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
    model = GraphContextDivisionModel().to(device)
    load_backbone_checkpoint(
        model,
        torch.load(initial_model_path, map_location="cpu", weights_only=True),
    )
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.")
    ]
    optimizer = torch.optim.AdamW(
        (
            {
                "params": model.backbone.parameters(),
                "lr": args.learning_rate * args.backbone_lr_multiplier,
            },
            {"params": head_parameters, "lr": args.learning_rate},
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
    train_targets, train_weights, train_eligible = train_data[4:7]
    selection_targets, selection_eligible, selection_inventory = (
        selection_data[4],
        selection_data[6],
        selection_data[7],
    )
    for step in range(1, args.steps + 1):
        model.train()
        rows = balanced_rows(train_targets, train_eligible, args.batch_size, generator)
        patches, geometry, context = augment_batch(
            train_data[0][rows], train_data[1][rows], train_data[2][rows], generator
        )
        batch = [
            value.to(device=device, non_blocking=True)
            for value in (patches, geometry, context, train_data[3][rows])
        ]
        target = train_targets[rows].to(device=device, non_blocking=True)
        weight = train_weights[rows].to(device=device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(*batch)
            loss = focal_loss(logits, target, weight)
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
            print(
                json.dumps(
                    {
                        "member": member_name,
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        "head_learning_rate": head_lr,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if step % args.validation_every == 0 or step == args.steps:
            scores = predict(
                ema,
                selection_data,
                batch_size=args.validation_batch_size,
                device=device,
            )
            metrics = eligible_metrics(
                selection_targets,
                scores,
                selection_eligible,
                selection_inventory,
            )
            improved = best_metrics is None or selection_utility(metrics) > selection_utility(best_metrics)
            history.append({"step": step, "metrics": metrics, "improved": improved})
            atomic_json(member_root / "selection_latest.json", history[-1])
            if improved:
                best_metrics = metrics
                best_state = state_dict_cpu(ema)
                best_step = step
    if best_metrics is None or best_state is None:
        raise RuntimeError("graph-context training produced no selection checkpoint")
    model.load_state_dict(best_state, strict=True)
    checkpoint = member_root / "graph_context_model.pt"
    torch.save(model.state_dict(), checkpoint)
    selection_scores = predict(
        model,
        selection_data,
        batch_size=args.validation_batch_size,
        device=device,
    )
    try:
        frozen = select_frozen_threshold(
            selection_targets[selection_eligible],
            selection_scores[selection_eligible],
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--initial-model", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", default="613111,713117,813121,913127")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--validation-batch-size", type=int, default=20)
    parser.add_argument("--validation-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=6e-5)
    parser.add_argument("--minimum-learning-rate", type=float, default=3e-7)
    parser.add_argument("--backbone-lr-multiplier", type=float, default=0.15)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--gradient-clip", type=float, default=2.0)
    parser.add_argument("--required-gpu-name", default="A10G")
    parser.add_argument(
        "--resume-completed",
        action="store_true",
        help="reuse only hash-verified, pre-audit worker terminals in an existing output root",
    )
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if len(args.initial_model) != 2 or len(seeds) != 4:
        raise ValueError("graph-context sweep requires two initial models and four seeds")
    if len({sha256_file(path) for path in args.initial_model}) != 2:
        raise ValueError("graph-context sweep requires distinct initial backbones")
    if min(
        args.steps,
        args.batch_size,
        args.validation_batch_size,
        args.validation_every,
        args.log_every,
    ) <= 0:
        raise ValueError("graph-context training counts must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("graph-context training requires one Antelume GPU")
    gpu_name = torch.cuda.get_device_name(0)
    if args.required_gpu_name.lower() not in gpu_name.lower():
        raise RuntimeError(
            f"required Antelume GPU {args.required_gpu_name!r}, saw {gpu_name!r}"
        )
    manifest_path = args.data_root / "graph_context_relational_patch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    optimization = load_role(args.data_root, manifest, "optimization")
    selection = load_role(args.data_root, manifest, "selection")
    optimization_stems = {row["stem"] for row in optimization[7]}
    selection_stems = {row["stem"] for row in selection[7]}
    audit_stems = {
        record["stem"] for record in manifest["records"] if record["role"] == "audit"
    }
    if optimization_stems & selection_stems or (optimization_stems | selection_stems) & audit_stems:
        raise RuntimeError("graph-context optimization, selection, and audit movies overlap")
    args.output_root.mkdir(parents=True, exist_ok=args.resume_completed)
    device = torch.device("cuda:0")
    started = time.monotonic()
    terminals = []
    resumed_members: list[str] = []
    for seed in seeds:
        for initial_index, initial_model in enumerate(args.initial_model):
            member_name = f"seed-{seed}-init-{initial_index + 1}"
            member_seed = seed + 10_003 * initial_index
            member_root = args.output_root / member_name
            if args.resume_completed and member_root.exists():
                resumed_members.append(member_name)
                terminals.append(
                    load_completed_member(
                        member_root=member_root,
                        member_name=member_name,
                        seed=member_seed,
                        initial_model_path=initial_model,
                        steps=args.steps,
                    )
                )
            else:
                terminals.append(
                    train_member(
                        member_name=member_name,
                        seed=member_seed,
                        initial_model_path=initial_model,
                        train_data=optimization,
                        selection_data=selection,
                        output_root=args.output_root,
                        args=args,
                        device=device,
                    )
                )
    accepted = [row for row in terminals if row["selection_gate_passed"]]
    selection_scores: dict[str, torch.Tensor] = {}
    for row in accepted:
        model = GraphContextDivisionModel().to(device)
        checkpoint = args.output_root / row["member"] / "graph_context_model.pt"
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True), strict=True
        )
        selection_scores[row["member"]] = predict(
            model, selection, batch_size=args.validation_batch_size, device=device
        )
        del model
        torch.cuda.empty_cache()
    strongest = max(accepted, key=lambda row: selection_utility(row["selection"])) if accepted else None
    selection_ensemble = None
    if len(accepted) >= 2:
        ensemble_scores = calibration_free_equal_rank_ensemble(
            [selection_scores[row["member"]] for row in accepted]
        )
        selection_ensemble = eligible_metrics(
            selection[4], ensemble_scores, selection[6], selection[7]
        )
    ensemble_selected = bool(
        selection_ensemble is not None
        and strongest is not None
        and selection_ensemble["average_precision"]
        >= strongest["selection"]["average_precision"] + 0.01
        and selection_ensemble["true_positives_before_first_false_positive"] >= 2
        and all(
            row["average_precision"] >= 0.45
            for row in selection_ensemble["by_embryo"].values()
        )
    )
    precommitted_policy = (
        "equal_rank_selection_admitted_ensemble"
        if ensemble_selected
        else ("strongest_selection_individual" if strongest else None)
    )
    precommitted_members = (
        [row["member"] for row in accepted]
        if ensemble_selected
        else ([strongest["member"]] if strongest else [])
    )

    audit_results = []
    audit_scores: dict[str, torch.Tensor] = {}
    audit: GraphContextData | None = None
    if accepted:
        audit = load_role(args.data_root, manifest, "audit")
        for row in accepted:
            model = GraphContextDivisionModel().to(device)
            checkpoint = args.output_root / row["member"] / "graph_context_model.pt"
            model.load_state_dict(
                torch.load(checkpoint, map_location=device, weights_only=True), strict=True
            )
            scores = predict(
                model, audit, batch_size=args.validation_batch_size, device=device
            )
            audit_scores[row["member"]] = scores
            metrics = eligible_metrics(audit[4], scores, audit[6], audit[7])
            decisions = threshold_decisions(
                audit[4][audit[6]],
                scores[audit[6]],
                row["selection_frozen_threshold"]["threshold"],
            )
            passed = bool(
                passes_selection_gate(metrics)
                and decisions["tp"] >= 2
                and decisions["precision"] >= 0.80
            )
            result = {
                "member": row["member"],
                "metrics": metrics,
                "frozen_selection_threshold_decisions": decisions,
                "audit_gate_passed": passed,
                "model_sha256": row["model_sha256"],
            }
            audit_results.append(result)
            atomic_json(args.output_root / row["member"] / "audit_terminal.json", result)
            del model
            torch.cuda.empty_cache()
    independently_strong = {
        row["member"] for row in audit_results if row["audit_gate_passed"]
    }
    audit_ensemble = None
    if ensemble_selected and audit is not None:
        audit_ensemble = eligible_metrics(
            audit[4],
            calibration_free_equal_rank_ensemble(
                [audit_scores[name] for name in precommitted_members]
            ),
            audit[6],
            audit[7],
        )
    policy_audit_passed = bool(
        precommitted_policy is not None
        and set(precommitted_members) <= independently_strong
        and (
            precommitted_policy == "strongest_selection_individual"
            or (audit_ensemble is not None and passes_selection_gate(audit_ensemble))
        )
    )
    aggregate = {
        "schema_version": 1,
        "status": "completed",
        "run_id": RUN_ID,
        "family": GRAPH_CONTEXT_DIVISION_FAMILY,
        "gpu_name": gpu_name,
        "gpu_count": 1,
        "execution_policy": "eight independent 74.7M models sequentially on Antelume A10G",
        "elapsed_seconds": time.monotonic() - started,
        "planned_model_count": 8,
        "completed_model_count": len(terminals),
        "resumed_completed_member_count": len(resumed_members),
        "resumed_completed_members": resumed_members,
        "partial_checkpoint_resumed": False,
        "steps_per_model": args.steps,
        "selection_accepted_members": [row["member"] for row in accepted],
        "selection_ensemble": selection_ensemble,
        "ensemble_members_precommitted_before_audit": True,
        "precommitted_policy": precommitted_policy,
        "precommitted_members": precommitted_members,
        "audit_opened": bool(accepted),
        "audit_results": audit_results,
        "audit_ensemble": audit_ensemble,
        "independently_strong_members": sorted(independently_strong),
        "policy_audit_passed": policy_audit_passed,
        "deployment_policy": precommitted_policy if policy_audit_passed else None,
        "deployment_members": precommitted_members if policy_audit_passed else [],
        "ensemble_eligible": bool(policy_audit_passed and ensemble_selected),
        "absolute_threshold_used_for_deployment": False,
        "model_subset_searched_on_audit": False,
        "final_probe_opened": False,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
        **architecture_contract(),
    }
    atomic_json(args.output_root / "graph_context_division_sweep_terminal.json", aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)
    if not policy_audit_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
