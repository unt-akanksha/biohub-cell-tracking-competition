#!/usr/bin/env python
"""Train independently gated temporal node localizers on Synthetic256 CC0.

The split is immutable: sequences 0--11 optimize weights, 12--13 rank
checkpoints, and 14--15 remain sealed until a checkpoint has been selected and
serialized.  No competition labels, predictions, leaderboard scores, or public
notebook code participate in this stage.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.synthetic_pretrain.data import POOLED_VOXEL_UM, SequenceSample, corrected_sequence_sample
from research.temporal_contrastive.patch_model import sample_physical_patches
from research.temporal_localization.model import (
    EXPECTED_PARAMETER_COUNT,
    FAMILY,
    HALF_EXTENT_UM,
    PATCH_SHAPE,
    TemporalNodeLocalizationModel,
    localization_loss,
    parameter_count,
)


RUN_ID = "synthetic256-temporal-node-localizer-v1"
TRAIN_INDICES = tuple(range(240))
SELECTION_INDICES = tuple(range(240, 248))
AUDIT_INDICES = tuple(range(248, 256))
DEFAULT_SEEDS = (41_021, 41_029, 41_039, 41_047)


@dataclass(frozen=True)
class SequenceState:
    index: int
    sample: SequenceSample
    eligible_rows: np.ndarray
    division_critical_rows: np.ndarray
    predecessor: np.ndarray
    successors: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class FixedExamples:
    rows_by_sequence: dict[int, np.ndarray]
    jitter_by_sequence_um: dict[int, np.ndarray]
    inventory_sha256: str

    @property
    def count(self) -> int:
        return sum(len(value) for value in self.rows_by_sequence.values())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def state_dict_half(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: (value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu())
        for key, value in model.state_dict().items()
    }


def discover_sequence_paths(root: Path) -> list[Path]:
    paths = [root / "sequences" / f"seq_{index:04d}.npz" for index in range(256)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Synthetic256 sequence inventory is incomplete: {missing}")
    extras = sorted((root / "sequences").glob("seq_*.npz"))
    if extras != paths:
        raise ValueError("Synthetic256 sequence inventory changed from the frozen 256 files")
    return paths


def build_sequence_state(index: int, sample: SequenceSample) -> SequenceState:
    node_count = len(sample.nodes)
    predecessor = np.full(node_count, -1, dtype=np.int64)
    successor_lists: list[list[int]] = [[] for _ in range(node_count)]
    for source, target in sample.edges.tolist():
        source, target = int(source), int(target)
        if predecessor[target] >= 0 and predecessor[target] != source:
            raise ValueError("synthetic lineage has multiple parents")
        predecessor[target] = source
        successor_lists[source].append(target)
    times = sample.nodes[:, 0].astype(np.int64)
    eligible = np.flatnonzero((times >= 1) & (times < sample.volumes.shape[0] - 1)).astype(np.int64)
    if not len(eligible):
        raise ValueError(f"sequence {index} has no interior-frame nodes")
    successors = tuple(np.asarray(rows, dtype=np.int64) for rows in successor_lists)
    division_critical = np.zeros(node_count, dtype=bool)
    for parent, children in enumerate(successors):
        if len(children) >= 2:
            division_critical[parent] = True
            division_critical[children] = True
    division_critical_rows = eligible[division_critical[eligible]]
    if not len(division_critical_rows):
        raise ValueError(f"sequence {index} has no interior division-critical nodes")
    return SequenceState(
        index,
        sample,
        eligible,
        division_critical_rows,
        predecessor,
        successors,
    )


def load_states(paths: list[Path], indices: Iterable[int]) -> list[SequenceState]:
    return [build_sequence_state(index, corrected_sequence_sample(paths[index])) for index in indices]


def random_jitter_um(rng: np.random.Generator, count: int, *, maximum_um: float = 10.0) -> np.ndarray:
    if count <= 0 or maximum_um <= 0:
        raise ValueError("jitter count and maximum must be positive")
    directions = rng.normal(size=(count, 3)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(1e-6)
    radii = rng.uniform(0.0, maximum_um, size=(count, 1)).astype(np.float32)
    # Exact centers teach a stable no-op and anchor the inference safety gate.
    radii[rng.random(count) < 0.1] = 0.0
    return directions * radii


def fixed_examples(
    states: list[SequenceState],
    *,
    seed: int,
    count: int,
    division_critical_only: bool = False,
) -> FixedExamples:
    if count < len(states):
        raise ValueError("fixed evaluation count must cover every sequence")
    rng = np.random.default_rng(seed)
    base, remainder = divmod(count, len(states))
    rows_by_sequence: dict[int, np.ndarray] = {}
    jitter_by_sequence: dict[int, np.ndarray] = {}
    inventory: list[dict[str, Any]] = []
    for order, state in enumerate(states):
        sequence_count = base + int(order < remainder)
        inventory_rows = (
            state.division_critical_rows
            if division_critical_only
            else state.eligible_rows
        )
        if not len(inventory_rows):
            raise ValueError(f"sequence {state.index} has no rows for the requested stratum")
        rows = rng.choice(inventory_rows, size=sequence_count, replace=True).astype(np.int64)
        jitter = random_jitter_um(rng, sequence_count)
        rows_by_sequence[state.index] = rows
        jitter_by_sequence[state.index] = jitter
        inventory.append(
            {
                "sequence": state.index,
                "stratum": "division_critical" if division_critical_only else "global",
                "rows": rows.tolist(),
                "jitter_um": jitter.tolist(),
            }
        )
    digest = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return FixedExamples(rows_by_sequence, jitter_by_sequence, digest)


def graph_motion_features(
    state: SequenceState,
    rows: np.ndarray,
    proposals_zyx_voxel: np.ndarray,
) -> np.ndarray:
    """Return bounded motion/context features available in an inferred graph."""

    rows = np.asarray(rows, dtype=np.int64)
    proposals = np.asarray(proposals_zyx_voxel, dtype=np.float32)
    if proposals.shape != (len(rows), 3):
        raise ValueError("proposal coordinates do not align with node rows")
    coords = state.sample.nodes[:, 1:4].astype(np.float32)
    times = state.sample.nodes[:, 0].astype(np.int64)
    scale = np.asarray(POOLED_VOXEL_UM, dtype=np.float32)
    output = np.zeros((len(rows), 12), dtype=np.float32)
    for batch_row, (node_row, proposal) in enumerate(zip(rows.tolist(), proposals, strict=True)):
        parent = int(state.predecessor[node_row])
        children = state.successors[node_row]
        if parent >= 0:
            output[batch_row, 0:3] = (proposal - coords[parent]) * scale / 10.0
            output[batch_row, 6] = 1.0
        if len(children):
            child_center = coords[children].mean(axis=0)
            output[batch_row, 3:6] = (child_center - proposal) * scale / 10.0
            output[batch_row, 7] = 1.0
        output[batch_row, 8] = min(len(children), 2) / 2.0
        same_frame = np.flatnonzero(times == times[node_row])
        same_frame = same_frame[same_frame != node_row]
        if len(same_frame):
            distances = np.linalg.norm((coords[same_frame] - proposal[None]) * scale[None], axis=1)
            output[batch_row, 9] = min(float(distances.min()) / 20.0, 2.0)
        output[batch_row, 10] = float(times[node_row]) / max(state.sample.volumes.shape[0] - 1, 1)
        boundary_um = np.minimum(proposal, np.asarray(state.sample.volumes.shape[1:]) - 1 - proposal) * scale
        output[batch_row, 11] = float(np.clip(boundary_um.min() / 10.0, 0.0, 2.0))
    return np.clip(output, -3.0, 3.0)


def make_patches(
    state: SequenceState,
    rows: np.ndarray,
    jitter_um: np.ndarray,
    device: torch.device,
    *,
    augment: bool,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nodes = state.sample.nodes[rows]
    proposals = nodes[:, 1:4].astype(np.float32) + jitter_um / POOLED_VOXEL_UM[None]
    graph = graph_motion_features(state, rows, proposals)
    times = nodes[:, 0].astype(np.int64)
    patches = torch.empty((len(rows), 3, *PATCH_SHAPE), dtype=torch.float32, device=device)
    for timepoint in np.unique(times).tolist():
        selected = np.flatnonzero(times == timepoint)
        context = torch.as_tensor(
            state.sample.volumes[[timepoint - 1, timepoint, timepoint + 1]],
            dtype=torch.float32,
            device=device,
        )
        patches[selected] = sample_physical_patches(
            context,
            torch.as_tensor(proposals[selected], dtype=torch.float32, device=device),
            voxel_size_zyx_um=POOLED_VOXEL_UM,
            output_shape_zyx=PATCH_SHAPE,
            half_extent_zyx_um=HALF_EXTENT_UM,
            chunk_size=len(selected),
        )
    if augment:
        if generator is None:
            raise ValueError("augmentation requires a deterministic generator")
        gain = 0.85 + 0.30 * torch.rand(
            (len(patches), 3, 1, 1, 1), generator=generator, device=device
        )
        noise = 0.03 * torch.randn(
            patches.shape, generator=generator, device=device, dtype=patches.dtype
        )
        patches = (patches * gain + noise).clamp(-6.0, 6.0)
    return (
        patches,
        torch.as_tensor(graph, dtype=torch.float32, device=device),
        torch.as_tensor(-jitter_um, dtype=torch.float32, device=device),
    )


def baseline_metrics(examples: FixedExamples) -> dict[str, Any]:
    target = np.concatenate(list(examples.jitter_by_sequence_um.values()), axis=0)
    distances = np.linalg.norm(target, axis=1)
    return {
        "examples": len(distances),
        "mean_residual_um": float(distances.mean()),
        "p90_residual_um": float(np.quantile(distances, 0.9)),
        "axis_mae_um": np.abs(target).mean(axis=0).tolist(),
        "within_5um_rate": float(np.mean(distances <= 5.0)),
    }


@torch.inference_mode()
def evaluate(
    model: TemporalNodeLocalizationModel,
    states: list[SequenceState],
    examples: FixedExamples,
    device: torch.device,
    *,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    residuals: list[np.ndarray] = []
    uncertainties: list[np.ndarray] = []
    safe_predictions: list[np.ndarray] = []
    by_index = {state.index: state for state in states}
    for sequence_index in sorted(examples.rows_by_sequence):
        rows = examples.rows_by_sequence[sequence_index]
        jitter = examples.jitter_by_sequence_um[sequence_index]
        for start in range(0, len(rows), batch_size):
            patches, graph, target = make_patches(
                by_index[sequence_index],
                rows[start : start + batch_size],
                jitter[start : start + batch_size],
                device,
                augment=False,
            )
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                predicted, log_variance, safe = model(patches, graph)
            residuals.append((predicted.float() - target).cpu().numpy())
            uncertainties.append(torch.exp(0.5 * log_variance.float()).mean(dim=1).cpu().numpy())
            safe_predictions.append(safe.float().cpu().numpy())
    residual = np.concatenate(residuals, axis=0)
    distances = np.linalg.norm(residual, axis=1)
    uncertainty = np.concatenate(uncertainties)
    safe = np.concatenate(safe_predictions)
    return {
        "examples": len(distances),
        "mean_residual_um": float(distances.mean()),
        "p90_residual_um": float(np.quantile(distances, 0.9)),
        "axis_mae_um": np.abs(residual).mean(axis=0).tolist(),
        "within_5um_rate": float(np.mean(distances <= 5.0)),
        "mean_predicted_sigma_um": float(uncertainty.mean()),
        "mean_safe_probability": float(safe.mean()),
    }


def improvement_gate(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    axis_nonregressive = all(
        float(right) <= float(left) + 1e-8
        for left, right in zip(baseline["axis_mae_um"], candidate["axis_mae_um"], strict=True)
    )
    passed = bool(
        int(candidate["examples"]) == int(baseline["examples"])
        and float(candidate["mean_residual_um"]) < 0.8 * float(baseline["mean_residual_um"])
        and float(candidate["p90_residual_um"]) < float(baseline["p90_residual_um"])
        and float(candidate["within_5um_rate"]) > float(baseline["within_5um_rate"])
        and axis_nonregressive
    )
    return {
        "passed": passed,
        "mean_relative_gain": 1.0 - float(candidate["mean_residual_um"]) / float(baseline["mean_residual_um"]),
        "p90_gain_um": float(baseline["p90_residual_um"]) - float(candidate["p90_residual_um"]),
        "within_5um_gain": float(candidate["within_5um_rate"]) - float(baseline["within_5um_rate"]),
        "axis_mae_nonregressive": axis_nonregressive,
    }


def stratified_improvement_gate(
    baseline_global: dict[str, Any],
    candidate_global: dict[str, Any],
    baseline_division_critical: dict[str, Any],
    candidate_division_critical: dict[str, Any],
) -> dict[str, Any]:
    global_gate = improvement_gate(baseline_global, candidate_global)
    division_gate = improvement_gate(
        baseline_division_critical, candidate_division_critical
    )
    return {
        "passed": bool(global_gate["passed"] and division_gate["passed"]),
        "global": global_gate,
        "division_critical": division_gate,
        "division_critical_gate_required": True,
    }


def update_ema(model: torch.nn.Module, ema: torch.nn.Module, *, decay: float) -> None:
    with torch.no_grad():
        for ema_value, value in zip(ema.state_dict().values(), model.state_dict().values(), strict=True):
            if ema_value.is_floating_point():
                ema_value.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                ema_value.copy_(value)


def train_member(
    args: argparse.Namespace,
    paths: list[Path],
    train_states: list[SequenceState],
    selection_states: list[SequenceState],
    *,
    member_index: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    started = time.monotonic()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device).manual_seed(seed + 700_001)
    output_dir = args.output_dir / f"member_{member_index:02d}_seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_examples = fixed_examples(
        selection_states,
        seed=91_000 + member_index,
        count=args.validation_examples,
    )
    selection_division_examples = fixed_examples(
        selection_states,
        seed=121_000 + member_index,
        count=args.division_validation_examples,
        division_critical_only=True,
    )
    baseline_selection = baseline_metrics(selection_examples)
    baseline_selection_division = baseline_metrics(selection_division_examples)

    model = TemporalNodeLocalizationModel().to(device)
    if parameter_count(model) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("temporal localizer parameter inventory changed")
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    weights = np.asarray([len(state.eligible_rows) for state in train_states], dtype=np.float64)
    weights /= weights.sum()
    history: list[dict[str, Any]] = []
    best_step = 0
    best_metrics = evaluate(
        ema,
        selection_states,
        selection_examples,
        device,
        batch_size=args.validation_batch_size,
    )
    best_division_metrics = evaluate(
        ema,
        selection_states,
        selection_division_examples,
        device,
        batch_size=args.validation_batch_size,
    )
    best_relative_residual = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    completed_step = 0

    atomic_json(
        output_dir / "training_config.json",
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "family": FAMILY,
            "member_index": member_index,
            "seed": seed,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "training_sequences": list(TRAIN_INDICES),
            "selection_sequences": list(SELECTION_INDICES),
            "audit_sequences_declared_but_unopened": list(AUDIT_INDICES),
            "selection_inventory_sha256": selection_examples.inventory_sha256,
            "selection_division_critical_inventory_sha256": (
                selection_division_examples.inventory_sha256
            ),
            "division_critical_training_rows": int(
                sum(len(state.division_critical_rows) for state in train_states)
            ),
            "division_critical_per_batch": args.division_critical_per_batch,
            "division_critical_selection_gate_required": True,
            "division_critical_audit_gate_required": True,
            "optimization_source_files": {
                paths[index].name: sha256_file(paths[index])
                for index in (*TRAIN_INDICES, *SELECTION_INDICES)
            },
            "competition_data_read": False,
            "public_code_copied": False,
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
        },
    )

    model.train()
    for step in range(1, args.steps + 1):
        if time.monotonic() - started >= args.member_max_wall_seconds - args.finalization_reserve_seconds:
            break
        completed_step = step
        state = train_states[int(rng.choice(len(train_states), p=weights))]
        global_count = args.batch_size - args.division_critical_per_batch
        rows = np.concatenate(
            (
                rng.choice(
                    state.eligible_rows, size=global_count, replace=True
                ).astype(np.int64),
                rng.choice(
                    state.division_critical_rows,
                    size=args.division_critical_per_batch,
                    replace=True,
                ).astype(np.int64),
            )
        )
        rng.shuffle(rows)
        jitter = random_jitter_um(rng, args.batch_size)
        patches, graph, target = make_patches(
            state, rows, jitter, device, augment=True, generator=generator
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            predicted, log_variance, safe = model(patches, graph)
            loss, components = localization_loss(
                predicted.float(), log_variance.float(), safe.float(), target.float()
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        update_ema(model, ema, decay=args.ema_decay)
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
                        "member": member_index,
                        "seed": seed,
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        **{key: float(value.cpu()) for key, value in components.items()},
                        "learning_rate": learning_rate,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if step % args.validation_every == 0 or step == args.steps:
            metrics = evaluate(
                ema, selection_states, selection_examples, device, batch_size=args.validation_batch_size
            )
            division_metrics = evaluate(
                ema,
                selection_states,
                selection_division_examples,
                device,
                batch_size=args.validation_batch_size,
            )
            gate = stratified_improvement_gate(
                baseline_selection,
                metrics,
                baseline_selection_division,
                division_metrics,
            )
            relative_residual = (
                float(metrics["mean_residual_um"])
                / float(baseline_selection["mean_residual_um"])
                + float(division_metrics["mean_residual_um"])
                / float(baseline_selection_division["mean_residual_um"])
            )
            row = {
                "step": step,
                "global_metrics": metrics,
                "division_critical_metrics": division_metrics,
                "relative_residual_sum": relative_residual,
                "gate": gate,
            }
            history.append(row)
            atomic_json(output_dir / "selection_latest.json", row)
            if gate["passed"] and relative_residual < best_relative_residual:
                best_step = step
                best_metrics = metrics
                best_division_metrics = division_metrics
                best_relative_residual = relative_residual
                best_state = state_dict_half(ema)
            model.train()

    atomic_json(output_dir / "selection_history.json", {"rows": history})
    selection_gate = stratified_improvement_gate(
        baseline_selection,
        best_metrics,
        baseline_selection_division,
        best_division_metrics,
    )
    terminal: dict[str, Any] = {
        "schema_version": 1,
        "status": "rejected_at_selection",
        "run_id": RUN_ID,
        "family": FAMILY,
        "member_index": member_index,
        "seed": seed,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "elapsed_seconds": time.monotonic() - started,
        "completed_step": completed_step,
        "best_step": best_step,
        "baseline_selection": baseline_selection,
        "best_selection": best_metrics,
        "baseline_selection_division_critical": baseline_selection_division,
        "best_selection_division_critical": best_division_metrics,
        "selection_gate": selection_gate,
        "selection_gate_passed": bool(selection_gate["passed"]),
        "division_critical_selection_gate_passed": bool(
            selection_gate["division_critical"]["passed"]
        ),
        "checkpoint_frozen_before_audit": False,
        "audit_opened": False,
        "competition_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    if best_state is not None and selection_gate["passed"]:
        checkpoint = output_dir / "localization_model.pt"
        torch.save(best_state, checkpoint)
        checkpoint_hash = sha256_file(checkpoint)
        serialized_state = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        ema.load_state_dict(serialized_state, strict=True)
        serialized_selection = evaluate(
            ema,
            selection_states,
            selection_examples,
            device,
            batch_size=args.validation_batch_size,
        )
        serialized_selection_division = evaluate(
            ema,
            selection_states,
            selection_division_examples,
            device,
            batch_size=args.validation_batch_size,
        )
        serialized_selection_gate = stratified_improvement_gate(
            baseline_selection,
            serialized_selection,
            baseline_selection_division,
            serialized_selection_division,
        )
        terminal.update(
            {
                "status": "rejected_after_checkpoint_serialization",
                "model_sha256": checkpoint_hash,
                "model_bytes": checkpoint.stat().st_size,
                "checkpoint_frozen_before_audit": True,
                "serialized_checkpoint_selection": serialized_selection,
                "serialized_checkpoint_selection_division_critical": (
                    serialized_selection_division
                ),
                "serialized_checkpoint_selection_gate": serialized_selection_gate,
                "serialized_checkpoint_selection_gate_passed": bool(
                    serialized_selection_gate["passed"]
                ),
            }
        )
        if serialized_selection_gate["passed"]:
            terminal["best_selection"] = serialized_selection
            terminal["best_selection_division_critical"] = serialized_selection_division
            terminal["selection_gate"] = serialized_selection_gate
            terminal["selection_gate_passed"] = True
            terminal["division_critical_selection_gate_passed"] = True
            # Audit data is opened only after the exact serialized checkpoint
            # has been hashed and re-passed both frozen selection strata.
            audit_states = load_states(paths, AUDIT_INDICES)
            audit_examples = fixed_examples(
                audit_states,
                seed=191_000 + member_index,
                count=args.audit_examples,
            )
            audit_division_examples = fixed_examples(
                audit_states,
                seed=221_000 + member_index,
                count=args.division_audit_examples,
                division_critical_only=True,
            )
            baseline_audit = baseline_metrics(audit_examples)
            baseline_audit_division = baseline_metrics(audit_division_examples)
            final_audit = evaluate(
                ema,
                audit_states,
                audit_examples,
                device,
                batch_size=args.validation_batch_size,
            )
            final_audit_division = evaluate(
                ema,
                audit_states,
                audit_division_examples,
                device,
                batch_size=args.validation_batch_size,
            )
            audit_gate = stratified_improvement_gate(
                baseline_audit,
                final_audit,
                baseline_audit_division,
                final_audit_division,
            )
            terminal.update(
                {
                    "status": "completed" if audit_gate["passed"] else "rejected_at_audit",
                    "audit_opened": True,
                    "audit_inventory_sha256": audit_examples.inventory_sha256,
                    "audit_division_critical_inventory_sha256": (
                        audit_division_examples.inventory_sha256
                    ),
                    "audit_source_files": {
                        paths[index].name: sha256_file(paths[index])
                        for index in AUDIT_INDICES
                    },
                    "baseline_audit": baseline_audit,
                    "final_audit": final_audit,
                    "baseline_audit_division_critical": baseline_audit_division,
                    "final_audit_division_critical": final_audit_division,
                    "audit_gate": audit_gate,
                    "audit_gate_passed": bool(audit_gate["passed"]),
                    "division_critical_audit_gate_passed": bool(
                        audit_gate["division_critical"]["passed"]
                    ),
                }
            )
    atomic_json(output_dir / "worker_terminal.json", terminal)
    del model, ema, optimizer
    torch.cuda.empty_cache()
    return terminal


def run(args: argparse.Namespace) -> None:
    started = time.monotonic()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Antelume localizer training requires exactly one GPU, saw {torch.cuda.device_count()}")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    if args.required_gpu_name not in gpu_name:
        raise RuntimeError(f"required AWS GPU {args.required_gpu_name!r}, saw {gpu_name!r}")
    paths = discover_sequence_paths(args.synthetic_root)
    # Audit sequences are deliberately not loaded here.
    train_states = load_states(paths, TRAIN_INDICES)
    selection_states = load_states(paths, SELECTION_INDICES)
    terminals: list[dict[str, Any]] = []
    for member_index, seed in enumerate(args.seeds):
        if time.monotonic() - started >= args.total_max_wall_seconds - args.finalization_reserve_seconds:
            break
        terminals.append(
            train_member(
                args,
                paths,
                train_states,
                selection_states,
                member_index=member_index,
                seed=seed,
                device=device,
            )
        )
    completed = [row for row in terminals if row.get("status") == "completed"]
    terminal = {
        "schema_version": 1,
        "status": "completed" if len(terminals) == len(args.seeds) else "wall_time_exhausted",
        "run_id": RUN_ID,
        "family": FAMILY,
        "gpu_name": gpu_name,
        "gpu_count": 1,
        "requested_members": len(args.seeds),
        "finished_members": len(terminals),
        "individually_accepted_members": len(completed),
        "all_members_individually_accepted": len(completed) == len(args.seeds),
        "members": terminals,
        "elapsed_seconds": time.monotonic() - started,
        "ensemble_policy": "equal mean offsets only after every included member independently passes selection and sealed audit; no subset or weight search",
        "competition_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    atomic_json(args.output_dir / "training_terminal.json", terminal)
    print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)
    if not completed:
        raise RuntimeError("no temporal localization member passed both synthetic gates")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--synthetic-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    result.add_argument("--steps", type=int, default=20_000)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--validation-examples", type=int, default=1_024)
    result.add_argument("--audit-examples", type=int, default=1_024)
    result.add_argument("--division-validation-examples", type=int, default=512)
    result.add_argument("--division-audit-examples", type=int, default=512)
    result.add_argument("--division-critical-per-batch", type=int, default=4)
    result.add_argument("--validation-batch-size", type=int, default=24)
    result.add_argument("--validation-every", type=int, default=1_000)
    result.add_argument("--log-every", type=int, default=100)
    result.add_argument("--learning-rate", type=float, default=2e-4)
    result.add_argument("--minimum-learning-rate", type=float, default=2e-6)
    result.add_argument("--weight-decay", type=float, default=0.03)
    result.add_argument("--ema-decay", type=float, default=0.997)
    result.add_argument("--member-max-wall-seconds", type=int, default=10_800)
    result.add_argument("--total-max-wall-seconds", type=int, default=43_200)
    result.add_argument("--finalization-reserve-seconds", type=int, default=900)
    result.add_argument("--required-gpu-name", default="A10G")
    return result


def main() -> None:
    args = parser().parse_args()
    counts = (
        args.steps,
        args.batch_size,
        args.validation_examples,
        args.audit_examples,
        args.division_validation_examples,
        args.division_audit_examples,
        args.division_critical_per_batch,
        args.validation_batch_size,
        args.validation_every,
        args.log_every,
        args.member_max_wall_seconds,
        args.total_max_wall_seconds,
        args.finalization_reserve_seconds,
    )
    if (
        min(counts) <= 0
        or args.division_critical_per_batch >= args.batch_size
        or len(set(args.seeds)) != len(args.seeds)
    ):
        raise ValueError("training counts must be positive and member seeds unique")
    run(args)


if __name__ == "__main__":
    main()
