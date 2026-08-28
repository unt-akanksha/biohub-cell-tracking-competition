#!/usr/bin/env python
"""Conservatively profile v4 training memory without consuming GPU quota."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_SHARD_ROOT = (
    ROOT
    / ".biohub"
    / "staging"
    / "biohub-zebrahub-contextual-shards-v1-balanced"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "profiles"
    / "temporal-multiscale-training-memory-v1.json"
)
EXPECTED_SHARD_MANIFEST_SHA256 = (
    "b35738f215413f1ece403ba5c0601adea82e2540c65f37e6465de0d0755cb7bf"
)
EXPECTED_PARAMETER_COUNTS = {"v3": 20_747_761, "v4": 46_386_607}
TRANSFER_MAX_SOURCES = 48
TRANSFER_MAX_TARGETS = 128
TRANSFER_MAX_NODES = TRANSFER_MAX_SOURCES + TRANSFER_MAX_TARGETS
EXPECTED_EXTERNAL_SHARDS = 80
EXPECTED_EXTERNAL_MAX_SOURCES = 64
EXPECTED_EXTERNAL_MAX_TARGETS = 96
EXPECTED_EXTERNAL_MAX_CANDIDATE_EDGES = 3_551
FP32_BYTES = 4
PERSISTENT_PARAMETER_COPIES = 5  # model, EMA, gradient, Adam first/second moments
ACTIVATION_WORKSPACE_MULTIPLIER = 1.25
FIXED_SAFETY_ALLOWANCE_BYTES = 2 * 1024**3
T4_CONSERVATIVE_TRAINING_BUDGET_BYTES = 12 * 1024**3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def storage_identity(tensor: torch.Tensor) -> tuple[str, int]:
    storage = tensor.untyped_storage()
    return str(tensor.device), int(storage.data_ptr())


def saved_tensor_inventory(
    operation: Callable[[], torch.Tensor],
    *,
    excluded_tensors: tuple[torch.Tensor, ...] = (),
) -> dict[str, int]:
    """Inventory tensors retained by autograd, excluding model parameters.

    Logical bytes deliberately count repeated saved views. Unique-storage bytes
    describe the lower physical bound. The final safety calculation uses the
    larger logical value, making view reuse and CPU fp32 profiling conservative
    for the fp16 Kaggle training path.
    """

    excluded = {storage_identity(tensor) for tensor in excluded_tensors}
    seen: set[tuple[str, int]] = set()
    logical_bytes = 0
    unique_storage_bytes = 0
    tensor_count = 0

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal logical_bytes, unique_storage_bytes, tensor_count
        identity = storage_identity(tensor)
        if identity not in excluded:
            tensor_count += 1
            logical_bytes += int(tensor.numel()) * int(tensor.element_size())
            if identity not in seen:
                seen.add(identity)
                unique_storage_bytes += int(tensor.untyped_storage().nbytes())
        return tensor

    def unpack(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        loss = operation()
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError("profile operation did not produce a finite scalar loss")
    if logical_bytes <= 0 or unique_storage_bytes <= 0:
        raise RuntimeError("autograd saved-tensor inventory is empty")
    return {
        "logical_bytes": logical_bytes,
        "unique_storage_bytes": unique_storage_bytes,
        "saved_tensor_count": tensor_count,
    }


def node_saved_tensors(model: torch.nn.Module) -> dict[str, int]:
    model.train()
    parameters = tuple(model.parameters())

    def operation() -> torch.Tensor:
        embeddings, divisions = model(torch.zeros(1, 3, 17, 17, 17))
        return embeddings.square().mean() + divisions.square().mean()

    return saved_tensor_inventory(operation, excluded_tensors=parameters)


def dense_context_head_saved_tensors(
    model: torch.nn.Module,
    *,
    source_count: int = TRANSFER_MAX_SOURCES,
    target_count: int = TRANSFER_MAX_TARGETS,
) -> dict[str, int]:
    """Profile the full contextual objective at the densest transfer cap."""

    from research.temporal_contrastive.contextual_pair_fusion import (
        contextual_bidirectional_pair_nll,
    )
    from research.temporal_contrastive.model import (
        masked_multi_positive_info_nce,
    )

    if source_count <= 0 or target_count <= 1:
        raise ValueError("context-head profile needs sources and hard negatives")
    model.train()
    parameters = tuple(model.parameters())
    generator = torch.Generator().manual_seed(840_917)
    source = torch.randn(source_count, 256, generator=generator, requires_grad=True)
    target = torch.randn(target_count, 256, generator=generator, requires_grad=True)
    source_coords = torch.randn(source_count, 3, generator=generator)
    target_coords = torch.randn(target_count, 3, generator=generator)
    divisions = torch.randn(source_count, generator=generator, requires_grad=True)
    candidates = torch.ones(source_count, target_count, dtype=torch.bool)
    positives = torch.zeros_like(candidates)
    positives[torch.arange(source_count), torch.arange(source_count) % target_count] = True
    context = torch.randn(
        source_count, target_count, 18, generator=generator
    )
    division_targets = torch.zeros(source_count)

    def operation() -> torch.Tensor:
        logits = model.candidate_pair_logits(
            source,
            target,
            source_coords,
            target_coords,
            divisions,
            candidates,
            context,
        )
        pair_loss = contextual_bidirectional_pair_nll(
            logits, positives, candidates
        )
        embedding_loss = masked_multi_positive_info_nce(
            source,
            target,
            positives,
            candidates,
            temperature=0.10,
        )
        division_loss = F.binary_cross_entropy_with_logits(
            divisions, division_targets
        )
        return pair_loss + 0.25 * embedding_loss + 0.1 * division_loss

    result = saved_tensor_inventory(operation, excluded_tensors=parameters)
    result.update(
        {
            "source_count": source_count,
            "target_count": target_count,
            "candidate_edges": source_count * target_count,
        }
    )
    return result


def external_inventory(shard_root: Path) -> dict[str, Any]:
    root = shard_root.expanduser().resolve()
    aggregate = root / "DATASET_MANIFEST.json"
    if not aggregate.is_file():
        raise FileNotFoundError(f"external shard manifest is missing: {aggregate}")
    aggregate_hash = sha256_file(aggregate)
    if aggregate_hash != EXPECTED_SHARD_MANIFEST_SHA256:
        raise RuntimeError(f"external shard manifest changed: {aggregate_hash}")
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.rglob("*.manifest.json"))
    ]
    result = {
        "dataset_manifest_sha256": aggregate_hash,
        "shards": len(rows),
        "max_source_nodes": max(int(row["source_nodes"]) for row in rows),
        "max_target_nodes": max(int(row["target_nodes"]) for row in rows),
        "max_candidate_edges": max(int(row["candidate_edges"]) for row in rows),
    }
    expected = {
        "shards": EXPECTED_EXTERNAL_SHARDS,
        "max_source_nodes": EXPECTED_EXTERNAL_MAX_SOURCES,
        "max_target_nodes": EXPECTED_EXTERNAL_MAX_TARGETS,
        "max_candidate_edges": EXPECTED_EXTERNAL_MAX_CANDIDATE_EDGES,
    }
    if any(result[name] != value for name, value in expected.items()):
        raise RuntimeError(f"external shard inventory changed: {result}")
    result["max_nodes"] = result["max_source_nodes"] + result["max_target_nodes"]
    return result


def projected_training_bytes(
    *,
    parameter_count: int,
    node_count: int,
    node_logical_bytes: int,
    context_head_logical_bytes: int,
) -> dict[str, int | float | bool]:
    persistent = (
        int(parameter_count) * FP32_BYTES * PERSISTENT_PARAMETER_COPIES
    )
    saved_activations = (
        int(node_count) * int(node_logical_bytes)
        + int(context_head_logical_bytes)
    )
    projected = (
        persistent
        + int(saved_activations * ACTIVATION_WORKSPACE_MULTIPLIER)
        + FIXED_SAFETY_ALLOWANCE_BYTES
    )
    return {
        "node_count": int(node_count),
        "persistent_model_optimizer_ema_bytes": persistent,
        "fp32_logical_saved_activation_bytes": saved_activations,
        "activation_workspace_multiplier": ACTIVATION_WORKSPACE_MULTIPLIER,
        "fixed_safety_allowance_bytes": FIXED_SAFETY_ALLOWANCE_BYTES,
        "projected_training_bytes": projected,
        "projected_training_gib": projected / 1024**3,
        "conservative_t4_budget_bytes": T4_CONSERVATIVE_TRAINING_BUDGET_BYTES,
        "headroom_bytes": T4_CONSERVATIVE_TRAINING_BUDGET_BYTES - projected,
        "within_conservative_t4_budget": (
            projected <= T4_CONSERVATIVE_TRAINING_BUDGET_BYTES
        ),
    }


def profile(shard_root: Path) -> dict[str, Any]:
    from research.temporal_contrastive.contextual_pair_fusion import (
        ContextualPairFusionAssociationModel,
    )
    from research.temporal_contrastive.multiscale_contextual_pair_fusion import (
        MultiscaleContextualPairFusionAssociationModel,
    )

    inventory = external_inventory(shard_root)
    measurements: dict[str, Any] = {}
    for name, factory in (
        ("v3", ContextualPairFusionAssociationModel),
        ("v4", MultiscaleContextualPairFusionAssociationModel),
    ):
        model = factory()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != EXPECTED_PARAMETER_COUNTS[name]:
            raise RuntimeError(f"{name} parameter inventory changed: {parameter_count}")
        measurements[name] = {
            "parameter_count": parameter_count,
            "parameter_bytes_fp32": parameter_count * FP32_BYTES,
            "node_saved_tensors_fp32": node_saved_tensors(model),
        }
        if name == "v4":
            measurements[name]["dense_context_objective_saved_tensors_fp32"] = (
                dense_context_head_saved_tensors(model)
            )
        del model
        gc.collect()

    v4 = measurements["v4"]
    per_node = int(v4["node_saved_tensors_fp32"]["logical_bytes"])
    dense_head = int(
        v4["dense_context_objective_saved_tensors_fp32"]["logical_bytes"]
    )
    cases = {
        "zebrahub_pretraining_observed_max": projected_training_bytes(
            parameter_count=int(v4["parameter_count"]),
            node_count=int(inventory["max_nodes"]),
            node_logical_bytes=per_node,
            # The 6,144-edge transfer profile upper-bounds the observed 3,551 edges.
            context_head_logical_bytes=dense_head,
        ),
        "biohub_transfer_declared_cap": projected_training_bytes(
            parameter_count=int(v4["parameter_count"]),
            node_count=TRANSFER_MAX_NODES,
            node_logical_bytes=per_node,
            context_head_logical_bytes=dense_head,
        ),
    }
    passed = all(
        bool(case["within_conservative_t4_budget"]) for case in cases.values()
    )
    return {
        "schema_version": 1,
        "status": "profiled",
        "run_id": "temporal-multiscale-training-memory-profile-v1",
        "method": (
            "CPU fp32 autograd saved-tensor logical bytes, parameter storages "
            "excluded; full fp32 model+EMA+gradient+Adam states, 1.25x activation "
            "workspace multiplier, and 2 GiB fixed safety allowance"
        ),
        "kaggle_training_dtype": "cuda autocast float16",
        "external_inventory": inventory,
        "transfer_caps": {
            "max_sources": TRANSFER_MAX_SOURCES,
            "max_targets": TRANSFER_MAX_TARGETS,
            "max_nodes": TRANSFER_MAX_NODES,
            "dense_candidate_edges": TRANSFER_MAX_SOURCES * TRANSFER_MAX_TARGETS,
        },
        "measurements": measurements,
        "cases": cases,
        "decision": (
            "retain_published_v4_training_configuration"
            if passed
            else "reduce_transition_inventory_before_launch"
        ),
        "passed": passed,
        "gpu_used": False,
        "public_leaderboard_used_for_selection": False,
        "competition_submission_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, default=DEFAULT_SHARD_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite training profile: {output}")
    result = profile(args.shard_root)
    atomic_json(output, result)
    print(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
