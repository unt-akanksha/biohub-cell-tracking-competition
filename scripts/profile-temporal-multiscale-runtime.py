#!/usr/bin/env python
"""Profile v3/v4 node MACs and all reciprocal final-sharding cases on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_BASE = (
    ROOT
    / ".biohub"
    / "cache"
    / "kernel-outputs"
    / "public-0927-clean-repro-v2"
    / "submission.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "profiles"
    / "temporal-multiscale-runtime-v1.json"
)
EXPECTED_BASE_SHA256 = (
    "33c179b0449b9cdd186f06a653cddc8cf12359f008982f6713cdf30784a52e6a"
)
EXPECTED_V3_PARAMETERS = 20_747_761
EXPECTED_V4_PARAMETERS = 46_386_607
MAXIMUM_ACCEPTED_PROJECTED_LOAD_RATIO = 1.01


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


def model_macs(model: nn.Module) -> int:
    """Count Conv/Linear multiply-accumulates for one 17-cube node patch."""

    total = 0
    hooks: list[Any] = []

    def record(module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        nonlocal total
        tensor = output if isinstance(output, torch.Tensor) else output[0]
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            kernel_elements = 1
            for value in module.kernel_size:
                kernel_elements *= int(value)
            total += int(tensor.numel()) * kernel_elements * (
                int(module.in_channels) // int(module.groups)
            )
        elif isinstance(module, nn.Linear):
            total += int(tensor.numel()) * int(module.in_features)

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            hooks.append(module.register_forward_hook(record))
    try:
        model.eval()
        with torch.inference_mode():
            model(torch.zeros(1, 3, 17, 17, 17))
    finally:
        for hook in hooks:
            hook.remove()
    if total <= 0:
        raise RuntimeError("model MAC inventory is empty")
    return total


def evaluated_plan(
    planner: Any,
    videos: dict[str, Any],
    encoder_counts: dict[str, int],
    *,
    planning_node_cost: float,
    evaluation_node_cost: float,
) -> dict[str, Any]:
    original = float(planner.APPEARANCE_NODE_COST)
    try:
        planner.APPEARANCE_NODE_COST = float(planning_node_cost)
        plan = planner.build_transition_work_plan(
            videos,
            ("0", "1"),
            encoder_count_by_stem=encoder_counts,
            pair_fusion=True,
        )
        planner.APPEARANCE_NODE_COST = float(evaluation_node_cost)
        loads = []
        for shard in plan["shards"]:
            loads.append(
                sum(
                    planner.transition_block_weight(
                        videos[str(unit["stem"])],
                        tuple(map(int, unit["transition_starts"])),
                        encoder_count=encoder_counts[str(unit["stem"])],
                        pair_fusion=True,
                    )
                    for unit in shard["units"]
                )
            )
    finally:
        planner.APPEARANCE_NODE_COST = original
    ratio = max(loads) / min(loads)
    return {
        "projected_true_loads": loads,
        "projected_true_load_ratio": ratio,
        "work_plan_sha256": plan["work_plan_sha256"],
        "partition_kind": plan["partition_kind"],
        "dominant_split": plan["dominant_split"],
    }


def profile(base_submission: Path) -> dict[str, Any]:
    from research.temporal_contrastive.contextual_pair_fusion import (
        ContextualPairFusionAssociationModel,
    )
    from research.temporal_contrastive.multiscale_contextual_pair_fusion import (
        MultiscaleContextualPairFusionAssociationModel,
    )
    from research.temporal_contrastive import (
        dual_fold_appearance_submission as planner,
    )
    from research.trackastra_graph import rerank_submission

    base_submission = base_submission.expanduser().resolve()
    if not base_submission.is_file():
        raise FileNotFoundError(f"base submission is missing: {base_submission}")
    base_hash = sha256_file(base_submission)
    if base_hash != EXPECTED_BASE_SHA256:
        raise RuntimeError(f"base submission changed: {base_hash}")

    models = {
        "v3": ContextualPairFusionAssociationModel(),
        "v4": MultiscaleContextualPairFusionAssociationModel(),
    }
    parameters = {
        name: sum(parameter.numel() for parameter in model.parameters())
        for name, model in models.items()
    }
    if parameters != {
        "v3": EXPECTED_V3_PARAMETERS,
        "v4": EXPECTED_V4_PARAMETERS,
    }:
        raise RuntimeError(f"appearance parameter inventory changed: {parameters}")
    macs = {name: model_macs(model) for name, model in models.items()}
    mac_ratio = float(macs["v4"]) / float(macs["v3"])
    current_node_cost = float(planner.APPEARANCE_NODE_COST)
    adjusted_node_cost = current_node_cost * mac_ratio
    videos = rerank_submission.read_submission(base_submission)

    cases: dict[str, Any] = {}
    for count_44b6 in (1, 2):
        for count_6bba in (1, 2):
            encoder_counts = {
                stem: count_44b6 if stem.startswith("44b6_") else count_6bba
                for stem in videos
            }
            case_name = f"44b6_{count_44b6}x__6bba_{count_6bba}x"
            cases[case_name] = {
                "encoder_counts": {
                    "44b6": count_44b6,
                    "6bba": count_6bba,
                },
                "current_scheduler": evaluated_plan(
                    planner,
                    videos,
                    encoder_counts,
                    planning_node_cost=current_node_cost,
                    evaluation_node_cost=adjusted_node_cost,
                ),
                "mac_adjusted_scheduler": evaluated_plan(
                    planner,
                    videos,
                    encoder_counts,
                    planning_node_cost=adjusted_node_cost,
                    evaluation_node_cost=adjusted_node_cost,
                ),
            }
    current_worst = max(
        float(case["current_scheduler"]["projected_true_load_ratio"])
        for case in cases.values()
    )
    adjusted_worst = max(
        float(case["mac_adjusted_scheduler"]["projected_true_load_ratio"])
        for case in cases.values()
    )
    retain_current = current_worst <= MAXIMUM_ACCEPTED_PROJECTED_LOAD_RATIO
    return {
        "schema_version": 1,
        "status": "profiled",
        "run_id": "temporal-multiscale-runtime-profile-v1",
        "base_submission_sha256": base_hash,
        "movies": len(videos),
        "parameters_per_fold": parameters,
        "macs_per_node": macs,
        "v4_to_v3_parameter_ratio": parameters["v4"] / parameters["v3"],
        "v4_to_v3_macs_ratio": mac_ratio,
        "current_appearance_node_cost": current_node_cost,
        "mac_adjusted_appearance_node_cost": adjusted_node_cost,
        "maximum_accepted_projected_load_ratio": (
            MAXIMUM_ACCEPTED_PROJECTED_LOAD_RATIO
        ),
        "current_scheduler_worst_projected_load_ratio": current_worst,
        "mac_adjusted_scheduler_worst_projected_load_ratio": adjusted_worst,
        "scheduler_decision": (
            "retain_current_transition_plan_no_runtime_mutation"
            if retain_current
            else "use_mac_adjusted_node_cost"
        ),
        "cases": cases,
        "gpu_used": False,
        "public_leaderboard_used_for_selection": False,
        "competition_submission_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-submission", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite runtime profile: {output}")
    result = profile(args.base_submission)
    atomic_json(output, result)
    print(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
