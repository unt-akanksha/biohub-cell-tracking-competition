#!/usr/bin/env python
"""Score a sealed graph-context policy on the exact EMA development probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.temporal_contrastive.graph_context_division_inference import (
    EXPECTED_PARAMETER_COUNT,
    calibration_free_scores,
    member_scores,
)
from research.temporal_contrastive.graph_context_division_model import (
    GRAPH_CONTEXT_DIVISION_FAMILY,
    GraphContextDivisionModel,
    architecture_contract,
)
from research.temporal_contrastive.score_relational_division_development_probe import (
    GEOMETRY_FIELDS,
    VOXEL_SIZE_ZYX_UM,
    atomic_json,
    ranking_metrics,
    verify_cache,
)
from research.temporal_contrastive.train_real_division_gate import sha256_file
from research.temporal_contrastive.patch_model import sample_physical_patches


RUN_ID = "competition-graph-context-division-development-probe-v1"
V1_TRAIN_RUN_ID = "competition-graph-context-division-sweep-v1"
V2_TRAIN_RUN_ID = "competition-graph-context-division-frozen-ensemble-v2"
V2_POLICY_CONTRACT = "all-selection-admitted-equal-rank-ensemble-v2"
INVENTORY_RUN_ID = "competition-graph-context-development-inventory-v1"
INVENTORY_SHA256 = "c8883e77abcf76c5a837c5fd0b21afdfefb2e51cb8e550e3a81f69d70562f311"


def validate_policy(results_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    terminal_path = results_root / "graph_context_division_sweep_terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    members = [str(value) for value in terminal.get("deployment_members", [])]
    policy = terminal.get("deployment_policy")
    train_run_id = terminal.get("run_id")
    is_v2 = train_run_id == V2_TRAIN_RUN_ID
    independently_strong = set(terminal.get("independently_strong_members", []))
    audit_ensemble = terminal.get("audit_ensemble") or {}
    v2_audit_unit_passed = bool(
        is_v2
        and terminal.get("policy_contract") == V2_POLICY_CONTRACT
        and terminal.get("policy_unit_audited") is True
        and terminal.get("constituent_audit_gate_required") is False
        and policy == "equal_rank_selection_admitted_ensemble"
        and len(members) >= 2
        and audit_ensemble.get("average_precision", 0.0) >= 0.55
        and audit_ensemble.get("true_positives_before_first_false_positive", 0) >= 2
        and all(
            row.get("average_precision", 0.0) >= 0.40
            for row in audit_ensemble.get("by_embryo", {}).values()
        )
        and len(audit_ensemble.get("by_embryo", {})) == 2
    )
    v1_constituents_passed = bool(
        train_run_id == V1_TRAIN_RUN_ID
        and set(members) <= independently_strong
        and terminal.get("constituent_audit_gate_required", True) is True
    )
    if not (
        terminal.get("schema_version") == 1
        and terminal.get("status") == "completed"
        and train_run_id in {V1_TRAIN_RUN_ID, V2_TRAIN_RUN_ID}
        and terminal.get("family") == GRAPH_CONTEXT_DIVISION_FAMILY
        and terminal.get("parameter_count") == EXPECTED_PARAMETER_COUNT
        and terminal.get("planned_model_count") == 8
        and terminal.get("completed_model_count") == 8
        and terminal.get("steps_per_model") == 20_000
        and terminal.get("ensemble_members_precommitted_before_audit") is True
        and terminal.get("policy_audit_passed") is True
        and policy
        in {
            "equal_rank_selection_admitted_ensemble",
            "strongest_selection_individual",
        }
        and 1 <= len(members) <= 8
        and len(set(members)) == len(members)
        and members == terminal.get("precommitted_members")
        and (v1_constituents_passed or v2_audit_unit_passed)
        and terminal.get("absolute_threshold_used_for_deployment") is False
        and terminal.get("model_subset_searched_on_audit") is False
        and terminal.get("final_probe_opened") is False
        and terminal.get("competition_test_data_read") is False
        and terminal.get("public_leaderboard_used_for_selection") is False
        and terminal.get("authorized_for_submission") is False
        and (
            (
                policy == "equal_rank_selection_admitted_ensemble"
                and len(members) >= 2
                and terminal.get("ensemble_eligible") is True
            )
            or (policy == "strongest_selection_individual" and len(members) == 1)
        )
    ):
        raise ValueError("graph-context deployment policy is ineligible")
    records = []
    for name in members:
        member_root = results_root / name
        worker = json.loads((member_root / "worker_terminal.json").read_text())
        audit = json.loads((member_root / "audit_terminal.json").read_text())
        checkpoint = member_root / "graph_context_model.pt"
        checkpoint_hash = sha256_file(checkpoint)
        if not (
            worker.get("status") == "accepted_at_selection"
            and worker.get("run_id") == train_run_id
            and worker.get("family") == GRAPH_CONTEXT_DIVISION_FAMILY
            and worker.get("member") == name
            and worker.get("parameter_count") == EXPECTED_PARAMETER_COUNT
            and worker.get("selection_gate_passed") is True
            and worker.get("model_sha256") == checkpoint_hash
            and worker.get("audit_opened") is False
            and worker.get("final_probe_opened") is False
            and worker.get("competition_test_data_read") is False
            and worker.get("public_leaderboard_used_for_selection") is False
            and worker.get("authorized_for_audit") is True
            and worker.get("authorized_for_submission") is False
            and audit.get("member") == name
            and audit.get("model_sha256") == checkpoint_hash
            and (
                audit.get("audit_gate_passed") is True
                or (is_v2 and isinstance(audit.get("metrics"), dict))
            )
        ):
            raise ValueError(f"graph-context deployment member is ineligible: {name}")
        records.append(
            {
                "member": name,
                "seed": int(worker["seed"]),
                "model_sha256": checkpoint_hash,
                "checkpoint": checkpoint,
                "selection_average_precision": float(
                    worker["selection"]["average_precision"]
                ),
                "audit_average_precision": float(audit["metrics"]["average_precision"]),
                "audit_gate_passed": audit.get("audit_gate_passed") is True,
            }
        )
    return terminal, records


def load_models(
    records: list[dict[str, Any]], device: torch.device
) -> list[GraphContextDivisionModel]:
    if architecture_contract()["parameter_count"] != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("graph-context architecture inventory changed")
    models = []
    for record in records:
        model = GraphContextDivisionModel().to(device)
        model.load_state_dict(
            torch.load(record["checkpoint"], map_location=device, weights_only=True),
            strict=True,
        )
        models.append(model.requires_grad_(False).eval())
    return models


@torch.inference_mode()
def build_rows(
    inventory: dict[str, Any],
    cache_root: Path,
    models: list[GraphContextDivisionModel],
    *,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    import zarr

    rows = []
    for movie in inventory["movies"]:
        stem = str(movie["stem"])
        array = zarr.open_group(str(cache_root / "train" / f"{stem}.zarr"), mode="r")["0"]
        eligible = [
            row for row in movie["candidates"] if row["inference_geometry_eligible"]
        ]
        by_time: dict[int, list[dict[str, Any]]] = {}
        for row in eligible:
            by_time.setdefault(int(row["timepoint"]), []).append(row)
        scored: dict[int, tuple[list[float], float]] = {}
        for timepoint, candidates in sorted(by_time.items()):
            temporal = np.stack(
                [
                    np.asarray(array[max(0, min(len(array) - 1, timepoint + offset))])
                    for offset in (-1, 0, 1)
                ]
            )
            centers = np.concatenate(
                [
                    np.asarray(
                        [
                            row["centers_zyx_voxel"][name]
                            for name in ("parent", "existing_child", "proposed_child")
                        ],
                        dtype=np.float32,
                    )
                    for row in candidates
                ],
                axis=0,
            )
            patches = sample_physical_patches(
                temporal,
                centers,
                voxel_size_zyx_um=VOXEL_SIZE_ZYX_UM,
                output_shape_zyx=(17, 17, 17),
                half_extent_zyx_um=(8.0, 8.0, 8.0),
                chunk_size=max(3, batch_size * 3),
            ).reshape(len(candidates), 3, 3, 17, 17, 17)
            geometry = torch.as_tensor(
                np.asarray(
                    [
                        [
                            np.nan if row[name] is None else float(row[name])
                            for name in GEOMETRY_FIELDS
                        ]
                        for row in candidates
                    ],
                    dtype=np.float32,
                )
            )
            context = torch.as_tensor(
                np.asarray(
                    [row["graph_context_features"] for row in candidates],
                    dtype=np.float16,
                )
            )
            context_mask = torch.as_tensor(
                np.asarray(
                    [row["graph_context_mask"] for row in candidates], dtype=np.bool_
                )
            )
            values = member_scores(
                models,
                patches,
                geometry,
                context,
                context_mask,
                device=device,
                batch_size=batch_size,
            )
            parent_ids = [int(row["parent_id"]) for row in candidates]
            consensus = calibration_free_scores(values, parent_ids)
            for index, parent_id in enumerate(parent_ids):
                scored[parent_id] = (
                    [float(member[index]) for member in values],
                    float(consensus[parent_id]),
                )
            del patches, geometry, context, context_mask
        for row in eligible:
            parent_id = int(row["parent_id"])
            result = dict(row)
            result.pop("graph_context_features", None)
            result.pop("graph_context_mask", None)
            result["stem"] = stem
            result["member_logits"], result["ensemble_logit"] = scored[parent_id]
            rows.append(result)
    if not (
        len(rows) == 9
        and sum(bool(row["safe_recovery_positive"]) for row in rows) == 3
    ):
        raise RuntimeError("graph-context development scoring inventory changed")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--required-gpu-name", default="A10G")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("graph-context probe batch size must be positive")
    if sha256_file(args.inventory) != INVENTORY_SHA256:
        raise ValueError("graph-context development inventory changed")
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    if not (
        inventory.get("schema_version") == 1
        and inventory.get("status") == "complete"
        and inventory.get("run_id") == INVENTORY_RUN_ID
        and inventory.get("summary", {}).get("rows") == 225
        and inventory.get("summary", {}).get("inference_geometry_eligible_rows") == 9
        and inventory.get("summary", {}).get("inference_geometry_eligible_positives") == 3
        and inventory.get("graph_context_edges_read") is False
        and inventory.get("graph_context_labels_used") is False
        and inventory.get("authorized_for_graph_context_probe_scoring") is True
        and inventory.get("authorized_for_submission") is False
    ):
        raise ValueError("graph-context development inventory is ineligible")
    cache_manifest = json.loads(
        (args.cache_root / "probe_cache_manifest.json").read_text(encoding="utf-8")
    )
    verify_cache(args.cache_root, cache_manifest)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("graph-context development probe requires one Antelume GPU")
    gpu_name = torch.cuda.get_device_name(0)
    if args.required_gpu_name.lower() not in gpu_name.lower():
        raise RuntimeError(f"required GPU {args.required_gpu_name!r}, saw {gpu_name!r}")
    terminal, records = validate_policy(args.results_root)
    device = torch.device("cuda:0")
    models = load_models(records, device)
    rows = build_rows(
        inventory, args.cache_root, models, device=device, batch_size=args.batch_size
    )
    result = {
        "schema_version": 1,
        "status": "development_probe_complete",
        "run_id": RUN_ID,
        "gpu_name": gpu_name,
        "inventory_sha256": sha256_file(args.inventory),
        "selection_audit_terminal_sha256": sha256_file(
            args.results_root / "graph_context_division_sweep_terminal.json"
        ),
        "selection_policy": terminal["deployment_policy"],
        "training_run_id": terminal["run_id"],
        "policy_contract": terminal.get("policy_contract"),
        "member_count": len(records),
        "members": [
            {key: value for key, value in row.items() if key != "checkpoint"}
            for row in records
        ],
        "metrics": ranking_metrics(rows, "ensemble_logit"),
        "rows": rows,
        "absolute_threshold_used": False,
        "weights_searched_on_probe": False,
        "model_subset_searched_on_probe": False,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "final_probe_opened": True,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_graph_context_development_evaluation": True,
        "authorized_for_submission": False,
    }
    atomic_json(args.output, result)
    print(json.dumps(result["metrics"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
