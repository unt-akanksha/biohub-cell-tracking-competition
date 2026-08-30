#!/usr/bin/env python
"""Score the precommitted relational policy on the exact EMA development probe."""

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

from research.temporal_contrastive.relational_division_inference import (
    calibration_free_parent_scores,
)
from research.temporal_contrastive.relational_division_model import (
    RELATIONAL_DIVISION_FAMILY,
    RelationalDivisionModel,
    architecture_contract,
)
from research.temporal_contrastive.train_real_division_gate import sha256_file
from research.temporal_contrastive.patch_model import sample_physical_patches


RUN_ID = "competition-relational-division-development-probe-v1"
TRAIN_RUN_ID = "competition-relational-division-sweep-v1"
INVENTORY_RUN_ID = "competition-relational-division-development-inventory-v1"
INVENTORY_SHA256 = "be8ff10d3355e2918a3480cb30a4de57c39a94edbb854aba5c9447da02ab30ce"
EXPECTED_PARAMETER_COUNT = 48_313_050
GEOMETRY_FIELDS = (
    "parent_distance_um",
    "sister_distance_um",
    "existing_distance_um",
    "daughter_midpoint_distance_um",
    "daughter_opposition_cosine",
    "daughter_step_ratio",
    "biological_geometry_score",
    "parent_velocity_um",
    "constant_velocity_midpoint_error_um",
)
VOXEL_SIZE_ZYX_UM = (1.625, 0.40625, 0.40625)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def verify_cache(cache_root: Path, manifest: dict[str, Any]) -> None:
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("status") == "complete"
        and manifest.get("run_id") == "competition-division-probe-frame-cache-v1"
        and manifest.get("summary", {}).get("movies") == 4
        and manifest.get("summary", {}).get("frames") == 15
        and manifest.get("competition_test_data_read") is False
        and manifest.get("authorized_for_submission") is False
    ):
        raise ValueError("relational development frame cache is ineligible")
    for record in manifest["files"]:
        relative = str(record["local_relative_path"])
        path = cache_root / relative
        if not (
            relative == str(record["remote_path"])
            and relative.startswith("train/")
            and path.is_file()
            and path.stat().st_size == int(record["bytes"])
            and sha256_file(path) == record["sha256"]
        ):
            raise ValueError(f"relational development cache changed: {relative}")


def ranking_metrics(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    labels = np.asarray([row["safe_recovery_positive"] for row in rows], dtype=bool)
    scores = np.asarray([row[score_key] for row in rows], dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    positives = int(labels.sum())
    precision = np.cumsum(labels[order]) / np.arange(1, len(labels) + 1)
    event_ranks = []
    for stem in sorted({str(row["stem"]) for row in rows}):
        local = [row for row in rows if row["stem"] == stem]
        positive = [row for row in local if row["safe_recovery_positive"]]
        if not positive:
            continue
        ranked = sorted(
            local, key=lambda row: (-float(row[score_key]), int(row["parent_id"]))
        )
        rank = 1 + next(
            index
            for index, row in enumerate(ranked)
            if int(row["parent_id"]) == int(positive[0]["parent_id"])
        )
        event_ranks.append(
            {
                "stem": stem,
                "parent_id": int(positive[0]["parent_id"]),
                "rank": rank,
                "candidates": len(local),
                "top_fraction": rank / len(local),
            }
        )
    return {
        "rows": len(rows),
        "positives": positives,
        "average_precision": float(precision[labels[order]].mean()),
        "precision_at_positive_count": float(labels[order[:positives]].mean()),
        "event_ranks": event_ranks,
        "all_positive_events_ranked_first": all(row["rank"] == 1 for row in event_ranks),
    }


def validate_policy(results_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    terminal_path = results_root / "relational_division_sweep_terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    members = [str(value) for value in terminal.get("deployment_members", [])]
    policy = terminal.get("deployment_policy")
    if not (
        terminal.get("schema_version") == 1
        and terminal.get("status") == "completed"
        and terminal.get("run_id") == TRAIN_RUN_ID
        and terminal.get("family") == RELATIONAL_DIVISION_FAMILY
        and terminal.get("planned_model_count") == 8
        and terminal.get("completed_model_count") == 8
        and terminal.get("steps_per_model") == 15_000
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
        and set(members) <= set(terminal.get("independently_strong_members", []))
        and terminal.get("absolute_threshold_used_for_deployment") is False
        and terminal.get("model_subset_searched_on_audit") is False
        and terminal.get("final_probe_opened") is False
        and terminal.get("competition_test_data_read") is False
        and terminal.get("public_code_copied") is False
        and terminal.get("public_predictions_copied") is False
        and terminal.get("public_leaderboard_used_for_selection") is False
        and terminal.get("submission_created") is False
        and terminal.get("authorized_for_submission") is False
        and (
            (policy == "equal_rank_selection_admitted_ensemble" and len(members) >= 2 and terminal.get("ensemble_eligible") is True)
            or (policy == "strongest_selection_individual" and len(members) == 1)
        )
    ):
        raise ValueError("relational deployment policy is ineligible")
    records = []
    for name in members:
        member_root = results_root / name
        worker = json.loads((member_root / "worker_terminal.json").read_text())
        audit = json.loads((member_root / "audit_terminal.json").read_text())
        checkpoint = member_root / "relational_model.pt"
        checkpoint_hash = sha256_file(checkpoint)
        if not (
            worker.get("status") == "accepted_at_selection"
            and worker.get("run_id") == TRAIN_RUN_ID
            and worker.get("family") == RELATIONAL_DIVISION_FAMILY
            and worker.get("member") == name
            and worker.get("parameter_count") == EXPECTED_PARAMETER_COUNT
            and worker.get("selection_gate_passed") is True
            and worker.get("model_sha256") == checkpoint_hash
            and worker.get("audit_opened") is False
            and worker.get("final_probe_opened") is False
            and worker.get("competition_test_data_read") is False
            and worker.get("public_leaderboard_used_for_selection") is False
            and worker.get("submission_created") is False
            and worker.get("authorized_for_audit") is True
            and worker.get("authorized_for_submission") is False
            and audit.get("member") == name
            and audit.get("model_sha256") == checkpoint_hash
            and audit.get("audit_gate_passed") is True
        ):
            raise ValueError(f"relational deployment member is ineligible: {name}")
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
            }
        )
    return terminal, records


def load_models(records: list[dict[str, Any]], device: torch.device) -> list[torch.nn.Module]:
    if architecture_contract()["parameter_count"] != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("relational architecture inventory changed")
    models = []
    for record in records:
        model = RelationalDivisionModel().to(device)
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
    models: list[torch.nn.Module],
    *,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    import zarr

    rows = []
    for movie in inventory["movies"]:
        stem = str(movie["stem"])
        array = zarr.open_group(
            str(cache_root / "train" / f"{stem}.zarr"), mode="r"
        )["0"]
        eligible = [
            row for row in movie["candidates"] if row["inference_geometry_eligible"]
        ]
        member_scores = [dict() for _ in models]
        by_time: dict[int, list[dict[str, Any]]] = {}
        for row in eligible:
            by_time.setdefault(int(row["timepoint"]), []).append(row)
        for timepoint, candidates in sorted(by_time.items()):
            context = np.stack(
                [np.asarray(array[max(0, min(len(array) - 1, timepoint + offset))]) for offset in (-1, 0, 1)]
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
            sampled = sample_physical_patches(
                context,
                centers,
                voxel_size_zyx_um=VOXEL_SIZE_ZYX_UM,
                output_shape_zyx=(17, 17, 17),
                half_extent_zyx_um=(8.0, 8.0, 8.0),
                chunk_size=max(3, batch_size * 3),
            ).reshape(len(candidates), 3, 3, 17, 17, 17)
            geometry = torch.as_tensor(
                np.asarray(
                    [
                        [np.nan if row[name] is None else float(row[name]) for name in GEOMETRY_FIELDS]
                        for row in candidates
                    ],
                    dtype=np.float32,
                )
            )
            for member_index, model in enumerate(models):
                parts = []
                for start in range(0, len(candidates), batch_size):
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        parts.append(
                            model(
                                sampled[start : start + batch_size].to(device),
                                geometry[start : start + batch_size].to(device),
                            ).float().cpu()
                        )
                values = torch.cat(parts).numpy()
                for row, value in zip(candidates, values, strict=True):
                    member_scores[member_index][int(row["parent_id"])] = float(value)
            del sampled, geometry
        parents = sorted(int(row["parent_id"]) for row in eligible)
        consensus = calibration_free_parent_scores(member_scores, parents)
        for row in eligible:
            parent_id = int(row["parent_id"])
            result = dict(row)
            result["stem"] = stem
            result["ensemble_logit"] = consensus[parent_id]
            result["member_logits"] = [scores[parent_id] for scores in member_scores]
            rows.append(result)
    if not (
        len(rows) == 9
        and sum(bool(row["safe_recovery_positive"]) for row in rows) == 3
    ):
        raise RuntimeError("relational development scoring inventory changed")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--required-gpu-name", default="A10G")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("relational probe batch size must be positive")
    if sha256_file(args.inventory) != INVENTORY_SHA256:
        raise ValueError("relational development inventory changed")
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    if not (
        inventory.get("schema_version") == 1
        and inventory.get("status") == "complete"
        and inventory.get("run_id") == INVENTORY_RUN_ID
        and inventory.get("summary", {}).get("rows") == 225
        and inventory.get("summary", {}).get("inference_geometry_eligible_rows") == 9
        and inventory.get("summary", {}).get("inference_geometry_eligible_positives") == 3
        and inventory.get("authorized_for_relational_probe_scoring") is True
        and inventory.get("authorized_for_submission") is False
    ):
        raise ValueError("relational development inventory is ineligible")
    cache_manifest = json.loads(
        (args.cache_root / "probe_cache_manifest.json").read_text(encoding="utf-8")
    )
    verify_cache(args.cache_root, cache_manifest)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("relational development probe requires one Antelume GPU")
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
            args.results_root / "relational_division_sweep_terminal.json"
        ),
        "selection_policy": terminal["deployment_policy"],
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
        "authorized_for_relational_development_evaluation": True,
        "authorized_for_submission": False,
    }
    atomic_json(args.output, result)
    print(json.dumps(result["metrics"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
