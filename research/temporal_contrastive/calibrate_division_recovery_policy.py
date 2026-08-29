#!/usr/bin/env python
"""Freeze a second-daughter recovery policy on external ZebraHub windows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from multiscale_contextual_pair_fusion import (
        EXPECTED_PARAMETER_COUNT,
        MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        MultiscaleContextualPairFusionAssociationModel,
    )
    from train_zebrahub_contextual_pretrain import (
        discover_shards,
        encode_patches,
        load_shard,
        partition_validation_records,
    )
    from verify_multiscale_pretraining_output import verify_output
except ModuleNotFoundError:
    from research.temporal_contrastive.multiscale_contextual_pair_fusion import (
        EXPECTED_PARAMETER_COUNT,
        MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        MultiscaleContextualPairFusionAssociationModel,
    )
    from research.temporal_contrastive.train_zebrahub_contextual_pretrain import (
        discover_shards,
        encode_patches,
        load_shard,
        partition_validation_records,
    )
    from research.temporal_contrastive.verify_multiscale_pretraining_output import (
        verify_output,
    )


RUN_ID = "external-division-recovery-policy-v1"
FOLDS = ("target_44b6", "target_6bba")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recovery_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    decisions = [row for row in rows if float(row["division_logit"]) >= threshold]
    truth_events = sum(bool(row["is_division"]) for row in rows)
    event_tp = sum(bool(row["is_division"]) for row in decisions)
    event_fp = len(decisions) - event_tp
    event_fn = truth_events - event_tp
    edge_tp = sum(bool(row["second_is_positive"]) for row in decisions)
    edge_fp = len(decisions) - edge_tp
    edge_fn = truth_events - edge_tp
    division_tp = sum(bool(row["top_two_are_daughters"]) for row in decisions)
    division_fp = len(decisions) - division_tp
    division_fn = truth_events - division_tp

    def jaccard(tp: int, fp: int, fn: int) -> float:
        denominator = tp + fp + fn
        return tp / denominator if denominator else 0.0

    edge_jaccard = jaccard(edge_tp, edge_fp, edge_fn)
    division_jaccard = jaccard(division_tp, division_fp, division_fn)
    event_jaccard = jaccard(event_tp, event_fp, event_fn)
    return {
        "threshold": float(threshold),
        "rows": len(rows),
        "truth_events": truth_events,
        "decisions": len(decisions),
        "event_tp": event_tp,
        "event_fp": event_fp,
        "event_fn": event_fn,
        "event_precision": event_tp / len(decisions) if decisions else 1.0,
        "event_jaccard": event_jaccard,
        "edge_tp": edge_tp,
        "edge_fp": edge_fp,
        "edge_fn": edge_fn,
        "edge_precision": edge_tp / len(decisions) if decisions else 1.0,
        "edge_jaccard": edge_jaccard,
        "division_tp": division_tp,
        "division_fp": division_fp,
        "division_fn": division_fn,
        "division_precision": division_tp / len(decisions) if decisions else 1.0,
        "division_jaccard": division_jaccard,
        "recovery_composite": edge_jaccard + 0.10 * division_jaccard,
    }


def threshold_candidates(rows: list[dict[str, Any]]) -> list[float]:
    values = sorted({float(row["division_logit"]) for row in rows})
    if not values:
        raise ValueError("division recovery selection rows are empty")
    boundaries = [float("-inf"), float("inf")]
    boundaries.extend((left + right) / 2.0 for left, right in zip(values, values[1:]))
    boundaries.extend(values)
    return sorted(set(boundaries))


def select_threshold(rows: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    candidates = [recovery_metrics(rows, value) for value in threshold_candidates(rows)]
    eligible = [
        row
        for row in candidates
        if int(row["decisions"]) > 0
        and float(row["edge_precision"]) >= 0.80
        and int(row["division_tp"]) > 0
    ]
    if not eligible:
        raise RuntimeError("no external division-recovery threshold passed precision")
    selected = max(
        eligible,
        key=lambda row: (
            float(row["recovery_composite"]),
            float(row["division_jaccard"]),
            -int(row["decisions"]),
            float(row["threshold"]),
        ),
    )
    return float(selected["threshold"]), selected


def load_models(root: Path, device: torch.device) -> tuple[list[torch.nn.Module], dict[str, str]]:
    verified = verify_output(root, strict_checkpoint=False)
    if (
        verified.get("status") != "verified"
        or verified.get("appearance_family")
        != MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY
    ):
        raise ValueError("v4 pretraining output is ineligible for policy calibration")
    models: list[torch.nn.Module] = []
    hashes: dict[str, str] = {}
    training_root = Path(str(verified["training_root"]))
    for fold in FOLDS:
        path = training_root / fold / "pretrained_model.pt"
        model = MultiscaleContextualPairFusionAssociationModel().to(device)
        model.load_state_dict(
            torch.load(path, map_location=device, weights_only=True), strict=True
        )
        if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETER_COUNT:
            raise RuntimeError("v4 recovery model parameter inventory changed")
        model.requires_grad_(False).eval()
        models.append(model)
        hashes[fold] = sha256_file(path)
    return models, hashes


@torch.inference_mode()
def score_records(
    models: list[torch.nn.Module],
    records: list[Any],
    device: torch.device,
    *,
    patch_batch_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        batch = load_shard(record.path, device)
        fold_logits: list[torch.Tensor] = []
        fold_divisions: list[torch.Tensor] = []
        for model in models:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                source, divisions = encode_patches(
                    model,
                    batch["source_patches"],
                    batch_size=patch_batch_size,
                )
                target, _target_divisions = encode_patches(
                    model,
                    batch["target_patches"],
                    batch_size=patch_batch_size,
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
            fold_logits.append(logits.float())
            fold_divisions.append(divisions.float())
        logits = torch.stack(fold_logits).mean(dim=0).cpu().numpy()
        divisions = torch.stack(fold_divisions).mean(dim=0).cpu().numpy()
        candidates = batch["candidate_mask"].cpu().numpy()
        positives = batch["positive_mask"].cpu().numpy()
        division_targets = batch["division_target"].cpu().numpy() > 0.5
        source_coordinates = batch["source_coords_um"].cpu().numpy()
        target_coordinates = batch["target_coords_um"].cpu().numpy()
        for source_row in range(len(divisions)):
            candidate_rows = np.flatnonzero(candidates[source_row])
            if len(candidate_rows) < 2:
                continue
            ranked_by_model = candidate_rows[
                np.argsort(logits[source_row, candidate_rows], kind="stable")[::-1]
            ]
            existing_child = int(ranked_by_model[0])
            remaining = candidate_rows[candidate_rows != existing_child]
            distances = np.linalg.norm(
                target_coordinates[remaining] - source_coordinates[source_row],
                axis=1,
            )
            second_child = int(
                remaining[np.lexsort((remaining, distances))[0]]
            )
            positive = positives[source_row]
            rows.append(
                {
                    "shard": record.path.name,
                    "csv_timepoint": int(record.csv_timepoint),
                    "source_row": int(source_row),
                    "division_logit": float(divisions[source_row]),
                    "existing_pair_logit": float(
                        logits[source_row, existing_child]
                    ),
                    "second_pair_logit": float(logits[source_row, second_child]),
                    "second_parent_distance_um": float(
                        np.linalg.norm(
                            target_coordinates[second_child]
                            - source_coordinates[source_row]
                        )
                    ),
                    "is_division": bool(division_targets[source_row]),
                    "existing_is_positive": bool(positive[existing_child]),
                    "second_is_positive": bool(positive[second_child]),
                    "top_two_are_daughters": bool(
                        division_targets[source_row]
                        and positive[existing_child]
                        and positive[second_child]
                    ),
                }
            )
        del batch
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretraining-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-batch-size", type=int, default=32)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("external division policy calibration requires one GPU")
    if args.patch_batch_size <= 0:
        raise ValueError("patch batch size must be positive")
    device = torch.device("cuda:0")
    records = discover_shards(
        args.data_root / "validation",
        expected_source="ZSNS005",
        expected_role="external_validation",
    )
    selection_records, audit_records = partition_validation_records(records)
    models, model_hashes = load_models(args.pretraining_root, device)
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
    audit_rows = score_records(
        models,
        audit_records,
        device,
        patch_batch_size=args.patch_batch_size,
    )
    audit_inventory = {(row["shard"], row["source_row"]) for row in audit_rows}
    if selection_inventory & audit_inventory:
        raise RuntimeError("external policy selection and audit rows overlap")
    audit = recovery_metrics(audit_rows, threshold)
    accepted = bool(
        int(audit["division_tp"]) > 0
        and float(audit["edge_precision"]) >= 0.75
        and float(audit["recovery_composite"]) > 0.0
    )
    result = {
        "schema_version": 1,
        "status": "accepted" if accepted else "rejected",
        "run_id": RUN_ID,
        "appearance_family": MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        "model_sha256": model_hashes,
        "ensemble": "mean of two independently trained fold logits",
        "selection_policy": (
            "maximize external recovery composite with >=0.80 second-edge "
            "precision; tie-break by division Jaccard, fewer decisions, threshold"
        ),
        "frozen_division_logit_threshold": threshold,
        "selection": selection,
        "audit": audit,
        "audit_opened_after_threshold_freeze": True,
        "competition_data_read": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_competition_graph_evaluation": accepted,
        "authorized_for_submission": False,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(args.output)
    print(rendered, end="")
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
