#!/usr/bin/env python
"""Assemble two selection-approved focused models and open the audit once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from calibrate_division_recovery_policy import (
        recovery_metrics,
        score_records,
        select_threshold,
        sha256_file,
    )
    from multiscale_contextual_pair_fusion import (
        EXPECTED_PARAMETER_COUNT,
        MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        MultiscaleContextualPairFusionAssociationModel,
    )
    from train_dual_fold_division_localization import (
        AUDIT_TIMEPOINTS,
        SELECTION_TIMEPOINTS,
        discover_shards,
    )
    from train_focused_division_gate import FAMILY, POLICY_RUN_ID, RUN_ID
except ModuleNotFoundError:
    from research.temporal_contrastive.calibrate_division_recovery_policy import (
        recovery_metrics,
        score_records,
        select_threshold,
        sha256_file,
    )
    from research.temporal_contrastive.multiscale_contextual_pair_fusion import (
        EXPECTED_PARAMETER_COUNT,
        MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        MultiscaleContextualPairFusionAssociationModel,
    )
    from research.temporal_contrastive.train_dual_fold_division_localization import (
        AUDIT_TIMEPOINTS,
        SELECTION_TIMEPOINTS,
        discover_shards,
    )
    from research.temporal_contrastive.train_focused_division_gate import (
        FAMILY,
        POLICY_RUN_ID,
        RUN_ID,
    )


OUTPUT_FOLDS = ("target_44b6", "target_6bba")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_selected_source(root: Path, fold: str) -> dict[str, Any]:
    worker_path = root / fold / "worker_terminal.json"
    config_path = root / fold / "training_config.json"
    checkpoint = root / fold / "division_model.pt"
    worker = json.loads(worker_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    digest = sha256_file(checkpoint)
    initialization = worker.get("initialization", {})
    if not (
        worker.get("schema_version") == 1
        and worker.get("status") == "accepted_at_selection"
        and worker.get("run_id") == RUN_ID
        and worker.get("family") == FAMILY
        and worker.get("appearance_family")
        == MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY
        and worker.get("fold") == fold
        and worker.get("parameter_count") == EXPECTED_PARAMETER_COUNT
        and int(worker.get("best_step", 0)) > 0
        and worker.get("selection_gate_passed") is True
        and worker.get("audit_opened") is False
        and worker.get("checkpoint_frozen_before_audit") is True
        and worker.get("model_sha256") == digest
        and initialization.get("parent_association_gate_passed") is False
        and initialization.get("initial_predictions_numerically_preserved") is True
        and initialization.get("authorized_for_focused_division_initialization_only")
        is True
        and config.get("audit_arrays_read") is False
        and worker.get("competition_data_read") is False
        and worker.get("public_code_copied") is False
        and worker.get("public_predictions_copied") is False
        and worker.get("public_leaderboard_used_for_selection") is False
        and worker.get("submission_created") is False
    ):
        raise ValueError(f"focused division source is ineligible: {root}/{fold}")
    return {
        "root": str(root.resolve()),
        "fold": fold,
        "model_sha256": digest,
        "worker_terminal_sha256": sha256_file(worker_path),
        "training_config_sha256": sha256_file(config_path),
        "best_step": int(worker["best_step"]),
        "initial_selection": worker["initial_selection"],
        "best_selection": worker["best_selection"],
        "selection_gate": worker["selection_gate"],
        "initialization": initialization,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-root", type=Path, required=True)
    parser.add_argument("--first-fold", choices=OUTPUT_FOLDS, required=True)
    parser.add_argument("--second-root", type=Path, required=True)
    parser.add_argument("--second-fold", choices=OUTPUT_FOLDS, required=True)
    parser.add_argument("--localization-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--patch-batch-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("focused ensemble assembly requires one Antelume GPU")
    if args.patch_batch_size <= 0:
        raise ValueError("patch batch size must be positive")
    first = validate_selected_source(args.first_root, args.first_fold)
    second = validate_selected_source(args.second_root, args.second_fold)
    if first["model_sha256"] == second["model_sha256"]:
        raise ValueError("focused ensemble models must be independently optimized")
    args.output_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    sources = (first, second)
    source_paths = (
        args.first_root / args.first_fold / "division_model.pt",
        args.second_root / args.second_fold / "division_model.pt",
    )
    models = []
    model_hashes: dict[str, str] = {}
    workers: dict[str, Any] = {}
    device = torch.device("cuda:0")
    for output_fold, source, source_path in zip(
        OUTPUT_FOLDS, sources, source_paths, strict=True
    ):
        fold_root = args.output_root / output_fold
        fold_root.mkdir()
        checkpoint = fold_root / "division_model.pt"
        shutil.copy2(source_path, checkpoint)
        digest = sha256_file(checkpoint)
        if digest != source["model_sha256"]:
            raise RuntimeError("focused model changed during ensemble staging")
        model = MultiscaleContextualPairFusionAssociationModel().to(device)
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True), strict=True
        )
        if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETER_COUNT:
            raise RuntimeError("focused ensemble parameter inventory changed")
        models.append(model.requires_grad_(False).eval())
        model_hashes[output_fold] = digest
        workers[output_fold] = {
            "schema_version": 1,
            "status": "accepted_at_selection",
            "run_id": RUN_ID,
            "family": FAMILY,
            "appearance_family": MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
            "fold": output_fold,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "best_step": source["best_step"],
            "initial_selection": source["initial_selection"],
            "best_selection": source["best_selection"],
            "selection_gate": source["selection_gate"],
            "selection_gate_passed": True,
            "audit_opened": False,
            "checkpoint_frozen_before_audit": True,
            "model_sha256": digest,
            "source_selection_approved_model": source,
            "competition_data_read": False,
            "public_code_copied": False,
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
        }
        atomic_json(fold_root / "worker_terminal.json", workers[output_fold])

    selection_records = discover_shards(
        args.localization_root / "selection",
        expected_source="ZSNS005",
        expected_role="external_validation",
        expected_timepoints=SELECTION_TIMEPOINTS,
    )
    selection_rows = score_records(
        models,
        selection_records,
        device,
        patch_batch_size=args.patch_batch_size,
    )
    threshold, selection = select_threshold(selection_rows)
    selection_inventory = {(row["shard"], row["source_row"]) for row in selection_rows}
    del selection_rows

    audit_records = discover_shards(
        args.localization_root / "audit",
        expected_source="ZSNS005",
        expected_role="external_validation",
        expected_timepoints=AUDIT_TIMEPOINTS,
    )
    audit_rows = score_records(
        models,
        audit_records,
        device,
        patch_batch_size=args.patch_batch_size,
    )
    audit_inventory = {(row["shard"], row["source_row"]) for row in audit_rows}
    if selection_inventory & audit_inventory:
        raise RuntimeError("focused ensemble selection and audit rows overlap")
    audit = recovery_metrics(audit_rows, threshold)
    accepted = bool(
        int(audit["division_tp"]) > 0
        and float(audit["edge_precision"]) >= 0.75
        and float(audit["recovery_composite"]) > 0.0
    )
    policy = {
        "schema_version": 1,
        "status": "accepted" if accepted else "rejected",
        "run_id": POLICY_RUN_ID,
        "model_training_run_id": RUN_ID,
        "appearance_family": MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        "focused_division_family": FAMILY,
        "model_sha256": model_hashes,
        "ensemble": "mean of two independently optimized selection-approved focused models",
        "selection_policy": (
            "maximize external recovery composite with >=0.80 second-edge "
            "precision; tie-break by division Jaccard, fewer decisions, threshold"
        ),
        "frozen_division_logit_threshold": threshold,
        "selection": selection,
        "audit": audit,
        "selection_timepoints": list(SELECTION_TIMEPOINTS),
        "audit_timepoints": list(AUDIT_TIMEPOINTS),
        "audit_opened_after_threshold_freeze": True,
        "competition_data_read": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_competition_graph_evaluation": accepted,
        "authorized_for_submission": False,
    }
    atomic_json(args.output_root / "division-recovery-policy.json", policy)
    terminal = {
        "schema_version": 1,
        "status": "accepted" if accepted else "rejected_at_audit",
        "run_id": RUN_ID,
        "family": FAMILY,
        "appearance_family": MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        "execution_gpu_count": 1,
        "execution_policy": "selection-approved cross-run ensemble assembled on Antelume A10G",
        "folds": workers,
        "both_folds_selected": True,
        "audit_opened": True,
        "audit_opened_after_threshold_freeze": True,
        "frozen_division_logit_threshold": threshold,
        "selection": selection,
        "audit": audit,
        "model_sha256": model_hashes,
        "elapsed_seconds": time.monotonic() - started,
        "authorized_for_competition_graph_evaluation": accepted,
        "authorized_for_submission": False,
        "competition_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    atomic_json(args.output_root / "focused_division_gate_terminal.json", terminal)
    print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
