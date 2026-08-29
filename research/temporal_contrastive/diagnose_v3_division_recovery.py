#!/usr/bin/env python
"""Diagnose division-recovery separability of frozen contextual-v3 folds.

This is an evidence-only control for the v4 recovery lane.  It deliberately
cannot authorize competition evaluation or submission because the external
audit partition was already opened by the original v3 training run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from calibrate_division_recovery_policy import (
        FOLDS,
        recovery_metrics,
        score_records,
        select_threshold,
        sha256_file,
    )
    from contextual_pair_fusion import (
        CONTEXTUAL_PAIR_FUSION_FAMILY,
        EXPECTED_PARAMETER_COUNT,
        ContextualPairFusionAssociationModel,
    )
    from train_zebrahub_contextual_pretrain import (
        discover_shards,
        partition_validation_records,
    )
except ModuleNotFoundError:
    from research.temporal_contrastive.calibrate_division_recovery_policy import (
        FOLDS,
        recovery_metrics,
        score_records,
        select_threshold,
        sha256_file,
    )
    from research.temporal_contrastive.contextual_pair_fusion import (
        CONTEXTUAL_PAIR_FUSION_FAMILY,
        EXPECTED_PARAMETER_COUNT,
        ContextualPairFusionAssociationModel,
    )
    from research.temporal_contrastive.train_zebrahub_contextual_pretrain import (
        discover_shards,
        partition_validation_records,
    )


RUN_ID = "contextual-v3-division-recovery-diagnostic-v1"


def load_v3_models(
    root: Path, device: torch.device
) -> tuple[list[torch.nn.Module], dict[str, str]]:
    terminal_path = root / "pretraining_terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if not (
        terminal.get("status") == "completed"
        and terminal.get("both_folds_improved") is True
        and terminal.get("appearance_family") == CONTEXTUAL_PAIR_FUSION_FAMILY
        and terminal.get("competition_data_read") is False
        and terminal.get("public_leaderboard_used_for_selection") is False
        and terminal.get("public_predictions_copied") is False
        and terminal.get("submission_created") is False
    ):
        raise ValueError("contextual-v3 training evidence is ineligible")

    models: list[torch.nn.Module] = []
    hashes: dict[str, str] = {}
    for fold in FOLDS:
        fold_terminal = terminal.get("folds", {}).get(fold, {})
        path = root / fold / "pretrained_model.pt"
        digest = sha256_file(path)
        if not (
            fold_terminal.get("status") == "completed"
            and fold_terminal.get("selection_gate_passed") is True
            and fold_terminal.get("audit_gate_passed") is True
            and fold_terminal.get("model_sha256") == digest
        ):
            raise ValueError(f"contextual-v3 fold evidence is ineligible: {fold}")
        model = ContextualPairFusionAssociationModel().to(device)
        model.load_state_dict(
            torch.load(path, map_location=device, weights_only=True), strict=True
        )
        if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETER_COUNT:
            raise RuntimeError("contextual-v3 parameter inventory changed")
        model.requires_grad_(False).eval()
        models.append(model)
        hashes[fold] = digest
    return models, hashes


def diagnostic_result(
    *,
    model_hashes: dict[str, str],
    threshold: float,
    selection: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "diagnostic_only",
        "run_id": RUN_ID,
        "appearance_family": CONTEXTUAL_PAIR_FUSION_FAMILY,
        "model_sha256": model_hashes,
        "ensemble": "mean of two independently trained fold logits",
        "frozen_division_logit_threshold": float(threshold),
        "selection": selection,
        "audit": audit,
        "audit_reuse_disclosure": (
            "the v3 training run already opened this disjoint audit partition; "
            "results are directional evidence only"
        ),
        "competition_data_read": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_competition_graph_evaluation": False,
        "authorized_for_submission": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretraining-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-batch-size", type=int, default=32)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("v3 division diagnostic requires exactly one GPU")
    if args.patch_batch_size <= 0:
        raise ValueError("patch batch size must be positive")

    device = torch.device("cuda:0")
    records = discover_shards(
        args.data_root / "validation",
        expected_source="ZSNS005",
        expected_role="external_validation",
    )
    selection_records, audit_records = partition_validation_records(records)
    models, model_hashes = load_v3_models(args.pretraining_root, device)
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
        raise RuntimeError("diagnostic selection and audit rows overlap")
    result = diagnostic_result(
        model_hashes=model_hashes,
        threshold=threshold,
        selection=selection,
        audit=recovery_metrics(audit_rows, threshold),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
