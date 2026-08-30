#!/usr/bin/env python
"""Open the sealed movie-disjoint audit for a frozen hard-negative gate once."""

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

from research.temporal_contrastive import train_real_division_gate as base
from research.temporal_contrastive.multiscale_contextual_pair_fusion import (
    MultiscaleContextualPairFusionAssociationModel,
)
from research.temporal_contrastive.train_real_division_gate_v2 import (
    DATA_RUN_ID,
    FAMILY,
    FOLDS,
    RUN_ID as TRAINING_RUN_ID,
    frozen_decisions,
    load_role,
    validate_manifest,
)


RUN_ID = "competition-real-division-hard-negative-audit-v2"


def validate_terminal(
    terminal: dict[str, Any], model_paths: list[Path]
) -> tuple[float, list[float]]:
    folds = terminal.get("folds", {})
    weights = terminal.get("ensemble_weights", {})
    threshold = terminal.get("frozen_division_logit_threshold")
    hashes = [base.sha256_file(path) for path in model_paths]
    if not (
        len(model_paths) == 2
        and terminal.get("schema_version") == 1
        and terminal.get("status") == "accepted_at_selection"
        and terminal.get("run_id") == TRAINING_RUN_ID
        and terminal.get("family") == FAMILY
        and terminal.get("selection_gate_passed") is True
        and terminal.get("audit_opened") is False
        and terminal.get("checkpoint_frozen_before_audit") is True
        and terminal.get("authorized_for_audit") is True
        and terminal.get("competition_test_data_read") is False
        and terminal.get("public_leaderboard_used_for_selection") is False
        and terminal.get("submission_created") is False
        and terminal.get("authorized_for_submission") is False
        and isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
        and np.isfinite(threshold)
        and isinstance(weights, dict)
        and set(weights) == set(FOLDS)
        and all(
            isinstance(weights[fold], (int, float))
            and not isinstance(weights[fold], bool)
            and np.isfinite(weights[fold])
            and float(weights[fold]) >= 0.0
            for fold in FOLDS
        )
        and np.isclose(sum(float(weights[fold]) for fold in FOLDS), 1.0)
        and [folds.get(fold, {}).get("model_sha256") for fold in FOLDS] == hashes
    ):
        raise ValueError("hard-negative training terminal is ineligible for audit")
    return float(threshold), [float(weights[fold]) for fold in FOLDS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--training-terminal", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--required-gpu-name", default="A10G")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("audit batch size must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("hard-negative audit requires one Antelume GPU")
    gpu_name = torch.cuda.get_device_name(0)
    if args.required_gpu_name.lower() not in gpu_name.lower():
        raise RuntimeError(f"required Antelume GPU {args.required_gpu_name!r}, saw {gpu_name!r}")
    terminal = json.loads(args.training_terminal.read_text(encoding="utf-8"))
    threshold, weights = validate_terminal(terminal, args.model_path)
    manifest_path = args.data_root / "real_division_hard_negative_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    if manifest.get("run_id") != DATA_RUN_ID:
        raise ValueError("hard-negative audit data run changed")
    device = torch.device("cuda:0")
    patches, targets, _, inventory = load_role(
        args.data_root, manifest, "audit", device
    )
    models = []
    for path in args.model_path:
        model = MultiscaleContextualPairFusionAssociationModel().to(device)
        model.load_state_dict(
            torch.load(path, map_location=device, weights_only=True), strict=True
        )
        models.append(model.requires_grad_(False).eval())
    fold_logits = [base.predict(model, patches, batch_size=args.batch_size) for model in models]
    scores = sum(weight * values for weight, values in zip(weights, fold_logits, strict=True))
    metrics = base.threshold_metrics(targets, scores)
    decisions = frozen_decisions(targets, scores, inventory, threshold)
    by_embryo = {}
    for embryo in ("44b6", "6bba"):
        indices = torch.as_tensor(
            [index for index, row in enumerate(inventory) if row["embryo"] == embryo],
            device=device,
        )
        embryo_metrics = base.threshold_metrics(targets[indices], scores[indices])
        embryo_inventory = [row for row in inventory if row["embryo"] == embryo]
        embryo_decisions = frozen_decisions(
            targets[indices], scores[indices], embryo_inventory, threshold
        )
        by_embryo[embryo] = {
            "ranking": embryo_metrics,
            "decisions": embryo_decisions,
        }
    accepted = bool(
        metrics["average_precision"] >= 0.55
        and decisions["tp"] >= 3
        and decisions["precision"] >= 0.90
        and decisions["jaccard"] > 0.0
        and decisions["by_frame_role"]["no_division_hard_negative"]["false_positives"]
        == 0
        and all(row["ranking"]["average_precision"] >= 0.40 for row in by_embryo.values())
        and all(row["decisions"]["tp"] >= 1 for row in by_embryo.values())
    )
    result = {
        "schema_version": 1,
        "status": "accepted" if accepted else "rejected",
        "run_id": RUN_ID,
        "model_training_run_id": TRAINING_RUN_ID,
        "gpu_name": gpu_name,
        "training_terminal_sha256": base.sha256_file(args.training_terminal),
        "manifest_sha256": base.sha256_file(manifest_path),
        "model_sha256": [base.sha256_file(path) for path in args.model_path],
        "ensemble_weights_frozen_before_audit": {
            fold: weight for fold, weight in zip(FOLDS, weights, strict=True)
        },
        "model_threshold_frozen_before_audit": threshold,
        "ranking": metrics,
        "decisions": decisions,
        "by_embryo": by_embryo,
        "audit_opened": True,
        "audit_opened_after_checkpoint_and_policy_freeze": True,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_competition_graph_evaluation": accepted,
        "authorized_for_submission": False,
    }
    base.atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
