#!/usr/bin/env python
"""Freeze embryo-specific blends and thresholds on hard-negative selection data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.temporal_contrastive import train_real_division_gate as base
from research.temporal_contrastive.multiscale_contextual_pair_fusion import (
    MultiscaleContextualPairFusionAssociationModel,
)
from research.temporal_contrastive.train_real_division_gate_v2 import (
    FAMILY,
    FOLDS,
    RUN_ID as TRAINING_RUN_ID,
    frozen_decisions,
    load_role,
    validate_manifest,
)


RUN_ID = "competition-real-division-embryo-policy-v2"
EMBRYOS = ("44b6", "6bba")


def validate_training(root: Path) -> tuple[dict[str, Any], list[Path]]:
    terminal_path = root / "real_division_hard_negative_terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    paths = [root / fold / "division_model.pt" for fold in FOLDS]
    if not (
        terminal.get("schema_version") == 1
        and terminal.get("status") in {"accepted_at_selection", "rejected_at_selection"}
        and terminal.get("run_id") == TRAINING_RUN_ID
        and terminal.get("family") == FAMILY
        and terminal.get("audit_opened") is False
        and terminal.get("checkpoint_frozen_before_audit") is True
        and terminal.get("competition_test_data_read") is False
        and terminal.get("public_leaderboard_used_for_selection") is False
        and terminal.get("submission_created") is False
        and terminal.get("authorized_for_submission") is False
        and [base.sha256_file(path) for path in paths]
        == [terminal.get("folds", {}).get(fold, {}).get("model_sha256") for fold in FOLDS]
    ):
        raise ValueError("hard-negative checkpoints are ineligible for embryo calibration")
    return terminal, paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--required-gpu-name", default="A10G")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("calibration batch size must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("embryo calibration requires one Antelume GPU")
    gpu_name = torch.cuda.get_device_name(0)
    if args.required_gpu_name.lower() not in gpu_name.lower():
        raise RuntimeError(f"required Antelume GPU {args.required_gpu_name!r}, saw {gpu_name!r}")
    training, model_paths = validate_training(args.training_root)
    manifest_path = args.data_root / "real_division_hard_negative_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    device = torch.device("cuda:0")
    patches, targets, _, inventory = load_role(
        args.data_root, manifest, "selection", device
    )
    models = []
    for path in model_paths:
        model = MultiscaleContextualPairFusionAssociationModel().to(device)
        model.load_state_dict(
            torch.load(path, map_location=device, weights_only=True), strict=True
        )
        models.append(model.requires_grad_(False).eval())
    fold_logits = [base.predict(model, patches, batch_size=args.batch_size) for model in models]
    policies = {}
    accepted = True
    for embryo in EMBRYOS:
        integer_indices = [
            index for index, row in enumerate(inventory) if row["embryo"] == embryo
        ]
        indices = torch.as_tensor(integer_indices, device=device)
        embryo_inventory = [inventory[index] for index in integer_indices]
        blend = base.select_model_blend(
            [values[indices] for values in fold_logits], targets[indices]
        )
        frozen = blend["frozen_threshold"]
        decisions = (
            frozen_decisions(
                targets[indices],
                blend["scores"],
                embryo_inventory,
                frozen["threshold"],
            )
            if frozen is not None
            else None
        )
        passed = bool(
            frozen is not None
            and decisions is not None
            and blend["metrics"]["average_precision"] >= 0.40
            and frozen["fp"] == 0
            and frozen["tp"] >= 2
            and decisions["by_frame_role"]["no_division_hard_negative"]["false_positives"]
            == 0
        )
        policies[embryo] = {
            "status": "accepted" if passed else "rejected",
            "ensemble_weights": blend["weights"],
            "ensemble_weight_grid": blend["grid"],
            "ranking": blend["metrics"],
            "frozen_threshold": frozen,
            "frozen_decisions": decisions,
        }
        accepted = accepted and passed
    result = {
        "schema_version": 1,
        "status": "accepted_at_selection" if accepted else "rejected_at_selection",
        "run_id": RUN_ID,
        "model_training_run_id": TRAINING_RUN_ID,
        "gpu_name": gpu_name,
        "training_terminal_sha256": base.sha256_file(
            args.training_root / "real_division_hard_negative_terminal.json"
        ),
        "manifest_sha256": base.sha256_file(manifest_path),
        "model_sha256": [base.sha256_file(path) for path in model_paths],
        "policies": policies,
        "selection_gate_passed": accepted,
        "audit_opened": False,
        "checkpoint_and_policy_frozen_before_audit": True,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_audit": accepted,
        "authorized_for_submission": False,
    }
    base.atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
