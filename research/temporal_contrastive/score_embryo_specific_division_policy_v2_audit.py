#!/usr/bin/env python
"""Score a frozen embryo-specific division policy on the sealed audit movies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.temporal_contrastive import train_real_division_gate as base
from research.temporal_contrastive.calibrate_embryo_specific_division_policy_v2 import (
    EMBRYOS,
    RUN_ID as POLICY_RUN_ID,
    validate_training,
)
from research.temporal_contrastive.multiscale_contextual_pair_fusion import (
    MultiscaleContextualPairFusionAssociationModel,
)
from research.temporal_contrastive.train_real_division_gate_v2 import (
    FOLDS,
    frozen_decisions,
    load_role,
    validate_manifest,
)


RUN_ID = "competition-real-division-embryo-audit-v2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--required-gpu-name", default="A10G")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("embryo-specific audit requires one Antelume GPU")
    gpu_name = torch.cuda.get_device_name(0)
    if args.required_gpu_name.lower() not in gpu_name.lower():
        raise RuntimeError(f"required Antelume GPU {args.required_gpu_name!r}, saw {gpu_name!r}")
    _, model_paths = validate_training(args.training_root)
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    if not (
        policy.get("schema_version") == 1
        and policy.get("status") == "accepted_at_selection"
        and policy.get("run_id") == POLICY_RUN_ID
        and policy.get("selection_gate_passed") is True
        and policy.get("audit_opened") is False
        and policy.get("checkpoint_and_policy_frozen_before_audit") is True
        and policy.get("authorized_for_audit") is True
        and policy.get("model_sha256") == [base.sha256_file(path) for path in model_paths]
        and set(policy.get("policies", {})) == set(EMBRYOS)
    ):
        raise ValueError("embryo-specific policy is ineligible for audit")
    manifest_path = args.data_root / "real_division_hard_negative_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    device = torch.device("cuda:0")
    patches, targets, _, inventory = load_role(args.data_root, manifest, "audit", device)
    models = []
    for path in model_paths:
        model = MultiscaleContextualPairFusionAssociationModel().to(device)
        model.load_state_dict(
            torch.load(path, map_location=device, weights_only=True), strict=True
        )
        models.append(model.requires_grad_(False).eval())
    fold_logits = [base.predict(model, patches, batch_size=args.batch_size) for model in models]
    margin_scores = torch.empty_like(targets)
    by_embryo = {}
    for embryo in EMBRYOS:
        integer_indices = [
            index for index, row in enumerate(inventory) if row["embryo"] == embryo
        ]
        indices = torch.as_tensor(integer_indices, device=device)
        embryo_inventory = [inventory[index] for index in integer_indices]
        frozen = policy["policies"][embryo]["frozen_threshold"]
        weights = policy["policies"][embryo]["ensemble_weights"]
        scores = sum(
            float(weights[fold]) * values[indices]
            for fold, values in zip(FOLDS, fold_logits, strict=True)
        )
        threshold = float(frozen["threshold"])
        margin_scores[indices] = scores - threshold
        by_embryo[embryo] = {
            "ranking": base.threshold_metrics(targets[indices], scores),
            "decisions": frozen_decisions(
                targets[indices], scores, embryo_inventory, threshold
            ),
        }
    ranking = base.threshold_metrics(targets, margin_scores)
    tp = sum(row["decisions"]["tp"] for row in by_embryo.values())
    fp = sum(row["decisions"]["fp"] for row in by_embryo.values())
    fn = sum(row["decisions"]["fn"] for row in by_embryo.values())
    negative_frame_fp = sum(
        row["decisions"]["by_frame_role"]["no_division_hard_negative"]["false_positives"]
        for row in by_embryo.values()
    )
    decisions = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "jaccard": float(tp / max(tp + fp + fn, 1)),
        "no_division_hard_negative_false_positives": negative_frame_fp,
    }
    accepted = bool(
        ranking["average_precision"] >= 0.55
        and decisions["tp"] >= 3
        and decisions["precision"] >= 0.90
        and decisions["jaccard"] > 0.0
        and negative_frame_fp == 0
        and all(row["ranking"]["average_precision"] >= 0.40 for row in by_embryo.values())
        and all(row["decisions"]["tp"] >= 1 for row in by_embryo.values())
    )
    result = {
        "schema_version": 1,
        "status": "accepted" if accepted else "rejected",
        "run_id": RUN_ID,
        "policy_run_id": POLICY_RUN_ID,
        "gpu_name": gpu_name,
        "policy_sha256": base.sha256_file(args.policy),
        "manifest_sha256": base.sha256_file(manifest_path),
        "model_sha256": [base.sha256_file(path) for path in model_paths],
        "ranking_by_frozen_margin": ranking,
        "decisions": decisions,
        "by_embryo": by_embryo,
        "audit_opened": True,
        "audit_opened_after_checkpoint_and_policy_freeze": True,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
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
