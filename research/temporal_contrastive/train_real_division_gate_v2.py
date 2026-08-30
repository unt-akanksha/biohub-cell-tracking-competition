#!/usr/bin/env python
"""Train a real-domain division gate with no-division hard negatives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.temporal_contrastive import train_real_division_gate as base
from research.temporal_contrastive.multiscale_contextual_pair_fusion import (
    MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
    MultiscaleContextualPairFusionAssociationModel,
)


RUN_ID = "competition-real-division-hard-negative-gate-v2"
FAMILY = "competition_real_hard_negative_temporal_division_gate_v2"
DATA_RUN_ID = "competition-real-division-hard-negative-patches-v2"
FOLDS = base.FOLDS
FRAME_ROLES = ("division", "no_division_hard_negative")


def validate_manifest(manifest: dict[str, Any]) -> None:
    summary = manifest.get("summary", {})
    split = summary.get("by_embryo_role", {})
    strata_valid = all(
        split.get(embryo, {}).get(role, {}).get("division_positives", 0) > 0
        and split.get(embryo, {}).get(role, {}).get("negative_frames", 0) > 0
        for embryo in ("44b6", "6bba")
        for role in ("optimization", "selection", "audit")
    )
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("status") == "complete"
        and manifest.get("run_id") == DATA_RUN_ID
        and manifest.get("audit_opened") is False
        and manifest.get("competition_train_data_read") is True
        and manifest.get("competition_test_data_read") is False
        and manifest.get("public_leaderboard_used_for_selection") is False
        and manifest.get("submission_created") is False
        and manifest.get("authorized_for_submission") is False
        and summary.get("movies") == 199
        and summary.get("excluded_final_probe_movies") == 4
        and summary.get("division_positives") == 146
        and summary.get("negative_frames", 0) >= 150
        and len(manifest.get("final_probe_stems", [])) == 4
        and strata_valid
    ):
        raise ValueError("hard-negative real division manifest is ineligible")


def load_role(
    root: Path,
    manifest: dict[str, Any],
    role: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    if role not in {"optimization", "selection", "audit"}:
        raise ValueError(f"unsupported hard-negative role: {role}")
    patches: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    inventory: list[dict[str, Any]] = []
    records = [record for record in manifest["shards"] if record["role"] == role]
    if not records:
        raise ValueError(f"hard-negative {role} inventory is empty")
    for record in records:
        path = root / str(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or base.sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"hard-negative shard verification failed: {path}")
        with np.load(path, allow_pickle=False) as data:
            source = np.asarray(data["source_patches"], dtype=np.float32)
            target = np.asarray(data["division_target"], dtype=np.float32)
            weight = np.asarray(data["label_weight"], dtype=np.float32)
            metadata = json.loads(str(data["metadata_json"].item()))
        frame_role = str(record["frame_role"])
        negative_weight = 0.50 if frame_role == "no_division_hard_negative" else 0.25
        if not (
            frame_role in FRAME_ROLES
            and source.ndim == 5
            and source.shape[1:] == (3, 17, 17, 17)
            and target.shape == weight.shape == (len(source),)
            and np.isin(target, (0.0, 1.0)).all()
            and np.all(weight[target > 0.5] == 1.0)
            and np.all(weight[target < 0.5] == negative_weight)
            and (
                frame_role != "no_division_hard_negative"
                or not np.any(target > 0.5)
            )
            and metadata.get("run_id") == DATA_RUN_ID
            and metadata.get("stem") == record["stem"]
            and metadata.get("role") == role
            and metadata.get("frame_role") == frame_role
            and metadata.get("competition_test_data_read") is False
        ):
            raise ValueError(f"hard-negative shard contract changed: {path}")
        patches.append(torch.as_tensor(source, device=device))
        targets.append(torch.as_tensor(target, device=device))
        weights.append(torch.as_tensor(weight, device=device))
        inventory.extend(
            {
                "stem": record["stem"],
                "embryo": record["embryo"],
                "timepoint": int(record["timepoint"]),
                "frame_role": frame_role,
                "source_row": index,
            }
            for index in range(len(source))
        )
    result_patches = torch.cat(patches)
    result_targets = torch.cat(targets)
    result_weights = torch.cat(weights)
    if not torch.any(result_targets > 0.5) or not torch.any(result_targets < 0.5):
        raise RuntimeError(f"hard-negative {role} inventory lost a class")
    return result_patches, result_targets, result_weights, inventory


def frozen_decisions(
    targets: torch.Tensor,
    scores: torch.Tensor,
    inventory: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    labels = targets.detach().float().cpu().numpy() > 0.5
    values = scores.detach().float().cpu().numpy()
    selected = values >= float(threshold)
    tp = int(np.sum(selected & labels))
    fp = int(np.sum(selected & ~labels))
    positives = int(labels.sum())
    by_frame_role = {
        frame_role: {
            "rows": int(
                sum(row["frame_role"] == frame_role for row in inventory)
            ),
            "selected": int(
                sum(
                    bool(selected[index]) and row["frame_role"] == frame_role
                    for index, row in enumerate(inventory)
                )
            ),
            "false_positives": int(
                sum(
                    bool(selected[index])
                    and not bool(labels[index])
                    and row["frame_role"] == frame_role
                    for index, row in enumerate(inventory)
                )
            ),
        }
        for frame_role in FRAME_ROLES
    }
    return {
        "threshold": float(threshold),
        "selected": int(selected.sum()),
        "tp": tp,
        "fp": fp,
        "fn": positives - tp,
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(positives, 1)),
        "jaccard": float(tp / max(tp + fp + positives - tp, 1)),
        "by_frame_role": by_frame_role,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--target-44b6-initial-model", type=Path, required=True)
    parser.add_argument("--target-6bba-initial-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-mode", choices=tuple(base.TRAINABLE_PREFIXES), default="focused")
    parser.add_argument("--seed", type=int, default=170_113)
    parser.add_argument("--steps", type=int, default=4_000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--validation-batch-size", type=int, default=96)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--minimum-learning-rate", type=float, default=2e-7)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--required-gpu-name", default="A10G")
    args = parser.parse_args()
    if min(
        args.steps,
        args.batch_size,
        args.validation_batch_size,
        args.validation_every,
        args.log_every,
    ) <= 0:
        raise ValueError("hard-negative training counts must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("hard-negative division training requires one Antelume GPU")
    gpu_name = torch.cuda.get_device_name(0)
    if args.required_gpu_name.lower() not in gpu_name.lower():
        raise RuntimeError(f"required Antelume GPU {args.required_gpu_name!r}, saw {gpu_name!r}")

    manifest_path = args.data_root / "real_division_hard_negative_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    args.output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    train_patches, train_targets, train_weights, train_inventory = load_role(
        args.data_root, manifest, "optimization", device
    )
    selection_patches, selection_targets, _, selection_inventory = load_role(
        args.data_root, manifest, "selection", device
    )
    train_stems = {row["stem"] for row in train_inventory}
    selection_stems = {row["stem"] for row in selection_inventory}
    audit_stems = {
        record["stem"] for record in manifest["shards"] if record["role"] == "audit"
    }
    if train_stems & selection_stems or (train_stems | selection_stems) & audit_stems:
        raise RuntimeError("hard-negative optimization, selection, and audit overlap")

    initial_models = {
        "target_44b6": args.target_44b6_initial_model,
        "target_6bba": args.target_6bba_initial_model,
    }
    initial_hashes = {
        fold: base.sha256_file(path) for fold, path in initial_models.items()
    }
    if len(set(initial_hashes.values())) != 2:
        raise ValueError("hard-negative models require independent initial checkpoints")
    base.RUN_ID = RUN_ID
    base.FAMILY = FAMILY
    started = time.monotonic()
    terminals = {
        fold: base.train_fold(
            fold=fold,
            initial_model_path=initial_models[fold],
            train_patches=train_patches,
            train_targets=train_targets,
            train_weights=train_weights,
            selection_patches=selection_patches,
            selection_targets=selection_targets,
            output_root=args.output_root,
            args=args,
            device=device,
        )
        for fold in FOLDS
    }
    models = []
    for fold in FOLDS:
        model = MultiscaleContextualPairFusionAssociationModel().to(device)
        model.load_state_dict(
            torch.load(
                args.output_root / fold / "division_model.pt",
                map_location=device,
                weights_only=True,
            ),
            strict=True,
        )
        models.append(model.requires_grad_(False).eval())
    fold_logits = [
        base.predict(model, selection_patches, batch_size=args.validation_batch_size)
        for model in models
    ]
    blend = base.select_model_blend(fold_logits, selection_targets)
    frozen = blend["frozen_threshold"]
    ensemble_logits = blend["scores"]
    selection_by_embryo = {}
    for embryo in ("44b6", "6bba"):
        indices = torch.as_tensor(
            [
                index
                for index, row in enumerate(selection_inventory)
                if row["embryo"] == embryo
            ],
            device=device,
        )
        selection_by_embryo[embryo] = base.threshold_metrics(
            selection_targets[indices], ensemble_logits[indices]
        )
    decisions = (
        frozen_decisions(
            selection_targets,
            ensemble_logits,
            selection_inventory,
            frozen["threshold"],
        )
        if frozen is not None
        else None
    )
    accepted = bool(
        frozen is not None
        and decisions is not None
        and frozen["fp"] == 0
        and frozen["tp"] >= 3
        and blend["metrics"]["average_precision"] >= 0.55
        and all(
            metrics["average_precision"] >= 0.40
            for metrics in selection_by_embryo.values()
        )
        and decisions["by_frame_role"]["no_division_hard_negative"]["false_positives"]
        == 0
    )
    terminal = {
        "schema_version": 1,
        "status": "accepted_at_selection" if accepted else "rejected_at_selection",
        "run_id": RUN_ID,
        "family": FAMILY,
        "appearance_family": MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        "gpu_name": gpu_name,
        "execution_gpu_count": 1,
        "execution_policy": "two independent models sequentially on Antelume A10G",
        "elapsed_seconds": time.monotonic() - started,
        "train_mode": args.train_mode,
        "train_rows": len(train_targets),
        "train_positives": int((train_targets > 0.5).sum()),
        "selection_rows": len(selection_targets),
        "selection_positives": int((selection_targets > 0.5).sum()),
        "train_movie_count": len(train_stems),
        "selection_movie_count": len(selection_stems),
        "sealed_audit_movie_count": len(audit_stems),
        "manifest_sha256": base.sha256_file(manifest_path),
        "folds": terminals,
        "ensemble_selection": blend["metrics"],
        "ensemble_weights": blend["weights"],
        "ensemble_weight_grid": blend["grid"],
        "selection_by_embryo": selection_by_embryo,
        "frozen_division_logit_threshold": (
            frozen["threshold"] if frozen is not None else None
        ),
        "threshold_selection": frozen,
        "selection_frozen_decisions": decisions,
        "selection_gate_passed": accepted,
        "audit_opened": False,
        "checkpoint_frozen_before_audit": True,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_audit": accepted,
        "authorized_for_submission": False,
    }
    base.atomic_json(args.output_root / "real_division_hard_negative_terminal.json", terminal)
    print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
