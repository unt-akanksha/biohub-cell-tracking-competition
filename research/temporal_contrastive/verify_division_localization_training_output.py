#!/usr/bin/env python
"""Verify accepted dual-GPU division-localization training output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

try:
    from multiscale_division_localization import (
        DIVISION_LOCALIZATION_FAMILY,
        EXPECTED_PARAMETER_COUNT,
        MultiscaleDivisionLocalizationModel,
    )
    from train_dual_fold_division_localization import FOLDS, RUN_ID, sha256_file
except ModuleNotFoundError:
    from research.temporal_contrastive.multiscale_division_localization import (
        DIVISION_LOCALIZATION_FAMILY,
        EXPECTED_PARAMETER_COUNT,
        MultiscaleDivisionLocalizationModel,
    )
    from research.temporal_contrastive.train_dual_fold_division_localization import (
        FOLDS,
        RUN_ID,
        sha256_file,
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def verify_output(root: Path, *, strict_checkpoint: bool = True) -> dict[str, Any]:
    root = root.resolve()
    aggregate_path = root / "localization_terminal.json"
    aggregate = read_json(aggregate_path)
    if not (
        aggregate.get("schema_version") == 1
        and aggregate.get("status") == "completed"
        and aggregate.get("run_id") == RUN_ID
        and aggregate.get("family") == DIVISION_LOCALIZATION_FAMILY
        and aggregate.get("gpu_count") == 2
        and aggregate.get("both_folds_passed") is True
        and set(aggregate.get("folds", {})) == set(FOLDS)
        and aggregate.get("competition_data_read") is False
        and aggregate.get("public_predictions_copied") is False
        and aggregate.get("public_leaderboard_used_for_selection") is False
        and aggregate.get("submission_created") is False
    ):
        raise ValueError("division localization aggregate is ineligible")
    forbidden = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and ("submission" in path.name.lower() or path.suffix.lower() in {".csv", ".zip"})
    ]
    if forbidden:
        raise ValueError(f"localization training emitted forbidden artifacts: {forbidden}")

    verified: dict[str, Any] = {}
    for fold in FOLDS:
        worker_path = root / fold / "worker_terminal.json"
        config_path = root / fold / "training_config.json"
        model_path = root / fold / "localization_model.pt"
        worker = read_json(worker_path)
        config = read_json(config_path)
        initialization = worker.get("initialization", {})
        if not (
            worker == aggregate["folds"][fold]
            and worker.get("status") == "completed"
            and worker.get("run_id") == RUN_ID
            and worker.get("family") == DIVISION_LOCALIZATION_FAMILY
            and worker.get("fold") == fold
            and worker.get("parameter_count") == EXPECTED_PARAMETER_COUNT
            and 0 < int(worker.get("best_step", 0)) <= int(worker.get("completed_step", 0))
            and worker.get("selection_gate_passed") is True
            and worker.get("audit_gate_passed") is True
            and worker.get("audit_opened") is True
            and worker.get("checkpoint_frozen_before_audit") is True
            and worker.get("competition_data_read") is False
            and worker.get("public_predictions_copied") is False
            and worker.get("public_leaderboard_used_for_selection") is False
            and worker.get("submission_created") is False
            and initialization.get("run_id")
            == "zebrahub-multiscale-contextual-pretrain-v1"
            and initialization.get("appearance_family")
            == "temporal_multiscale_contextual_pair_fusion_v4"
            and initialization.get("fold") == fold
            and initialization.get("parent_stage") == "external_pretraining"
            and initialization.get("association_predictions_numerically_preserved") is True
            and config.get("audit_inventory_read") is False
        ):
            raise ValueError(f"division localization worker is ineligible: {fold}")
        if not model_path.is_file() or sha256_file(model_path) != worker.get("model_sha256"):
            raise ValueError(f"division localization checkpoint changed: {fold}")
        if worker.get("train_inventory_sha256") == worker.get("selection_inventory_sha256"):
            raise ValueError(f"localization train and selection inventories collide: {fold}")
        if worker.get("selection_inventory_sha256") == worker.get("audit_inventory_sha256"):
            raise ValueError(f"localization selection and audit inventories collide: {fold}")
        if strict_checkpoint:
            model = MultiscaleDivisionLocalizationModel()
            model.load_state_dict(
                torch.load(model_path, map_location="cpu", weights_only=True),
                strict=True,
            )
            if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETER_COUNT:
                raise ValueError(f"localization checkpoint parameter count changed: {fold}")
        verified[fold] = {
            "model_sha256": worker["model_sha256"],
            "worker_terminal_sha256": sha256_file(worker_path),
            "selection_gate": worker["selection_gate"],
            "audit_gate": worker["audit_gate"],
        }
    return {
        "run_id": RUN_ID,
        "family": DIVISION_LOCALIZATION_FAMILY,
        "gpu_count": 2,
        "folds": verified,
        "terminal_sha256": sha256_file(aggregate_path),
        "authorized_for_graph_evaluation": True,
        "authorized_for_submission": False,
        "competition_submission_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--skip-strict-checkpoint", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify_output(
                args.root, strict_checkpoint=not args.skip_strict_checkpoint
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
