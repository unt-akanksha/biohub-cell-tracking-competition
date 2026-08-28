#!/usr/bin/env python
"""Strictly verify a completed two-GPU multiscale ZebraHub pretraining output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import torch

try:
    from appearance_family import verify_appearance_metadata
    from multiscale_contextual_pair_fusion import (
        EXPECTED_PARAMETER_COUNT,
        MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        MultiscaleContextualPairFusionAssociationModel,
    )
    from train_zebrahub_contextual_pretrain import (
        VALIDATION_PARTITION_POLICY,
        validation_improvement_gate,
    )
except ModuleNotFoundError:
    from research.temporal_contrastive.appearance_family import (
        verify_appearance_metadata,
    )
    from research.temporal_contrastive.multiscale_contextual_pair_fusion import (
        EXPECTED_PARAMETER_COUNT,
        MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        MultiscaleContextualPairFusionAssociationModel,
    )
    from research.temporal_contrastive.train_zebrahub_contextual_pretrain import (
        VALIDATION_PARTITION_POLICY,
        validation_improvement_gate,
    )


RUN_ID = "zebrahub-multiscale-contextual-pretrain-v1"
FOLDS = {"target_44b6", "target_6bba"}
EXPECTED_PARENT_MODEL_SHA256 = {
    "target_44b6": "c817a4ab00fd1318c086d4b5619ba7a8d861fd2ef721492be091c9b3c03a9e14",
    "target_6bba": "633f9600b53e345d412053ea925fd5b37ea0447e9f77aa89ea568562cbde9014",
}
EXPECTED_DATASET_MANIFEST_SHA256 = (
    "b35738f215413f1ece403ba5c0601adea82e2540c65f37e6465de0d0755cb7bf"
)
EXPECTED_SELECTION_INVENTORY_SHA256 = (
    "750d05c1d5a26a4ae878672391de9430d27854a4c454d7b512439ef49424c4bc"
)
EXPECTED_AUDIT_INVENTORY_SHA256 = (
    "518303e2aae0eb536c3d8aa5e89297755f5625d4de225eca33b040db021a1733"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_terminal(root: Path, name: str, run_id: str) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in root.rglob(name):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("run_id") == run_id:
            matches.append((path, payload))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {name} for {run_id}, found {[str(path) for path, _ in matches]}"
        )
    return matches[0]


def recompute_gates(worker: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    selection = validation_improvement_gate(
        worker.get("initial_selection", {}), worker.get("best_selection", {})
    )
    audit = validation_improvement_gate(
        worker.get("initial_audit", {}), worker.get("final_audit", {})
    )
    return selection, audit


def verify_output(root: Path, *, strict_checkpoint: bool = True) -> dict[str, Any]:
    root = root.expanduser().resolve()
    aggregate_path, aggregate = unique_terminal(root, "pretraining_terminal.json", RUN_ID)
    launcher_path, launcher = unique_terminal(root, "launcher_terminal.json", RUN_ID)
    training_root = aggregate_path.parent
    aggregate_hash = sha256_file(aggregate_path)
    if not (
        launcher.get("schema_version") == 1
        and launcher.get("status") == "completed"
        and launcher.get("declared_budget_seconds") == 24_000
        and launcher.get("trainer_max_wall_seconds") == 21_600
        and launcher.get("trainer_hard_stop_seconds") == 22_800
        and launcher.get("pretraining_terminal_exists") is True
        and launcher.get("pretraining_terminal_sha256") == aggregate_hash
        and launcher.get("competition_data_read") is False
        and launcher.get("public_leaderboard_used_for_selection") is False
        and launcher.get("submission_created") is False
    ):
        raise RuntimeError("multiscale launcher evidence is invalid")
    folds = aggregate.get("folds")
    if not (
        aggregate.get("schema_version") == 1
        and aggregate.get("status") == "completed"
        and aggregate.get("appearance_family")
        == MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY
        and aggregate.get("gpu_count") == 2
        and aggregate.get("both_folds_improved") is True
        and isinstance(folds, dict)
        and set(folds) == FOLDS
        and aggregate.get("competition_data_read") is False
        and aggregate.get("public_code_copied") is False
        and aggregate.get("public_predictions_copied") is False
        and aggregate.get("public_leaderboard_used_for_selection") is False
        and aggregate.get("submission_created") is False
    ):
        raise RuntimeError("multiscale aggregate evidence is invalid")

    verified_folds: dict[str, Any] = {}
    for fold in sorted(FOLDS):
        worker_path = training_root / fold / "worker_terminal.json"
        model_path = training_root / fold / "pretrained_model.pt"
        if not worker_path.is_file() or not model_path.is_file():
            raise FileNotFoundError(f"multiscale fold output is incomplete: {fold}")
        worker = json.loads(worker_path.read_text(encoding="utf-8"))
        if worker != folds[fold]:
            raise RuntimeError(f"aggregate and worker evidence diverge: {fold}")
        selection_gate, audit_gate = recompute_gates(worker)
        initialization = worker.get("initialization", {})
        model_hash = sha256_file(model_path)
        if not (
            worker.get("schema_version") == 1
            and worker.get("status") == "completed"
            and worker.get("run_id") == RUN_ID
            and worker.get("fold") == fold
            and verify_appearance_metadata(worker)
            == MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY
            and worker.get("parameter_count") == EXPECTED_PARAMETER_COUNT
            and int(worker.get("completed_step", 0)) == 12_000
            and int(worker.get("best_step", 0)) > 0
            and worker.get("selection_gate") == selection_gate
            and worker.get("audit_gate") == audit_gate
            and worker.get("selection_gate_passed") is True
            and worker.get("audit_gate_passed") is True
            and selection_gate.get("passed") is True
            and audit_gate.get("passed") is True
            and worker.get("validation_partition_policy")
            == VALIDATION_PARTITION_POLICY
            and worker.get("selection_inventory_sha256")
            == EXPECTED_SELECTION_INVENTORY_SHA256
            and worker.get("audit_inventory_sha256")
            == EXPECTED_AUDIT_INVENTORY_SHA256
            and worker.get("dataset_manifest_sha256")
            == EXPECTED_DATASET_MANIFEST_SHA256
            and worker.get("model_sha256") == model_hash
            and worker.get("external_training_source") == "ZSNS004"
            and worker.get("external_validation_source") == "ZSNS005"
            and worker.get("competition_data_read") is False
            and worker.get("public_code_copied") is False
            and worker.get("public_predictions_copied") is False
            and worker.get("public_leaderboard_used_for_selection") is False
            and worker.get("submission_created") is False
            and initialization.get("policy")
            == "accepted contextual v3 plus zero-residual multiscale expansion"
            and initialization.get("run_id") == "zebrahub-contextual-pretrain-v1"
            and initialization.get("source_family")
            == "temporal_contextual_pair_fusion_v3"
            and initialization.get("target_family")
            == MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY
            and initialization.get("model_sha256")
            == EXPECTED_PARENT_MODEL_SHA256[fold]
            and initialization.get("initial_predictions_numerically_preserved") is True
            and int(initialization.get("new_parameter_keys", 0)) > 0
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(initialization.get("new_parameter_keys_sha256", "")),
            )
        ):
            raise RuntimeError(f"multiscale worker evidence is invalid: {fold}")
        if strict_checkpoint:
            model = MultiscaleContextualPairFusionAssociationModel()
            model.load_state_dict(
                torch.load(model_path, map_location="cpu", weights_only=True),
                strict=True,
            )
            if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETER_COUNT:
                raise RuntimeError(f"multiscale checkpoint architecture changed: {fold}")
            del model
        verified_folds[fold] = {
            "model_sha256": model_hash,
            "worker_terminal_sha256": sha256_file(worker_path),
            "best_step": int(worker["best_step"]),
            "selection_composite_gain": float(selection_gate["gains"]["composite"]),
            "audit_composite_gain": float(audit_gate["gains"]["composite"]),
        }
    return {
        "schema_version": 1,
        "status": "verified",
        "run_id": RUN_ID,
        "appearance_family": MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        "gpu_count": 2,
        "training_root": str(training_root),
        "pretraining_terminal_sha256": aggregate_hash,
        "launcher_terminal_sha256": sha256_file(launcher_path),
        "folds": verified_folds,
        "strict_checkpoint_loaded": strict_checkpoint,
        "competition_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--no-strict-checkpoint", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify_output(
                args.root, strict_checkpoint=not args.no_strict_checkpoint
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
