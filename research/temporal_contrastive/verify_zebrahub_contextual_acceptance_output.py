#!/usr/bin/env python
"""Independently verify downloaded one-shot ZSNS001 acceptance output."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

try:
    from contextual_pair_fusion import (
        CONTEXTUAL_PAIR_FUSION_FAMILY,
        EXPECTED_PARAMETER_COUNT,
        ContextualPairFusionAssociationModel,
    )
    from evaluate_zebrahub_contextual_acceptance import (
        ACCEPTANCE_ROLE,
        ACCEPTANCE_SHARDS,
        ACCEPTANCE_SOURCE,
        EXPECTED_ACCEPTANCE_INVENTORY_SHA256,
        EXPECTED_ACCEPTANCE_MANIFEST_SHA256,
        EXPECTED_ACCEPTANCE_RECORD_INVENTORY_SHA256,
        FOLDS,
        FOLD_SEEDS,
        PRETRAINING_RUN_ID,
        RUN_ID,
        seed_initialization,
        sha256_file,
        state_dict_sha256,
        verify_pretraining_source,
    )
    from train_zebrahub_contextual_pretrain import validation_improvement_gate
except ModuleNotFoundError:
    from research.temporal_contrastive.contextual_pair_fusion import (
        CONTEXTUAL_PAIR_FUSION_FAMILY,
        EXPECTED_PARAMETER_COUNT,
        ContextualPairFusionAssociationModel,
    )
    from research.temporal_contrastive.evaluate_zebrahub_contextual_acceptance import (
        ACCEPTANCE_ROLE,
        ACCEPTANCE_SHARDS,
        ACCEPTANCE_SOURCE,
        EXPECTED_ACCEPTANCE_INVENTORY_SHA256,
        EXPECTED_ACCEPTANCE_MANIFEST_SHA256,
        EXPECTED_ACCEPTANCE_RECORD_INVENTORY_SHA256,
        FOLDS,
        FOLD_SEEDS,
        PRETRAINING_RUN_ID,
        RUN_ID,
        seed_initialization,
        sha256_file,
        state_dict_sha256,
        verify_pretraining_source,
    )
    from research.temporal_contrastive.train_zebrahub_contextual_pretrain import (
        validation_improvement_gate,
    )


PROHIBITED_OUTPUT_SUFFIXES = frozenset({".csv", ".ckpt", ".npz", ".pt", ".pth"})
METRIC_NAMES = ("composite", "top1", "mrr", "division_top2")
COUNT_NAMES = ("rows", "division_rows", "transitions")


def read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required acceptance artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"acceptance artifact is not a JSON object: {path}")
    return payload


def unique_path(root: Path, name: str) -> Path:
    matches = sorted(path for path in root.rglob(name) if path.is_file())
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {name}, found "
            f"{[path.relative_to(root).as_posix() for path in matches]}"
        )
    return matches[0]


def terminal_with_hash(root: Path, expected_sha256: str) -> Path:
    matches = sorted(
        path
        for path in root.rglob("acceptance_terminal.json")
        if path.is_file() and sha256_file(path) == expected_sha256
    )
    if len(matches) != 1:
        raise ValueError("launcher does not bind exactly one acceptance aggregate")
    return matches[0]


def finite_metrics(payload: Any, *, fold: str, stage: str) -> dict[str, float | int]:
    if not isinstance(payload, dict):
        raise ValueError(f"{fold} {stage} metrics are missing")
    result: dict[str, float | int] = {}
    for name in METRIC_NAMES:
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{fold} {stage} metric is malformed: {name}")
        parsed = float(value)
        if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
            raise ValueError(f"{fold} {stage} metric is outside [0, 1]: {name}")
        result[name] = parsed
    for name in COUNT_NAMES:
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{fold} {stage} inventory is malformed: {name}")
        result[name] = value
    return result


def initial_state_hash(seed: int) -> str:
    seed_initialization(seed, torch.device("cpu"))
    model = ContextualPairFusionAssociationModel()
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("acceptance baseline parameter count changed")
    return state_dict_sha256(model)


def strict_load_checkpoint(path: Path) -> None:
    model = ContextualPairFusionAssociationModel()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("accepted pretraining checkpoint architecture changed")


def verify_fold(
    *,
    fold: str,
    aggregate_row: Any,
    aggregate_root: Path,
    pretraining: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(aggregate_row, dict):
        raise ValueError(f"acceptance aggregate fold is malformed: {fold}")
    terminal_path = aggregate_root / fold / "acceptance_terminal.json"
    terminal = read_object(terminal_path)
    terminal_hash = sha256_file(terminal_path)
    aggregate_copy = dict(aggregate_row)
    if aggregate_copy.pop("terminal_sha256", None) != terminal_hash:
        raise ValueError(f"acceptance aggregate does not bind fold terminal: {fold}")
    if aggregate_copy != terminal:
        raise ValueError(f"acceptance fold terminal diverges from aggregate: {fold}")

    source = pretraining["folds"][fold]
    strict_load_checkpoint(Path(source["model_path"]))
    initial = finite_metrics(terminal.get("initial"), fold=fold, stage="initial")
    final = finite_metrics(terminal.get("final"), fold=fold, stage="final")
    recomputed_gate = validation_improvement_gate(initial, final)
    elapsed = terminal.get("elapsed_seconds")
    valid = bool(
        terminal.get("schema_version") == 1
        and terminal.get("status") == "completed"
        and terminal.get("run_id") == RUN_ID
        and terminal.get("fold") == fold
        and terminal.get("seed") == FOLD_SEEDS[fold]
        and isinstance(elapsed, (int, float))
        and not isinstance(elapsed, bool)
        and math.isfinite(float(elapsed))
        and 0.0 < float(elapsed) < 3_600.0
        and terminal.get("appearance_family") == CONTEXTUAL_PAIR_FUSION_FAMILY
        and terminal.get("parameter_count") == EXPECTED_PARAMETER_COUNT
        and terminal.get("acceptance_source") == ACCEPTANCE_SOURCE
        and terminal.get("acceptance_role") == ACCEPTANCE_ROLE
        and terminal.get("acceptance_shards") == ACCEPTANCE_SHARDS
        and isinstance(terminal.get("acceptance_cache_bytes"), int)
        and terminal["acceptance_cache_bytes"] > 0
        and terminal.get("acceptance_manifest_sha256")
        == EXPECTED_ACCEPTANCE_MANIFEST_SHA256
        and terminal.get("acceptance_inventory_sha256")
        == EXPECTED_ACCEPTANCE_INVENTORY_SHA256
        and terminal.get("acceptance_record_inventory_sha256")
        == EXPECTED_ACCEPTANCE_RECORD_INVENTORY_SHA256
        and terminal.get("pretraining_terminal_sha256")
        == pretraining["terminal_sha256"]
        and terminal.get("pretrained_model_sha256") == source["model_sha256"]
        and terminal.get("pretraining_worker_terminal_sha256")
        == source["worker_terminal_sha256"]
        and terminal.get("initial_state_sha256") == initial_state_hash(FOLD_SEEDS[fold])
        and terminal.get("gate") == recomputed_gate
        and terminal.get("gate_passed") is True
        and recomputed_gate["passed"] is True
        and terminal.get("selection_or_checkpoint_redirect_permitted") is False
        and terminal.get("competition_data_read") is False
        and terminal.get("public_predictions_copied") is False
        and terminal.get("public_leaderboard_used_for_selection") is False
        and terminal.get("submission_created") is False
    )
    if not valid:
        raise ValueError(f"invalid downloaded ZSNS001 acceptance fold: {fold}")
    return {
        "terminal_sha256": terminal_hash,
        "pretrained_model_sha256": source["model_sha256"],
        "initial_state_sha256": terminal["initial_state_sha256"],
        "gains": recomputed_gate["gains"],
    }


def verify_acceptance_output(output_root: Path, pretraining_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(f"acceptance output root is missing: {output_root}")
    prohibited = sorted(
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in PROHIBITED_OUTPUT_SUFFIXES
    )
    if prohibited:
        raise ValueError(f"acceptance output contains prohibited artifacts: {prohibited}")

    launcher_path = unique_path(output_root, "launcher_terminal.json")
    launcher = read_object(launcher_path)
    aggregate_hash = launcher.get("acceptance_terminal_sha256")
    if not isinstance(aggregate_hash, str) or len(aggregate_hash) != 64:
        raise ValueError("launcher acceptance-terminal hash is malformed")
    aggregate_path = terminal_with_hash(output_root, aggregate_hash)
    aggregate = read_object(aggregate_path)
    pretraining = verify_pretraining_source(pretraining_root)
    folds = aggregate.get("folds")
    if not isinstance(folds, dict) or set(folds) != set(FOLDS):
        raise ValueError("acceptance aggregate fold inventory changed")
    fold_evidence = {
        fold: verify_fold(
            fold=fold,
            aggregate_row=folds[fold],
            aggregate_root=aggregate_path.parent,
            pretraining=pretraining,
        )
        for fold in FOLDS
    }

    aggregate_elapsed = aggregate.get("elapsed_seconds")
    aggregate_valid = bool(
        aggregate.get("schema_version") == 1
        and aggregate.get("status") == "completed"
        and aggregate.get("run_id") == RUN_ID
        and isinstance(aggregate_elapsed, (int, float))
        and not isinstance(aggregate_elapsed, bool)
        and math.isfinite(float(aggregate_elapsed))
        and 0.0 < float(aggregate_elapsed) < 3_600.0
        and aggregate.get("gpu_count") == 2
        and aggregate.get("pretraining_run_id") == PRETRAINING_RUN_ID
        and aggregate.get("pretraining_terminal_sha256")
        == pretraining["terminal_sha256"]
        and aggregate.get("acceptance_source") == ACCEPTANCE_SOURCE
        and aggregate.get("acceptance_manifest_sha256")
        == EXPECTED_ACCEPTANCE_MANIFEST_SHA256
        and aggregate.get("acceptance_inventory_sha256")
        == EXPECTED_ACCEPTANCE_INVENTORY_SHA256
        and aggregate.get("acceptance_record_inventory_sha256")
        == EXPECTED_ACCEPTANCE_RECORD_INVENTORY_SHA256
        and aggregate.get("both_folds_improved") is True
        and aggregate.get("selection_or_checkpoint_redirect_permitted") is False
        and aggregate.get("competition_data_read") is False
        and aggregate.get("public_predictions_copied") is False
        and aggregate.get("public_leaderboard_used_for_selection") is False
        and aggregate.get("submission_created") is False
    )
    launcher_elapsed = launcher.get("elapsed_seconds")
    launcher_valid = bool(
        launcher.get("schema_version") == 1
        and launcher.get("run_id") == RUN_ID
        and launcher.get("status") == "completed"
        and isinstance(launcher_elapsed, (int, float))
        and not isinstance(launcher_elapsed, bool)
        and math.isfinite(float(launcher_elapsed))
        and 0.0 < float(launcher_elapsed) <= 3_600.0
        and launcher.get("declared_budget_seconds") == 3_600
        and launcher.get("evaluator_hard_stop_seconds") == 3_300
        and launcher.get("acceptance_terminal_exists") is True
        and launcher.get("gpu_count_required") == 2
        and launcher.get("competition_data_read") is False
        and launcher.get("public_predictions_copied") is False
        and launcher.get("public_leaderboard_used_for_selection") is False
        and launcher.get("submission_created") is False
    )
    if not aggregate_valid:
        raise ValueError("invalid downloaded ZSNS001 acceptance aggregate")
    if not launcher_valid:
        raise ValueError("invalid downloaded ZSNS001 acceptance launcher")
    return {
        "schema_version": 1,
        "status": "accepted_verified",
        "run_id": RUN_ID,
        "launcher_terminal_sha256": sha256_file(launcher_path),
        "acceptance_terminal_sha256": aggregate_hash,
        "pretraining_terminal_sha256": pretraining["terminal_sha256"],
        "both_folds_improved": True,
        "folds": fold_evidence,
        "gpu_count": 2,
        "competition_data_read": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pretraining-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_acceptance_output(args.output_root, args.pretraining_root),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
