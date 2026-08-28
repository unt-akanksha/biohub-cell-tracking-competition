#!/usr/bin/env python
"""Fail-closed audit of downloaded reciprocal appearance training output."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    from appearance_family import (
        COSINE_FAMILY,
        CONTEXTUAL_FAMILIES,
        CONTEXTUAL_PAIR_FUSION_FAMILY,
        FAMILIES,
        PAIR_FUSION_FAMILY,
        PARAMETER_COUNT_BY_FAMILY,
        TRAINING_RUN_BY_FAMILY,
        build_appearance_model,
        verify_appearance_metadata,
    )
except ModuleNotFoundError:
    from research.temporal_contrastive.appearance_family import (
        COSINE_FAMILY,
        CONTEXTUAL_FAMILIES,
        CONTEXTUAL_PAIR_FUSION_FAMILY,
        FAMILIES,
        PAIR_FUSION_FAMILY,
        PARAMETER_COUNT_BY_FAMILY,
        TRAINING_RUN_BY_FAMILY,
        build_appearance_model,
        verify_appearance_metadata,
    )


FOLDS = {
    "target_44b6": {
        "source_prefix": "6bba",
        "target_prefix": "44b6",
        "seed": 45_427,
        "real_train_count": 96,
    },
    "target_6bba": {
        "source_prefix": "44b6",
        "target_prefix": "6bba",
        "seed": 47_627,
        "real_train_count": 45,
    },
}
OPENED_ACCEPTANCE_STEMS = frozenset(
    {
        "44b6_12dfb391",
        "44b6_267148e4",
        "6bba_062c8d37",
        "6bba_07e24132",
    }
)
EXPECTED_SYNTHETIC_MANIFEST_SHA256 = (
    "e8b5376b2ac6fdd55bd6e45d1b07b401339d375b211b6f67be93fb0de4d8ce14"
)
EXPECTED_SYNTHETIC_TRAIN_COUNT = 1_900
EXPECTED_SYNTHETIC_VALIDATION_COUNT = 128
EXPECTED_REAL_VALIDATION_COUNT = 12
EXPECTED_REAL_CALIBRATION_COUNT = 12
MINIMUM_REAL_TOP1 = 0.70
MINIMUM_SYNTHETIC_TOP1 = 0.85
MINIMUM_REAL_COMPOSITE_GAIN = 0.005
MAXIMUM_SYNTHETIC_METRIC_REGRESSION = 0.01


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def finite_float(payload: dict[str, Any], key: str, *, fold: str) -> float:
    value = float(payload.get(key, float("nan")))
    if not math.isfinite(value):
        raise ValueError(f"{fold} has a non-finite {key}")
    return value


def _family_from_aggregate(aggregate: dict[str, Any]) -> str:
    declared = aggregate.get("appearance_family")
    if declared is not None:
        family = str(declared)
        if family not in FAMILIES:
            raise ValueError("aggregate declares an unknown appearance family")
        return family
    rows = aggregate.get("folds", {})
    if not isinstance(rows, dict) or not rows:
        raise ValueError("aggregate has no appearance folds")
    families = set()
    for row in rows.values():
        if not isinstance(row, dict):
            raise ValueError("aggregate appearance fold is not an object")
        try:
            families.add(verify_appearance_metadata(row, require_training_run=True))
        except (TypeError, ValueError) as error:
            raise ValueError("aggregate appearance metadata is invalid") from error
    if families != {COSINE_FAMILY}:
        raise ValueError("aggregate omits an unambiguous appearance family")
    return COSINE_FAMILY


def _verify_checkpoint_strict(path: Path, family: str) -> None:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("appearance checkpoint is not a state dictionary")
    model = build_appearance_model(family)
    model.load_state_dict(state, strict=True)
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
            raise ValueError(f"appearance checkpoint tensor is invalid: {name}")


def verify_finetuning_gate(
    worker: dict[str, Any], config: dict[str, Any], *, fold: str
) -> None:
    """Recompute the contextual transfer gate from terminal metrics."""

    initial_real = worker.get("initial_real")
    best_real = worker.get("best_real")
    initial_synthetic = worker.get("initial_synthetic")
    best_synthetic = worker.get("best_synthetic")
    gate = worker.get("finetuning_gate")
    initialization = worker.get("initialization")
    if not all(
        isinstance(value, dict)
        for value in (
            initial_real,
            best_real,
            initial_synthetic,
            best_synthetic,
            gate,
            initialization,
        )
    ):
        raise ValueError(f"contextual transfer evidence is missing: {fold}")
    if config.get("initialization") != initialization:
        raise ValueError(f"contextual initialization evidence diverges: {fold}")
    if not (
        initialization.get("policy") == "hash-bound ZebraHub external pretraining"
        and initialization.get("run_id") == "zebrahub-contextual-pretrain-v1"
        and initialization.get("fold") == fold
        and len(str(initialization.get("model_sha256", ""))) == 64
        and initialization.get("external_training_source") == "ZSNS004"
        and initialization.get("external_validation_source") == "ZSNS005"
        and config.get("minimum_real_composite_gain")
        == MINIMUM_REAL_COMPOSITE_GAIN
        and config.get("maximum_synthetic_metric_regression")
        == MAXIMUM_SYNTHETIC_METRIC_REGRESSION
        and worker.get("finetuning_gate_passed") is True
        and gate.get("passed") is True
        and gate.get("minimum_real_composite_gain")
        == MINIMUM_REAL_COMPOSITE_GAIN
        and gate.get("maximum_synthetic_metric_regression")
        == MAXIMUM_SYNTHETIC_METRIC_REGRESSION
        and gate.get("real_inventory_unchanged") is True
        and gate.get("synthetic_inventory_unchanged") is True
    ):
        raise ValueError(f"contextual transfer contract changed: {fold}")
    metrics = ("composite", "top1", "mrr", "division_top2")
    real_gains = {
        name: finite_float(best_real, name, fold=fold)
        - finite_float(initial_real, name, fold=fold)
        for name in metrics
    }
    synthetic_gains = {
        name: finite_float(best_synthetic, name, fold=fold)
        - finite_float(initial_synthetic, name, fold=fold)
        for name in metrics
    }
    recorded_real = gate.get("real_gains")
    recorded_synthetic = gate.get("synthetic_gains")
    if not isinstance(recorded_real, dict) or not isinstance(
        recorded_synthetic, dict
    ):
        raise ValueError(f"contextual transfer gains are missing: {fold}")
    if any(
        not math.isclose(
            float(recorded[name]), gain, rel_tol=1e-9, abs_tol=1e-12
        )
        for recorded, observed in (
            (recorded_real, real_gains),
            (recorded_synthetic, synthetic_gains),
        )
        for name, gain in observed.items()
    ):
        raise ValueError(f"contextual transfer gains diverge: {fold}")
    inventory_names = ("rows", "division_rows", "transitions")
    inventory_unchanged = all(
        int(initial[name]) == int(candidate[name])
        for initial, candidate in (
            (initial_real, best_real),
            (initial_synthetic, best_synthetic),
        )
        for name in inventory_names
    )
    if not (
        inventory_unchanged
        and real_gains["composite"] >= MINIMUM_REAL_COMPOSITE_GAIN
        and real_gains["top1"] > 0.0
        and real_gains["mrr"] > 0.0
        and real_gains["division_top2"] >= 0.0
        and all(
            value >= -MAXIMUM_SYNTHETIC_METRIC_REGRESSION
            for value in synthetic_gains.values()
        )
    ):
        raise ValueError(f"contextual transfer gate failed: {fold}")


def verify_output(
    root: Path,
    *,
    expected_family: str | None = None,
    strict_checkpoint: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    aggregate_path = root / "training_terminal.json"
    aggregate = read_json(aggregate_path)
    family = _family_from_aggregate(aggregate)
    if expected_family is not None and family != expected_family:
        raise ValueError("downloaded appearance family does not match expectation")
    expected_run = TRAINING_RUN_BY_FAMILY[family]
    expected_folds = set(FOLDS)
    if not (
        aggregate.get("schema_version") == 1
        and aggregate.get("status") == "completed"
        and aggregate.get("run_id") == expected_run
        and aggregate.get("gpu_count") == 2
        and aggregate.get("both_folds_trained") is True
        and (
            family not in CONTEXTUAL_FAMILIES
            or aggregate.get("both_folds_improved") is True
        )
        and aggregate.get("public_predictions_copied") is False
        and aggregate.get("public_leaderboard_used_for_selection") is False
        and aggregate.get("submission_created") is False
        and set(aggregate.get("folds", {})) == expected_folds
    ):
        raise ValueError("aggregate appearance terminal is not eligible")

    forbidden = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and (
            "submission" in path.name.lower()
            or path.suffix.lower() in {".csv", ".zip"}
        )
    ]
    if forbidden:
        raise ValueError(f"training output contains competition artifacts: {forbidden}")

    verified: dict[str, dict[str, Any]] = {}
    split_by_prefix: dict[str, dict[str, set[str]]] = {
        prefix: {"train": set(), "validation": set(), "calibration": set()}
        for prefix in {"44b6", "6bba"}
    }
    synthetic_partitions: list[tuple[list[str], list[str]]] = []
    for fold, expected in FOLDS.items():
        fold_dir = root / fold
        worker_path = fold_dir / "worker_terminal.json"
        config_path = fold_dir / "training_config.json"
        model_path = fold_dir / "appearance_model.pt"
        worker = read_json(worker_path)
        config = read_json(config_path)
        if worker != aggregate["folds"].get(fold):
            raise ValueError(f"aggregate and appearance worker diverge: {fold}")
        try:
            worker_family = verify_appearance_metadata(
                worker, require_training_run=True
            )
            config_family = verify_appearance_metadata(
                config, require_training_run=True
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"appearance metadata changed: {fold}") from error
        completed_step = int(worker.get("completed_step", 0))
        best_step = int(worker.get("best_step", 0))
        if not (
            worker.get("schema_version") == 1
            and worker.get("status") == "completed"
            and worker.get("run_id") == expected_run
            and worker.get("fold") == fold
            and worker_family == family == config_family
            and 0 < best_step <= completed_step <= 30_000
            and int(worker.get("optimizer_steps", 0)) > 0
            and int(worker.get("parameter_count", 0))
            == PARAMETER_COUNT_BY_FAMILY[family]
            and worker.get("public_predictions_copied") is False
            and worker.get("public_leaderboard_used_for_selection") is False
            and worker.get("submission_created") is False
        ):
            raise ValueError(f"appearance worker is not eligible: {fold}")
        if not model_path.is_file() or sha256_file(model_path) != worker.get(
            "model_sha256"
        ):
            raise ValueError(f"appearance model hash mismatch: {fold}")
        if strict_checkpoint:
            _verify_checkpoint_strict(model_path, family)

        best_real = worker.get("best_real")
        best_synthetic = worker.get("best_synthetic")
        if not isinstance(best_real, dict) or not isinstance(best_synthetic, dict):
            raise ValueError(f"appearance validation metrics are missing: {fold}")
        real_top1 = finite_float(best_real, "top1", fold=fold)
        synthetic_top1 = finite_float(best_synthetic, "top1", fold=fold)
        real_composite = finite_float(best_real, "composite", fold=fold)
        synthetic_composite = finite_float(best_synthetic, "composite", fold=fold)
        best_score = finite_float(worker, "best_score", fold=fold)
        expected_score = 0.85 * real_composite + 0.15 * synthetic_composite
        if (
            real_top1 + 1e-12 < MINIMUM_REAL_TOP1
            or synthetic_top1 + 1e-12 < MINIMUM_SYNTHETIC_TOP1
            or not math.isclose(best_score, expected_score, rel_tol=1e-9, abs_tol=1e-12)
        ):
            raise ValueError(f"appearance checkpoint gate failed: {fold}")

        synthetic_train = config.get("synthetic_train_names")
        synthetic_validation = config.get("synthetic_validation_names")
        real_train = config.get("real_train_stems")
        real_validation = config.get("real_validation_stems")
        real_calibration = config.get("real_calibration_stems_reserved")
        if not (
            config.get("schema_version") == 1
            and config.get("run_id") == expected_run
            and config.get("fold") == fold
            and config.get("seed") == expected["seed"]
            and config.get("source_prefix") == expected["source_prefix"]
            and config.get("target_prefix") == expected["target_prefix"]
            and config.get("requested_real_train_movies") == 96
            and config.get("effective_real_train_movies")
            == expected["real_train_count"]
            and config.get("synthetic_manifest_sha256")
            == EXPECTED_SYNTHETIC_MANIFEST_SHA256
            and config.get("base_channels") == 64
            and config.get("embedding_channels") == 256
            and config.get("candidate_radius_um") == 32.0
            and config.get("patch_shape") == [17, 17, 17]
            and config.get("patch_half_extent_um") == [8.0, 8.0, 8.0]
            and config.get("calibration_ground_truth_read") is False
            and config.get("opened_acceptance_stems_excluded")
            == sorted(OPENED_ACCEPTANCE_STEMS)
            and config.get("public_predictions_copied") is False
            and config.get("public_leaderboard_used_for_selection") is False
            and config.get("submission_created") is False
            and isinstance(synthetic_train, list)
            and len(synthetic_train) == EXPECTED_SYNTHETIC_TRAIN_COUNT
            and len(set(map(str, synthetic_train))) == len(synthetic_train)
            and isinstance(synthetic_validation, list)
            and len(synthetic_validation) == EXPECTED_SYNTHETIC_VALIDATION_COUNT
            and len(set(map(str, synthetic_validation))) == len(synthetic_validation)
            and isinstance(real_train, list)
            and len(real_train) == expected["real_train_count"]
            and isinstance(real_validation, list)
            and len(real_validation) == EXPECTED_REAL_VALIDATION_COUNT
            and isinstance(real_calibration, list)
            and len(real_calibration) == EXPECTED_REAL_CALIBRATION_COUNT
        ):
            raise ValueError(f"appearance training configuration changed: {fold}")
        if family in CONTEXTUAL_FAMILIES:
            verify_finetuning_gate(worker, config, fold=fold)
        synthetic_names = set(map(str, synthetic_train))
        synthetic_validation_names = set(map(str, synthetic_validation))
        if synthetic_names & synthetic_validation_names:
            raise ValueError(f"synthetic split overlaps: {fold}")
        synthetic_partitions.append(
            (list(map(str, synthetic_train)), list(map(str, synthetic_validation)))
        )
        real_sets = {
            "train": set(map(str, real_train)),
            "validation": set(map(str, real_validation)),
            "calibration": set(map(str, real_calibration)),
        }
        if (
            real_sets["train"] & real_sets["validation"]
            or real_sets["train"] & real_sets["calibration"]
            or real_sets["validation"] & real_sets["calibration"]
            or set().union(*real_sets.values()) & OPENED_ACCEPTANCE_STEMS
        ):
            raise ValueError(f"real split overlaps or leaks acceptance: {fold}")
        for role, stems in real_sets.items():
            prefix = (
                str(expected["source_prefix"])
                if role == "train"
                else str(expected["target_prefix"])
            )
            if any(not stem.startswith(f"{prefix}_") for stem in stems):
                raise ValueError(f"real split prefix mismatch: {fold}")
            split_by_prefix[prefix][role].update(stems)

        verified[fold] = {
            "completed_step": completed_step,
            "best_step": best_step,
            "optimizer_steps": int(worker["optimizer_steps"]),
            "model_sha256": worker["model_sha256"],
            "real_top1": real_top1,
            "synthetic_top1": synthetic_top1,
            "best_score": best_score,
            "worker_terminal_sha256": sha256_file(worker_path),
            "training_config_sha256": sha256_file(config_path),
        }

    if synthetic_partitions[0] != synthetic_partitions[1]:
        raise ValueError("reciprocal workers used different synthetic partitions")
    for prefix, split in split_by_prefix.items():
        if (
            split["train"] & split["validation"]
            or split["train"] & split["calibration"]
            or split["validation"] & split["calibration"]
        ):
            raise ValueError(f"global real partition overlaps for prefix {prefix}")

    return {
        "schema_version": 1,
        "status": "verified",
        "run_id": expected_run,
        "appearance_family": family,
        "root": str(root),
        "gpu_count": 2,
        "folds": verified,
        "training_terminal_sha256": sha256_file(aggregate_path),
        "strict_checkpoint_loaded": strict_checkpoint,
        "competition_artifacts_found": False,
        "authorized_for_calibration": True,
        "authorized_for_submission": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-family", choices=FAMILIES)
    parser.add_argument("--strict-checkpoint", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify_output(
                args.root,
                expected_family=args.expected_family,
                strict_checkpoint=args.strict_checkpoint,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
