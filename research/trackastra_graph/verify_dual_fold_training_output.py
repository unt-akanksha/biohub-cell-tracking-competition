#!/usr/bin/env python
"""Fail-closed audit of downloaded reciprocal Trackastra training output."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


RUN_ID = "trackastra-dual-fold-synthetic-v1"
FOLDS = {
    "target_44b6": {
        "source_prefix": "6bba",
        "target_prefix": "44b6",
        "real_train_count": 96,
    },
    "target_6bba": {
        "source_prefix": "44b6",
        "target_prefix": "6bba",
        "real_train_count": 69,
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
EXPECTED_PARAMETER_COUNT = 27_456_880
EXPECTED_PRETRAINED_SHA256 = (
    "24c290ce74289ee6dac952a3355c9fb7a0a477fda3012656a2c6d80af892981f"
)
EXPECTED_SYNTHETIC_MANIFEST_SHA256 = (
    "e8b5376b2ac6fdd55bd6e45d1b07b401339d375b211b6f67be93fb0de4d8ce14"
)
MINIMUM_REAL_GAIN = 0.001
MAXIMUM_SYNTHETIC_REGRESSION = 0.005


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


def finite_metric(payload: dict[str, Any], key: str, *, fold: str) -> float:
    value = float(payload.get(key, float("nan")))
    if not math.isfinite(value):
        raise ValueError(f"{fold} has a non-finite {key}")
    return value


def terminal_source_policy(aggregate: dict[str, Any]) -> str:
    """Classify one coherent two-fold adapted or predeclared-control source."""

    rows = aggregate.get("folds")
    if not isinstance(rows, dict) or set(rows) != set(FOLDS):
        raise ValueError("aggregate Trackastra terminal does not cover both folds")
    if aggregate.get("both_folds_improved") is True and all(
        isinstance(row, dict)
        and int(row.get("best_step", 0)) > 0
        and row.get("pretrained_initialization_retained") is False
        for row in rows.values()
    ):
        return "adapted_dual_fold"
    control_hashes = {
        str(row.get("model_sha256", ""))
        for row in rows.values()
        if isinstance(row, dict)
    }
    if aggregate.get("both_folds_improved") is False and all(
        isinstance(row, dict)
        and int(row.get("best_step", -1)) == 0
        and row.get("pretrained_initialization_retained") is True
        and row.get("best_real") == row.get("initial_real")
        and row.get("best_synthetic") == row.get("initial_synthetic")
        and float(row.get("best_selection_score", float("nan")))
        == float(row.get("initial_selection_score", float("nan")))
        for row in rows.values()
    ) and len(control_hashes) == 1 and "" not in control_hashes:
        return "predeclared_pretrained_control"
    raise ValueError("aggregate Trackastra terminal mixes or rejects source policies")


def verify_output(
    root: Path, *, allow_pretrained_control: bool = False
) -> dict[str, Any]:
    root = root.resolve()
    aggregate_path = root / "training_terminal.json"
    aggregate = read_json(aggregate_path)
    expected_folds = set(FOLDS)
    if not (
        aggregate.get("schema_version") == 1
        and aggregate.get("status") == "completed"
        and aggregate.get("run_id") == RUN_ID
        and aggregate.get("gpu_count") == 2
        and set(aggregate.get("whole_fold_coverage", [])) == expected_folds
        and aggregate.get("submission_created") is False
        and set(aggregate.get("folds", {})) == expected_folds
    ):
        raise ValueError("aggregate Trackastra terminal is not eligible")
    source_policy = terminal_source_policy(aggregate)
    if (
        source_policy == "predeclared_pretrained_control"
        and not allow_pretrained_control
    ):
        raise ValueError("aggregate Trackastra terminal is not eligible")
    control_model_hash: str | None = None
    if source_policy == "predeclared_pretrained_control":
        control_hashes = {
            str(aggregate["folds"][fold].get("model_sha256", ""))
            for fold in expected_folds
        }
        if len(control_hashes) != 1 or "" in control_hashes:
            raise ValueError("pretrained control folds do not share one model hash")
        control_model_hash = next(iter(control_hashes))

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

    verified: dict[str, Any] = {}
    for fold, expected in FOLDS.items():
        fold_dir = root / fold
        worker_path = fold_dir / "worker_terminal.json"
        config_path = fold_dir / "training_config.json"
        model_path = fold_dir / "model.pt"
        model_config_path = fold_dir / "config.yaml"
        worker = read_json(worker_path)
        config = read_json(config_path)
        if worker != aggregate["folds"][fold]:
            raise ValueError(f"aggregate and worker terminal diverge: {fold}")
        common_worker_valid = bool(
            worker.get("schema_version") == 1
            and worker.get("status") == "completed"
            and worker.get("run_id") == RUN_ID
            and worker.get("fold") == fold
            and int(worker.get("completed_step", 0)) > 0
            and int(worker.get("parameter_count", 0)) == EXPECTED_PARAMETER_COUNT
            and worker.get("submission_created") is False
        )
        adapted_worker_valid = bool(
            source_policy == "adapted_dual_fold"
            and 0 < int(worker.get("best_step", 0)) <= int(worker["completed_step"])
            and worker.get("pretrained_initialization_retained") is False
        )
        control_worker_valid = bool(
            source_policy == "predeclared_pretrained_control"
            and int(worker.get("best_step", -1)) == 0
            and worker.get("pretrained_initialization_retained") is True
            and worker.get("model_sha256") == control_model_hash
            and worker.get("best_real") == worker.get("initial_real")
            and worker.get("best_synthetic") == worker.get("initial_synthetic")
            and float(worker.get("best_selection_score", float("nan")))
            == float(worker.get("initial_selection_score", float("nan")))
        )
        if not (common_worker_valid and (adapted_worker_valid or control_worker_valid)):
            raise ValueError(f"worker terminal is not eligible: {fold}")
        if not model_path.is_file() or sha256_file(model_path) != worker.get(
            "model_sha256"
        ):
            raise ValueError(f"model hash mismatch: {fold}")
        if not model_config_path.is_file():
            raise FileNotFoundError(f"Trackastra config is missing: {fold}")

        initial_real = finite_metric(worker["initial_real"], "composite", fold=fold)
        best_real = finite_metric(worker["best_real"], "composite", fold=fold)
        initial_synthetic = finite_metric(
            worker["initial_synthetic"], "composite", fold=fold
        )
        best_synthetic = finite_metric(
            worker["best_synthetic"], "composite", fold=fold
        )
        initial_selection = finite_metric(
            worker, "initial_selection_score", fold=fold
        )
        best_selection = finite_metric(worker, "best_selection_score", fold=fold)
        if source_policy == "adapted_dual_fold":
            if best_real + 1e-12 < initial_real + MINIMUM_REAL_GAIN:
                raise ValueError(f"real reciprocal gain gate failed: {fold}")
            if best_synthetic + 1e-12 < initial_synthetic - MAXIMUM_SYNTHETIC_REGRESSION:
                raise ValueError(f"synthetic regression gate failed: {fold}")
            if best_selection <= initial_selection:
                raise ValueError(f"selection score did not improve: {fold}")

        real_train = config.get("real_train_stems")
        real_validation = config.get("real_validation_stems")
        if not (
            config.get("schema_version") == 1
            and config.get("run_id") == RUN_ID
            and config.get("fold") == fold
            and config.get("source_prefix") == expected["source_prefix"]
            and config.get("target_prefix") == expected["target_prefix"]
            and config.get("pretrained_sha256") == EXPECTED_PRETRAINED_SHA256
            and config.get("synthetic_manifest_sha256")
            == EXPECTED_SYNTHETIC_MANIFEST_SHA256
            and config.get("synthetic_native_geometry_restored") is True
            and config.get("synthetic_train_count") == 2046
            and config.get("synthetic_validation_count") == 128
            and config.get("steps_target") == 75000
            and isinstance(real_train, list)
            and len(real_train) == expected["real_train_count"]
            and isinstance(real_validation, list)
            and len(real_validation) == 12
        ):
            raise ValueError(f"training configuration changed: {fold}")
        inventory = set(map(str, real_train + real_validation))
        if inventory & OPENED_ACCEPTANCE_STEMS:
            raise ValueError(f"opened acceptance data entered training: {fold}")
        if any(
            not stem.startswith(f"{expected['source_prefix']}_")
            for stem in real_train
        ) or any(
            not stem.startswith(f"{expected['target_prefix']}_")
            for stem in real_validation
        ):
            raise ValueError(f"real split prefix mismatch: {fold}")

        verified[fold] = {
            "completed_step": int(worker["completed_step"]),
            "best_step": int(worker["best_step"]),
            "model_sha256": worker["model_sha256"],
            "real_composite_gain": best_real - initial_real,
            "synthetic_composite_delta": best_synthetic - initial_synthetic,
            "selection_score_gain": best_selection - initial_selection,
            "worker_terminal_sha256": sha256_file(worker_path),
            "training_config_sha256": sha256_file(config_path),
        }

    return {
        "schema_version": 1,
        "status": "verified",
        "run_id": RUN_ID,
        "root": str(root),
        "gpu_count": 2,
        "source_policy": source_policy,
        "folds": verified,
        "training_terminal_sha256": sha256_file(aggregate_path),
        "synthetic_native_geometry_restored": True,
        "competition_artifacts_found": False,
        "authorized_for_submission": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--allow-pretrained-control", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify_output(
                args.root, allow_pretrained_control=args.allow_pretrained_control
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
