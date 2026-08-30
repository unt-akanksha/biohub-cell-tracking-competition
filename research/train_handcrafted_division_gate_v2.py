#!/usr/bin/env python
"""Train a CPU morphology gate on hard-negative real-division patches."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.calibrate_handcrafted_division_gate import select_threshold
from research.temporal_contrastive.train_real_division_gate_v2 import (
    DATA_RUN_ID,
    FRAME_ROLES,
    validate_manifest,
)
from research.train_handcrafted_division_gate import (
    FEATURE_FAMILY,
    atomic_json,
    estimator_grid,
    fit_with_weights,
    high_precision_metrics,
    patch_features,
    sha256_file,
)


RUN_ID = "competition-real-handcrafted-hard-negative-gate-v2"


def load_role(
    root: Path, manifest: dict[str, Any], role: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    patches = []
    targets = []
    weights = []
    groups = []
    inventory: list[dict[str, Any]] = []
    records = [record for record in manifest["shards"] if record["role"] == role]
    for record in records:
        path = root / record["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"handcrafted hard-negative shard changed: {path}")
        with np.load(path, allow_pickle=False) as data:
            source = np.asarray(data["source_patches"], dtype=np.float32)
            target = np.asarray(data["division_target"], dtype=np.int64)
            weight = np.asarray(data["label_weight"], dtype=np.float64)
        frame_role = str(record["frame_role"])
        if frame_role not in FRAME_ROLES:
            raise ValueError("handcrafted hard-negative frame role changed")
        patches.append(source)
        targets.append(target)
        weights.append(weight)
        groups.extend([record["stem"]] * len(target))
        inventory.extend(
            {
                "stem": record["stem"],
                "embryo": record["embryo"],
                "timepoint": int(record["timepoint"]),
                "frame_role": frame_role,
            }
            for _ in range(len(target))
        )
    if not patches:
        raise ValueError(f"handcrafted hard-negative {role} inventory is empty")
    return (
        patch_features(np.concatenate(patches)),
        np.concatenate(targets),
        np.concatenate(weights),
        np.asarray(groups),
        inventory,
    )


def score_payload(payload: dict[str, Any], features: np.ndarray) -> np.ndarray:
    return np.mean(
        np.stack(
            [row["model"].predict_proba(features)[:, 1] for row in payload["models"]]
        ),
        axis=0,
    )


def decision_summary(
    targets: np.ndarray,
    scores: np.ndarray,
    inventory: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    labels = targets.astype(bool)
    selected = scores >= threshold
    tp = int(np.sum(selected & labels))
    fp = int(np.sum(selected & ~labels))
    positives = int(labels.sum())
    return {
        "threshold": float(threshold),
        "selected": int(selected.sum()),
        "tp": tp,
        "fp": fp,
        "fn": positives - tp,
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(positives, 1)),
        "jaccard": float(tp / max(fp + positives, 1)),
        "no_division_hard_negative_false_positives": int(
            sum(
                bool(selected[index])
                and not bool(labels[index])
                and row["frame_role"] == "no_division_hard_negative"
                for index, row in enumerate(inventory)
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=190_127)
    args = parser.parse_args()
    manifest_path = args.data_root / "real_division_hard_negative_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    if manifest.get("run_id") != DATA_RUN_ID:
        raise ValueError("handcrafted hard-negative data run changed")
    args.output_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    train_x, train_y, train_weights, groups, train_inventory = load_role(
        args.data_root, manifest, "optimization"
    )
    selection_x, selection_y, _, _, selection_inventory = load_role(
        args.data_root, manifest, "selection"
    )
    audit_stems = {
        record["stem"] for record in manifest["shards"] if record["role"] == "audit"
    }
    if set(groups) & audit_stems or {
        row["stem"] for row in selection_inventory
    } & audit_stems:
        raise RuntimeError("handcrafted hard-negative audit movies leaked")
    class_balance = float(
        (train_weights * (train_y == 0)).sum() / max(int(train_y.sum()), 1)
    )
    fit_weights = train_weights * np.where(train_y == 1, class_balance, 1.0)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    candidates = []
    for name, estimator in estimator_grid(args.seed):
        oof = np.zeros(len(train_y), dtype=np.float64)
        for fit, score in splitter.split(train_x, train_y, groups):
            model = fit_with_weights(
                clone(estimator), train_x[fit], train_y[fit], fit_weights[fit]
            )
            oof[score] = model.predict_proba(train_x[score])[:, 1]
        candidates.append(
            {
                "name": name,
                "estimator": estimator,
                "oof": high_precision_metrics(train_y, oof),
            }
        )
        print(json.dumps({"name": name, "oof": candidates[-1]["oof"]}), flush=True)
    ranked = sorted(
        candidates,
        key=lambda row: (
            row["oof"]["recall_at_zero_false_positives"],
            row["oof"]["recall_at_precision_0_95"],
            row["oof"]["average_precision"],
        ),
        reverse=True,
    )
    selected = ranked[:2]
    payload = {
        "run_id": RUN_ID,
        "feature_family": FEATURE_FAMILY,
        "feature_count": train_x.shape[1],
        "models": [
            {
                "name": row["name"],
                "model": fit_with_weights(
                    clone(row["estimator"]), train_x, train_y, fit_weights
                ),
            }
            for row in selected
        ],
    }
    scores = score_payload(payload, selection_x)
    metrics = high_precision_metrics(selection_y, scores)
    frozen = select_threshold(selection_y, scores)
    decisions = decision_summary(
        selection_y, scores, selection_inventory, frozen["threshold"]
    )
    by_embryo = {}
    for embryo in ("44b6", "6bba"):
        indices = np.asarray(
            [index for index, row in enumerate(selection_inventory) if row["embryo"] == embryo]
        )
        by_embryo[embryo] = high_precision_metrics(
            selection_y[indices], scores[indices]
        )
    accepted = bool(
        metrics["average_precision"] >= 0.55
        and frozen["fp"] == 0
        and frozen["tp"] >= 3
        and decisions["no_division_hard_negative_false_positives"] == 0
        and all(row["average_precision"] >= 0.40 for row in by_embryo.values())
    )
    model_path = args.output_root / "handcrafted_hard_negative_gate.joblib"
    joblib.dump(payload, model_path, compress=3)
    result = {
        "schema_version": 1,
        "status": "accepted_at_selection" if accepted else "rejected_at_selection",
        "run_id": RUN_ID,
        "feature_family": FEATURE_FAMILY,
        "feature_count": train_x.shape[1],
        "elapsed_seconds": time.monotonic() - started,
        "train_rows": len(train_y),
        "train_positives": int(train_y.sum()),
        "selection_rows": len(selection_y),
        "selection_positives": int(selection_y.sum()),
        "sealed_audit_movie_count": len(audit_stems),
        "cross_validation": [
            {"name": row["name"], "metrics": deepcopy(row["oof"])} for row in ranked
        ],
        "selected_models": [row["name"] for row in selected],
        "selection": metrics,
        "selection_by_embryo": by_embryo,
        "frozen_division_probability_threshold": frozen["threshold"],
        "threshold_selection": frozen,
        "selection_frozen_decisions": decisions,
        "selection_gate_passed": accepted,
        "model_sha256": sha256_file(model_path),
        "manifest_sha256": sha256_file(manifest_path),
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
    atomic_json(args.output_root / "handcrafted_hard_negative_terminal.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
