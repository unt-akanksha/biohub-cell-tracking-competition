#!/usr/bin/env python
"""Freeze a high-precision threshold for the completed CPU morphology gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.train_handcrafted_division_gate import (
    FEATURE_FAMILY,
    RUN_ID as TRAINING_RUN_ID,
    atomic_json,
    high_precision_metrics,
    load_role,
    sha256_file,
)


RUN_ID = "competition-real-handcrafted-division-policy-v1"


def select_threshold(targets: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    order = np.argsort(-scores, kind="stable")
    labels = targets[order].astype(bool)
    ranked_scores = scores[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(~labels)
    fn = int(targets.sum()) - tp
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / int(targets.sum())
    jaccard = tp / np.maximum(tp + fp + fn, 1)
    eligible = np.flatnonzero((fp == 0) & (tp >= 2))
    if len(eligible):
        index = int(eligible[-1])
        policy = "maximum recall with zero false positives and at least two positives"
    else:
        eligible = np.flatnonzero((precision >= 0.95) & (tp > 0))
        if not len(eligible):
            raise RuntimeError("handcrafted selection has no high-precision threshold")
        index = int(eligible[np.argmax(jaccard[eligible])])
        policy = "maximum Jaccard among thresholds with at least 0.95 precision"
    return {
        "threshold": float(ranked_scores[index]),
        "policy": policy,
        "tp": int(tp[index]),
        "fp": int(fp[index]),
        "fn": int(fn[index]),
        "precision": float(precision[index]),
        "recall": float(recall[index]),
        "jaccard": float(jaccard[index]),
    }


def validate_training(root: Path) -> tuple[dict[str, Any], Path]:
    terminal_path = root / "handcrafted_division_gate_terminal.json"
    model_path = root / "handcrafted_division_gate.joblib"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if not (
        terminal.get("schema_version") == 1
        and terminal.get("status") == "completed"
        and terminal.get("run_id") == TRAINING_RUN_ID
        and terminal.get("feature_family") == FEATURE_FAMILY
        and terminal.get("feature_count") == 132
        and terminal.get("train_rows") == 1154
        and terminal.get("train_positives") == 115
        and terminal.get("selection_rows") == 239
        and terminal.get("selection_positives") == 31
        and terminal.get("competition_train_data_read") is True
        and terminal.get("competition_test_data_read") is False
        and terminal.get("final_probe_opened") is False
        and terminal.get("public_code_copied") is False
        and terminal.get("public_predictions_copied") is False
        and terminal.get("public_leaderboard_used_for_selection") is False
        and terminal.get("submission_created") is False
        and terminal.get("authorized_for_submission") is False
        and model_path.is_file()
        and sha256_file(model_path) == terminal.get("model_sha256")
    ):
        raise ValueError("handcrafted division training is ineligible")
    return terminal, model_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    terminal, model_path = validate_training(args.training_root)
    manifest_path = args.data_root / "real_division_patch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection_x, selection_y, _, _ = load_role(args.data_root, manifest, "selection")
    payload = joblib.load(model_path)
    if not (
        payload.get("run_id") == TRAINING_RUN_ID
        and payload.get("feature_family") == FEATURE_FAMILY
        and payload.get("feature_count") == 132
        and len(payload.get("models", [])) == 2
    ):
        raise ValueError("handcrafted division model payload changed")
    scores = np.mean(
        np.stack(
            [row["model"].predict_proba(selection_x)[:, 1] for row in payload["models"]]
        ),
        axis=0,
    )
    metrics = high_precision_metrics(selection_y, scores)
    if any(
        not math.isclose(float(metrics[key]), float(terminal["selection"][key]), abs_tol=1e-12)
        for key in (
            "average_precision",
            "recall_at_zero_false_positives",
            "recall_at_precision_0_95",
        )
    ):
        raise ValueError("handcrafted selection predictions are not reproducible")
    frozen = select_threshold(selection_y, scores)
    accepted = bool(
        metrics["average_precision"] >= 0.60
        and metrics["recall_at_zero_false_positives"] >= 0.20
        and frozen["fp"] == 0
        and frozen["tp"] >= 2
    )
    result = {
        "schema_version": 1,
        "status": "accepted_at_selection" if accepted else "rejected_at_selection",
        "run_id": RUN_ID,
        "model_training_run_id": TRAINING_RUN_ID,
        "feature_family": FEATURE_FAMILY,
        "model_sha256": sha256_file(model_path),
        "training_terminal_sha256": sha256_file(
            args.training_root / "handcrafted_division_gate_terminal.json"
        ),
        "manifest_sha256": sha256_file(manifest_path),
        "selection": metrics,
        "frozen_threshold": frozen,
        "frozen_division_probability_threshold": frozen["threshold"],
        "selection_gate_passed": accepted,
        "final_probe_opened": False,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_final_probe": accepted,
        "authorized_for_submission": False,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
