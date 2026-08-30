#!/usr/bin/env python
"""Open the sealed hard-negative audit for an accepted CPU morphology gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.temporal_contrastive.train_real_division_gate_v2 import validate_manifest
from research.train_handcrafted_division_gate import (
    FEATURE_FAMILY,
    atomic_json,
    high_precision_metrics,
    sha256_file,
)
from research.train_handcrafted_division_gate_v2 import (
    RUN_ID as TRAINING_RUN_ID,
    decision_summary,
    load_role,
    score_payload,
)


RUN_ID = "competition-real-handcrafted-hard-negative-audit-v2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    terminal_path = args.training_root / "handcrafted_hard_negative_terminal.json"
    model_path = args.training_root / "handcrafted_hard_negative_gate.joblib"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if not (
        terminal.get("schema_version") == 1
        and terminal.get("status") == "accepted_at_selection"
        and terminal.get("run_id") == TRAINING_RUN_ID
        and terminal.get("feature_family") == FEATURE_FAMILY
        and terminal.get("selection_gate_passed") is True
        and terminal.get("audit_opened") is False
        and terminal.get("checkpoint_and_policy_frozen_before_audit") is True
        and terminal.get("authorized_for_audit") is True
        and terminal.get("competition_test_data_read") is False
        and terminal.get("public_leaderboard_used_for_selection") is False
        and terminal.get("authorized_for_submission") is False
        and model_path.is_file()
        and sha256_file(model_path) == terminal.get("model_sha256")
    ):
        raise ValueError("handcrafted hard-negative gate is ineligible for audit")
    manifest_path = args.data_root / "real_division_hard_negative_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    features, targets, _, _, inventory = load_role(args.data_root, manifest, "audit")
    payload = joblib.load(model_path)
    if not (
        payload.get("run_id") == TRAINING_RUN_ID
        and payload.get("feature_family") == FEATURE_FAMILY
        and payload.get("feature_count") == 132
        and len(payload.get("models", [])) == 2
    ):
        raise ValueError("handcrafted hard-negative model payload changed")
    scores = score_payload(payload, features)
    threshold = float(terminal["frozen_division_probability_threshold"])
    metrics = high_precision_metrics(targets, scores)
    decisions = decision_summary(targets, scores, inventory, threshold)
    by_embryo = {}
    for embryo in ("44b6", "6bba"):
        indices = np.asarray(
            [index for index, row in enumerate(inventory) if row["embryo"] == embryo]
        )
        embryo_inventory = [inventory[index] for index in indices]
        by_embryo[embryo] = {
            "ranking": high_precision_metrics(targets[indices], scores[indices]),
            "decisions": decision_summary(
                targets[indices], scores[indices], embryo_inventory, threshold
            ),
        }
    accepted = bool(
        metrics["average_precision"] >= 0.55
        and decisions["tp"] >= 3
        and decisions["precision"] >= 0.90
        and decisions["jaccard"] > 0.0
        and decisions["no_division_hard_negative_false_positives"] == 0
        and all(row["ranking"]["average_precision"] >= 0.40 for row in by_embryo.values())
        and all(row["decisions"]["tp"] >= 1 for row in by_embryo.values())
    )
    result = {
        "schema_version": 1,
        "status": "accepted" if accepted else "rejected",
        "run_id": RUN_ID,
        "model_training_run_id": TRAINING_RUN_ID,
        "model_sha256": sha256_file(model_path),
        "training_terminal_sha256": sha256_file(terminal_path),
        "manifest_sha256": sha256_file(manifest_path),
        "model_threshold_frozen_before_audit": threshold,
        "ranking": metrics,
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
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
