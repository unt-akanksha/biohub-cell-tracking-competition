from __future__ import annotations

import numpy as np
import pytest
import torch

from research.temporal_contrastive.score_real_division_gate_v2_audit import (
    validate_terminal,
)
from research.temporal_contrastive.train_real_division_gate_v2 import (
    frozen_decisions,
    validate_manifest,
)


def eligible_manifest() -> dict:
    strata = {
        embryo: {
            role: {"division_positives": 2, "negative_frames": 3}
            for role in ("optimization", "selection", "audit")
        }
        for embryo in ("44b6", "6bba")
    }
    return {
        "schema_version": 1,
        "status": "complete",
        "run_id": "competition-real-division-hard-negative-patches-v2",
        "audit_opened": False,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
        "final_probe_stems": ["a", "b", "c", "d"],
        "summary": {
            "movies": 199,
            "excluded_final_probe_movies": 4,
            "division_positives": 146,
            "negative_frames": 190,
            "by_embryo_role": strata,
        },
    }


def test_manifest_requires_a_sealed_movie_disjoint_audit() -> None:
    manifest = eligible_manifest()
    validate_manifest(manifest)
    manifest["audit_opened"] = True
    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_frozen_decisions_report_negative_frame_false_positives() -> None:
    targets = torch.tensor([1.0, 0.0, 0.0, 1.0])
    scores = torch.tensor([2.0, 1.8, -1.0, 1.5])
    inventory = [
        {"frame_role": "division"},
        {"frame_role": "no_division_hard_negative"},
        {"frame_role": "no_division_hard_negative"},
        {"frame_role": "division"},
    ]

    result = frozen_decisions(targets, scores, inventory, 1.4)

    assert result["tp"] == 2
    assert result["fp"] == 1
    assert result["by_frame_role"]["no_division_hard_negative"]["false_positives"] == 1


def test_audit_terminal_binds_models_threshold_and_weights(tmp_path) -> None:
    model_paths = [tmp_path / "a.pt", tmp_path / "b.pt"]
    model_paths[0].write_bytes(b"a")
    model_paths[1].write_bytes(b"b")
    import hashlib

    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in model_paths]
    terminal = {
        "schema_version": 1,
        "status": "accepted_at_selection",
        "run_id": "competition-real-division-hard-negative-gate-v2",
        "family": "competition_real_hard_negative_temporal_division_gate_v2",
        "selection_gate_passed": True,
        "audit_opened": False,
        "checkpoint_frozen_before_audit": True,
        "authorized_for_audit": True,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
        "frozen_division_logit_threshold": 0.75,
        "ensemble_weights": {"target_44b6": 0.25, "target_6bba": 0.75},
        "folds": {
            "target_44b6": {"model_sha256": hashes[0]},
            "target_6bba": {"model_sha256": hashes[1]},
        },
    }

    threshold, weights = validate_terminal(terminal, model_paths)

    assert threshold == 0.75
    np.testing.assert_allclose(weights, [0.25, 0.75])
