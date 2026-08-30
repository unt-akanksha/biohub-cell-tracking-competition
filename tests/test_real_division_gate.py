from __future__ import annotations

import torch

from research.temporal_contrastive.train_real_division_gate import (
    balanced_rows,
    select_frozen_threshold,
    selection_utility,
    threshold_metrics,
    validate_manifest,
    weighted_focal_loss,
)


def eligible_manifest() -> dict:
    return {
        "schema_version": 1,
        "status": "complete",
        "run_id": "competition-real-division-patches-v1",
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
            "ordinary_unlabeled_controls": 1247,
            "by_embryo_role": {
                "44b6": {
                    "optimization": {"division_positives": 19},
                    "selection": {"division_positives": 5},
                },
                "6bba": {
                    "optimization": {"division_positives": 96},
                    "selection": {"division_positives": 26},
                },
            },
        },
    }


def test_manifest_binds_final_probe_exclusion_and_exact_counts() -> None:
    manifest = eligible_manifest()
    validate_manifest(manifest)
    manifest["summary"]["division_positives"] = 147
    try:
        validate_manifest(manifest)
    except ValueError:
        pass
    else:
        raise AssertionError("changed positive inventory must fail")


def test_balanced_rows_matches_pu_weighted_one_to_four_policy() -> None:
    targets = torch.tensor([1.0] * 10 + [0.0] * 30)
    rows = balanced_rows(targets, 20, torch.Generator().manual_seed(7))
    assert len(rows) == 20
    assert int((targets[rows] > 0.5).sum()) == 4


def test_threshold_freeze_prefers_last_zero_false_positive_cutoff() -> None:
    labels = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0])
    scores = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    selected = select_frozen_threshold(labels, scores)
    assert selected["threshold"] == 4.0
    assert selected["tp"] == 2
    assert selected["fp"] == 0


def test_metrics_and_utility_reward_high_precision_recall_first() -> None:
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    weaker = threshold_metrics(labels, torch.tensor([2.0, -1.0, 1.0, 0.0]))
    stronger = threshold_metrics(labels, torch.tensor([2.0, 1.0, 0.0, -1.0]))
    assert selection_utility(stronger) > selection_utility(weaker)


def test_pu_weight_reduces_negative_loss_contribution() -> None:
    logits = torch.tensor([0.0, 2.0])
    targets = torch.tensor([1.0, 0.0])
    weighted = weighted_focal_loss(logits, targets, torch.tensor([1.0, 0.25]))
    unweighted = weighted_focal_loss(logits, targets, torch.ones(2))
    assert weighted < unweighted
