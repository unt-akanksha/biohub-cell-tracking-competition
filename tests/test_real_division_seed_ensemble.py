from pathlib import Path
import runpy

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "research/temporal_contrastive/evaluate_real_division_seed_ensemble.py"
MODULE = runpy.run_path(str(SCRIPT))


def test_movie_rank_percentiles_are_calibration_free_and_grouped() -> None:
    scores = torch.tensor([100.0, 10.0, -5.0, 0.2, 0.1])
    inventory = [
        {"stem": "a"},
        {"stem": "a"},
        {"stem": "a"},
        {"stem": "b"},
        {"stem": "b"},
    ]

    ranked = MODULE["movie_rank_percentiles"](scores, inventory)

    assert torch.allclose(ranked, torch.tensor([1.0, 0.5, 0.0, 1.0, 0.0]))
    shifted = MODULE["movie_rank_percentiles"](scores * 50.0 - 800.0, inventory)
    assert torch.equal(ranked, shifted)


def test_movie_top_metrics_select_exactly_one_row_per_movie() -> None:
    targets = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0])
    scores = torch.tensor([0.0, 2.0, 1.0, 3.0, 2.0])
    inventory = [
        {"stem": "a"},
        {"stem": "a"},
        {"stem": "a"},
        {"stem": "b"},
        {"stem": "b"},
    ]

    metrics = MODULE["movie_top_metrics"](targets, scores, inventory)

    assert metrics == {
        "selected_movies": 2,
        "positive_movies": 2,
        "true_positive_tops": 1,
        "false_positive_tops": 1,
        "precision": 0.5,
        "positive_movie_recall": 0.5,
    }


def test_individual_gate_requires_every_component_to_be_strong() -> None:
    pooled = {"average_precision": 0.61}
    embryos = {
        "44b6": {"average_precision": 0.55},
        "6bba": {"average_precision": 0.58},
    }
    frozen = {"tp": 3, "fp": 0}

    assert MODULE["individual_admitted"](pooled, embryos, frozen)
    embryos["6bba"]["average_precision"] = 0.39
    assert not MODULE["individual_admitted"](pooled, embryos, frozen)


def test_stronger_individual_can_progress_without_ensemble_gain() -> None:
    reference = MODULE["REFERENCE_SELECTION_AP"]
    required_gain = MODULE["MINIMUM_INDIVIDUAL_AP_GAIN"]
    candidate = {
        "status": "admitted",
        "selection": {
            "average_precision": reference + required_gain + 1e-6,
        },
    }

    assert MODULE["stronger_than_reference"](candidate)
    candidate["selection"]["average_precision"] = reference + required_gain - 1e-6
    assert not MODULE["stronger_than_reference"](candidate)
    candidate["status"] = "rejected"
    candidate["selection"]["average_precision"] = reference + 1.0
    assert not MODULE["stronger_than_reference"](candidate)


def test_reference_voter_is_hash_and_metric_bound() -> None:
    payload = {
        "schema_version": 1,
        "status": "accepted_at_selection",
        "run_id": MODULE["TRAIN_RUN_ID"],
        "selection_gate_passed": True,
        "ensemble_weights": {"target_44b6": 0.0, "target_6bba": 1.0},
        "ensemble_selection": {
            "average_precision": MODULE["REFERENCE_SELECTION_AP"],
        },
        "folds": {
            "target_6bba": {
                "model_sha256": MODULE["REFERENCE_MODEL_SHA256"],
            },
        },
        "competition_test_data_read": False,
        "final_probe_opened": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }

    MODULE["validate_reference_terminal"](payload)
    payload["folds"]["target_6bba"]["model_sha256"] = "0" * 64
    try:
        MODULE["validate_reference_terminal"](payload)
    except ValueError as error:
        assert "reference voter" in str(error)
    else:
        raise AssertionError("changed reference checkpoint was accepted")


def test_training_config_binds_independent_effective_seed_and_recipe() -> None:
    seed = 205_043
    fold = "target_6bba"
    payload = {
        "schema_version": 1,
        "run_id": MODULE["TRAIN_RUN_ID"],
        "family": "competition_real_temporal_multiscale_division_gate_v1",
        "fold": fold,
        "seed": seed + MODULE["SEED_OFFSETS"][fold],
        "train_mode": "focused",
        "steps": 50_000,
        "batch_size": 48,
        "learning_rate": 2e-5,
        "minimum_learning_rate": 2e-7,
        "weight_decay": 1e-4,
        "ema_decay": 0.995,
        "initial_model_sha256": MODULE["INITIAL_MODEL_SHA256"][fold],
        "trainable_parameters": 25_178_047,
        "frozen_parameters": 21_208_560,
        "competition_test_data_read": False,
        "final_probe_opened": False,
        "submission_created": False,
    }

    assert MODULE["validate_training_config"](payload, seed, fold) == 215_046
    payload["seed"] += 1
    try:
        MODULE["validate_training_config"](payload, seed, fold)
    except ValueError as error:
        assert "training config" in str(error)
    else:
        raise AssertionError("reused or shifted training seed was accepted")


def test_byte_identical_checkpoints_are_not_counted_as_diverse_members() -> None:
    owners = {}
    model_hash = "a" * 64

    assert (
        MODULE["register_checkpoint_owner"](
            owners, model_hash, seed=205_043, fold="target_44b6"
        )
        is None
    )
    duplicate = MODULE["register_checkpoint_owner"](
        owners, model_hash, seed=305_047, fold="target_44b6"
    )

    assert duplicate == {"seed": 205_043, "fold": "target_44b6"}
    assert len(owners) == 1
