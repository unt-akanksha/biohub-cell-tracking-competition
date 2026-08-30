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
