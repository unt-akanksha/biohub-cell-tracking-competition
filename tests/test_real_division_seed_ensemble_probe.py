from pathlib import Path
import runpy
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "research/temporal_contrastive/score_real_division_seed_ensemble_probe.py"
sys.modules.setdefault("zarr", types.ModuleType("zarr"))
MODULE = runpy.run_path(str(SCRIPT))


def admitted(seed: int, fold: str, model_hash: str, ap: float) -> dict:
    return {
        "seed": seed,
        "fold": fold,
        "status": "admitted",
        "model_sha256": model_hash,
        "selection": {"average_precision": ap},
        "selection_by_embryo": {
            "44b6": {"average_precision": 0.50},
            "6bba": {"average_precision": 0.51},
        },
        "selection_frozen_threshold_diagnostic": {"tp": 3, "fp": 0},
    }


def terminal() -> dict:
    return {
        "schema_version": 1,
        "run_id": MODULE["ENSEMBLE_RUN_ID"],
        "status": "ensemble_eligible_for_development_probe",
        "authorized_for_development_probe": True,
        "competition_test_data_read": False,
        "final_probe_opened": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
        "ensemble_eligible_for_development_probe": True,
        "ensemble": {
            "model_count": 2,
            "absolute_threshold_authorized": False,
            "policy": "equal average of within-movie percentile ranks from every independently admitted model",
        },
        "candidates": [
            admitted(1, "target_44b6", "a" * 64, 0.59),
            admitted(2, "target_6bba", "b" * 64, 0.58),
        ],
        "stronger_individuals": [],
    }


def test_eligible_ensemble_uses_every_independently_admitted_member() -> None:
    policy, members = MODULE["select_precommitted_members"](terminal())

    assert policy == "equal_rank_admitted_ensemble"
    assert [row["model_sha256"] for row in members] == ["a" * 64, "b" * 64]


def test_duplicate_checkpoint_cannot_vote_twice() -> None:
    payload = terminal()
    payload["candidates"][1]["model_sha256"] = "a" * 64

    with pytest.raises(ValueError, match="byte-identical"):
        MODULE["select_precommitted_members"](payload)


def test_fallback_uses_only_strongest_predeclared_individual() -> None:
    payload = terminal()
    reference = MODULE["REFERENCE_SELECTION_AP"]
    first = admitted(3, "target_44b6", "c" * 64, reference + 0.03)
    second = admitted(4, "target_6bba", "d" * 64, reference + 0.02)
    payload.update(
        {
            "status": "stronger_individuals_eligible_for_development_probe",
            "ensemble_eligible_for_development_probe": False,
            "ensemble": None,
            "candidates": [second, first],
            "stronger_individuals": [
                {
                    "seed": 3,
                    "fold": "target_44b6",
                    "model_sha256": "c" * 64,
                },
                {
                    "seed": 4,
                    "fold": "target_6bba",
                    "model_sha256": "d" * 64,
                },
            ],
        }
    )

    policy, members = MODULE["select_precommitted_members"](payload)

    assert policy == "strongest_individual_rank"
    assert len(members) == 1
    assert members[0]["model_sha256"] == "c" * 64


def test_movie_ranks_are_calibration_free_and_do_not_cross_movies() -> None:
    raw = np.asarray([[100.0, 10.0, 0.0, -500.0], [0.1, 0.2, 9.0, 8.0]])
    stems = ["a", "a", "b", "b"]

    ranked = MODULE["movie_rank_percentiles"](raw, stems)
    shifted = MODULE["movie_rank_percentiles"](raw * 200.0 - 7.0, stems)

    assert np.array_equal(ranked, shifted)
    assert np.array_equal(ranked[0], np.asarray([1.0, 0.0, 1.0, 0.0]))
    assert np.array_equal(ranked[1], np.asarray([0.0, 1.0, 1.0, 0.0]))
