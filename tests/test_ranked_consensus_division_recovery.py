from research.evaluate_ranked_consensus_division_recovery import (
    eligible_deep_probe,
    select_consensus_rows,
)


def test_consensus_requires_geometry_and_agreement_without_absolute_threshold() -> None:
    deep = [
        {
            "stem": "a",
            "timepoint": 3,
            "parent_id": 1,
            "geometry_division_score": 4.0,
            "ensemble_logit": -20.0,
        },
        {
            "stem": "a",
            "timepoint": 3,
            "parent_id": 2,
            "geometry_division_score": 5.0,
            "ensemble_logit": -21.0,
        },
        {
            "stem": "b",
            "timepoint": 5,
            "parent_id": 3,
            "geometry_division_score": 2.0,
            "ensemble_logit": 100.0,
        },
    ]
    morphology = [
        {"stem": "a", "timepoint": 3, "parent_id": 1, "ensemble_logit": 0.1},
        {"stem": "a", "timepoint": 3, "parent_id": 2, "ensemble_logit": 0.0},
        {"stem": "b", "timepoint": 5, "parent_id": 3, "ensemble_logit": 0.9},
    ]

    selected = select_consensus_rows(deep, morphology)

    assert [(row["stem"], row["parent_id"]) for row in selected] == [("a", 1)]
    assert selected[0]["ensemble_logit"] == -20.0


def test_overnight_probe_requires_precommitted_threshold_free_policy() -> None:
    payload = {
        "run_id": "competition-real-division-seed-ensemble-probe-v1",
        "status": "development_probe_complete",
        "authorized_for_ranked_consensus_development_evaluation": True,
        "absolute_threshold_used": False,
        "weights_searched_on_probe": False,
        "model_subset_searched_on_probe": False,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }

    assert eligible_deep_probe(payload)
    payload["weights_searched_on_probe"] = True
    assert not eligible_deep_probe(payload)
