from research.evaluate_ranked_consensus_division_recovery import (
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
