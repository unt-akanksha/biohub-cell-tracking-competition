from __future__ import annotations

from research.evaluate_relational_division_development import (
    select_agreement_rows,
)


def row(stem: str, parent: int, relational: float, positive: bool = False) -> dict:
    return {
        "stem": stem,
        "timepoint": 5,
        "parent_id": parent,
        "second_child_id": parent + 100,
        "ensemble_logit": relational,
        "safe_recovery_positive": positive,
    }


def test_agreement_selects_at_most_one_candidate_per_movie() -> None:
    relational = [row("a", 1, 0.9, True), row("a", 2, 0.2)]
    morphology = [row("a", 1, 0.8, True), row("a", 2, 0.1)]

    selected = select_agreement_rows(relational, morphology)

    assert len(selected) == 1
    assert selected[0]["parent_id"] == 1
    assert selected[0]["morphology_probability"] == 0.8


def test_disagreement_adds_no_edge() -> None:
    relational = [row("a", 1, 0.9), row("a", 2, 0.2)]
    morphology = [row("a", 1, 0.1), row("a", 2, 0.8)]

    assert select_agreement_rows(relational, morphology) == []
