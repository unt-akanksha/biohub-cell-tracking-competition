from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "research/build_competition_relational_division_inventory.py"
MODULE = runpy.run_path(str(SCRIPT))


def node(t: int, z: float, y: float, x: float) -> tuple[float, ...]:
    return (float(t), z, y, x)


def test_relational_examples_include_true_pair_and_candidate_specific_negative() -> None:
    nodes = {
        1: node(0, 5, 20, 20),
        2: node(1, 5, 20, 10),
        3: node(1, 5, 20, 30),
        4: node(0, 5, 50, 50),
        5: node(1, 5, 50, 40),
        6: node(1, 5, 50, 60),
    }
    edges = [(1, 2), (1, 3), (4, 5)]

    rows = MODULE["movie_examples"]("44b6_fixture", nodes, edges)

    positives = [row for row in rows if row["division_recovery_target"]]
    negatives = [row for row in rows if not row["division_recovery_target"]]
    assert len(positives) == 1
    assert positives[0]["parent_id"] == 1
    assert {positives[0]["existing_child_id"], positives[0]["proposed_child_id"]} == {
        2,
        3,
    }
    assert negatives
    assert all(row["daughter_order_invariant"] is True for row in rows)
    assert positives[0]["inference_geometry_eligible"] is True


def test_relational_examples_exclude_geometry_ineligible_pairs() -> None:
    nodes = {
        1: node(0, 5, 5, 5),
        2: node(1, 5, 5, 50),
        3: node(1, 5, 5, 90),
    }

    rows = MODULE["movie_examples"]("6bba_fixture", nodes, [(1, 2), (1, 3)])

    assert rows == []


def test_role_allocation_is_disjoint_and_keeps_both_embryos() -> None:
    movies = []
    for embryo in ("44b6", "6bba"):
        for index in range(10):
            movies.append(
                {
                    "stem": f"{embryo}_{index:08d}",
                    "embryo": embryo,
                    "positives": 1,
                    "hard_negatives": 1,
                    "inference_eligible_positives": 1,
                    "inference_eligible_hard_negatives": 1,
                }
            )

    roles = MODULE["allocate_roles"](movies)

    assert set(roles.values()) == {"optimization", "selection", "audit"}
    for embryo in ("44b6", "6bba"):
        observed = {
            role for stem, role in roles.items() if stem.startswith(embryo + "_")
        }
        assert observed == {"optimization", "selection", "audit"}


def test_role_allocation_can_supplement_positive_movies_with_negative_only_movies() -> None:
    movies = []
    for embryo in ("44b6", "6bba"):
        for index in range(6):
            movies.append(
                {
                    "stem": f"{embryo}_positive_{index}",
                    "embryo": embryo,
                    "positives": 1,
                    "hard_negatives": 0,
                    "inference_eligible_positives": 1,
                    "inference_eligible_hard_negatives": 0,
                }
            )
        for index in range(6):
            movies.append(
                {
                    "stem": f"{embryo}_negative_{index}",
                    "embryo": embryo,
                    "positives": 0,
                    "hard_negatives": 8,
                    "inference_eligible_positives": 0,
                    "inference_eligible_hard_negatives": 8,
                }
            )

    roles = MODULE["allocate_roles"](movies)

    for embryo in ("44b6", "6bba"):
        for role in ("selection", "audit"):
            assigned = [
                movie
                for movie in movies
                if movie["embryo"] == embryo and roles[movie["stem"]] == role
            ]
            assert sum(movie["positives"] for movie in assigned) > 0
            assert sum(movie["hard_negatives"] for movie in assigned) > 0


def test_negative_cap_retains_inference_eligible_distractors_first() -> None:
    rows = [
        {
            "parent_id": index,
            "proposed_child_id": index + 100,
            "inference_geometry_eligible": False,
        }
        for index in range(40)
    ]
    rows.append(
        {
            "parent_id": 999,
            "proposed_child_id": 1000,
            "inference_geometry_eligible": True,
        }
    )

    selected = MODULE["select_negative_examples"]("44b6_fixture", rows, 32)

    assert len(selected) == 32
    assert any(row["parent_id"] == 999 for row in selected)
