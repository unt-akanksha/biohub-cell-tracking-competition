from __future__ import annotations

from research.build_competition_division_training_inventory import (
    movie_examples,
    stratified_selection_stems,
)


def test_same_event_frame_examples_keep_safe_negatives_and_divisions() -> None:
    nodes = {
        1: (4, 0.0, 0.0, 0.0),
        2: (4, 1.0, 1.0, 1.0),
        3: (4, 2.0, 2.0, 2.0),
        4: (5, 0.0, 1.0, 0.0),
        5: (5, 0.0, -1.0, 0.0),
        6: (5, 1.0, 2.0, 1.0),
    }
    edges = [(1, 4), (1, 5), (2, 6)]

    rows = movie_examples(nodes, edges)

    assert [row["node_id"] for row in rows] == [1, 2]
    assert [row["division_target"] for row in rows] == [True, False]
    assert rows[0]["child_ids"] == [4, 5]


def test_frames_without_divisions_do_not_create_negative_labels() -> None:
    nodes = {1: (4, 0.0, 0.0, 0.0), 2: (5, 0.0, 1.0, 0.0)}

    assert movie_examples(nodes, [(1, 2)]) == []


def test_selection_is_deterministic_and_division_stratified() -> None:
    movies = []
    for embryo in ("44b6", "6bba"):
        for index in range(10):
            movies.append(
                {
                    "stem": f"{embryo}_{index:08x}",
                    "embryo": embryo,
                    "role": "unassigned",
                    "summary": {"division_positives": 1},
                }
            )

    selected = stratified_selection_stems(movies)

    assert selected == stratified_selection_stems(movies)
    assert sum(stem.startswith("44b6_") for stem in selected) == 2
    assert sum(stem.startswith("6bba_") for stem in selected) == 2
