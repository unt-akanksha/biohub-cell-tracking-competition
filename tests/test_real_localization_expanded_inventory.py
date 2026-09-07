from pathlib import Path

import pytest

from research import build_competition_real_localization_expanded_inventory as expanded


def test_temporal_expansion_preserves_existing_and_spreads_centers() -> None:
    centers = expanded.temporally_spread_centers(
        list(range(1, 10)), [5], maximum=5
    )
    assert centers == [1, 3, 5, 7, 9]


def test_temporal_expansion_never_removes_dense_event_inventory() -> None:
    centers = expanded.temporally_spread_centers(
        list(range(1, 10)), [1, 2, 3, 4, 5, 6], maximum=5
    )
    assert centers == [1, 2, 3, 4, 5, 6]


def test_temporal_expansion_rejects_unannotated_center() -> None:
    with pytest.raises(ValueError, match="eligible annotated"):
        expanded.temporally_spread_centers([1, 2, 3], [4], maximum=5)


def test_only_optimization_centers_expand(tmp_path: Path, monkeypatch) -> None:
    stems = ("44b6_train", "6bba_select", "6bba_audit")
    for stem in stems:
        (tmp_path / f"{stem}.geff").mkdir()

    movies = [
        {
            "stem": stem,
            "embryo": stem.split("_")[0],
            "role": role,
            "center_frames": [5],
            "required_frames": [4, 5, 6],
            "annotated_nodes_at_centers": 1,
        }
        for stem, role in zip(
            stems, ("optimization", "selection", "sealed_audit"), strict=True
        )
    ]

    def graph_loader(_path: Path):
        return {index: (float(index), 0.0, 0.0, 0.0) for index in range(11)}, []

    monkeypatch.setattr(expanded, "FINAL_PROBE_STEMS", set())
    result = expanded.expand_movies(movies, tmp_path, graph_loader=graph_loader)
    by_role = {row["role"]: row for row in result}
    assert by_role["optimization"]["center_frames"] == [1, 3, 5, 7, 9]
    assert by_role["selection"]["center_frames"] == [5]
    assert by_role["sealed_audit"]["center_frames"] == [5]
