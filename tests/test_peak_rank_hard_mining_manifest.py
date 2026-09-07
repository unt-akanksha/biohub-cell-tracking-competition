from pathlib import Path

import pytest

from research.peak_rank_detection.build_hard_mining_manifest import (
    assign_multiplicities,
)


def _rows(embryo: str, count: int) -> list[dict]:
    return [
        {
            "identity": f"{embryo}_{index:03d}:t1",
            "embryo": embryo,
            "top64_recall_at_2_5_voxels": float(index),
            "mean_capped_distance_voxels": float(count - index),
        }
        for index in range(count)
    ]


def test_balances_embryos_and_repeats_hardest_first() -> None:
    entries = assign_multiplicities(
        [*_rows("44b6", 2), *_rows("6bba", 3)],
        expected_unique_counts={"44b6": 2, "6bba": 3},
        target_per_embryo=5,
    )
    by_id = {row["identity"]: row for row in entries}
    assert sum(row["sampling_multiplicity"] for row in entries[:2]) == 5
    assert sum(row["sampling_multiplicity"] for row in entries[2:]) == 5
    assert by_id["44b6_000:t1"]["sampling_multiplicity"] == 3
    assert by_id["44b6_001:t1"]["sampling_multiplicity"] == 2
    assert by_id["6bba_000:t1"]["sampling_multiplicity"] == 2
    assert by_id["6bba_002:t1"]["sampling_multiplicity"] == 1


def test_rejects_duplicate_or_changed_inventory() -> None:
    rows = [*_rows("44b6", 2), *_rows("6bba", 3)]
    with pytest.raises(ValueError, match="unique"):
        assign_multiplicities(
            [*rows, dict(rows[0])],
            expected_unique_counts={"44b6": 3, "6bba": 3},
            target_per_embryo=5,
        )
    with pytest.raises(ValueError, match="unexpected 44b6"):
        assign_multiplicities(
            rows,
            expected_unique_counts={"44b6": 3, "6bba": 3},
            target_per_embryo=5,
        )


def test_source_fails_closed_on_nonoptimization_roles() -> None:
    source = Path(
        "research/peak_rank_detection/build_hard_mining_manifest.py"
    ).read_text(encoding="utf-8")
    assert "bundle.extract(" not in source
    assert '"selection_data_read": False' in source
    assert '"sealed_audit_data_read": False' in source
    assert '"competition_test_data_read": False' in source
    assert '"public_leaderboard_used_for_selection": False' in source
