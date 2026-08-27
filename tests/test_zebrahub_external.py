from __future__ import annotations

import numpy as np
import pytest

from research.temporal_contrastive.zebrahub_external import (
    EXPECTED_LEVEL0_SPACING_UM,
    ORGANIZER_AUTHORIZATION,
    SOURCE_SPECS,
    select_dense_transition,
    validate_public_url,
)


def test_public_url_is_strictly_scoped() -> None:
    valid = (
        "https://public.czbiohub.org/royerlab/zebrahub/imaging/"
        "single-objective/ZSNS004_tracks.csv"
    )
    assert validate_public_url(valid) == valid
    with pytest.raises(ValueError, match="public CZ Biohub"):
        validate_public_url("https://example.com/ZSNS004_tracks.csv")
    with pytest.raises(ValueError, match="outside"):
        validate_public_url("https://public.czbiohub.org/royerlab/private/file")
    with pytest.raises(ValueError, match="query"):
        validate_public_url(valid + "?download=1")


def test_external_split_is_frozen_and_authorized() -> None:
    assert SOURCE_SPECS["ZSNS004"]["role"] == "external_pretraining"
    assert SOURCE_SPECS["ZSNS005"]["role"] == "external_validation"
    assert "ZSNS003" not in SOURCE_SPECS
    assert ORGANIZER_AUTHORIZATION.endswith("/discussion/734330")


def test_dense_transition_retains_division_and_hard_negatives() -> None:
    spacing = np.asarray(EXPECTED_LEVEL0_SPACING_UM, dtype=np.float32)
    source_ids = np.asarray([10, 20, 30], dtype=np.int64)
    source_um = np.asarray(
        [[10.0, 10.0, 10.0], [40.0, 10.0, 10.0], [70.0, 10.0, 10.0]],
        dtype=np.float32,
    )
    target_ids = np.asarray([101, 102, 201, 301, 999], dtype=np.int64)
    target_um = np.asarray(
        [
            [11.0, 9.0, 10.0],
            [11.0, 12.0, 10.0],
            [41.0, 10.0, 10.0],
            [71.0, 10.0, 10.0],
            [15.0, 10.0, 10.0],
        ],
        dtype=np.float32,
    )
    parents = np.asarray([10, 10, 20, 30, -1], dtype=np.int64)
    result = select_dense_transition(
        source_ids,
        source_um / spacing,
        target_ids,
        target_um / spacing,
        parents,
        voxel_size_level0_um=spacing,
        radius_um=32.0,
        max_sources=3,
        max_targets=5,
        seed=7,
    )
    result.validate()
    division_row = int(np.flatnonzero(result.source_ids == 10)[0])
    assert result.positive_mask[division_row].sum() == 2
    assert result.division_target[division_row] == 1.0
    assert np.all(result.candidate_mask.sum(axis=1) > result.positive_mask.sum(axis=1))
    assert set(result.target_ids[result.positive_mask[division_row]]) == {101, 102}


def test_dense_transition_rejects_positive_outside_radius() -> None:
    with pytest.raises(ValueError, match="omitted selected ZebraHub links"):
        select_dense_transition(
            np.asarray([1]),
            np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            np.asarray([2, 3]),
            np.asarray([[100.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            np.asarray([1, -1]),
            voxel_size_level0_um=(1.0, 1.0, 1.0),
            radius_um=32.0,
            max_sources=1,
            max_targets=2,
        )
