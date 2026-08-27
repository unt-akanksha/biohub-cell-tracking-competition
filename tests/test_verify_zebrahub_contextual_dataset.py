from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from research.temporal_contrastive.verify_zebrahub_contextual_dataset import (
    SPLIT_CONTRACT,
    safe_file,
    validate_arrays,
)


def write_arrays(path: Path, *, invalid_positive: bool = False) -> None:
    candidates = np.asarray(
        [[True, True, False], [False, True, True]], dtype=bool
    )
    positives = np.asarray(
        [[True, False, False], [False, False, True]], dtype=bool
    )
    if invalid_positive:
        positives[0, 2] = True
    np.savez_compressed(
        path,
        source_patches=np.zeros((2, 3, 17, 17, 17), dtype=np.float16),
        target_patches=np.zeros((3, 3, 17, 17, 17), dtype=np.float16),
        source_ids=np.asarray([10, 20], dtype=np.int64),
        target_ids=np.asarray([30, 40, 50], dtype=np.int64),
        source_coords_um=np.zeros((2, 3), dtype=np.float32),
        target_coords_um=np.zeros((3, 3), dtype=np.float32),
        candidate_mask=candidates,
        positive_mask=positives,
        division_target=(positives.sum(axis=1) >= 2).astype(np.float32),
        transition_context=np.zeros(8, dtype=np.float32),
        candidate_context=np.zeros((2, 3, 18), dtype=np.float32),
    )


def test_derived_array_verifier_checks_complete_contextual_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "transition.npz"
    write_arrays(path)

    summary = validate_arrays(path)

    assert summary == {
        "source_nodes": 2,
        "target_nodes": 3,
        "candidate_edges": 4,
        "positive_edges": 2,
        "division_sources": 0,
    }
    write_arrays(path, invalid_positive=True)
    with pytest.raises(ValueError, match="positives escape candidates"):
        validate_arrays(path)


def test_dataset_verifier_has_frozen_64_16_source_roles() -> None:
    assert SPLIT_CONTRACT["training"] == {
        "directory": "train",
        "source": "ZSNS004",
        "source_role": "external_pretraining",
        "count": 64,
    }
    assert SPLIT_CONTRACT["validation"] == {
        "directory": "validation",
        "source": "ZSNS005",
        "source_role": "external_validation",
        "count": 16,
    }


def test_array_verifier_rejects_division_only_sampling(tmp_path: Path) -> None:
    path = tmp_path / "division-heavy.npz"
    source_count = 4
    target_count = 9
    candidates = np.ones((source_count, target_count), dtype=bool)
    positives = np.zeros_like(candidates)
    for row in range(source_count):
        positives[row, 2 * row : 2 * row + 2] = True
    np.savez_compressed(
        path,
        source_patches=np.zeros((source_count, 3, 17, 17, 17), dtype=np.float16),
        target_patches=np.zeros((target_count, 3, 17, 17, 17), dtype=np.float16),
        source_ids=np.arange(source_count, dtype=np.int64),
        target_ids=np.arange(target_count, dtype=np.int64),
        source_coords_um=np.zeros((source_count, 3), dtype=np.float32),
        target_coords_um=np.zeros((target_count, 3), dtype=np.float32),
        candidate_mask=candidates,
        positive_mask=positives,
        division_target=np.ones(source_count, dtype=np.float32),
        transition_context=np.zeros(8, dtype=np.float32),
        candidate_context=np.zeros(
            (source_count, target_count, 18), dtype=np.float32
        ),
    )

    with pytest.raises(ValueError, match="division sampling is imbalanced"):
        validate_arrays(path)


def test_dataset_verifier_rejects_path_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-derived-shard.npz"
    outside.write_bytes(b"outside")
    try:
        with pytest.raises(ValueError, match="escapes"):
            safe_file(tmp_path.resolve(), "../outside-derived-shard.npz")
    finally:
        outside.unlink()
