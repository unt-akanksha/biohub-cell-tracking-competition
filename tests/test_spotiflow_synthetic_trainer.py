from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.spotiflow_biohub.train_synthetic_detector import (
    EXPECTED_NATIVE_SHAPE,
    build_sequences,
    find_manifest,
    load_static_records,
    split_records,
)


def make_source(tmp_path: Path, count: int = 5) -> Path:
    static = tmp_path / "static"
    static.mkdir()
    records = []
    for index in range(count):
        path = static / f"vol_{index:05d}.npz"
        np.savez(
            path,
            volume=np.zeros(EXPECTED_NATIVE_SHAPE, dtype=np.uint8),
            centroids=np.asarray([[3.0, 128.0, 128.0]], dtype=np.float32),
            voxel_um=np.asarray([1.625, 0.40625, 0.40625], dtype=np.float32),
        )
        records.append(
            {
                "file": f"static/{path.name}",
                "n_cells": 1,
                "shape": list(EXPECTED_NATIVE_SHAPE),
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"static": records}), encoding="utf-8")
    return manifest


def test_manifest_split_and_lazy_sequences(tmp_path: Path) -> None:
    manifest = make_source(tmp_path)
    assert find_manifest(tmp_path) == manifest
    records = load_static_records(manifest)
    train, validation = split_records(records, validation_count=2, train_limit=2)
    assert len(train) == 2
    assert len(validation) == 2
    assert {item["path"] for item in train}.isdisjoint(
        item["path"] for item in validation
    )
    images, points = build_sequences(train)
    assert images[0].shape == (64, 64, 64)
    assert points[0].shape == (1, 3)
    np.testing.assert_allclose(np.asarray(points[0]), [[3.0, 32.0, 32.0]])


def test_manifest_rejects_unexpected_geometry(tmp_path: Path) -> None:
    manifest = make_source(tmp_path, count=1)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["static"][0]["shape"] = [32, 256, 256]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected synthetic shape"):
        load_static_records(manifest)
