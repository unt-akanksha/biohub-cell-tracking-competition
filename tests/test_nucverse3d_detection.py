from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.nucverse3d_detection.screen_pretrained import (
    MODEL_SHAPE,
    ONNX_SHA256,
    attractor_centroids,
    compatibility_passed,
    physical_patch,
    stable_subset,
    validate_optimization_receipt,
)


def test_physical_patch_preserves_center_mapping() -> None:
    volume = np.zeros((64, 64, 64), dtype=np.float32)
    point = np.array([31.25, 30.5, 29.75], dtype=np.float32)
    patch, target = physical_patch(volume + np.indices(volume.shape).sum(axis=0), point)
    assert patch.shape == MODEL_SHAPE
    assert patch.dtype == np.float32
    assert np.allclose(target, [34.5, 67.5, 68.5])


def test_attractor_decoder_finds_known_center(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "research.nucverse3d_detection.screen_pretrained.MAX_FOREGROUND_POINTS", 50_000
    )
    shape = (17, 19, 21)
    center = np.array([8.0, 9.0, 10.0], dtype=np.float32)
    grid = np.indices(shape, dtype=np.float32).transpose(1, 2, 3, 0)
    delta = center - grid
    distance = np.linalg.norm(delta, axis=-1)
    probability = (distance <= 5).astype(np.float32)
    gradient = delta / np.maximum(distance[..., None], 1.0)
    centers, diagnostics = attractor_centroids(
        probability, gradient, iterations=60
    )
    assert diagnostics["foreground_voxels"] > 0
    assert len(centers) >= 1
    assert np.linalg.norm(centers - center, axis=1).min() <= 1.5


def test_stable_subset_is_balanced_and_order_independent() -> None:
    rows = [
        {"embryo": embryo, "path": f"optimization/{embryo}_{index}.npz"}
        for embryo in ("44b6", "6bba")
        for index in range(10)
    ]
    forward = stable_subset(rows, per_embryo=3)
    reverse = stable_subset(reversed(rows), per_embryo=3)
    assert forward == reverse
    assert [row["embryo"] for row in forward].count("44b6") == 3
    assert [row["embryo"] for row in forward].count("6bba") == 3


def test_selection_receipt_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    receipt = {
        "run_id": "nucverse3d-generalized-physical-compatibility-v1",
        "phase": "optimization",
        "compatibility_passed": True,
        "onnx_sha256": ONNX_SHA256,
        "public_leaderboard_used_for_selection": False,
        "competition_test_data_read": False,
    }
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert validate_optimization_receipt(path, ONNX_SHA256) == receipt
    receipt["compatibility_passed"] = False
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot authorize"):
        validate_optimization_receipt(path, ONNX_SHA256)


def test_compatibility_gate_is_frozen() -> None:
    passing = {
        "points": 16,
        "foreground_support_recall": 0.85,
        "attractor_recall": 0.80,
        "finite_attractor_fraction": 0.90,
        "mean_finite_attractor_distance_model_voxels": 8.0,
    }
    assert compatibility_passed(passing)
    for key, value in {
        "points": 15,
        "foreground_support_recall": 0.84,
        "attractor_recall": 0.79,
        "finite_attractor_fraction": 0.89,
        "mean_finite_attractor_distance_model_voxels": 8.01,
    }.items():
        failing = dict(passing)
        failing[key] = value
        assert not compatibility_passed(failing)
