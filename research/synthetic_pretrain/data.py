"""Streaming, geometry-safe access to the public physical synthetic dataset.

The source stores static images and points in native Biohub coordinates.  Its
temporal images are already XY-pooled, but their nodes remain native.  These
helpers make the conversion explicit and validate it before training.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


NATIVE_VOXEL_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float32)
POOLED_VOXEL_UM = np.asarray((1.625, 1.625, 1.625), dtype=np.float32)
DEFAULT_XY_STRIDE = 4
REAL_DIVISION_RATE = 0.0026
SYNTHETIC_DIVISION_RATE = 0.0407440315209261


@dataclass(frozen=True)
class StaticSample:
    volume: np.ndarray
    centroids: np.ndarray
    voxel_um: np.ndarray


@dataclass(frozen=True)
class SequenceSample:
    volumes: np.ndarray
    nodes: np.ndarray
    edges: np.ndarray
    divisions: np.ndarray
    voxel_um: np.ndarray


def _points_in_bounds(points: np.ndarray, shape: Sequence[int]) -> bool:
    if len(points) == 0:
        return True
    spatial_shape = np.asarray(tuple(shape), dtype=np.float32)
    return bool(np.all(points >= 0.0) and np.all(points < spatial_shape[None]))


def corrected_static_sample(path: str | Path, *, xy_stride: int = DEFAULT_XY_STRIDE) -> StaticSample:
    """Load a native static sample and convert image plus points to pooled space."""

    if xy_stride <= 0:
        raise ValueError("xy_stride must be positive")
    with np.load(Path(path), allow_pickle=False) as payload:
        volume = np.asarray(payload["volume"])
        centroids = np.asarray(payload["centroids"], dtype=np.float32)
        voxel_um = np.asarray(payload["voxel_um"], dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"static volume must be 3D, got {volume.shape}")
    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError(f"static centroids must have shape (N,3), got {centroids.shape}")
    if not np.allclose(voxel_um, NATIVE_VOXEL_UM, atol=1e-5):
        raise ValueError(f"unexpected native voxel scale: {voxel_um.tolist()}")

    pooled = np.ascontiguousarray(volume[:, ::xy_stride, ::xy_stride])
    pooled_centroids = centroids.copy()
    pooled_centroids[:, 1:] /= float(xy_stride)
    if not _points_in_bounds(pooled_centroids, pooled.shape):
        raise ValueError("corrected static centroids fall outside the pooled image")
    return StaticSample(pooled, pooled_centroids, POOLED_VOXEL_UM.copy())


def corrected_sequence_sample(path: str | Path, *, xy_stride: int = DEFAULT_XY_STRIDE) -> SequenceSample:
    """Load an already-pooled sequence and repair native Y/X node coordinates."""

    if xy_stride <= 0:
        raise ValueError("xy_stride must be positive")
    with np.load(Path(path), allow_pickle=False) as payload:
        volumes = np.asarray(payload["volumes"])
        nodes = np.asarray(payload["nodes"], dtype=np.float32)
        edges = np.asarray(payload["edges"], dtype=np.int64)
        divisions = np.asarray(payload["divisions"], dtype=np.int64)
        voxel_um = np.asarray(payload["voxel_um_pooled"], dtype=np.float32)
    if volumes.ndim != 4:
        raise ValueError(f"sequence volumes must have shape (T,Z,Y,X), got {volumes.shape}")
    if nodes.ndim != 2 or nodes.shape[1] < 5:
        raise ValueError(f"sequence nodes must have columns (t,z,y,x,tid), got {nodes.shape}")
    if edges.size and (edges.ndim != 2 or edges.shape[1] != 2):
        raise ValueError(f"sequence edges must have shape (E,2), got {edges.shape}")
    if not np.allclose(voxel_um, POOLED_VOXEL_UM, atol=1e-5):
        raise ValueError(f"unexpected pooled voxel scale: {voxel_um.tolist()}")

    corrected_nodes = nodes.copy()
    corrected_nodes[:, 2:4] /= float(xy_stride)
    if np.any(corrected_nodes[:, 0] < 0) or np.any(corrected_nodes[:, 0] >= volumes.shape[0]):
        raise ValueError("sequence node time lies outside the volume")
    if not _points_in_bounds(corrected_nodes[:, 1:4], volumes.shape[1:]):
        raise ValueError("corrected sequence nodes fall outside the pooled image")
    if edges.size and (np.any(edges < 0) or np.any(edges >= len(nodes))):
        raise ValueError("sequence edge references an invalid node row")
    if divisions.size and (np.any(divisions < 0) or np.any(divisions >= len(nodes))):
        raise ValueError("sequence division references an invalid node row")
    return SequenceSample(
        np.ascontiguousarray(volumes),
        corrected_nodes,
        edges.reshape(-1, 2),
        divisions.reshape(-1),
        POOLED_VOXEL_UM.copy(),
    )


def division_prior_weight(
    *, real_rate: float = REAL_DIVISION_RATE, synthetic_rate: float = SYNTHETIC_DIVISION_RATE
) -> float:
    """Importance weight for the inflated synthetic positive division prior."""

    if not 0.0 < real_rate < 1.0 or not 0.0 < synthetic_rate < 1.0:
        raise ValueError("division rates must lie strictly between zero and one")
    return float(real_rate / synthetic_rate)


def split_static_paths(
    paths: Sequence[str | Path], *, validation_count: int, salt: str = "biohub-synthetic-v1"
) -> tuple[list[Path], list[Path]]:
    """Deterministically split samples without relying on filesystem ordering."""

    resolved = [Path(path) for path in paths]
    if validation_count <= 0 or validation_count >= len(resolved):
        raise ValueError("validation_count must leave non-empty train and validation splits")
    ranked = sorted(
        resolved,
        key=lambda path: hashlib.sha256(f"{salt}:{path.name}".encode("utf-8")).hexdigest(),
    )
    validation = ranked[:validation_count]
    return ranked[validation_count:], validation


class SyntheticStaticStore:
    """Small-cache store shared by lazy image and point sequences."""

    def __init__(self, paths: Sequence[str | Path], *, xy_stride: int = DEFAULT_XY_STRIDE):
        self.paths = tuple(Path(path) for path in paths)
        self.xy_stride = int(xy_stride)
        if not self.paths:
            raise ValueError("at least one synthetic path is required")

    def __len__(self) -> int:
        return len(self.paths)

    @lru_cache(maxsize=2)
    def load(self, index: int) -> StaticSample:
        if index < 0:
            index += len(self.paths)
        if index < 0 or index >= len(self.paths):
            raise IndexError(index)
        return corrected_static_sample(self.paths[index], xy_stride=self.xy_stride)


class PooledImageSequence(Sequence[np.ndarray]):
    def __init__(self, store: SyntheticStaticStore):
        self.store = store

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.store.load(index).volume


class PointSequence(Sequence[np.ndarray]):
    def __init__(self, store: SyntheticStaticStore):
        self.store = store

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.store.load(index).centroids
