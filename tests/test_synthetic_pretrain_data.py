from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from research.synthetic_pretrain.data import (
    PointSequence,
    PooledImageSequence,
    SyntheticStaticStore,
    corrected_sequence_sample,
    corrected_sequence_graph,
    corrected_static_sample,
    division_prior_weight,
    split_static_paths,
)


def write_static(path: Path, *, x: float = 252.0) -> None:
    np.savez(
        path,
        volume=np.zeros((8, 256, 256), dtype=np.uint16),
        centroids=np.asarray([[3.0, 128.0, x]], dtype=np.float32),
        voxel_um=np.asarray([1.625, 0.40625, 0.40625], dtype=np.float32),
    )


def test_static_coordinates_and_image_are_pooled_together(tmp_path: Path) -> None:
    path = tmp_path / "vol_00000.npz"
    write_static(path)
    sample = corrected_static_sample(path)
    assert sample.volume.shape == (8, 64, 64)
    np.testing.assert_allclose(sample.centroids, [[3.0, 32.0, 63.0]])
    np.testing.assert_allclose(sample.voxel_um, [1.625, 1.625, 1.625])


def test_temporal_native_coordinates_are_repaired_once(tmp_path: Path) -> None:
    path = tmp_path / "seq_0000.npz"
    np.savez(
        path,
        volumes=np.zeros((2, 8, 64, 64), dtype=np.uint16),
        nodes=np.asarray([[0, 3, 120, 200, 7], [1, 4, 124, 204, 7]], dtype=np.float32),
        edges=np.asarray([[0, 1]], dtype=np.int32),
        divisions=np.asarray([0], dtype=np.int32),
        voxel_um_pooled=np.asarray([1.625, 1.625, 1.625], dtype=np.float32),
    )
    sample = corrected_sequence_sample(path)
    np.testing.assert_allclose(sample.nodes[:, 1:4], [[3, 30, 50], [4, 31, 51]])
    np.testing.assert_array_equal(sample.edges, [[0, 1]])
    np.testing.assert_array_equal(sample.divisions, [0])


def test_temporal_out_of_range_graph_reference_fails(tmp_path: Path) -> None:
    path = tmp_path / "seq_bad.npz"
    np.savez(
        path,
        volumes=np.zeros((1, 8, 64, 64), dtype=np.uint16),
        nodes=np.asarray([[0, 3, 120, 200, 7]], dtype=np.float32),
        edges=np.asarray([[0, 2]], dtype=np.int32),
        divisions=np.asarray([], dtype=np.int32),
        voxel_um_pooled=np.asarray([1.625, 1.625, 1.625], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="invalid node"):
        corrected_sequence_sample(path)


def test_graph_only_loader_does_not_require_volume_member(tmp_path: Path) -> None:
    path = tmp_path / "seq_graph_only.npz"
    np.savez(
        path,
        nodes=np.asarray([[0, 3, 120, 200, 7], [1, 4, 124, 204, 7]], dtype=np.float32),
        edges=np.asarray([[0, 1]], dtype=np.int32),
        divisions=np.asarray([], dtype=np.int32),
        voxel_um_pooled=np.asarray([1.625, 1.625, 1.625], dtype=np.float32),
    )
    graph = corrected_sequence_graph(path, pooled_shape=(8, 64, 64), timepoints=2)
    np.testing.assert_allclose(graph.nodes[:, 1:4], [[3, 30, 50], [4, 31, 51]])
    np.testing.assert_array_equal(graph.edges, [[0, 1]])


def test_lazy_sequences_share_the_corrected_store(tmp_path: Path) -> None:
    path = tmp_path / "vol_00001.npz"
    write_static(path, x=200.0)
    store = SyntheticStaticStore([path])
    images = PooledImageSequence(store)
    points = PointSequence(store)
    assert images[0].shape == (8, 64, 64)
    np.testing.assert_allclose(points[0], [[3.0, 32.0, 50.0]])
    assert store.load.cache_info().hits >= 1


def test_metadata_lazy_sequences_avoid_loading_until_materialized(tmp_path: Path) -> None:
    path = tmp_path / "vol_00002.npz"
    write_static(path, x=180.0)
    store = SyntheticStaticStore(
        [path], pooled_shape=(8, 64, 64), point_counts=[1]
    )
    images = PooledImageSequence(store, lazy=True)
    points = PointSequence(store, lazy=True)
    misses_before = store.load.cache_info().misses

    assert images[0].shape == (8, 64, 64)
    assert images[0].ndim == 3
    assert points[0].shape == (1, 3)
    assert points[0].ndim == 2
    assert len(list(images)) == 1
    assert len(list(points)) == 1
    assert store.load.cache_info().misses == misses_before

    np.testing.assert_allclose(np.asarray(points[0]), [[3.0, 32.0, 45.0]])
    assert store.load.cache_info().misses == misses_before + 1
    assert np.asarray(images[0]).shape == (8, 64, 64)
    assert store.load.cache_info().hits >= 1


def test_split_and_prior_weight_are_deterministic() -> None:
    paths = [Path(f"vol_{index:05d}.npz") for index in range(10)]
    train_a, val_a = split_static_paths(paths, validation_count=2)
    train_b, val_b = split_static_paths(list(reversed(paths)), validation_count=2)
    assert train_a == train_b
    assert val_a == val_b
    assert set(train_a).isdisjoint(val_a)
    assert division_prior_weight() == pytest.approx(0.063812, rel=1e-4)
