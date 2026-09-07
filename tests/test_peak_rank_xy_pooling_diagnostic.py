import numpy as np

from research.peak_rank_detection.diagnose_xy_pooling import (
    available_chunk,
    truth_points,
    xy_reductions,
)


def test_xy_reductions_preserve_shape_and_expected_statistics() -> None:
    frame = np.arange(64 * 256 * 256, dtype=np.float32).reshape(64, 256, 256)
    rows = xy_reductions(frame)
    assert set(rows) == {
        "phase00_decimate",
        "area_mean4",
        "area_rms4",
        "area_mean75_max25",
        "area_mean50_max50",
        "area_max4",
    }
    assert all(values.shape == (64, 64, 64) for values in rows.values())
    assert np.array_equal(rows["phase00_decimate"], frame[:, ::4, ::4])
    assert np.all(
        rows["area_mean4"]
        <= rows["area_rms4"] + np.finfo(np.float32).eps * rows["area_mean4"]
    )
    assert np.all(rows["area_rms4"] <= rows["area_max4"])
    assert np.allclose(
        rows["area_mean75_max25"],
        0.75 * rows["area_mean4"] + 0.25 * rows["area_max4"],
    )


def test_truth_points_use_isotropic_pooled_coordinates() -> None:
    nodes = np.asarray([[1.0, 12.5, 40.0, 80.0, 1.0]], dtype=np.float32)
    assert np.allclose(truth_points(nodes), [[12.5, 10.0, 20.0]])


def test_available_chunk_is_fail_closed(tmp_path) -> None:
    assert available_chunk(tmp_path, "44b6_deadbeef", 7) is None
    chunk = tmp_path / "44b6_deadbeef.zarr" / "0" / "c" / "7" / "0" / "0" / "0"
    chunk.parent.mkdir(parents=True)
    chunk.write_bytes(b"payload")
    assert available_chunk(tmp_path, "44b6_deadbeef", 7) == chunk
