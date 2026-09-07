from pathlib import Path

import numpy as np

from research.peak_rank_detection.diagnose_multiscale_blob_prior import (
    difference_of_averages,
    peak_coordinates,
    point_distances,
    response_maps,
    standardized_response,
)


def test_difference_of_averages_prefers_compact_bright_center() -> None:
    values = np.zeros((17, 17, 17), dtype=np.float32)
    values[7:10, 7:10, 7:10] = 1.0
    response = difference_of_averages(values, 3, 9)
    assert response[8, 8, 8] > 0.9
    assert response[8, 8, 8] > response[0, 0, 0]


def test_response_maps_include_single_and_fixed_fusions() -> None:
    volumes = np.linspace(0.0, 0.1, 3 * 64**3, dtype=np.float32).reshape(
        3, 64, 64, 64
    )
    volumes[:, 31:34, 31:34, 31:34] = 1.0
    maps = response_maps(volumes, ((3, 9), (5, 13), (7, 15)))
    assert set(maps) == {
        "avg3-avg9",
        "avg5-avg13",
        "avg7-avg15",
        "equal-z:avg3-avg9+avg5-avg13",
        "equal-z:avg3-avg9+avg7-avg15",
        "equal-z:avg5-avg13+avg7-avg15",
        "equal-z:avg3-avg9+avg5-avg13+avg7-avg15",
    }
    for response in maps.values():
        assert response.shape == (64, 64, 64)
        assert np.isfinite(response).all()


def test_peak_coordinates_and_distances_use_local_top_k() -> None:
    response = np.zeros((9, 9, 9), dtype=np.float32)
    response[2, 3, 4] = 2.0
    response[7, 6, 5] = 1.0
    peaks = peak_coordinates(response, maximum_predictions=2)
    assert tuple(peaks[0]) == (2.0, 3.0, 4.0)
    distances = point_distances(peaks, np.asarray([[2.0, 3.0, 4.0]]))
    assert distances.tolist() == [0.0]


def test_standardized_response_sets_background_median_to_zero() -> None:
    values = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    standardized = standardized_response(values)
    assert standardized[1, 1, 1] == 0.0
    assert standardized[2, 2, 2] > 0.0


def test_module_does_not_extract_or_reference_forbidden_roles() -> None:
    source = Path(
        "research/peak_rank_detection/diagnose_multiscale_blob_prior.py"
    ).read_text(encoding="utf-8")
    assert "bundle.extract(" not in source
    assert '"selection_data_read": False' in source
    assert '"sealed_audit_data_read": False' in source
    assert '"competition_test_data_read": False' in source
