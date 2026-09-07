from pathlib import Path

import numpy as np

from research.peak_rank_detection.diagnose_local_snr_prior import (
    candidate_response_maps,
    local_snr_band,
)


def test_local_snr_band_prefers_compact_bright_center() -> None:
    values = np.zeros((17, 17, 17), dtype=np.float32)
    values[7:10, 7:10, 7:10] = 1.0
    response = local_snr_band(values, 3, 9)
    assert np.isfinite(response).all()
    assert response[8, 8, 8] > response[0, 0, 0]


def test_candidate_maps_are_fixed_finite_fields() -> None:
    volumes = np.linspace(0.0, 0.1, 3 * 64**3, dtype=np.float32).reshape(
        3, 64, 64, 64
    )
    volumes[:, 31:34, 31:34, 31:34] = 1.0
    maps = candidate_response_maps(volumes)
    assert set(maps) == {
        f"{reducer}:{family}"
        for reducer in ("mean", "median", "minimum", "maximum")
        for family in ("fixed-multiscale", "local-snr-multiscale")
    }
    assert all(response.shape == (64, 64, 64) for response in maps.values())
    assert all(np.isfinite(response).all() for response in maps.values())


def test_module_refuses_nonoptimization_roles() -> None:
    source = Path(
        "research/peak_rank_detection/diagnose_local_snr_prior.py"
    ).read_text(encoding="utf-8")
    assert "bundle.extract(" not in source
    assert '"selection_data_read": False' in source
    assert '"sealed_audit_data_read": False' in source
    assert '"competition_test_data_read": False' in source
