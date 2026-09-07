from pathlib import Path

import numpy as np

from research.peak_rank_detection.diagnose_motion_supported_prior import (
    PRIOR_LOCAL_SNR_BANDS,
    motion_supported_maps,
    per_frame_local_snr,
    reduced_local_snr_control,
)
from research.peak_rank_detection.diagnose_local_snr_prior import (
    candidate_response_maps,
)


def test_per_frame_local_snr_keeps_three_finite_fields() -> None:
    volumes = np.linspace(0.0, 0.1, 3 * 64**3, dtype=np.float32).reshape(
        3, 64, 64, 64
    )
    volumes[:, 30:35, 30:35, 30:35] = 1.0
    fields = per_frame_local_snr(volumes)
    assert fields.shape == volumes.shape
    assert np.isfinite(fields).all()
    assert np.all(fields[:, 32, 32, 32] > fields[:, 0, 0, 0])


def test_motion_supported_maps_are_fixed_center_anchored_fields() -> None:
    volumes = np.linspace(0.0, 0.1, 3 * 64**3, dtype=np.float32).reshape(
        3, 64, 64, 64
    )
    volumes[0, 29:34, 31:36, 31:36] = 1.0
    volumes[1, 30:35, 30:35, 30:35] = 1.0
    volumes[2, 31:36, 29:34, 30:35] = 1.0
    maps = motion_supported_maps(volumes)
    assert set(maps) == {
        "current:per-frame-local-snr",
        "mean:per-frame-local-snr",
        "minimum:per-frame-local-snr",
        "motion-mean-r1:per-frame-local-snr",
        "motion-minimum-r1:per-frame-local-snr",
        "motion-mean-r2:per-frame-local-snr",
        "motion-minimum-r2:per-frame-local-snr",
        "control:minimum-reduced-local-snr-prior-bands",
        "control:minimum-reduced-local-snr-deployed-bands",
    }
    assert all(response.shape == (64, 64, 64) for response in maps.values())
    assert all(np.isfinite(response).all() for response in maps.values())


def test_prior_scale_control_exactly_matches_original_diagnostic() -> None:
    rng = np.random.default_rng(11)
    volumes = rng.normal(size=(3, 64, 64, 64)).astype(np.float32)
    expected = candidate_response_maps(volumes)["minimum:local-snr-multiscale"]
    actual = reduced_local_snr_control(volumes, PRIOR_LOCAL_SNR_BANDS)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_module_refuses_nonoptimization_roles() -> None:
    source = Path(
        "research/peak_rank_detection/diagnose_motion_supported_prior.py"
    ).read_text(encoding="utf-8")
    assert "bundle.extract(" not in source
    assert '"selection_data_read": False' in source
    assert '"sealed_audit_data_read": False' in source
    assert '"competition_test_data_read": False' in source
    assert "ThreadPoolExecutor" in source
