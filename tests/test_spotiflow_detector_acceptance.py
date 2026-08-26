from __future__ import annotations

import numpy as np

from research.spotiflow_biohub.evaluate_pretrained_detector import (
    FramePeaks,
    density_threshold,
    match_frame,
)


def test_density_threshold_selects_requested_sample_count() -> None:
    sampled = [
        FramePeaks(0, np.zeros((3, 3)), np.array([0.9, 0.8, 0.7])),
        FramePeaks(5, np.zeros((3, 3)), np.array([0.95, 0.85, 0.75])),
    ]
    threshold, projected = density_threshold(sampled, estimated_count=20, n_frames=10)
    assert 0.75 < threshold < 0.8
    assert projected == 20.0


def test_physical_matching_accounts_for_biohub_downsample() -> None:
    # Input-space y=10 corresponds to original-space y=40 and 16.25 um.
    predicted = np.array([[3.0, 10.0, 20.0]])
    gt = np.array([[3.0, 40.0, 80.0]])
    matched, annotated, distances = match_frame(predicted, gt)
    assert (matched, annotated) == (1, 1)
    assert distances == [0.0]


def test_matching_is_one_to_one_and_radius_gated() -> None:
    predicted = np.array([[0.0, 0.0, 0.0], [40.0, 40.0, 40.0]])
    gt = np.array([[0.0, 0.0, 0.0], [63.0, 255.0, 255.0]])
    matched, annotated, distances = match_frame(predicted, gt)
    assert matched == 1
    assert annotated == 2
    assert distances == [0.0]
