from __future__ import annotations

import numpy as np

from research.calibrate_handcrafted_division_gate import select_threshold


def test_handcrafted_threshold_freezes_at_last_zero_fp_positive() -> None:
    targets = np.asarray([1, 1, 0, 1, 0])
    scores = np.asarray([0.9, 0.8, 0.7, 0.6, 0.1])

    selected = select_threshold(targets, scores)

    assert selected["threshold"] == 0.8
    assert selected["tp"] == 2
    assert selected["fp"] == 0
