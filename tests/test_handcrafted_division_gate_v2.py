from __future__ import annotations

import numpy as np

from research.train_handcrafted_division_gate_v2 import decision_summary


def test_decision_summary_tracks_hard_negative_false_positives() -> None:
    targets = np.asarray([1, 0, 1, 0])
    scores = np.asarray([0.9, 0.8, 0.7, 0.1])
    inventory = [
        {"frame_role": "division"},
        {"frame_role": "no_division_hard_negative"},
        {"frame_role": "division"},
        {"frame_role": "no_division_hard_negative"},
    ]

    result = decision_summary(targets, scores, inventory, 0.6)

    assert result["tp"] == 2
    assert result["fp"] == 1
    assert result["no_division_hard_negative_false_positives"] == 1
