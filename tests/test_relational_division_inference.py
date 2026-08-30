from __future__ import annotations

import numpy as np
import pytest
import torch

from research.learned_division_recovery import DivisionRecoveryCandidate
from research.temporal_contrastive.relational_division_inference import (
    calibration_free_parent_scores,
    candidate_centers_zyx,
    candidate_geometry_features,
    score_relational_candidates,
)


def candidate(parent: int = 2) -> DivisionRecoveryCandidate:
    return DivisionRecoveryCandidate(
        parent_id=parent,
        existing_child_id=3,
        second_child_id=4,
        parent_distance_um=1.0,
        sister_distance_um=2.0,
        existing_distance_um=1.0,
        daughter_midpoint_distance_um=0.0,
        daughter_opposition_cosine=-1.0,
        daughter_step_ratio=1.0,
        biological_geometry_score=4.0,
    )


def nodes() -> dict[int, dict[str, float | int]]:
    return {
        1: {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        2: {"t": 1, "z": 0.0, "y": 0.0, "x": 1.0},
        3: {"t": 2, "z": 0.0, "y": 0.0, "x": 2.0},
        4: {"t": 2, "z": 0.0, "y": 0.0, "x": 0.0},
    }


def test_geometry_matches_training_order_and_velocity_definition() -> None:
    features = candidate_geometry_features(
        candidate(), nodes(), [{"source_id": 1, "target_id": 2}]
    )

    assert np.allclose(features[:7], [1.0, 2.0, 1.0, 0.0, -1.0, 1.0, 4.0])
    assert features[7] == pytest.approx(0.40625)
    assert features[8] == pytest.approx(0.40625)
    assert np.array_equal(
        candidate_centers_zyx(candidate(), nodes()),
        np.asarray([[0, 0, 1], [0, 0, 2], [0, 0, 0]], dtype=np.float64),
    )


def test_geometry_marks_ambiguous_velocity_missing() -> None:
    features = candidate_geometry_features(
        candidate(), nodes(), []
    )

    assert np.isnan(features[7:]).all()


def test_calibration_free_parent_scores_ignore_member_scale() -> None:
    result = calibration_free_parent_scores(
        [{10: 100.0, 20: 0.0}, {10: 0.51, 20: 0.49}], [10, 20]
    )

    assert result == {10: 1.0, 20: 0.0}


class FakeRelationalModel(torch.nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset

    def forward(self, patches: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        return geometry[:, 0] + self.offset


def test_candidate_scorer_uses_three_centers_and_three_frames() -> None:
    observed: dict[str, object] = {}

    def read_frame(index: int) -> np.ndarray:
        observed.setdefault("frames", []).append(index)
        return np.full((5, 5, 5), index, dtype=np.float32)

    def sample(temporal: torch.Tensor, centers: np.ndarray, **kwargs: object) -> torch.Tensor:
        observed["temporal_shape"] = tuple(temporal.shape)
        observed["centers"] = centers.copy()
        observed["kwargs"] = kwargs
        return torch.zeros(len(centers), 3, 17, 17, 17)

    scores, stats = score_relational_candidates(
        [FakeRelationalModel(0.0), FakeRelationalModel(10.0)],
        [candidate()],
        nodes(),
        [{"source_id": 1, "target_id": 2}],
        read_frame=read_frame,
        sample_physical_patches=sample,
        device=torch.device("cpu"),
        maximum_time=2,
        batch_size=2,
    )

    assert observed["frames"] == [0, 1, 2]
    assert observed["temporal_shape"] == (3, 5, 5, 5)
    assert np.asarray(observed["centers"]).shape == (3, 3)
    assert scores == {2: 1.0}
    assert stats["relational_member_count"] == 2
    assert stats["absolute_threshold_used"] is False
