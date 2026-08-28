"""Label-free t+2 continuation context for learned division ranking.

The extractor deliberately makes no division decision and contains no tuned
distance threshold. It selects the best distinct learned continuation for two
proposed daughters and emits scale-normalized physical evidence that a later
model may learn from under sealed reciprocal validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


FUTURE_DIVISION_FEATURE_WIDTH = 8


@dataclass(frozen=True)
class FutureDivisionEvidence:
    available: bool
    continuation_indices: tuple[int, int] | None
    current_separation_um: float
    future_separation_um: float
    separation_gain_um: float
    continuation_score_minimum: float
    continuation_score_mean: float
    assignment_margin: float
    separation_direction_cosine: float

    def feature_vector(self, *, candidate_radius_um: float) -> np.ndarray:
        radius = float(candidate_radius_um)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("candidate radius must be positive and finite")
        if not self.available:
            return np.zeros(FUTURE_DIVISION_FEATURE_WIDTH, dtype=np.float32)
        current = max(self.current_separation_um, radius * 1e-6)
        ratio = np.clip(self.future_separation_um / current, 1e-4, 1e4)
        result = np.asarray(
            [
                self.current_separation_um / radius,
                self.future_separation_um / radius,
                self.separation_gain_um / radius,
                math.log(ratio),
                self.continuation_score_minimum,
                self.continuation_score_mean,
                self.assignment_margin,
                self.separation_direction_cosine,
            ],
            dtype=np.float32,
        )
        if result.shape != (FUTURE_DIVISION_FEATURE_WIDTH,) or not np.isfinite(
            result
        ).all():
            raise RuntimeError("future division feature construction changed")
        return result


def _checked_coords(value: np.ndarray, shape: tuple[int | None, int], label: str) -> np.ndarray:
    coords = np.asarray(value, dtype=np.float64)
    expected_rows, expected_columns = shape
    if coords.ndim != 2 or coords.shape[1] != expected_columns:
        raise ValueError(f"{label} coordinates must have shape (N, 3)")
    if expected_rows is not None and len(coords) != expected_rows:
        raise ValueError(f"{label} coordinates must have shape ({expected_rows}, 3)")
    if not np.isfinite(coords).all():
        raise ValueError(f"{label} coordinates must be finite")
    return coords


def future_division_evidence(
    daughter_coords_zyx_um: np.ndarray,
    next_coords_zyx_um: np.ndarray,
    continuation_scores: np.ndarray,
    candidate_mask: np.ndarray,
) -> FutureDivisionEvidence:
    """Measure distinct learned continuations for two proposed daughters.

    ``continuation_scores`` has shape ``(2, K)`` and should carry the same
    learned association evidence used by the linker. The best injective pair is
    selected globally, preventing both daughters from claiming one future node.
    The assignment margin compares the selected total against the swapped pair
    when both swapped edges are eligible; it is zero when that alternative is
    unavailable. No ground truth, public prediction, or fixed decision threshold
    enters this function.
    """

    daughters = _checked_coords(
        daughter_coords_zyx_um, (2, 3), "daughter"
    )
    future = _checked_coords(next_coords_zyx_um, (None, 3), "future")
    scores = np.asarray(continuation_scores, dtype=np.float64)
    candidates = np.asarray(candidate_mask)
    if scores.shape != (2, len(future)):
        raise ValueError("continuation scores must have shape (2, K)")
    if candidates.shape != scores.shape or candidates.dtype != np.bool_:
        raise ValueError("candidate mask must be boolean with shape (2, K)")
    if not np.isfinite(scores[candidates]).all():
        raise ValueError("eligible continuation scores must be finite")
    unavailable = FutureDivisionEvidence(
        available=False,
        continuation_indices=None,
        current_separation_um=float(np.linalg.vector_norm(daughters[1] - daughters[0])),
        future_separation_um=0.0,
        separation_gain_um=0.0,
        continuation_score_minimum=0.0,
        continuation_score_mean=0.0,
        assignment_margin=0.0,
        separation_direction_cosine=0.0,
    )
    if len(future) < 2 or not candidates.all(axis=1).any():
        return unavailable

    finite = np.where(candidates, scores, -1e30)
    rows, columns = linear_sum_assignment(-finite)
    if len(rows) != 2 or len(set(columns.tolist())) != 2:
        return unavailable
    selected_by_row = np.full(2, -1, dtype=np.int64)
    selected_by_row[rows] = columns
    if (selected_by_row < 0).any() or not candidates[
        np.arange(2), selected_by_row
    ].all():
        return unavailable

    first, second = map(int, selected_by_row.tolist())
    selected_scores = scores[np.arange(2), selected_by_row]
    current_vector = daughters[1] - daughters[0]
    future_vector = future[second] - future[first]
    current_separation = float(np.linalg.vector_norm(current_vector))
    future_separation = float(np.linalg.vector_norm(future_vector))
    denominator = max(current_separation * future_separation, 1e-12)
    direction_cosine = float(
        np.clip(np.dot(current_vector, future_vector) / denominator, -1.0, 1.0)
    )
    swapped_available = bool(candidates[0, second] and candidates[1, first])
    selected_total = float(selected_scores.sum())
    swapped_total = (
        float(scores[0, second] + scores[1, first])
        if swapped_available
        else selected_total
    )
    return FutureDivisionEvidence(
        available=True,
        continuation_indices=(first, second),
        current_separation_um=current_separation,
        future_separation_um=future_separation,
        separation_gain_um=future_separation - current_separation,
        continuation_score_minimum=float(selected_scores.min()),
        continuation_score_mean=float(selected_scores.mean()),
        assignment_margin=max(0.0, selected_total - swapped_total),
        separation_direction_cosine=direction_cosine,
    )
