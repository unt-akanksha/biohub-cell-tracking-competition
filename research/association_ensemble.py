from __future__ import annotations

from typing import TypeAlias

import numpy as np


PairScores: TypeAlias = dict[
    int, tuple[np.ndarray, np.ndarray, np.ndarray]
]


def _logit(probability: np.ndarray, epsilon: float) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), epsilon, 1 - epsilon)
    return np.log(clipped) - np.log1p(-clipped)


def _sigmoid(logit: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -logit))


def _aligned_matrix(
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    other_source_ids: np.ndarray,
    other_target_ids: np.ndarray,
    other_scores: np.ndarray,
) -> np.ndarray:
    source_ids = np.asarray(source_ids, dtype=np.int64)
    target_ids = np.asarray(target_ids, dtype=np.int64)
    other_source_ids = np.asarray(other_source_ids, dtype=np.int64)
    other_target_ids = np.asarray(other_target_ids, dtype=np.int64)
    other_scores = np.asarray(other_scores, dtype=np.float64)
    if other_scores.shape != (len(other_source_ids), len(other_target_ids)):
        raise ValueError("Association matrix shape does not match its stable IDs")
    if len(np.unique(other_source_ids)) != len(other_source_ids):
        raise ValueError("Ensemble source IDs must be unique")
    if len(np.unique(other_target_ids)) != len(other_target_ids):
        raise ValueError("Ensemble target IDs must be unique")
    if set(source_ids.tolist()) != set(other_source_ids.tolist()):
        raise ValueError("Ensemble source-node sets differ")
    if set(target_ids.tolist()) != set(other_target_ids.tolist()):
        raise ValueError("Ensemble target-node sets differ")
    source_row = {int(node_id): row for row, node_id in enumerate(other_source_ids)}
    target_col = {int(node_id): col for col, node_id in enumerate(other_target_ids)}
    rows = np.asarray([source_row[int(node_id)] for node_id in source_ids])
    cols = np.asarray([target_col[int(node_id)] for node_id in target_ids])
    return other_scores[np.ix_(rows, cols)]


def blend_pair_scores(
    trackastra: PairScores,
    hoct: PairScores,
    *,
    trackastra_weight: float,
    epsilon: float = 1e-5,
    support_aware: bool = True,
) -> PairScores:
    """Blend independently learned association probabilities in log-odds space.

    Stable node IDs are aligned explicitly, so a harmless row/column ordering
    difference cannot silently corrupt the ensemble. Both tiled models use an
    exact zero for pairs outside their candidate neighborhoods while evaluated
    candidates have strictly positive softmax probabilities. With
    ``support_aware=True`` a score is therefore blended only where both models
    evaluated the pair; elsewhere the available model is preserved. This
    prevents one model's bounded candidate graph from vetoing useful evidence
    from the other model.
    """
    if not 0 <= trackastra_weight <= 1:
        raise ValueError("trackastra_weight must be in [0, 1]")
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be in (0, 0.5)")
    if set(trackastra) != set(hoct):
        raise ValueError("Ensemble frame-pair sets differ")

    blended: PairScores = {}
    for source_t in sorted(trackastra):
        source_ids, target_ids, trackastra_scores = trackastra[source_t]
        other_source_ids, other_target_ids, hoct_scores = hoct[source_t]
        source_ids = np.asarray(source_ids, dtype=np.int64)
        target_ids = np.asarray(target_ids, dtype=np.int64)
        trackastra_scores = np.asarray(trackastra_scores, dtype=np.float64)
        if trackastra_scores.shape != (len(source_ids), len(target_ids)):
            raise ValueError("Trackastra matrix shape does not match stable IDs")
        if len(np.unique(source_ids)) != len(source_ids):
            raise ValueError("Ensemble source IDs must be unique")
        if len(np.unique(target_ids)) != len(target_ids):
            raise ValueError("Ensemble target IDs must be unique")
        aligned_hoct = _aligned_matrix(
            source_ids,
            target_ids,
            other_source_ids,
            other_target_ids,
            hoct_scores,
        )
        if not np.isfinite(trackastra_scores).all() or not np.isfinite(
            aligned_hoct
        ).all():
            raise ValueError("Association probabilities must be finite")
        if (
            (trackastra_scores < 0).any()
            or (trackastra_scores > 1).any()
            or (aligned_hoct < 0).any()
            or (aligned_hoct > 1).any()
        ):
            raise ValueError("Association probabilities must be in [0, 1]")

        logit = trackastra_weight * _logit(trackastra_scores, epsilon)
        logit += (1 - trackastra_weight) * _logit(aligned_hoct, epsilon)
        matrix = _sigmoid(logit)
        if support_aware:
            trackastra_support = trackastra_scores > 0
            hoct_support = aligned_hoct > 0
            matrix = np.where(
                trackastra_support & ~hoct_support,
                trackastra_scores,
                matrix,
            )
            matrix = np.where(
                hoct_support & ~trackastra_support,
                aligned_hoct,
                matrix,
            )
            matrix = np.where(
                ~trackastra_support & ~hoct_support,
                0.0,
                matrix,
            )
        blended[source_t] = (
            source_ids.copy(),
            target_ids.copy(),
            matrix.astype(np.float32),
        )
    return blended
