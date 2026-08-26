from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class HybridLinkConfig:
    edge_threshold: float = 0.08
    base_lock_probability: float = 0.96
    base_keep_probability: float = 0.70
    base_bonus: float = 0.12
    division_threshold: float = 0.18
    division_ratio: float = 0.45
    base_division_keep_probability: float = 0.90


@dataclass(frozen=True)
class HybridLinkResult:
    edges: tuple[tuple[int, int], ...]
    locked_base_edges: int
    retained_base_edges: int
    replaced_base_edges: int
    new_primary_edges: int
    retained_base_divisions: int
    new_division_edges: int


def _base_arrays(
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    base_edges: list[tuple[int, int, float]],
) -> tuple[np.ndarray, np.ndarray]:
    row_by_id = {int(node_id): index for index, node_id in enumerate(source_ids)}
    col_by_id = {int(node_id): index for index, node_id in enumerate(target_ids)}
    probability = np.full((len(source_ids), len(target_ids)), np.nan, dtype=np.float32)
    child_count = np.zeros(len(source_ids), dtype=np.int32)
    for source_id, target_id, value in base_edges:
        row = row_by_id.get(int(source_id))
        col = col_by_id.get(int(target_id))
        if row is None or col is None:
            continue
        probability[row, col] = max(
            float(value),
            float(probability[row, col]) if np.isfinite(probability[row, col]) else 0.0,
        )
        child_count[row] += 1
    return probability, child_count


def hybrid_link_pair(
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    trackastra_scores: np.ndarray,
    base_edges: list[tuple[int, int, float]],
    config: HybridLinkConfig,
) -> HybridLinkResult:
    """Fuse a selected base graph with dense Trackastra association scores.

    High-confidence base edges are locked. Remaining one-to-one assignments are
    solved globally after adding a bounded bonus at base-edge positions. A
    second pass retains or proposes at most one division child per source.
    """
    source_ids = np.asarray(source_ids, dtype=np.int64)
    target_ids = np.asarray(target_ids, dtype=np.int64)
    scores = np.asarray(trackastra_scores, dtype=np.float32)
    if scores.shape != (len(source_ids), len(target_ids)):
        raise ValueError("Trackastra score matrix does not match source/target IDs")
    if not np.isfinite(scores[np.isfinite(scores)]).all():
        raise ValueError("Trackastra scores contain invalid finite values")

    base_probability, base_child_count = _base_arrays(source_ids, target_ids, base_edges)
    finite_scores = np.where(np.isfinite(scores), scores, -1e6)
    base_positions = np.isfinite(base_probability)
    fused = finite_scores.copy()
    fused[base_positions] += config.base_bonus * base_probability[base_positions]

    edges: list[tuple[int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    primary_col_by_row: dict[int, int] = {}
    locked = 0
    retained = 0
    new_primary = 0

    lock_candidates = [
        (float(base_probability[row, col]), row, col)
        for row, col in zip(*np.where(base_probability >= config.base_lock_probability))
    ]
    for _probability, row, col in sorted(lock_candidates, reverse=True):
        if row in used_rows or col in used_cols:
            continue
        edges.append((int(source_ids[row]), int(target_ids[col])))
        used_rows.add(row)
        used_cols.add(col)
        primary_col_by_row[row] = col
        locked += 1
        retained += 1

    available_rows = np.array(
        [row for row in range(len(source_ids)) if row not in used_rows], dtype=np.int64
    )
    available_cols = np.array(
        [col for col in range(len(target_ids)) if col not in used_cols], dtype=np.int64
    )
    if len(available_rows) and len(available_cols):
        local_rows, local_cols = linear_sum_assignment(
            -fused[np.ix_(available_rows, available_cols)]
        )
        for local_row, local_col in zip(local_rows.tolist(), local_cols.tolist()):
            row = int(available_rows[local_row])
            col = int(available_cols[local_col])
            is_base = bool(base_positions[row, col])
            if is_base:
                accepted = (
                    float(base_probability[row, col]) >= config.base_keep_probability
                    or float(finite_scores[row, col]) >= config.edge_threshold
                )
            else:
                accepted = float(finite_scores[row, col]) >= config.edge_threshold
            if not accepted:
                continue
            edges.append((int(source_ids[row]), int(target_ids[col])))
            used_rows.add(row)
            used_cols.add(col)
            primary_col_by_row[row] = col
            if is_base:
                retained += 1
            else:
                new_primary += 1

    retained_base_divisions = 0
    new_divisions = 0
    division_candidates: list[tuple[int, float, int, int, bool]] = []
    for row, primary_col in primary_col_by_row.items():
        primary_score = max(float(finite_scores[row, primary_col]), 1e-6)
        for col in np.argsort(fused[row])[::-1].tolist():
            if col == primary_col or col in used_cols:
                continue
            is_base = bool(base_positions[row, col])
            if is_base and base_child_count[row] >= 2:
                accepted = (
                    float(base_probability[row, col])
                    >= config.base_division_keep_probability
                )
                priority = 2
            else:
                accepted = (
                    float(finite_scores[row, col]) >= config.division_threshold
                    and float(finite_scores[row, col])
                    >= primary_score * config.division_ratio
                )
                priority = 1
            if accepted:
                division_candidates.append(
                    (priority, float(fused[row, col]), row, col, is_base)
                )
            break

    for _priority, _score, row, col, is_base in sorted(division_candidates, reverse=True):
        if col in used_cols:
            continue
        edges.append((int(source_ids[row]), int(target_ids[col])))
        used_cols.add(col)
        if is_base:
            retained_base_divisions += 1
        else:
            new_divisions += 1

    original_base = {
        (int(source_id), int(target_id)) for source_id, target_id, _ in base_edges
    }
    final = set(edges)
    primary_base_count = sum(base_child_count > 0)
    replaced = max(0, int(primary_base_count) - retained)
    return HybridLinkResult(
        edges=tuple(sorted(final)),
        locked_base_edges=locked,
        retained_base_edges=len(final & original_base),
        replaced_base_edges=replaced,
        new_primary_edges=new_primary,
        retained_base_divisions=retained_base_divisions,
        new_division_edges=new_divisions,
    )
