"""Selective soft-teacher distillation for a possible SpatialDINO PU v2.

This is staged separately from the active v1 experiment. It transfers only
teacher-supported probability structure, discounts seed disagreement, and does
not turn either public teacher's final predictions into submission output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DistillationTargets:
    probability: np.ndarray
    weights: np.ndarray
    support_mask: np.ndarray
    mean_agreement: float


def build_selective_distillation_targets(
    primary_probability: np.ndarray,
    secondary_probability: np.ndarray,
    *,
    support_threshold: float = 0.05,
    agreement_power: float = 2.0,
) -> DistillationTargets:
    primary = np.asarray(primary_probability, dtype=np.float32)
    secondary = np.asarray(secondary_probability, dtype=np.float32)
    if primary.shape != secondary.shape or primary.ndim != 3:
        raise ValueError("teacher probabilities must have the same 3D shape")
    if not np.isfinite(primary).all() or not np.isfinite(secondary).all():
        raise ValueError("teacher probabilities must be finite")
    if (
        primary.size
        and (
            float(primary.min()) < 0.0
            or float(primary.max()) > 1.0
            or float(secondary.min()) < 0.0
            or float(secondary.max()) > 1.0
        )
    ):
        raise ValueError("teacher probabilities must lie in [0, 1]")
    if not 0.0 < support_threshold < 1.0:
        raise ValueError("support_threshold must lie in (0, 1)")
    if agreement_power <= 0:
        raise ValueError("agreement_power must be positive")

    support = np.maximum(primary, secondary) >= support_threshold
    probability = 0.5 * (primary + secondary)
    agreement = np.power(1.0 - np.abs(primary - secondary), agreement_power)
    # Requiring both teachers to contribute confidence makes a lone high seed
    # weak supervision instead of a hard pseudo-label.
    confidence = np.sqrt(primary * secondary)
    weights = np.where(support, agreement * confidence, 0.0).astype(np.float32)
    mean_agreement = float(agreement[support].mean()) if np.any(support) else 0.0
    return DistillationTargets(
        probability=probability.astype(np.float32),
        weights=weights,
        support_mask=support,
        mean_agreement=mean_agreement,
    )


def selective_distillation_bce(logits, probability, weights):
    """Confidence-normalized soft BCE on teacher-supported voxels only."""

    import torch
    import torch.nn.functional as torch_functional

    if logits.shape != probability.shape or logits.shape != weights.shape:
        raise ValueError("logits, probability, and weights must have identical shapes")
    if torch.any(weights < 0):
        raise ValueError("distillation weights must be nonnegative")
    active = weights > 0
    if not torch.any(active):
        return logits.sum() * 0.0
    raw = torch_functional.binary_cross_entropy_with_logits(
        logits,
        probability.to(logits.dtype),
        reduction="none",
    )
    return (raw[active] * weights[active]).sum() / weights[active].sum().clamp_min(1e-8)
