"""Self-contained utilities used by graph-context division training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def state_dict_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def update_ema(model: torch.nn.Module, ema: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_value, value in zip(
            ema.state_dict().values(), model.state_dict().values(), strict=True
        ):
            if ema_value.is_floating_point():
                ema_value.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                ema_value.copy_(value)


def average_precision(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.detach().float().cpu() > 0.5
    scores = scores.detach().float().cpu()
    positives = int(labels.sum())
    if positives <= 0 or positives >= len(labels):
        raise ValueError("average precision requires both classes")
    order = torch.argsort(scores, descending=True, stable=True)
    ranked = labels[order]
    cumulative = torch.cumsum(ranked.to(torch.float64), dim=0)
    ranks = torch.arange(1, len(ranked) + 1, dtype=torch.float64)
    return float((cumulative[ranked] / ranks[ranked]).mean())


def threshold_metrics(
    labels: torch.Tensor, scores: torch.Tensor
) -> dict[str, Any]:
    labels = labels.detach().float().cpu() > 0.5
    scores = scores.detach().float().cpu()
    order = torch.argsort(scores, descending=True, stable=True)
    ranked_labels = labels[order]
    ranked_scores = scores[order]
    tp = torch.cumsum(ranked_labels.to(torch.int64), dim=0)
    fp = torch.cumsum((~ranked_labels).to(torch.int64), dim=0)
    positives = int(labels.sum())
    fn = positives - tp
    precision = tp.float() / (tp + fp).clamp_min(1).float()
    recall = tp.float() / max(positives, 1)
    jaccard = tp.float() / (tp + fp + fn).clamp_min(1).float()
    zero_fp = torch.nonzero((fp == 0) & (tp > 0), as_tuple=False).flatten()
    precise = torch.nonzero((precision >= 0.95) & (tp > 0), as_tuple=False).flatten()
    best_index = int(torch.argmax(jaccard))
    return {
        "rows": len(labels),
        "positives": positives,
        "average_precision": average_precision(labels.float(), scores),
        "best_jaccard": float(jaccard[best_index]),
        "best_jaccard_threshold": float(ranked_scores[best_index]),
        "recall_at_zero_false_positives": (
            float(recall[zero_fp[-1]]) if len(zero_fp) else 0.0
        ),
        "recall_at_precision_0_95": (
            float(recall[precise].max()) if len(precise) else 0.0
        ),
        "binary_cross_entropy": float(
            F.binary_cross_entropy_with_logits(scores, labels.float())
        ),
    }


def select_frozen_threshold(
    labels: torch.Tensor, scores: torch.Tensor
) -> dict[str, Any]:
    labels = labels.detach().float().cpu() > 0.5
    scores = scores.detach().float().cpu()
    order = torch.argsort(scores, descending=True, stable=True)
    ranked_labels = labels[order]
    ranked_scores = scores[order]
    tp = torch.cumsum(ranked_labels.to(torch.int64), dim=0)
    fp = torch.cumsum((~ranked_labels).to(torch.int64), dim=0)
    fn = int(labels.sum()) - tp
    precision = tp.float() / (tp + fp).clamp_min(1).float()
    recall = tp.float() / max(int(labels.sum()), 1)
    jaccard = tp.float() / (tp + fp + fn).clamp_min(1).float()
    eligible = torch.nonzero((fp == 0) & (tp >= 2), as_tuple=False).flatten()
    if len(eligible):
        index = int(eligible[-1])
        policy = "maximum recall with zero false positives and at least two positives"
    else:
        eligible = torch.nonzero(
            (precision >= 0.95) & (tp > 0), as_tuple=False
        ).flatten()
        if not len(eligible):
            raise RuntimeError("selection cannot provide a high-precision threshold")
        index = int(eligible[torch.argmax(jaccard[eligible])])
        policy = "maximum Jaccard among thresholds with at least 0.95 precision"
    return {
        "threshold": float(ranked_scores[index]),
        "policy": policy,
        "tp": int(tp[index]),
        "fp": int(fp[index]),
        "fn": int(fn[index]),
        "precision": float(precision[index]),
        "recall": float(recall[index]),
        "jaccard": float(jaccard[index]),
    }


def eligible_metrics(
    targets: torch.Tensor,
    scores: torch.Tensor,
    eligible: torch.Tensor,
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    indices = torch.nonzero(eligible, as_tuple=False).flatten()
    labels = targets[indices]
    values = scores[indices]
    pooled = threshold_metrics(labels, values)
    order = torch.argsort(values, descending=True, stable=True)
    ranked = labels[order] > 0.5
    first_false = torch.nonzero(~ranked, as_tuple=False).flatten()
    before_first_false = (
        int(ranked[: int(first_false[0])].sum())
        if len(first_false)
        else int(ranked.sum())
    )
    by_embryo: dict[str, dict[str, Any]] = {}
    for embryo in ("44b6", "6bba"):
        embryo_rows = torch.as_tensor(
            [
                row_index
                for row_index in indices.tolist()
                if inventory[row_index]["embryo"] == embryo
            ],
            dtype=torch.long,
        )
        by_embryo[embryo] = threshold_metrics(
            targets[embryo_rows], scores[embryo_rows]
        )
    return {
        **pooled,
        "true_positives_before_first_false_positive": before_first_false,
        "by_embryo": by_embryo,
    }


def selection_utility(metrics: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(metrics["true_positives_before_first_false_positive"]),
        min(float(row["average_precision"]) for row in metrics["by_embryo"].values()),
        float(metrics["average_precision"]),
        float(metrics["best_jaccard"]),
        -float(metrics["binary_cross_entropy"]),
    )


def calibration_free_equal_rank_ensemble(
    member_scores: list[torch.Tensor],
) -> torch.Tensor:
    if not member_scores:
        raise ValueError("graph-context rank ensemble requires at least one member")
    shape = member_scores[0].shape
    if any(scores.shape != shape or scores.ndim != 1 for scores in member_scores):
        raise ValueError("graph-context rank ensemble scores must be aligned vectors")
    ranks = []
    for scores in member_scores:
        if not torch.isfinite(scores).all():
            raise ValueError("graph-context rank ensemble scores must be finite")
        order = torch.argsort(scores, descending=True, stable=True)
        percentiles = (
            torch.ones(1, dtype=torch.float64)
            if len(scores) == 1
            else torch.linspace(1.0, 0.0, len(scores), dtype=torch.float64)
        )
        member_ranks = torch.empty(len(scores), dtype=torch.float64)
        member_ranks[order] = percentiles
        ranks.append(member_ranks)
    return torch.stack(ranks).mean(dim=0).float()


def focal_loss(
    logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    logits = logits.float()
    targets = targets.float()
    base = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = torch.sigmoid(logits)
    correct_probability = torch.where(
        targets > 0.5, probability, 1.0 - probability
    )
    losses = (1.0 - correct_probability).square() * base * weights.float()
    return losses.sum() / weights.sum().clamp_min(1e-6)


def threshold_decisions(
    labels: torch.Tensor, scores: torch.Tensor, threshold: float
) -> dict[str, Any]:
    labels = labels.float() > 0.5
    selected = scores >= float(threshold)
    tp = int((selected & labels).sum())
    fp = int((selected & ~labels).sum())
    fn = int(labels.sum()) - tp
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(int(labels.sum()), 1),
        "jaccard": tp / max(tp + fp + fn, 1),
    }
