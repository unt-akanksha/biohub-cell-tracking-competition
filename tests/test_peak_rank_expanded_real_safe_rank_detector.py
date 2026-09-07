from pathlib import Path

import torch

from research.peak_rank_detection import train_expanded_real_safe_rank_detector as safe
from research.peak_rank_detection.model_safe_rank import MODEL_FAMILY


def test_safe_background_rank_prefers_known_center_over_dark_shell() -> None:
    logits = torch.zeros(1, 1, 17, 17, 17, requires_grad=True)
    evidence = torch.ones_like(logits)
    evidence[:, :, :8] = -1.0
    points = torch.tensor([[8.0, 8.0, 8.0]])
    baseline = safe.safe_background_ranking_loss(logits, evidence, points)
    improved_logits = logits.detach().clone()
    improved_logits[0, 0, 8, 8, 8] = 3.0
    improved = safe.safe_background_ranking_loss(
        improved_logits, evidence, points
    )
    assert baseline > 0.0
    assert improved < baseline
    baseline.backward()
    assert torch.isfinite(logits.grad).all()


def test_bright_unlabeled_shell_voxels_are_not_negative() -> None:
    logits = torch.zeros(1, 1, 17, 17, 17)
    evidence = torch.zeros_like(logits)
    evidence[0, 0, 8, 8, 13] = 10.0
    logits[0, 0, 8, 8, 13] = 20.0
    points = torch.tensor([[8.0, 8.0, 8.0]])
    loss_with_bright_candidate = safe.safe_background_ranking_loss(
        logits, evidence, points, quantile=0.5
    )
    logits[0, 0, 8, 8, 13] = 0.0
    loss_without_candidate = safe.safe_background_ranking_loss(
        logits, evidence, points, quantile=0.5
    )
    assert torch.allclose(loss_with_bright_candidate, loss_without_candidate)


def test_terminal_freezes_optimization_only_safe_boundary() -> None:
    written = {}

    def writer(path: Path, payload: dict) -> None:
        written[path.name] = payload

    safe.tagged_atomic_json(Path("terminal.json"), {"status": "complete"}, writer=writer)
    terminal = written["terminal.json"]
    assert terminal["model_family"] == MODEL_FAMILY
    assert terminal["safe_negative_evidence_band"] == [5, 13]
    assert terminal["safe_negative_boundary_quantile"] == 0.5
    assert terminal["safe_negative_selection_role"] == "expanded_real_optimization_only"
    assert terminal["optimization_annotated_below_boundary_fraction"] == (
        0.02811804008908686
    )
