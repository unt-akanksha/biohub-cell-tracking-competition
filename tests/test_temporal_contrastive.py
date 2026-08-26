from __future__ import annotations

import pytest
import torch

from research.temporal_contrastive.model import (
    TemporalFusionHead,
    masked_link_info_nce,
)


def test_temporal_head_outputs_normalized_embedding_and_sparse_division_prior() -> None:
    torch.manual_seed(7)
    head = TemporalFusionHead(8, hidden_channels=16, embedding_channels=6)
    current = torch.randn(2, 8, 4, 4, 4)
    following = torch.randn_like(current)

    embeddings, division_logits = head(current, following)

    assert embeddings.shape == (2, 6, 4, 4, 4)
    assert division_logits.shape == (2, 1, 4, 4, 4)
    assert torch.allclose(
        torch.linalg.vector_norm(embeddings, dim=1),
        torch.ones(2, 4, 4, 4),
        atol=1e-5,
    )
    assert float(division_logits.detach().mean()) < -3.0


def test_masked_info_nce_rewards_correct_link_similarity() -> None:
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    positives = torch.tensor([0, 1])
    candidates = torch.ones((2, 3), dtype=torch.bool)
    aligned = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    reversed_sources = aligned.flip(0)

    good = masked_link_info_nce(aligned, targets, positives, candidates)
    bad = masked_link_info_nce(reversed_sources, targets, positives, candidates)

    assert float(good) < float(bad)


def test_masked_info_nce_excludes_unknown_sources_but_keeps_gradients() -> None:
    sources = torch.randn(3, 4, requires_grad=True)
    targets = torch.randn(3, 4)
    positives = torch.tensor([0, -1, 2])
    candidates = torch.ones((3, 3), dtype=torch.bool)

    loss = masked_link_info_nce(sources, targets, positives, candidates)
    loss.backward()

    assert torch.isfinite(loss)
    assert sources.grad is not None
    assert torch.count_nonzero(sources.grad[1]) == 0


def test_masked_info_nce_rejects_missing_positive_candidate() -> None:
    sources = torch.randn(1, 4)
    targets = torch.randn(2, 4)
    candidates = torch.tensor([[False, True]])

    with pytest.raises(ValueError, match="ground-truth link"):
        masked_link_info_nce(sources, targets, torch.tensor([0]), candidates)
