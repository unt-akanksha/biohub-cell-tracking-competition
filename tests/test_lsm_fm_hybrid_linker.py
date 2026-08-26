from __future__ import annotations

import numpy as np
import pytest
import torch

from research.lsm_fm_detection.hybrid_linker import (
    AssociationConsensusConfig,
    reciprocal_seed_consensus,
)


def test_association_config_is_frozen_to_compatible_grid() -> None:
    with pytest.raises(ValueError, match="downsample"):
        AssociationConsensusConfig(downsample=(1, 2, 2))
    with pytest.raises(ValueError, match="secondary_seed_weight"):
        AssociationConsensusConfig(secondary_seed_weight=0.5)


def test_consensus_preserves_unanimous_reciprocal_probabilities() -> None:
    forward = torch.tensor([[[2.0, -1.0], [-1.0, 2.0]]])
    reverse = forward.transpose(1, 2)
    result = reciprocal_seed_consensus(forward, reverse, forward, reverse)
    expected = torch.softmax(forward, dim=1)
    torch.testing.assert_close(result, expected)


def test_consensus_penalizes_one_direction_disagreement() -> None:
    forward = torch.tensor([[[4.0, 0.0], [0.0, 4.0]]])
    reverse_agree = forward.transpose(1, 2)
    reverse_disagree = torch.tensor([[[0.0, 4.0], [4.0, 0.0]]])
    agree = reciprocal_seed_consensus(
        forward, reverse_agree, forward, reverse_agree
    )
    disagree = reciprocal_seed_consensus(
        forward, reverse_disagree, forward, reverse_disagree
    )
    assert float(disagree[0, 0, 0]) < float(agree[0, 0, 0])
    np.testing.assert_allclose(disagree.sum(dim=1).numpy(), 1.0, atol=1e-6)


def test_consensus_penalizes_secondary_seed_disagreement() -> None:
    primary = torch.tensor([[[4.0], [0.0]]])
    secondary = torch.tensor([[[0.0], [4.0]]])
    primary_only = reciprocal_seed_consensus(
        primary, primary.transpose(1, 2), primary, primary.transpose(1, 2)
    )
    with_disagreement = reciprocal_seed_consensus(
        primary,
        primary.transpose(1, 2),
        secondary,
        secondary.transpose(1, 2),
    )
    assert float(with_disagreement[0, 0, 0]) < float(primary_only[0, 0, 0])


def test_consensus_rejects_misaligned_node_sets() -> None:
    forward = torch.zeros((1, 2, 3))
    reverse = torch.zeros((1, 3, 2))
    with pytest.raises(ValueError, match="forward seed"):
        reciprocal_seed_consensus(
            forward, reverse, torch.zeros((1, 3, 2)), reverse
        )
    with pytest.raises(ValueError, match="reverse logits"):
        reciprocal_seed_consensus(
            forward, torch.zeros((1, 2, 3)), forward, reverse
        )
