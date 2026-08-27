from __future__ import annotations

import torch

from research.temporal_contrastive.contextual_pair_fusion import (
    CONTEXTUAL_PAIR_FEATURE_WIDTH,
    EDGE_SET_FEATURE_WIDTH,
    EXPECTED_PARAMETER_COUNT,
    ContextualPairFusionAssociationModel,
)
from research.temporal_contrastive.transition_context import (
    CANDIDATE_CONTEXT_WIDTH,
)


def inputs():
    torch.manual_seed(19)
    source_embeddings = torch.randn(3, 256, requires_grad=True)
    target_embeddings = torch.randn(4, 256, requires_grad=True)
    source_coords = torch.tensor(
        [[0.0, 0.0, 0.0], [4.0, 3.0, 2.0], [8.0, 5.0, 4.0]]
    )
    target_coords = torch.tensor(
        [[1.0, 0.0, 0.0], [5.0, 3.0, 2.0], [9.0, 5.0, 4.0], [7.0, 7.0, 7.0]]
    )
    divisions = torch.tensor([0.2, -0.7, 1.3])
    candidates = torch.tensor(
        [
            [True, True, False, False],
            [False, True, True, True],
            [False, False, True, True],
        ]
    )
    context = torch.randn(3, 4, CANDIDATE_CONTEXT_WIDTH)
    context[~candidates] = 0.0
    return (
        source_embeddings,
        target_embeddings,
        source_coords,
        target_coords,
        divisions,
        candidates,
        context,
    )


def test_contextual_head_has_frozen_widths_and_finite_backward() -> None:
    model = ContextualPairFusionAssociationModel()
    values = inputs()
    logits = model.candidate_pair_logits(*values, chunk_size=3)

    assert model.pair_feature_width == CONTEXTUAL_PAIR_FEATURE_WIDTH == 1_047
    assert EDGE_SET_FEATURE_WIDTH == 1_536
    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETER_COUNT
    assert torch.isfinite(logits[values[5]]).all()
    assert torch.isneginf(logits[~values[5]]).all()
    loss = logits[values[5]].square().mean()
    loss.backward()
    populated = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert populated
    assert all(torch.isfinite(gradient).all() for gradient in populated)


def test_contextual_logits_are_source_and_target_permutation_equivariant() -> None:
    model = ContextualPairFusionAssociationModel().eval()
    values = inputs()
    source_order = torch.tensor([2, 0, 1])
    target_order = torch.tensor([3, 1, 0, 2])
    with torch.no_grad():
        original = model.candidate_pair_logits(*values)
        permuted = model.candidate_pair_logits(
            values[0][source_order],
            values[1][target_order],
            values[2][source_order],
            values[3][target_order],
            values[4][source_order],
            values[5][source_order][:, target_order],
            values[6][source_order][:, target_order],
        )
    expected = original[source_order][:, target_order]
    assert torch.allclose(permuted[torch.isfinite(permuted)], expected[torch.isfinite(expected)])
    assert torch.equal(torch.isfinite(permuted), torch.isfinite(expected))


def test_contextual_head_rejects_wrong_context_width() -> None:
    model = ContextualPairFusionAssociationModel()
    values = list(inputs())
    values[-1] = values[-1][..., :-1]
    try:
        model.candidate_pair_logits(*values)
    except ValueError as error:
        assert "wrong dense shape" in str(error)
    else:
        raise AssertionError("wrong candidate context width was accepted")
