from __future__ import annotations

import numpy as np
import pytest
import torch

from research.temporal_contrastive.contextual_pair_fusion import (
    ContextualPairFusionAssociationModel,
)
from research.temporal_contrastive.contextual_training import (
    contextual_pair_logits_for_transition,
    image_transition_context,
)
from research.temporal_contrastive.train_dual_fold_patch import TransitionBatch


def temporal_volumes(shift=(2, -3, 4)) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(27)
    central = rng.normal(0.0, 0.01, size=(17, 21, 25)).astype(np.float32)
    central[3:8, 6:12, 9:15] += 2.0
    moved = np.roll(central, shift, axis=(0, 1, 2))
    source = np.stack((central * 0.9, central, central * 1.1))
    target = np.stack((central, moved, moved * 1.1))
    return source, target


def test_central_frame_adapter_recovers_physical_shift() -> None:
    source, target = temporal_volumes()
    context = image_transition_context(
        source, target, voxel_size_zyx_um=(1.5, 0.5, 0.5)
    )
    assert context.global_shift_zyx_voxel == pytest.approx((2, -3, 4))
    assert context.global_shift_zyx_um == pytest.approx((3.0, -1.5, 2.0))
    assert context.aligned_ncc > 0.999


def test_complete_contextual_transition_has_finite_logits_and_gradients() -> None:
    torch.manual_seed(41)
    source_volume, target_volume = temporal_volumes()
    batch = TransitionBatch(
        source_coords=np.asarray([[4, 6, 7], [9, 11, 12]], dtype=np.float32),
        target_coords=np.asarray(
            [[6, 3, 11], [11, 8, 16], [8, 8, 8]], dtype=np.float32
        ),
        candidate_mask=np.asarray(
            [[True, False, True], [False, True, True]], dtype=bool
        ),
        positive_mask=np.asarray(
            [[True, False, False], [False, True, False]], dtype=bool
        ),
        division_target=np.zeros(2, dtype=np.float32),
        timepoint=5,
    )
    model = ContextualPairFusionAssociationModel()
    source_embeddings = torch.randn(2, 256, requires_grad=True)
    target_embeddings = torch.randn(3, 256, requires_grad=True)
    division_logits = torch.randn(2, requires_grad=True)

    logits, candidates, positives, context = contextual_pair_logits_for_transition(
        model,
        source_embeddings,
        target_embeddings,
        division_logits,
        batch,
        source_volume,
        target_volume,
        (1.0, 1.0, 1.0),
        torch.device("cpu"),
        pair_chunk_size=2,
    )

    assert torch.equal(candidates, torch.as_tensor(batch.candidate_mask))
    assert torch.equal(positives, torch.as_tensor(batch.positive_mask))
    assert torch.isfinite(logits[candidates]).all()
    assert torch.isneginf(logits[~candidates]).all()
    assert context.global_shift_zyx_voxel == pytest.approx((2, -3, 4))
    logits[candidates].mean().backward()
    assert torch.isfinite(source_embeddings.grad).all()
    assert torch.isfinite(target_embeddings.grad).all()
    assert torch.isfinite(division_logits.grad).all()


def test_adapter_rejects_non_temporal_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        image_transition_context(
            np.ones((2, 5, 5, 5)),
            np.ones((2, 5, 5, 5)),
            voxel_size_zyx_um=(1.0, 1.0, 1.0),
        )
