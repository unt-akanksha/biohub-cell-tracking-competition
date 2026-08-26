from __future__ import annotations

import pytest
import torch

from research.spatialdino_association.encoder import PatchEmbed, SpatialDinoViTS8


def test_patch_embed_rejects_misaligned_or_multichannel_volumes() -> None:
    embed = PatchEmbed(embed_dim=4, patch_size=8)
    with pytest.raises(ValueError, match="divisible"):
        embed(torch.zeros(1, 1, 8, 8, 9))
    with pytest.raises(ValueError, match="shape"):
        embed(torch.zeros(1, 2, 8, 8, 8))


def test_vits8_has_checkpoint_compatible_shape_and_parameters() -> None:
    model = SpatialDinoViTS8()

    assert len(model.state_dict()) == 174
    assert sum(parameter.numel() for parameter in model.parameters()) == 21_501_312
    assert model.patch_embed.proj.weight.shape == (384, 1, 8, 8, 8)
