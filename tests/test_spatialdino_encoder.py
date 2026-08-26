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


def test_intermediate_forward_keeps_final_output_equivalent() -> None:
    torch.manual_seed(3)
    model = SpatialDinoViTS8().eval()
    volume = torch.randn(1, 1, 8, 8, 8)
    with torch.inference_mode():
        final = model(volume)
        intermediate = model.forward_intermediates(volume, block_indices=(2, 5, 8, 11))

    assert len(intermediate) == 4
    assert all(grid.shape == (1, 384, 1, 1, 1) for grid in intermediate)
    torch.testing.assert_close(intermediate[-1], final, rtol=0, atol=0)


def test_intermediate_indices_must_be_ordered_and_in_range() -> None:
    model = SpatialDinoViTS8()
    volume = torch.zeros(1, 1, 8, 8, 8)
    with pytest.raises(ValueError, match="unique and increasing"):
        model.forward_intermediates(volume, block_indices=(5, 2))
    with pytest.raises(ValueError, match="outside"):
        model.forward_intermediates(volume, block_indices=(12,))
