from __future__ import annotations

import torch
import pytest

from research.temporal_contrastive.multiscale_contextual_pair_fusion import (
    MultiscaleContextualPairFusionAssociationModel,
)
from research.temporal_contrastive.multiscale_division_localization import (
    EXPECTED_PARAMETER_COUNT,
    MAXIMUM_CORRECTION_UM,
    MultiscaleDivisionLocalizationModel,
    architecture_contract,
    integer_jitter_crops,
    load_multiscale_v4_warm_start,
    localization_loss,
    parameter_count,
)


def coordinate_volume(batch: int = 2) -> torch.Tensor:
    z, y, x = torch.meshgrid(
        torch.arange(17), torch.arange(17), torch.arange(17), indexing="ij"
    )
    values = z * 100 + y * 10 + x
    return values.float()[None, None].expand(batch, 3, -1, -1, -1).clone()


def test_integer_jitter_crop_center_and_physical_target_are_exact() -> None:
    patches = coordinate_volume()
    shifts = torch.tensor([[2, -1, 0], [-2, 1, 2]], dtype=torch.int64)

    crops, targets = integer_jitter_crops(patches, shifts)

    assert crops.shape == (2, 3, 13, 13, 13)
    assert crops[0, 0, 6, 6, 6].item() == pytest.approx(10 * 100 + 7 * 10 + 8)
    assert crops[1, 0, 6, 6, 6].item() == pytest.approx(6 * 100 + 9 * 10 + 10)
    assert torch.equal(targets, -shifts.float())


def test_integer_jitter_crop_rejects_padding_or_wrong_dtype() -> None:
    with pytest.raises(ValueError, match="padding-free"):
        integer_jitter_crops(coordinate_volume(1), torch.tensor([[3, 0, 0]]))
    with pytest.raises(ValueError, match="integer"):
        integer_jitter_crops(coordinate_volume(1), torch.zeros((1, 3)))


def test_localization_model_is_zero_initialized_and_bounded() -> None:
    torch.manual_seed(7)
    model = MultiscaleDivisionLocalizationModel().eval()
    patches = torch.randn(2, 3, 13, 13, 13)

    with torch.inference_mode():
        offsets = model.localization_offsets_um(patches)

    assert torch.equal(offsets, torch.zeros_like(offsets))
    with torch.no_grad():
        model.localization_head[-1].bias.fill_(100.0)
        bounded = model.localization_offsets_um(patches)
    assert torch.all(bounded <= MAXIMUM_CORRECTION_UM)
    assert torch.all(bounded >= -MAXIMUM_CORRECTION_UM)


def test_v4_warm_start_preserves_association_predictions_exactly() -> None:
    torch.manual_seed(11)
    v4 = MultiscaleContextualPairFusionAssociationModel().eval()
    localization = MultiscaleDivisionLocalizationModel().eval()
    missing = load_multiscale_v4_warm_start(localization, v4.state_dict())
    patches = torch.randn(2, 3, 13, 13, 13)

    with torch.inference_mode():
        expected = v4(patches)
        actual = localization(patches)

    assert missing
    assert all(name.startswith("localization_head.") for name in missing)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_first_localization_update_unlocks_zero_adapter() -> None:
    torch.manual_seed(19)
    model = MultiscaleDivisionLocalizationModel().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    patches = torch.randn(2, 3, 13, 13, 13)
    target = torch.tensor([[1.0, -1.0, 0.5], [-1.0, 1.0, -0.5]])

    loss = localization_loss(model.localization_offsets_um(patches), target)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.localization_head[-1].weight.grad is not None
    assert torch.count_nonzero(model.localization_head[-1].weight.grad) > 0
    optimizer.step()
    assert torch.count_nonzero(model.localization_head[-1].weight) > 0


def test_localization_architecture_contract_is_stable() -> None:
    contract = architecture_contract()

    assert parameter_count(MultiscaleDivisionLocalizationModel()) == EXPECTED_PARAMETER_COUNT
    assert contract["parameter_count"] == EXPECTED_PARAMETER_COUNT
    assert contract["prediction_preserving_initialization"] is True

