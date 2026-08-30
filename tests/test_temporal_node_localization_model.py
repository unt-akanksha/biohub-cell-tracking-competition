from __future__ import annotations

import pytest
import torch

from research.temporal_localization.model import (
    EXPECTED_PARAMETER_COUNT,
    MAXIMUM_CORRECTION_UM,
    TemporalNodeLocalizationModel,
    architecture_contract,
    localization_loss,
    parameter_count,
)


def test_architecture_contract_is_large_owned_and_topology_preserving() -> None:
    contract = architecture_contract()
    assert contract["parameter_count"] == EXPECTED_PARAMETER_COUNT
    assert contract["parameter_count"] > 50_000_000
    assert contract["topology_preserving"] is True
    assert contract["node_count_preserving"] is True
    assert contract["public_code_copied"] is False
    assert parameter_count(TemporalNodeLocalizationModel()) == EXPECTED_PARAMETER_COUNT


def test_forward_is_bounded_and_loss_is_finite() -> None:
    model = TemporalNodeLocalizationModel().eval()
    patches = torch.randn(2, 3, 17, 17, 17)
    graph = torch.randn(2, 12)
    with torch.inference_mode():
        offsets, log_variance, safe = model(patches, graph)
    assert offsets.shape == (2, 3)
    assert log_variance.shape == (2, 3)
    assert safe.shape == (2,)
    assert torch.all(offsets.abs() <= MAXIMUM_CORRECTION_UM)
    assert torch.all((safe >= 0.0) & (safe <= 1.0))
    loss, parts = localization_loss(
        offsets, log_variance, safe, torch.zeros_like(offsets)
    )
    assert torch.isfinite(loss)
    assert set(parts) == {"robust", "heteroscedastic", "calibration"}


def test_forward_rejects_graph_inventory_mismatch() -> None:
    with pytest.raises(ValueError, match="graph features"):
        TemporalNodeLocalizationModel()(torch.randn(1, 3, 17, 17, 17), torch.randn(1, 11))
