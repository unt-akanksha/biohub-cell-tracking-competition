from __future__ import annotations

from research.temporal_contrastive.verify_division_localization_dataset import (
    SPLIT_CONTRACT,
    V3_V4_VALIDATION_TIMEPOINTS,
)


def test_localization_windows_are_frozen_disjoint_and_temporally_supported() -> None:
    selection = SPLIT_CONTRACT["selection"]
    audit = SPLIT_CONTRACT["audit"]
    combined = selection + audit

    assert len(selection) == 8
    assert len(audit) == 8
    assert len(set(combined)) == 16
    assert not set(selection) & set(audit)
    assert not set(combined) & V3_V4_VALIDATION_TIMEPOINTS
    assert min(combined) >= 2
    assert max(combined) <= 598

