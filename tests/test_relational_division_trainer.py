from __future__ import annotations

import torch

from research.temporal_contrastive.train_relational_division_sweep import (
    augment_relational_batch,
    balanced_rows,
    eligible_metrics,
    passes_selection_gate,
)


def test_balanced_rows_covers_every_class_eligibility_stratum() -> None:
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0] * 4)
    eligible = torch.tensor([True, False, True, False] * 4)
    rows = balanced_rows(targets, eligible, 12, torch.Generator().manual_seed(7))
    observed = {(bool(targets[index]), bool(eligible[index])) for index in rows}

    assert len(rows) == 12
    assert observed == {(True, True), (True, False), (False, True), (False, False)}


def test_augmentation_preserves_shape_and_swappable_geometry() -> None:
    patches = torch.zeros(8, 3, 3, 5, 5, 5, dtype=torch.float16)
    geometry = torch.arange(72, dtype=torch.float32).reshape(8, 9)
    augmented, output_geometry = augment_relational_batch(
        patches, geometry, torch.Generator().manual_seed(19)
    )

    assert augmented.shape == patches.shape
    assert output_geometry.shape == geometry.shape
    assert torch.equal(output_geometry[:, 1], geometry[:, 1])
    assert torch.equal(output_geometry[:, 3:], geometry[:, 3:])
    for row in range(len(geometry)):
        assert set(output_geometry[row, [0, 2]].tolist()) == set(
            geometry[row, [0, 2]].tolist()
        )


def test_eligible_metrics_and_gate_ignore_ineligible_hard_negatives() -> None:
    targets = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    scores = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 20.0, 19.0])
    eligible = torch.tensor([True, True, True, True, True, True, False, False])
    inventory = [
        {"embryo": "44b6"},
        {"embryo": "44b6"},
        {"embryo": "6bba"},
        {"embryo": "6bba"},
        {"embryo": "44b6"},
        {"embryo": "6bba"},
        {"embryo": "44b6"},
        {"embryo": "6bba"},
    ]

    metrics = eligible_metrics(targets, scores, eligible, inventory)

    assert metrics["average_precision"] == 1.0
    assert metrics["true_positives_before_first_false_positive"] == 4
    assert passes_selection_gate(metrics) is True
