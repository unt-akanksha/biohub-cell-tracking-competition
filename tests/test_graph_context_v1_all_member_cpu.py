from __future__ import annotations

import json
from pathlib import Path
import runpy

import torch

from research.temporal_contrastive.train_graph_context_division_sweep import (
    eligible_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "research/temporal_contrastive/evaluate_graph_context_v1_all_member_cpu.py"
)


def test_eligible_subset_preserves_the_reported_metrics() -> None:
    module = runpy.run_path(str(SCRIPT))
    eligible_subset = module["eligible_subset"]
    rows = 8
    eligible = torch.tensor([True, False, True, False, True, False, True, False])
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    weights = torch.ones(rows)
    inventory = [
        {"embryo": "44b6" if index < 4 else "6bba", "row": index}
        for index in range(rows)
    ]
    data = (
        torch.arange(rows).reshape(rows, 1),
        torch.arange(rows).reshape(rows, 1),
        torch.arange(rows).reshape(rows, 1),
        torch.ones((rows, 1), dtype=torch.bool),
        targets,
        weights,
        eligible,
        inventory,
    )
    scores = torch.tensor([0.9, 0.8, 0.2, 0.7, 0.7, 0.6, 0.1, 0.3])
    subset = eligible_subset(data)
    original_metrics = eligible_metrics(targets, scores, eligible, inventory)
    subset_metrics = eligible_metrics(
        subset[4], scores[eligible], subset[6], subset[7]
    )
    assert json.dumps(original_metrics, sort_keys=True) == json.dumps(
        subset_metrics, sort_keys=True
    )
