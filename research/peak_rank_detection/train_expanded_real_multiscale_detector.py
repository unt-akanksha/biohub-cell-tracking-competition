#!/usr/bin/env python
"""Train the optimization-selected multi-scale blob/global detector."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

try:
    import local_shape_base as local_shape
    from model_multiscale import (
        MODEL_FAMILY,
        MultiscaleBlobGlobalTemporalPeakRankDetector,
    )
except ModuleNotFoundError:
    from research.peak_rank_detection import (
        train_expanded_real_local_shape_detector as local_shape,
    )
    from research.peak_rank_detection.model_multiscale import (
        MODEL_FAMILY,
        MultiscaleBlobGlobalTemporalPeakRankDetector,
    )


RUN_ID = (
    "synthetic256-expanded-real-pu-faint-local-shape-multiscale-blob-global-"
    "peak-rank-v15"
)


def tagged_atomic_json(
    path: Path,
    payload: dict[str, Any],
    *,
    writer: Callable[[Path, dict[str, Any]], None],
) -> None:
    tagged = dict(payload)
    if path.name == "terminal.json":
        tagged["model_family"] = MODEL_FAMILY
        tagged["blob_bands"] = [[3, 9], [5, 13], [7, 15]]
        tagged["blob_scale_selection_role"] = "expanded_real_optimization_only"
    writer(path, tagged)


def main() -> None:
    base = local_shape.expanded.base
    original_writer = base.atomic_json

    def write_with_family(path: Path, payload: dict[str, Any]) -> None:
        tagged_atomic_json(path, payload, writer=original_writer)

    base.RUN_ID = RUN_ID
    base.TemporalPeakRankDetector = MultiscaleBlobGlobalTemporalPeakRankDetector
    base.validate_real_manifest = (
        local_shape.expanded.validate_expanded_real_manifest
    )
    base.augment_example = local_shape.faint.augment_example
    base.training_loss = local_shape.training_loss
    base.atomic_json = write_with_family
    base.main()


if __name__ == "__main__":
    main()
