#!/usr/bin/env python
"""Train the blob-aware global-context detector on expanded clean coverage."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

try:
    import local_shape_base as local_shape
    from model_global import MODEL_FAMILY, BlobGlobalTemporalPeakRankDetector
except ModuleNotFoundError:
    from research.peak_rank_detection import (
        train_expanded_real_local_shape_detector as local_shape,
    )
    from research.peak_rank_detection.model_global import (
        MODEL_FAMILY,
        BlobGlobalTemporalPeakRankDetector,
    )


RUN_ID = (
    "synthetic256-expanded-real-pu-faint-local-shape-blob-global-peak-rank-v13"
)


def tagged_atomic_json(
    path: Path,
    payload: dict[str, Any],
    *,
    writer: Callable[[Path, dict[str, Any]], None],
) -> None:
    """Record the non-default model family only in the final terminal."""

    tagged = dict(payload)
    if path.name == "terminal.json":
        tagged["model_family"] = MODEL_FAMILY
    writer(path, tagged)


def main() -> None:
    base = local_shape.expanded.base
    original_writer = base.atomic_json

    def write_with_family(path: Path, payload: dict[str, Any]) -> None:
        tagged_atomic_json(path, payload, writer=original_writer)

    base.RUN_ID = RUN_ID
    base.TemporalPeakRankDetector = BlobGlobalTemporalPeakRankDetector
    base.validate_real_manifest = (
        local_shape.expanded.validate_expanded_real_manifest
    )
    base.augment_example = local_shape.faint.augment_example
    base.training_loss = local_shape.training_loss
    base.atomic_json = write_with_family
    base.main()


if __name__ == "__main__":
    main()
