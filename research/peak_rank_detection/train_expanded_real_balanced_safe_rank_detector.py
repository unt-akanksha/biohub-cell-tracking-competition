#!/usr/bin/env python
"""Train the safe-rank detector with balanced real optimization sampling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

try:
    import safe_rank_base as safe
except ModuleNotFoundError:
    from research.peak_rank_detection import (
        train_expanded_real_safe_rank_detector as safe,
    )


RUN_ID = (
    "synthetic256-expanded-real-balanced-pu-faint-local-shape-multiscale-"
    "blob-global-safe-rank-peak-rank-v19"
)
ORIGINAL_OPTIMIZATION_COUNTS = {"44b6": 150, "6bba": 330}
BALANCED_OPTIMIZATION_COUNTS = {"44b6": 300, "6bba": 330}

_base_load_real_examples = safe.local_shape.expanded.base.load_real_examples


def load_embryo_balanced_real_examples(
    paths: Sequence[Path], *, role: str
) -> list[Any]:
    """Duplicate only 44b6 optimization crops; preserve held-out roles exactly."""

    examples = list(_base_load_real_examples(paths, role=role))
    if role != "optimization":
        return examples
    forty_four = [example for example in examples if example.identity.startswith("44b6_")]
    six_bba = [example for example in examples if example.identity.startswith("6bba_")]
    if len(forty_four) != 150 or len(six_bba) != 330:
        raise ValueError("expanded optimization embryo inventory changed")
    return [*examples, *forty_four]


def tagged_atomic_json(
    path: Path,
    payload: dict[str, Any],
    *,
    writer: Callable[[Path, dict[str, Any]], None],
) -> None:
    tagged = dict(payload)
    if path.name == "terminal.json":
        tagged.update(
            {
                "model_family": safe.MODEL_FAMILY,
                "safe_negative_evidence_band": [5, 13],
                "safe_negative_boundary_quantile": safe.SAFE_BACKGROUND_QUANTILE,
                "safe_negative_shell_radius_voxels": [
                    safe.SAFE_SHELL_MINIMUM_RADIUS,
                    safe.SAFE_SHELL_MAXIMUM_RADIUS,
                ],
                "safe_negative_selection_role": "expanded_real_optimization_only",
                "optimization_annotated_below_boundary_fraction": (
                    safe.OPTIMIZATION_ANNOTATED_BELOW_BOUNDARY_FRACTION
                ),
                "real_optimization_sampling_policy": (
                    "duplicate_44b6_once_balance_embryo_crops"
                ),
                "real_optimization_original_counts": ORIGINAL_OPTIMIZATION_COUNTS,
                "real_optimization_effective_counts": BALANCED_OPTIMIZATION_COUNTS,
                "selection_sampling_changed": False,
                "sealed_audit_sampling_changed": False,
            }
        )
    writer(path, tagged)


def main() -> None:
    base = safe.local_shape.expanded.base
    original_writer = base.atomic_json

    def write_with_contract(path: Path, payload: dict[str, Any]) -> None:
        tagged_atomic_json(path, payload, writer=original_writer)

    base.RUN_ID = RUN_ID
    base.TemporalPeakRankDetector = safe.SafeRankMultiscaleBlobGlobalDetector
    base.validate_real_manifest = (
        safe.local_shape.expanded.validate_expanded_real_manifest
    )
    base.load_real_examples = load_embryo_balanced_real_examples
    base.augment_example = safe.local_shape.faint.augment_example
    base.training_loss = safe.training_loss
    base.atomic_json = write_with_contract
    base.main()


if __name__ == "__main__":
    main()
