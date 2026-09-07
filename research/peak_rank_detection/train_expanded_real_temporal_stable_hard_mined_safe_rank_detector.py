#!/usr/bin/env python
"""Train V23 architecture with balanced optimization-only hard mining."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

try:
    import temporal_stable_base as temporal
except ModuleNotFoundError:
    from research.peak_rank_detection import (
        train_expanded_real_temporal_stable_balanced_safe_rank_detector as temporal,
    )


RUN_ID = (
    "synthetic256-expanded-real-temporal-min-balanced-hard-mined-pu-faint-"
    "local-shape-multiscale-blob-global-safe-rank-peak-rank-v27"
)
SAMPLING_MANIFEST_ENV = "BIOHUB_HARD_MINING_MANIFEST"
SAMPLING_MANIFEST_SHA256 = (
    "9967efa25021453b4f043a23e47e744153da678f17f5c00180ecd6ebd4dca900"
)
EXPECTED_UNIQUE_COUNTS = {"44b6": 150, "6bba": 330}
EXPECTED_EFFECTIVE_COUNTS = {"44b6": 495, "6bba": 495}

_base_load_real_examples = (
    temporal.balanced.safe.local_shape.expanded.base.load_real_examples
)
_sampling_receipt: dict[str, Any] | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repeat_examples_by_manifest(
    examples: Sequence[Any], manifest: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    """Validate the frozen inventory and materialize its exact multiplicities."""

    if not (
        manifest.get("schema_version") == 1
        and manifest.get("status") == "complete"
        and manifest.get("run_id")
        == "optimization-only-balanced-local-snr-hard-mining-v27"
        and manifest.get("expected_unique_counts") == EXPECTED_UNIQUE_COUNTS
        and manifest.get("effective_counts") == EXPECTED_EFFECTIVE_COUNTS
        and manifest.get("target_effective_count_per_embryo") == 495
        and manifest.get("local_snr_bands") == [[3, 9], [3, 11], [5, 13]]
        and manifest.get("competition_train_data_read") is True
        and manifest.get("competition_test_data_read") is False
        and manifest.get("selection_data_read") is False
        and manifest.get("sealed_audit_data_read") is False
        and manifest.get("public_predictions_read") is False
        and manifest.get("public_notebook_weights_read") is False
        and manifest.get("public_leaderboard_used_for_selection") is False
        and manifest.get("submission_created") is False
    ):
        raise ValueError("hard-mining manifest violates the frozen clean contract")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != sum(EXPECTED_UNIQUE_COUNTS.values()):
        raise ValueError("hard-mining manifest inventory changed")
    by_identity = {str(example.identity): example for example in examples}
    if len(by_identity) != len(examples):
        raise ValueError("optimization examples contain duplicate identities")
    entry_ids = [str(row.get("identity")) for row in entries]
    if len(entry_ids) != len(set(entry_ids)) or set(entry_ids) != set(by_identity):
        raise ValueError("hard-mining manifest and optimization examples differ")

    repeated: list[Any] = []
    observed_effective = Counter()
    multiplicity_histogram: dict[str, Counter[int]] = {
        embryo: Counter() for embryo in EXPECTED_UNIQUE_COUNTS
    }
    for row in sorted(entries, key=lambda item: str(item["identity"])):
        identity = str(row["identity"])
        embryo = str(row.get("embryo"))
        multiplicity = int(row.get("sampling_multiplicity", 0))
        if (
            embryo not in EXPECTED_UNIQUE_COUNTS
            or not identity.startswith(f"{embryo}_")
            or multiplicity <= 0
        ):
            raise ValueError("hard-mining entry is invalid")
        repeated.extend([by_identity[identity]] * multiplicity)
        observed_effective[embryo] += multiplicity
        multiplicity_histogram[embryo][multiplicity] += 1
    if dict(observed_effective) != EXPECTED_EFFECTIVE_COUNTS:
        raise ValueError("hard-mining effective counts changed")
    receipt = {
        "unique_counts": dict(EXPECTED_UNIQUE_COUNTS),
        "effective_counts": dict(observed_effective),
        "multiplicity_histogram": {
            embryo: {str(key): value for key, value in sorted(histogram.items())}
            for embryo, histogram in sorted(multiplicity_histogram.items())
        },
        "effective_examples": len(repeated),
    }
    return repeated, receipt


def load_hard_mined_real_examples(
    paths: Sequence[Path], *, role: str
) -> list[Any]:
    global _sampling_receipt
    examples = list(_base_load_real_examples(paths, role=role))
    if role != "optimization":
        return examples
    raw_path = os.environ.get(SAMPLING_MANIFEST_ENV)
    if not raw_path:
        raise RuntimeError(f"{SAMPLING_MANIFEST_ENV} is required")
    manifest_path = Path(raw_path).resolve()
    if not manifest_path.is_file() or sha256_file(manifest_path) != SAMPLING_MANIFEST_SHA256:
        raise ValueError("hard-mining sampling manifest hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repeated, receipt = repeat_examples_by_manifest(examples, manifest)
    _sampling_receipt = {
        **receipt,
        "manifest_sha256": SAMPLING_MANIFEST_SHA256,
        "manifest_run_id": manifest["run_id"],
    }
    return repeated


def tagged_atomic_json(
    path: Path,
    payload: dict[str, Any],
    *,
    writer: Callable[[Path, dict[str, Any]], None],
) -> None:
    tagged = dict(payload)
    if path.name == "terminal.json":
        if _sampling_receipt is None:
            raise RuntimeError("hard-mining receipt was not materialized")
        tagged.update(
            {
                "model_family": temporal.MODEL_FAMILY,
                "safe_negative_evidence_band": [5, 13],
                "safe_negative_boundary_quantile": (
                    temporal.balanced.safe.SAFE_BACKGROUND_QUANTILE
                ),
                "safe_negative_shell_radius_voxels": [
                    temporal.balanced.safe.SAFE_SHELL_MINIMUM_RADIUS,
                    temporal.balanced.safe.SAFE_SHELL_MAXIMUM_RADIUS,
                ],
                "optimization_annotated_below_boundary_fraction": (
                    temporal.balanced.safe.OPTIMIZATION_ANNOTATED_BELOW_BOUNDARY_FRACTION
                ),
                "real_optimization_sampling_policy": (
                    "embryo_balanced_495_each_v23_local_snr_hardest_first"
                ),
                "hard_mining_role": "expanded_real_optimization_only",
                "hard_mining_selection_sampling_changed": False,
                "hard_mining_sealed_audit_sampling_changed": False,
                "hard_mining_receipt": _sampling_receipt,
            }
        )
    writer(path, tagged)


def main() -> None:
    base = temporal.balanced.safe.local_shape.expanded.base
    original_writer = base.atomic_json

    def write_with_contract(path: Path, payload: dict[str, Any]) -> None:
        tagged_atomic_json(path, payload, writer=original_writer)

    temporal.balanced.safe.MODEL_FAMILY = temporal.MODEL_FAMILY
    base.RUN_ID = RUN_ID
    base.TemporalPeakRankDetector = temporal.TemporalMinimumLocalSnrSafeRankDetector
    base.validate_real_manifest = (
        temporal.balanced.safe.local_shape.expanded.validate_expanded_real_manifest
    )
    base.load_real_examples = load_hard_mined_real_examples
    base.augment_example = temporal.balanced.safe.local_shape.faint.augment_example
    base.training_loss = temporal.balanced.safe.training_loss
    base.atomic_json = write_with_contract
    base.main()


if __name__ == "__main__":
    main()
