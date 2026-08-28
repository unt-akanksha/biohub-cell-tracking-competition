#!/usr/bin/env python
"""Stage exact multiscale-v4 acceptance as a private Kaggle dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "scripts" / "stage-temporal-contextual-kaggle-artifacts.py"
EXPECTED_BASE_SHA256 = (
    "1bd2319310c2e24c637b0d2b0d909fe11f756af7d2e5e450cafc614fee4f98fa"
)


def transformed_source() -> str:
    source_bytes = BASE.read_bytes()
    observed = hashlib.sha256(source_bytes).hexdigest()
    if observed != EXPECTED_BASE_SHA256:
        raise RuntimeError(f"base contextual artifact stager changed: {observed}")
    source = source_bytes.decode("utf-8")
    replacements = (
        (
            "temporal-contextual-pair-fusion-v3",
            "temporal-multiscale-contextual-pair-fusion-v4",
            2,
        ),
        (
            "temporal_contextual_pair_fusion_v3",
            "temporal_multiscale_contextual_pair_fusion_v4",
            1,
        ),
        (
            "trackastra_contextual_pair_fusion_blend",
            "trackastra_multiscale_contextual_pair_fusion_blend",
            1,
        ),
        (
            "biohub-temporal-contextual-exact-acceptance-v3",
            "biohub-multiscale-exact-acceptance-v4",
            2,
        ),
        (
            "contextual_v3_exact_acceptance",
            "multiscale_contextual_v4_exact_acceptance",
            1,
        ),
        (
            "Biohub Temporal Contextual Exact Acceptance v3",
            "Biohub Multiscale Exact Acceptance v4",
            1,
        ),
        ("contextual-v3", "multiscale-contextual-v4", 6),
    )
    for old, new, expected_count in replacements:
        observed_count = source.count(old)
        if observed_count != expected_count:
            raise RuntimeError(
                f"base artifact stager drifted for {old!r}: "
                f"expected {expected_count}, saw {observed_count}"
            )
        source = source.replace(old, new)
    return source


def main() -> None:
    namespace = {
        "__name__": "__main__",
        "__file__": str(Path(__file__).resolve()),
        "__package__": None,
    }
    exec(compile(transformed_source(), str(Path(__file__).resolve()), "exec"), namespace)


if __name__ == "__main__":
    main()
