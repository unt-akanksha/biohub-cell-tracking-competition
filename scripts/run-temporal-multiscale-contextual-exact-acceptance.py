#!/usr/bin/env python
"""Run the pinned multiscale-contextual-v4 exact processed CPU gate."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "scripts" / "run-temporal-contextual-exact-acceptance.py"
EXPECTED_BASE_SHA256 = (
    "dafd6fb978f1abc8ad56612113ec480a54cbd490bc12ff2ba88c4dae8133012f"
)


def transformed_source() -> str:
    source_bytes = BASE.read_bytes()
    observed = hashlib.sha256(source_bytes).hexdigest()
    if observed != EXPECTED_BASE_SHA256:
        raise RuntimeError(f"base exact-acceptance runner changed: {observed}")
    source = source_bytes.decode("utf-8")
    replacements = (
        (
            "temporal-contextual-pair-fusion-processed-acceptance-v3",
            "temporal-multiscale-contextual-pair-fusion-processed-acceptance-v4",
            4,
        ),
        (
            "trackastra_contextual_pair_fusion_blend",
            "trackastra_multiscale_contextual_pair_fusion_blend",
            1,
        ),
        (
            "temporal_contextual_pair_fusion_v3",
            "temporal_multiscale_contextual_pair_fusion_v4",
            1,
        ),
        ("contextual-v3", "multiscale-contextual-v4", 3),
    )
    for old, new, expected_count in replacements:
        observed_count = source.count(old)
        if observed_count != expected_count:
            raise RuntimeError(
                f"base exact runner drifted for {old!r}: "
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
