#!/usr/bin/env python
"""Submit only a completed, verified multiscale-contextual-v4 kernel version."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "scripts" / "submit-temporal-contextual-kernel.py"
EXPECTED_BASE_SHA256 = (
    "7b43627a89b2d557c69665b3324da02ed8361a9f9a8b32f8df195725269ba6d6"
)


def transformed_source() -> str:
    source_bytes = BASE.read_bytes()
    observed = hashlib.sha256(source_bytes).hexdigest()
    if observed != EXPECTED_BASE_SHA256:
        raise RuntimeError(f"base submission verifier changed: {observed}")
    source = source_bytes.decode("utf-8")
    replacements = (
        (
            "temporal-contextual-pair-fusion-candidate-v3",
            "temporal-multiscale-contextual-pair-fusion-candidate-v4",
            1,
        ),
        (
            "biohub-temporal-contextual-submission-candidate-v3",
            "biohub-multiscale-submission-candidate-v4",
            1,
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
            "biohub-temporal-contextual-final-runtime-v1",
            "biohub-multiscale-contextual-final-v4",
            1,
        ),
        (
            "biohub-temporal-contextual-exact-acceptance-v3",
            "biohub-multiscale-exact-acceptance-v4",
            1,
        ),
        (
            "biohub-temporal-contextual-transfer-v3",
            "biohub-temporal-multiscale-transfer-v4",
            1,
        ),
        ("contextual-v3", "multiscale-contextual-v4", 1),
        ("Contextual v3", "Multiscale contextual v4", 1),
        ("contextual candidate", "multiscale contextual candidate", 4),
    )
    for old, new, expected_count in replacements:
        observed_count = source.count(old)
        if observed_count != expected_count:
            raise RuntimeError(
                f"base submission verifier drifted for {old!r}: "
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
