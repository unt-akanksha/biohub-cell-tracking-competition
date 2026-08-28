#!/usr/bin/env python
"""Build the guarded-launch preflight for the multiscale-contextual-v4 candidate."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "scripts" / "build-temporal-contextual-submission-candidate-preflight.py"
EXPECTED_BASE_SHA256 = (
    "b785a75f088c68ede25359a652510126a49a1758c9fb729ce956183576147b0f"
)


def transformed_source() -> str:
    source_bytes = BASE.read_bytes()
    observed = hashlib.sha256(source_bytes).hexdigest()
    if observed != EXPECTED_BASE_SHA256:
        raise RuntimeError(f"base candidate preflight changed: {observed}")
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
            2,
        ),
        (
            "build-temporal-contextual-submission-candidate-kernel.py",
            "build-temporal-multiscale-contextual-submission-kernel.py",
            1,
        ),
        (
            "biohub-temporal-contextual-transfer-v3",
            "biohub-temporal-multiscale-transfer-v4",
            1,
        ),
        (
            "biohub-temporal-contextual-exact-acceptance-v3",
            "biohub-multiscale-exact-acceptance-v4",
            1,
        ),
        (
            "temporal-contextual-pair-fusion-v3-exact-acceptance.json",
            "temporal-multiscale-contextual-pair-fusion-v4-exact-acceptance.json",
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
            "e69a20f10f56108818a6bf0715fe071e868fd176d1720cc2ffc04f2a645b41ff",
            "221ab4359e84706a306cd88b41e2790ebd1a4bad71d2a36ef4a1873499a570b4",
            1,
        ),
        (
            "biohub-temporal-contextual-final-runtime-v1",
            "biohub-multiscale-contextual-final-v4",
            1,
        ),
        (
            "temporal-contextual-pair-fusion-processed-acceptance-v3",
            "temporal-multiscale-contextual-pair-fusion-processed-acceptance-v4",
            3,
        ),
        (
            "contextual_v3_exact_acceptance",
            "multiscale_contextual_v4_exact_acceptance",
            1,
        ),
        (
            "temporal-contextual-pair-fusion-v3",
            "temporal-multiscale-contextual-pair-fusion-v4",
            4,
        ),
        (
            "test_temporal_contextual_submission_candidate_kernel_builder.py",
            "test_temporal_multiscale_downstream.py",
            1,
        ),
        ("contextual-v3", "multiscale-contextual-v4", 2),
        ("Contextual-v3", "Multiscale-contextual-v4", 1),
    )
    for old, new, expected_count in replacements:
        observed_count = source.count(old)
        if observed_count != expected_count:
            raise RuntimeError(
                f"base candidate preflight drifted for {old!r}: "
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
