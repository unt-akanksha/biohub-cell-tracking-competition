from __future__ import annotations

import numpy as np
import pytest

from research.hoct_graph.multibackbone import (
    build_multibackbone_variants,
    required_backbones,
)


def _pair(matrix):
    return {
        0: (
            np.asarray([10], dtype=np.int64),
            np.asarray([20, 21], dtype=np.int64),
            np.asarray(matrix, dtype=np.float32),
        )
    }


def _backbones():
    return {
        "general": {
            "pretrained": _pair([[0.8, 0.0]]),
            "biohub_probe": _pair([[0.7, 0.1]]),
        },
        "ctc": {
            "pretrained": _pair([[0.0, 0.6]]),
            "biohub_probe": _pair([[0.3, 0.9]]),
        },
    }


def test_default_multibackbone_grid_is_small_and_stable() -> None:
    variants = build_multibackbone_variants(_backbones())

    assert len(variants) == 10
    assert set(variants) >= {
        "general:pretrained",
        "general:biohub_probe",
        "ctc:pretrained",
        "ctc:biohub_probe",
        "general_ctc_blend_w0.5:pretrained",
        "general_ctc_blend_w0.5:biohub_probe",
    }


def test_multibackbone_blend_preserves_candidate_union() -> None:
    variants = build_multibackbone_variants(_backbones())

    np.testing.assert_allclose(
        variants["general_ctc_blend_w0.5:pretrained"][0][2],
        [[0.8, 0.6]],
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("general:pretrained", {"general"}),
        ("ctc:biohub_probe", {"ctc"}),
        ("general_ctc_blend_w0.25:pretrained", {"general", "ctc"}),
    ],
)
def test_required_backbones(variant: str, expected: set[str]) -> None:
    assert required_backbones(variant) == expected


def test_multibackbone_rejects_incomplete_sources() -> None:
    with pytest.raises(ValueError, match="exactly the general and ctc"):
        build_multibackbone_variants({"general": _backbones()["general"]})
