from __future__ import annotations

import numpy as np
import pytest

from research.hoct_graph.multibackbone import (
    build_multibackbone_variants,
    materialize_variant,
    required_backbones,
)
from research.hoct_graph.train_biohub_hoct_multibackbone import (
    multibackbone_configurations,
    robust_selection_key,
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


def test_materialize_variant_needs_only_selected_single_backbone() -> None:
    scores = {"ctc": _backbones()["ctc"]}

    selected = materialize_variant("ctc:biohub_probe", scores)

    np.testing.assert_allclose(selected[0][2], [[0.3, 0.9]], atol=1e-6)


def test_materialize_blend_matches_grid() -> None:
    scores = _backbones()
    expected = build_multibackbone_variants(scores)[
        "general_ctc_blend_w0.75:biohub_probe"
    ]

    selected = materialize_variant(
        "general_ctc_blend_w0.75:biohub_probe", scores
    )

    np.testing.assert_allclose(selected[0][2], expected[0][2], atol=1e-6)


def test_multibackbone_linker_grid_has_no_duplicate_configurations() -> None:
    variants = sorted(build_multibackbone_variants(_backbones()))

    configurations = multibackbone_configurations(variants)
    frozen = {tuple(sorted(row.items())) for row in configurations}

    assert len(configurations) == len(frozen)
    assert {row["variant"] for row in configurations} == set(variants)
    assert {row["method"] for row in configurations} == {
        "hoct_only",
        "raw_confidence_hybrid",
    }


def test_robust_selection_prioritizes_worst_embryo_delta() -> None:
    concentrated = {
        "selection_min_delta_vs_base": -0.02,
        "selection_summary": {
            "proxy_score": 1.10,
            "worst_movie": 0.70,
            "div_fp": 0,
        },
        "method": "hoct_only",
    }
    transferable = {
        "selection_min_delta_vs_base": 0.001,
        "selection_summary": {
            "proxy_score": 0.95,
            "worst_movie": 0.90,
            "div_fp": 2,
        },
        "method": "raw_confidence_hybrid",
    }

    assert robust_selection_key(transferable) > robust_selection_key(concentrated)
