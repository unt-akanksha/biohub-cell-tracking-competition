from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle/biohub-real-division-patches-v1"
SCRIPT = KERNEL / "extract.py"
SPEC = importlib.util.spec_from_file_location("real_division_patches", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
patches = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patches)


def test_kernel_is_private_cpu_train_only_extractor() -> None:
    metadata = json.loads((KERNEL / "kernel-metadata.json").read_text())
    source = SCRIPT.read_text()

    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is False
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert '"competition_test_data_read": False' in source
    assert "submission.csv" not in source


def test_movie_examples_use_only_division_frames() -> None:
    nodes = {
        1: (4, 0.0, 0.0, 0.0),
        2: (4, 1.0, 1.0, 1.0),
        3: (3, 2.0, 2.0, 2.0),
        4: (5, 0.0, 1.0, 0.0),
        5: (5, 0.0, -1.0, 0.0),
        6: (5, 1.0, 2.0, 1.0),
        7: (4, 2.0, 3.0, 2.0),
    }
    edges = [(1, 4), (1, 5), (2, 6), (3, 7)]

    rows = patches.movie_examples(nodes, edges)

    assert [row["node_id"] for row in rows] == [1, 2]
    assert [row["division_target"] for row in rows] == [True, False]


def test_physical_patch_sampler_returns_normalized_temporal_channels() -> None:
    volume = np.arange(3 * 9 * 9 * 9, dtype=np.float32).reshape(3, 9, 9, 9)
    result = patches.sample_physical_patches(
        volume,
        np.asarray([[4.0, 4.0, 4.0]], dtype=np.float32),
        output_shape_zyx=(5, 5, 5),
        half_extent_zyx_um=(2.0, 2.0, 2.0),
    )

    assert tuple(result.shape) == (1, 3, 5, 5, 5)
    np.testing.assert_allclose(
        result.mean(dim=(2, 3, 4)).numpy(), np.zeros((1, 3)), atol=1e-5
    )


def test_patch_selection_split_is_division_stratified() -> None:
    examples = {
        **{
            f"44b6_{index:08x}": [{"division_target": True}]
            for index in range(10)
        },
        **{
            f"6bba_{index:08x}": [{"division_target": True}]
            for index in range(10)
        },
    }

    selected = patches.stratified_selection_stems(examples)

    assert sum(stem.startswith("44b6_") for stem in selected) == 2
    assert sum(stem.startswith("6bba_") for stem in selected) == 2
