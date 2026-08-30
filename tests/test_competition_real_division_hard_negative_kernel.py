from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle/biohub-real-division-hard-negative-patches-v2"
SCRIPT = KERNEL / "extract.py"
SPEC = importlib.util.spec_from_file_location("real_division_hard_negatives", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
extractor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extractor)


def test_kernel_is_private_cpu_only_and_uses_pinned_runtime() -> None:
    metadata = json.loads((KERNEL / "kernel-metadata.json").read_text())
    source = SCRIPT.read_text()

    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is False
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["dataset_sources"] == [
        "indarkarhana/biohub-kaggle-codec-wheels-v1",
        "indarkarhana/biohub-real-division-extractor-runtime-v1",
    ]
    assert '"competition_test_data_read": False' in source
    assert "submission.csv" not in source


def test_hard_negative_frame_is_added_without_changing_division_labels() -> None:
    nodes = {
        1: (2, 0.0, 0.0, 0.0),
        2: (2, 1.0, 1.0, 1.0),
        3: (3, 0.0, 1.0, 0.0),
        4: (3, 0.0, -1.0, 0.0),
        5: (3, 1.0, 2.0, 1.0),
        10: (5, 0.0, 0.0, 0.0),
        11: (5, 1.0, 0.0, 0.0),
        12: (5, 2.0, 0.0, 0.0),
        13: (5, 3.0, 0.0, 0.0),
        20: (6, 0.0, 1.0, 0.0),
        21: (6, 1.0, 1.0, 0.0),
        22: (6, 2.0, 1.0, 0.0),
        23: (6, 3.0, 1.0, 0.0),
    }
    edges = [(1, 3), (1, 4), (2, 5), (10, 20), (11, 21), (12, 22), (13, 23)]

    frames = extractor.movie_frames("44b6_deadbeef", nodes, edges)

    assert frames[2]["frame_role"] == "division"
    assert sum(row["division_target"] for row in frames[2]["rows"]) == 1
    assert frames[5]["frame_role"] == "no_division_hard_negative"
    assert not any(row["division_target"] for row in frames[5]["rows"])


def test_roles_are_movie_disjoint_and_stratified_by_embryo() -> None:
    examples = {
        **{
            f"44b6_{index:08x}": {
                3: {"rows": [{"division_target": True}]}
            }
            for index in range(20)
        },
        **{
            f"6bba_{index:08x}": {
                3: {"rows": [{"division_target": True}]}
            }
            for index in range(20)
        },
    }

    roles = extractor.allocate_roles(examples)

    for embryo in ("44b6", "6bba"):
        embryo_roles = [role for stem, role in roles.items() if stem.startswith(embryo)]
        assert embryo_roles.count("audit") == 4
        assert embryo_roles.count("selection") == 4
        assert embryo_roles.count("optimization") == 12
