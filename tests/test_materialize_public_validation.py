from __future__ import annotations

import os
from pathlib import Path

from research.trackastra_graph.materialize_public_validation import (
    load_public_namespace,
)


def test_public_preset_executes_before_config_and_postprocess(
    tmp_path: Path, monkeypatch
) -> None:
    key = "BIOHUB_TEST_FROZEN_PRESET"
    monkeypatch.delenv(key, raising=False)
    preset = tmp_path / "preset.py"
    config = tmp_path / "config.py"
    postprocess = tmp_path / "postprocess.py"
    preset.write_text(
        f'import os\nos.environ["{key}"] = "frozen-value"\n', encoding="utf-8"
    )
    config.write_text(
        f'import os\nCONFIGURED = os.environ["{key}"]\n', encoding="utf-8"
    )
    postprocess.write_text(
        'POSTPROCESSED = CONFIGURED + "-postprocessed"\n', encoding="utf-8"
    )

    namespace = load_public_namespace(
        preset,
        config,
        postprocess,
        tmp_path / "competition",
    )

    assert namespace["CONFIGURED"] == "frozen-value"
    assert namespace["POSTPROCESSED"] == "frozen-value-postprocessed"
    assert namespace["TEST_DIR"] == tmp_path / "competition" / "train"
    assert os.environ[key] == "frozen-value"
