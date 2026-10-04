from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / 'scripts/replay-focus-bridge-official.py'))


def test_missing_manifest_rejected_before_truth_access(tmp_path):
    with pytest.raises(FileNotFoundError, match='prelabel_graph_manifest'):
        MODULE['replay'](tmp_path, tmp_path / 'absent_truth')


def test_current_and_packaged_scorer_pins_differ():
    assert MODULE['SOURCES']['current'][1] != MODULE['SOURCES']['packaged'][1]
