from pathlib import Path
import runpy
import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_cpu_only_one_shot_frozen_short_name_queue(monkeypatch):
    path=ROOT/'scripts/run-flow-spatial-tta-scoring-queue.py'
    module=runpy.run_path(str(path)); module['validate_staging']()
    source=path.read_text()
    assert "'--accelerator'" not in source and "'competitions','submit'" not in source
    assert "open('x'" in source and "helper.verify_push(output,scoring)" in source
    assert "helper.INSPECT(scoring)['present']" in source
    read=Path.read_bytes
    monkeypatch.setattr(Path,'read_bytes',lambda self: b'changed' if self.name=='biohub-flow-spatial-tta-scoring-v1.ipynb' else read(self))
    with pytest.raises(ValueError,match='Frozen motion notebook changed'): module['validate_staging']()
