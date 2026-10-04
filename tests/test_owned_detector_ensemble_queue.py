from pathlib import Path
import runpy
import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_frozen_cpu_only_followthrough_and_tamper_rejection(monkeypatch):
    path=ROOT/'scripts/run-owned-detector-ensemble-scoring-queue.py'
    module=runpy.run_path(str(path))
    module['validate_staging']()
    source=path.read_text()
    assert "'--accelerator'" not in source and "'competitions','submit'" not in source
    assert "open('x'" in source and "helper.verify_push(output,scoring)" in source
    assert "helper.wait_complete(selection,deadline)" in source
    assert "helper.INSPECT(scoring)['present']" in source
    read=Path.read_bytes
    monkeypatch.setattr(Path,'read_bytes',lambda self: b'corrupt' if self.name=='biohub-owned-detector-ensemble-selection-v1.ipynb' else read(self))
    with pytest.raises(ValueError,match='Frozen ensemble launch changed'): module['validate_staging']()
