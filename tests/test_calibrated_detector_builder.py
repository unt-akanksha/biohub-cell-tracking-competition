import ast
import json
from pathlib import Path
import runpy
import pytest

ROOT=Path(__file__).resolve().parents[1]
B=runpy.run_path(str(ROOT/'scripts/build-calibrated-detector-selection.py'))


def test_builder_requires_full_pass_and_pins_scoring_notebook(monkeypatch):
    report,_=runpy.run_path(str(ROOT/'tests/test_detector_confidence_calibration.py'))['fixture']()
    report_path=ROOT/'reports/experiments/detector-calibration-full-v1-result.json'
    original_bytes=Path.read_bytes; original_text=Path.read_text
    monkeypatch.setattr(Path,'read_bytes',lambda p,*a,**k: json.dumps(report).encode() if p==report_path else original_bytes(p,*a,**k))
    nb,meta,target=B['build']()
    assert nb['metadata']['codex']['detector_confidence_calibration']['threshold']==.75
    assert meta['enable_gpu'] and not meta['enable_internet']
    assert meta['kernel_sources']==['indarkarhana/biohub-owned-detector-fit-pair-v1/1']
    assert '--detector-calibration-json' in ''.join(nb['cells'][-1]['source'])
    path=target/meta['code_file']; content=json.dumps(nb)
    monkeypatch.setattr(Path,'read_text',lambda p,*a,**k: content if p==path else original_text(p,*a,**k))
    scorer,smeta,_=B['build'](True)
    source=''.join(scorer['cells'][-1]['source']); bundle=ast.literal_eval(B['assignment'](source,'scoring_sources').value)
    assert bundle['selection_launch.ipynb']==content
    assert 'research/detector_confidence_calibration.py' in bundle
    assert not smeta['enable_gpu'] and not smeta['enable_internet']
    assert smeta['kernel_sources']==['indarkarhana/biohub-owned-detector-calibrated-selection-v1/1']
    report['full_diagnostic_recall_preserved']=False
    with pytest.raises(ValueError): B['build']()
