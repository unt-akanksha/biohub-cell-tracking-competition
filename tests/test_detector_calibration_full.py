import ast
import copy
from pathlib import Path
import runpy
import pytest

ROOT=Path(__file__).resolve().parents[1]
G=runpy.run_path(str(ROOT/'research/detector_calibration_full_contract.py'))
B=runpy.run_path(str(ROOT/'scripts/build-detector-calibration-full.py'))


def test_exact_successful_small_probe_required():
    payload=(ROOT/'reports/experiments/detector-calibration-probe-v1-result.json').read_bytes()
    replay=G['verify_probe'](payload)
    assert len(replay)==18
    with pytest.raises(ValueError): G['verify_probe'](payload+b' ')
    row=copy.deepcopy(next(iter(replay.values())))
    assert G['verify_replay_row'](row,replay)
    row['input_sha256']='f'*64
    with pytest.raises(ValueError): G['verify_replay_row'](row,replay)


def test_full_builder_freezes_all_but_declared_collection_scope():
    path=ROOT/'kaggle/biohub-owned-detector-calibration-probe-v1/biohub-owned-detector-calibration-probe-v1.ipynb'
    before=path.read_bytes(); nb,meta=B['build']()
    assert path.read_bytes()==before
    codex=nb['metadata']['codex']
    assert codex['training_movies']==120 and codex['fitting_movies']==96 and codex['diagnostic_movies']==24
    assert codex['declared_budget_seconds']==3600 and not codex['authorized_for_submission']
    assert not codex['target_audit_opened'] and not meta['enable_internet'] and meta['enable_gpu']
    source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(node.value)
    runner=runtime['run_selection.py']
    assert 'groups=calibration_scope(split,probe=False)' in runner
    assert 'len(records)!=360 or replay_count!=18' in runner
    assert 'autocast' not in runner and 'optimizer' not in runner
    assert len(G['verify_probe'](runtime['frozen_calibration_probe_report.json'].encode()))==18
    with pytest.raises(ValueError): B['full_runtime'](runner)
