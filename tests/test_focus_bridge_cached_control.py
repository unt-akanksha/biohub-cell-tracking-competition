from pathlib import Path
import runpy
import pytest

ROOT=Path(__file__).resolve().parents[1]
M=runpy.run_path(str(ROOT/'scripts/build-focus-bridge-cached-control.py'))


def test_recovery_requires_hash_and_removes_inference():
    with pytest.raises(ValueError):
        M['build']('unknown')
    n=M['build']('a'*64)
    source='\n'.join(''.join(c['source']) for c in n['cells'])
    assert 'Prediction completed in' not in source
    assert 'SUBMISSION_PATH = _resume_candidates[0]' in source
    assert 'def deepcenter_score_point(' in source
    assert 'official.evaluate(graph, ds.tracks' in source
    assert 'with SUBMISSION_PATH.open("w"' not in source
    assert source.index('sys.path.insert(0, str(_official_source_root))') < source.index('import biohub_tracking.metrics as official')
    assert 'return [Path(_deepcenter_materialized_path)]' in source
    assert "rglob('control_validation.csv')" not in source
    assert "rglob('focus3d_bridge_proposals_terminal.json')" not in source
