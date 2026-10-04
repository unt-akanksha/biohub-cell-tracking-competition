from pathlib import Path
import runpy

import pytest


def test_builder_requires_hash_and_removes_base_postprocessing():
    root = Path(__file__).resolve().parents[1]
    build = runpy.run_path(str(root / 'scripts/build-focus-raw-learned-linker-v1.py'))['build']
    with pytest.raises(ValueError, match='SHA-256'):
        build('unverified')
    nb = build('a' * 64)
    code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    assert 'def filter_output_graph(' not in code
    assert 'def apply_focus_bridge(' not in code
    assert 'require_tracks=True' not in code
    assert 'Expected one hash-bound raw detection terminal' in code
    assert code.index('_focus_predictor_path.write_text') < code.index('start_time = time.time()')
    assert nb['metadata']['codex']['authorized_for_production_promotion'] is False
    assert 'graph = solve_links_keep_nodes(graph, solver)' in code
    assert "rglob('focus3d_raw_detections_terminal.json')" not in code
