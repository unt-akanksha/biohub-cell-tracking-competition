from pathlib import Path
import runpy
import json

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
M = runpy.run_path(str(ROOT / 'scripts/score-focus-raw-linker.py'))


def test_canonical_comparison_preserves_duplicate_multiplicity():
    rows = [[1, 2, 3, 4], [0, 1, 2, 3], [1, 2, 3, 4]]
    np.testing.assert_array_equal(M['canonical'](rows), M['canonical'](rows[::-1]))
    assert len(M['canonical'](rows)) == 3
    assert not np.array_equal(M['canonical'](rows), M['canonical'](rows[:2]))


def test_missing_predictions_fail_before_truth_access(tmp_path):
    with pytest.raises(FileNotFoundError, match='focus3d_raw_detections_terminal'):
        M['score'](tmp_path, tmp_path, tmp_path, tmp_path / 'truth')


@pytest.mark.parametrize('shift', [0., .5])
def test_real_geff_verification_checks_raw_coordinates(tmp_path, shift):
    pytest.importorskip('tracksdata')
    fixtures = runpy.run_path(str(ROOT / 'tests/test_verify_focus_raw_detections.py'))
    raw_root = tmp_path / 'raw'
    raw_root.mkdir()
    raw = fixtures['write_fixture'](raw_root)
    output_root = tmp_path / 'output'
    graph_root = output_root / 'predictions'
    graph_root.mkdir(parents=True)
    hashes = {}
    for record in raw['raw_detections']:
        graph = M['REPLAY']['prediction_graph']({
            'nodes': {'0': {'t': 0, 'z': 1., 'y': 2., 'x': 3. + shift}}, 'edges': []})
        path = graph_root / (record['stem'] + '.geff')
        graph.to_geff(path)
        hashes[record['stem']] = M['tree_hash'](path)
    (output_root / 'launcher_terminal.json').write_text(json.dumps({
        'status': 'completed', 'run_id': 'focus-raw-learned-linker-v1'}))
    (output_root / 'focus_raw_linked_graphs.json').write_text(json.dumps({
        'ground_truth_opened': False, 'authorized_for_submission': False,
        'graph_root': '/kaggle/working/predictions', 'graph_sha256': hashes}))
    if shift:
        with pytest.raises(ValueError, match='moved raw detections'):
            M['validate'](output_root, raw_root)
    else:
        prepared, _ = M['validate'](output_root, raw_root)
        assert set(prepared) == set(M['RAW']['STEMS'])
