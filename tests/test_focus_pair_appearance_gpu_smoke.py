import ast
import base64
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from research.focus_pair_appearance_cuda import CudaObjective, validate_layout
from research.focus_pair_appearance_gpu_smoke import choices, compare

ROOT = Path(__file__).resolve().parents[1]


def layout():
    x = np.zeros((8, 9), np.float64)
    x[[1, 2, 3, 5, 6], 0] = 1
    offset = np.zeros(8, np.float64)
    offset[[0, 4, 7]] = -4.5
    return [x, offset, np.array([1, 4, 3], np.int64),
            np.array([0, 2, 7], np.int64), np.array([0, 1, 0], np.int64)]


def test_variable_complete_groups_and_tie_order():
    data = layout()
    np.testing.assert_array_equal(validate_layout(*data), [0, 1, 5])
    np.testing.assert_array_equal(choices(np.zeros(8), np.array([0, 1, 5]), data[2]), [0, 0, 0])


@pytest.mark.parametrize('corruption', ['null_feature', 'null_offset', 'coverage', 'label', 'float32', 'nonfinite'])
def test_layout_rejects_corruption(corruption):
    data = layout()
    if corruption == 'null_feature':
        data[0][4, 0] = 1
    elif corruption == 'null_offset':
        data[1][4] = -4
    elif corruption == 'coverage':
        data[2][0] = 2
    elif corruption == 'label':
        data[4][0] = 1
    elif corruption == 'float32':
        data[0] = data[0].astype(np.float32)
    else:
        data[0][2, 0] = np.nan
    with pytest.raises(ValueError):
        validate_layout(*data)


def test_role_and_weight_fail_before_torch_import():
    with pytest.raises(ValueError, match='Only fitting'):
        CudaObjective(None, None, None, None, None, 1, 'diagnostic')
    with pytest.raises(ValueError, match='positive'):
        CudaObjective(*layout(), 0, 'fitting')


def test_comparison_does_not_hide_gradient_difference():
    assert compare((1., np.ones(9)), (1., np.ones(9)))['gradient_max_absolute_error'] == 0
    with pytest.raises(AssertionError):
        compare((1., np.ones(9)), (1., np.ones(9) + .001))


def test_real_builder_payload_hashes_scope_and_limits():
    builder = runpy.run_path(str(ROOT / 'scripts/build-focus-pair-appearance-gpu-smoke.py'))
    nb, meta = builder['build']()
    assert meta['is_private'] and meta['enable_gpu']
    assert not meta['enable_tpu'] and not meta['enable_internet']
    assert meta['competition_sources'] == ['biohub-cell-tracking-during-development']
    assert not meta['dataset_sources'] and not meta['kernel_sources'] and not meta['model_sources']
    assert len(json.dumps(nb).encode()) < 950000
    cells = [''.join(c['source']) for c in nb['cells']]
    trees = [ast.parse(c) for c in cells]
    values = {}
    for node in trees[1].body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('runtime_sources', 'packet_base64'):
                values[node.targets[0].id] = ast.literal_eval(node.value)
    runtime = values['runtime_sources']
    hashes = json.loads(runtime['source_hashes.json'])
    for name, value in runtime.items():
        if name != 'source_hashes.json':
            assert hashlib.sha256(value.encode()).hexdigest() == hashes[name]
    assert hashlib.sha256(base64.b64decode(values['packet_base64'])).hexdigest() == hashes['pair.npz']
    spec = json.loads(runtime['spec.json'])
    assert spec['declared_budget_seconds'] == 300
    assert spec['stem'] == '6bba_57b7cc1e' and spec['frame'] == 31
    assert 'threading.Timer(285, emergency_stop)' in cells[0]
    assert '240-(time.monotonic()-started)' in cells[2]
    assert 'submission_performed=False' in cells[2]
