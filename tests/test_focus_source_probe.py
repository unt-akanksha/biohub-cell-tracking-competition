import ast
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
M = runpy.run_path(str(ROOT / 'scripts/build-focus-source-probe.py'))


def fake_identity():
    return dict(status='matched', model_sha256=M['AUDIT']['EXPECTED_SHA'],
                model_size=M['AUDIT']['EXPECTED_SIZE'], weights_deserialized=False, files=[])


def test_frozen_scope_and_offline_two_gpu():
    nb, meta = M['build'](fake_identity())
    text = '\n'.join(''.join(c['source']) for c in nb['cells'])
    assert '/test/' not in text
    assert 'volume = zarr_arr[:3]' in text
    assert 'zarr_arr[:]' not in text
    assert 'from biohub_tracking.metrics import' not in text
    assert 'Timer(3300,' in text
    assert meta['enable_gpu'] and not meta['enable_internet'] and not meta['enable_tpu']
    assert nb['metadata']['codex']['frozen_stems'] == ['6bba_f1fde7e0', '6bba_23af9eeb']
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            ast.parse(''.join(c['source']))


def test_rejects_unverified_identity():
    identity = fake_identity()
    identity['model_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='identity'):
        M['build'](identity)


def test_requires_exact_old_raw_replay_and_runtime_bytes():
    nb, _ = M['build'](fake_identity())
    text = '\n'.join(''.join(c['source']) for c in nb['cells'])
    assert "_identity['files'] != _expected_identity['files']" in text
    assert 'np.array_equal(actual, expected)' in text
    assert 'Exact original raw reference required' in text


def test_source_replacement_fails_closed():
    with pytest.raises(ValueError, match='layout'):
        M['change']('twice twice', 'twice', 'once')
