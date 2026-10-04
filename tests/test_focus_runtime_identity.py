import hashlib
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = runpy.run_path(str(ROOT / 'scripts/audit-focus-runtime-identity.py'))
BUILD = runpy.run_path(str(ROOT / 'scripts/build-focus-runtime-identity.py'))


def test_hashes_bytes_without_model_loading(tmp_path):
    model = tmp_path / 'model_final_nuclei.pth'
    model.write_bytes(b'not a pickle or a model')
    (tmp_path / 'config.yaml').write_text('a: 1')
    result = AUDIT['inspect_runtime'](tmp_path, hashlib.sha256(model.read_bytes()).hexdigest(), model.stat().st_size)
    assert result['status'] == 'matched'
    assert result['weights_deserialized'] is False
    assert all('sha256' in r for r in result['files'])


def test_identity_mismatch_is_explicit(tmp_path):
    (tmp_path / 'model_final_nuclei.pth').write_bytes(b'wrong')
    assert AUDIT['inspect_runtime'](tmp_path)['status'] == 'mismatch'


def test_missing_or_duplicate_weights_rejected(tmp_path):
    with pytest.raises(ValueError, match='Exactly one'):
        AUDIT['inspect_runtime'](tmp_path)
    (tmp_path / 'model_final_nuclei.pth').touch()
    (tmp_path / 'other').mkdir()
    (tmp_path / 'other/model_final_nuclei.pth').touch()
    with pytest.raises(ValueError, match='Exactly one'):
        AUDIT['inspect_runtime'](tmp_path)


def test_offline_cpu_only_no_competition_or_predictions():
    meta = BUILD['build']()
    assert not any(meta[k] for k in ('enable_gpu', 'enable_tpu', 'enable_internet', 'competition_sources', 'kernel_sources'))
    assert meta['dataset_sources'] == ['qiweiyin/focus3d-nuclei-runtime']
