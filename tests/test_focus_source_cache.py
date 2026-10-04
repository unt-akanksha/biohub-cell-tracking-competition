import ast
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
M = runpy.run_path(str(ROOT / 'scripts/build-focus-source-cache.py'))


def inputs():
    probe = dict(status='verified_six_frame_focus_source_probe', cpu_exact_replay=True,
        terminal=dict(status='completed', model_sha256=M['M']['AUDIT']['EXPECTED_SHA']),
        notebook_sha256=M['SMOKE_SHA'], authorized_for_submission=False,
        records=[dict(stem=s, sha256='0'*64) for s in M['M']['STEMS']])
    return probe, (ROOT / 'research/independent_real_baseline_v1_split.json').read_bytes()


def test_full_cache_remains_source_only_and_replays_smoke():
    nb, meta = M['build'](*inputs())
    text = '\n'.join(''.join(c['source']) for c in nb['cells'])
    assert '/test/' not in text and '44b6_' not in text
    assert 'np.array_equal(actual, original' in text
    assert 'limit = 3 if' in text and 'else 100' in text
    assert 'Incomplete source movie' in text
    assert "glob('**/raw_detections/" not in text
    assert "'notebooks/indarkarhana/' + _reference_slug" in text
    assert len(nb['metadata']['codex']['source_stems']) == 8
    assert meta['kernel_sources'] == ['indarkarhana/biohub-focus-source-probe-v1/1']
    assert meta['enable_gpu'] and not meta['enable_internet']
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            ast.parse(''.join(c['source']))


def test_rejects_failed_probe_and_split_drift():
    probe, split = inputs()
    probe['cpu_exact_replay'] = False
    with pytest.raises(ValueError, match='smoke'):
        M['build'](probe, split)
    probe, split = inputs()
    with pytest.raises(ValueError, match='split'):
        M['build'](probe, split + b' ')


def test_reference_resolution_never_recurses_into_competition(tmp_path, monkeypatch):
    nb, _ = M['build'](*inputs())
    source = ''.join(nb['cells'][-1]['source'])
    source = source[source.index('_reference_slug ='):source.index('records = []')]
    source = source.replace("Path('/kaggle/input')", 'input_root')
    reference = tmp_path / 'notebooks/indarkarhana/biohub-focus-source-probe-v1/raw_detections'
    reference.mkdir(parents=True)
    def forbidden(*args, **kwargs):
        raise AssertionError('No filesystem traversal allowed for reference lookup')
    monkeypatch.setattr(Path, 'glob', forbidden)
    monkeypatch.setattr(Path, 'rglob', forbidden)
    namespace = dict(Path=Path, input_root=tmp_path)
    exec(compile(source, 'reference_lookup', 'exec'), namespace)
    assert namespace['_reference_root'] == reference.resolve()


def test_missing_explicit_reference_fails_without_fallback(tmp_path):
    nb, _ = M['build'](*inputs())
    source = ''.join(nb['cells'][-1]['source'])
    source = source[source.index('_reference_slug ='):source.index('records = []')]
    source = source.replace("Path('/kaggle/input')", 'input_root')
    with pytest.raises(RuntimeError, match='explicit'):
        exec(compile(source, 'reference_lookup', 'exec'), dict(Path=Path, input_root=tmp_path))
