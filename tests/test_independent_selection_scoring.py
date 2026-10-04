import copy
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
M = runpy.run_path(str(ROOT/'scripts/score-independent-selection.py'))


def fixture():
    stems = ['6bba_'+str(i) for i in range(8)]
    codex = dict(run_id='selection',checkpoint_sha256='1'*64)
    terminal = dict(status='completed',run_id='selection',elapsed_seconds=100)
    manifest = dict(status='completed',run_id='selection',checkpoint_sha256='1'*64,
        target_audit_opened=False,ground_truth_opened=False,authorized_for_submission=False,
        records=[dict(stem=s,image_shape=[5,10,20,20],processed_frames=5) for s in stems])
    return manifest,terminal,codex,dict(folds=[dict(selection=stems)])


def test_complete_selection_manifest():
    M['verify_manifest'](*fixture())


def test_manifest_identity_binds_to_verified_emitter_not_receipt():
    manifest,terminal,codex,split = fixture()
    manifest['run_id'] = 'legacy-runner'
    source = "def main():\n    result = dict(status='completed',run_id='legacy-runner')\n"
    expected = M['emitted_manifest_run_id'](source)
    with pytest.raises(ValueError,match='Incomplete selection manifest'):
        M['verify_manifest'](manifest,terminal,codex,split)
    M['verify_manifest'](manifest,terminal,codex,split,expected_manifest_run_id=expected)
    manifest['run_id'] = 'forged'
    with pytest.raises(ValueError,match='Incomplete selection manifest'):
        M['verify_manifest'](manifest,terminal,codex,split,expected_manifest_run_id=expected)
    manifest['run_id'] = expected
    terminal['run_id'] = expected
    with pytest.raises(ValueError,match='terminal'):
        M['verify_manifest'](manifest,terminal,codex,split,expected_manifest_run_id=expected)


@pytest.mark.parametrize('source',[
    'result = dict(status="completed")',
    'result = dict(run_id=dynamic_value)',
    'result = dict(run_id="one")\nresult = dict(run_id="two")',
])
def test_ambiguous_or_dynamic_emitter_rejected(source):
    with pytest.raises(ValueError):
        M['emitted_manifest_run_id'](source)


def test_pinned_scorer_accepts_only_line_ending_transport_change(tmp_path):
    source = ROOT/'.biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot'
    M['verify_scorer_sources'](source)
    for name in ('metrics.py','division_metrics.py'):
        (tmp_path/name).write_bytes((source/name).read_bytes().replace(b'\r\n',b'\n'))
    M['verify_scorer_sources'](tmp_path)
    (tmp_path/'metrics.py').write_bytes((tmp_path/'metrics.py').read_bytes()+b'\n# changed\n')
    with pytest.raises(ValueError,match='hash mismatch'):
        M['verify_scorer_sources'](tmp_path)


@pytest.mark.parametrize('fault',['missing','partial','hash','audit','timeout'])
def test_invalid_selection_rejected(fault):
    manifest,terminal,codex,split = copy.deepcopy(fixture())
    if fault == 'missing': manifest['records'].pop()
    if fault == 'partial': manifest['records'][0]['processed_frames'] = 4
    if fault == 'hash': manifest['checkpoint_sha256'] = '2'*64
    if fault == 'audit': manifest['target_audit_opened'] = True
    if fault == 'timeout': terminal['elapsed_seconds'] = 3601
    with pytest.raises(ValueError):
        M['verify_manifest'](manifest,terminal,codex,split)
