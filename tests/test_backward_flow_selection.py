import ast
import copy
import json
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD = runpy.run_path(str(ROOT/'scripts/build-backward-flow-selection.py'))['build']
VALIDATE = runpy.run_path(str(ROOT/'scripts/score-backward-flow-selection.py'))['validate_manifest']


def test_selection_builder_pins_checkpoint_and_exact_node_reference():
    report = dict(status='verified_full_motion_fit_not_tracking_validation',checkpoint_sha256='a'*64,
                  small_fit_gate_passed=True,probe_inputs_replayed=True,steps=1000)
    nb,meta = BUILD(report)
    assert meta['enable_gpu'] and not meta['enable_internet']
    assert meta['kernel_sources'] == ['indarkarhana/biohub-backward-flow-fit-v1/1',
                                      'indarkarhana/biohub-independent-joint-selection-v1/1']
    launch = ''.join(nb['cells'][-1]['source'])
    assert '--steps' not in launch and 'a'*64 in launch and "'--reference'" in launch
    assignment = next(n for n in ast.parse(''.join(nb['cells'][1]['source'])).body
        if isinstance(n,ast.Assign) and n.targets[0].id == 'runtime_sources')
    worker = ast.literal_eval(assignment.value)['run_pilot.py']
    assert 'require_tracks=False' in worker and 'No ground truth may be loaded' in worker
    assert 'no clamping or deletion' in worker and 'pairs != shape[0]-1' in worker
    report['small_fit_gate_passed'] = False
    with pytest.raises(ValueError):
        BUILD(report)


def fixture():
    reference = dict(checkpoint_sha256='b'*64,manifest_sha256='c'*64,
        records=[dict(stem=f'movie{i}',image_shape=[100,64,256,256],processed_frames=100,
                      predicted_nodes=1000,graph_sha256=str(i)*64) for i in range(8)])
    records = [dict(stem=p['stem'],image_shape=p['image_shape'],processed_frames=100,
                    processed_pairs=99,predicted_nodes=1000,all_nodes_covered=True,native_graph_sha256=p['graph_sha256'])
               for p in reference['records']]
    manifest = dict(status='completed',run_id='backward-flow-selection-v1',checkpoint_sha256='a'*64,
        detector_checkpoint_sha256='b'*64,split_sha256='c'*64,reference_manifest_sha256='d'*64,
        records=records,ground_truth_opened=False,target_audit_opened=False,authorized_for_submission=False)
    terminal = dict(status='completed',run_id='backward-flow-selection-v1',declared_budget_seconds=3600,
                    elapsed_seconds=200.,submission_performed=False)
    return manifest,terminal,dict(checkpoint_sha256='a'*64),reference,'d'*64


def test_exact_complete_motion_manifest():
    VALIDATE(*fixture())


@pytest.mark.parametrize('field,value',[('processed_pairs',98),('processed_frames',99),
    ('predicted_nodes',999),('all_nodes_covered',False),('native_graph_sha256','wrong')])
def test_incomplete_or_changed_nodes_rejected(field,value):
    args = fixture()
    args[0]['records'][0][field] = value
    with pytest.raises(ValueError):
        VALIDATE(*args)


def test_ground_truth_access_and_runtime_overrun_rejected():
    args = fixture()
    args[0]['ground_truth_opened'] = True
    with pytest.raises(ValueError):
        VALIDATE(*args)
    args = fixture()
    args[1]['elapsed_seconds'] = 3601
    with pytest.raises(ValueError):
        VALIDATE(*args)


def test_cpu_scorer_embeds_exact_both_launch_notebooks():
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-backward-flow-scoring.py'))['build']()
    assert meta['enable_gpu'] is False and meta['enable_internet'] is False
    assert meta['kernel_sources'] == ['indarkarhana/biohub-independent-joint-selection-v1/1',
                                      'indarkarhana/biohub-backward-flow-selection-v1/1']
    source = ''.join(nb['cells'][-1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'scoring_sources')
    bundle = ast.literal_eval(assignment.value)
    for name,slug in [('flow_launch.ipynb','biohub-backward-flow-selection-v1'),
                      ('selection_launch.ipynb','biohub-independent-joint-selection-v1')]:
        assert bundle[name] == (ROOT/'kaggle'/slug/(slug+'.ipynb')).read_text(encoding='utf-8')
    assert "truth,flow_root,scoring/'flow_launch.ipynb'" in source
    assert 'tests/test_backward_flow_linking.py' in source
