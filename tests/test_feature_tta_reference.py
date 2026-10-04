import ast
import copy
import json
from pathlib import Path
import runpy
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
from feature_tta_reference import same_coordinates, load_reference


def test_exact_nodes_allow_order_but_not_changes_duplicates_or_missing():
    a = [[0,1,2,3],[1,2,3,4]]
    assert same_coordinates(a,a[::-1])
    assert not same_coordinates(a,[[0,1,2,3],[1,2,3,4.00001]])
    assert not same_coordinates(a,[a[0],a[0]])
    assert not same_coordinates(a,a[:1])


def test_reference_must_be_complete_native_same_checkpoint_and_split(tmp_path):
    (tmp_path/'outputs').mkdir()
    manifest = dict(status='completed',checkpoint_sha256='a'*64,manifest_sha256='b'*64,
        records=[dict(stem='movie',processed_frames=100,image_shape=[100,2,3,4],graph_sha256='c'*64)],
        target_audit_opened=False,ground_truth_opened=False,authorized_for_submission=False)
    path = tmp_path/'outputs/selection_manifest.json'
    path.write_text(json.dumps(manifest))
    (tmp_path/'launcher_terminal.json').write_text(json.dumps(dict(status='completed')))
    assert load_reference(tmp_path,'a'*64,'b'*64,['movie'])['records']['movie']['processed_frames'] == 100
    for key,value in [('checkpoint_sha256','d'*64),('manifest_sha256','d'*64),('edge_feature_tta',{'views':8})]:
        bad = copy.deepcopy(manifest); bad[key] = value
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_reference(tmp_path,'a'*64,'b'*64,['movie'])


def test_feature_selection_and_scoring_keep_exact_node_reference(monkeypatch):
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-edge-feature-tta-selection.py'))['build']()
    assert meta['kernel_sources'] == ['indarkarhana/biohub-independent-joint-broad-v1/1',
                                      'indarkarhana/biohub-independent-joint-selection-v1/1']
    assert nb['metadata']['codex']['edge_feature_tta']
    source = ''.join(nb['cells'][-1]['source'])
    assert "'--edge-feature-tta','--node-reference'" in source
    assert "'independent_joint_selection'" in source
    frozen = json.dumps(nb)
    path = ROOT/'kaggle/biohub-edge-feature-tta-selection-v1/biohub-edge-feature-tta-selection-v1.ipynb'
    original = Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda self,*a,**kw:frozen if self == path else original(self,*a,**kw))
    scored,meta = runpy.run_path(str(ROOT/'scripts/build-edge-feature-tta-scoring.py'))['build']()
    assert not meta['enable_gpu']
    source = ''.join(scored['cells'][-1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'scoring_sources')
    assert ast.literal_eval(assignment.value)['selection_launch.ipynb'] == frozen
