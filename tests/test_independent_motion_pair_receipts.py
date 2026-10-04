import copy
import json
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
SUMMARIZE = runpy.run_path(str(ROOT/'scripts/summarize-independent-motion-pair.py'))['summarize']


def receipt(tmp_path):
    split = json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text())
    small = json.loads((ROOT/'reports/experiments/independent-motion-row-v1-summary.json').read_text())
    # Synthetic structural receipt; deliberately not represented as model evidence.
    result = dict(status='completed',detector_unchanged=True,frozen_detector_sha256='d'*64,
        checkpoint_sha256='c'*64,sample_hashes=['a'*64]*2000,
        identity=dict(training_profile='full',training_stems=split['folds'][0]['train'],max_steps=1000,
            initialization_sha256='b'*64,seed=20260909,augmentation_seed=20260909,
            optimizer={'lr':.0001},motion_residual={'version':1}),
        history=[dict(step=i,correct=1,positive_links=2,confident_correct=1,wrong_child_claims=1)
                 for i in range(10,1001,10)])
    for key,step in (('before',0),('step100_smoke',100),('after',1000)):
        result[key] = dict(status='passed',checkpoint_step=step,strict_reload=True,geff_round_trip=True,
                           checkpoint_sha256='c'*64,scorer_counts=small['both_after_smoke_counts'])
    pair = dict(status='completed',inputs_identical=True,sample_count=2000,results={})
    for arm,loss in [('control','sparse_parent_classification_with_null_v1'),
                     ('row','sparse_parent_and_row_hard_negative_v1')]:
        pair['results'][arm] = copy.deepcopy(result)
        pair['results'][arm]['identity']['loss'] = loss
    (tmp_path/'outputs').mkdir()
    (tmp_path/'launcher_terminal.json').write_text(json.dumps(dict(status='completed',
        run_id='independent-motion-broad-pair-v1',elapsed_seconds=1000,submission_performed=False)))
    return split,pair


def test_receipt_passes_only_as_training_evidence(tmp_path):
    split,pair = receipt(tmp_path)
    (tmp_path/'outputs/paired_result.json').write_text(json.dumps(pair))
    summary = SUMMARIZE(tmp_path,split)
    assert summary['authorized_for_submission'] is False
    assert summary['arms']['control']['training_totals']['correct'] == 100


@pytest.mark.parametrize('failure',['input','gate','coverage','checkpoint','initialization'])
def test_receipt_rejects_broken_pair(tmp_path,failure):
    split,pair = receipt(tmp_path)
    row = pair['results']['row']
    if failure == 'input':
        row['sample_hashes'][5] = 'e'*64
    elif failure == 'gate':
        row['step100_smoke']['status'] = 'failed'
    elif failure == 'coverage':
        row['history'].pop()
    elif failure == 'checkpoint':
        row['after']['checkpoint_sha256'] = 'e'*64
    else:
        row['identity']['seed'] += 1
    (tmp_path/'outputs/paired_result.json').write_text(json.dumps(pair))
    with pytest.raises(ValueError):
        SUMMARIZE(tmp_path,split)


@pytest.mark.parametrize('arm',['control','row'])
def test_scoring_builder_preserves_embedded_selection_notebook(monkeypatch,arm):
    selection = runpy.run_path(str(ROOT/'scripts/build-independent-motion-broad-selection.py'))['build']
    nb,selection_meta = selection('a'*64,1,arm)
    frozen = json.dumps(nb)
    selection_name = selection_meta['id'].split('/')[1]
    path = ROOT/'kaggle'/selection_name/selection_meta['code_file']
    original = Path.read_text
    def read(self,*args,**kwargs):
        return frozen if self == path else original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    scored,meta = runpy.run_path(str(ROOT/'scripts/build-independent-motion-broad-scoring.py'))['build'](arm,1)
    import ast
    source = ''.join(scored['cells'][-1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'scoring_sources')
    bundle = ast.literal_eval(assignment.value)
    assert bundle['selection_launch.ipynb'] == frozen
    assert meta['kernel_sources'] == [f'indarkarhana/{selection_name}/1']
    assert not meta['enable_gpu'] and not meta['enable_internet']
