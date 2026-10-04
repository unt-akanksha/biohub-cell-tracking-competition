import ast
import copy
import json
from pathlib import Path
import runpy
import pytest
ROOT=Path(__file__).resolve().parents[1]
G=runpy.run_path(str(ROOT/'research/embryo_audit_contract.py'))


def split():
    return json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text())


def test_frozen_gain_required_and_initial_movies_not_cherry_picked():
    report=(ROOT/'reports/experiments/detector-spatial-tta-selection-v1-score.json').read_bytes()
    original=(ROOT/'research/independent_real_baseline_v1_split.json').read_bytes()
    assert len(G['verify_gate'](report,original)['stems'])==4
    for a,b in [(report+b' ',original),(report,original+b' ')]:
        with pytest.raises(ValueError): G['verify_gate'](a,b)
    altered=split(); altered['folds'][0]['initial_complete_movie_audit'].reverse()
    with pytest.raises(ValueError): G['contract'](altered)


def test_audit_scoring_requires_registered_scope_and_complete_coverage():
    module=runpy.run_path(str(ROOT/'tests/test_independent_selection_scoring.py'))
    manifest,terminal,codex,_=module['fixture']()
    frozen=split(); contract=G['contract'](frozen)
    policy=dict(neural_weight=0.,spatial_weight=1.,null_logit=-4.5,
        flow_checkpoint_sha256='3006ee0f904640b16dd988d6404a1b3ac4f933ca68ca912d69fe4598f96d4788')
    codex.update(checkpoint_sha256=G['CHECKPOINT_SHA'],embryo_audit=contract,target_audit_opened=True,
        detector_spatial_tta=True,standalone_image_flow=policy)
    manifest.update(checkpoint_sha256=G['CHECKPOINT_SHA'],embryo_audit=contract,target_audit_opened=True,
        standalone_image_flow=policy,records=[dict(stem=s,image_shape=[100,64,256,256],processed_frames=100) for s in contract['stems']],
        detector_spatial_tta=dict(version=1,views=8,encode_calls=396,features='native unchanged',
            logits='inverse-aligned XY D4 arithmetic mean',maximum_mean_absolute_logit_delta=.1))
    module['M']['verify_manifest'](manifest,terminal,codex,frozen)
    for fault in ('undeclared','missing','receipt','checkpoint'):
        bad,meta=copy.deepcopy(manifest),copy.deepcopy(codex)
        if fault=='undeclared': meta.pop('embryo_audit')
        if fault=='missing': bad['records'].pop()
        if fault=='receipt': bad['embryo_audit']['selection_report_sha256']='f'*64
        if fault=='checkpoint': bad['checkpoint_sha256']='f'*64
        with pytest.raises(ValueError): module['M']['verify_manifest'](bad,terminal,meta,frozen)


def test_audit_builder_embeds_frozen_report_without_mutating_selection():
    selection=ROOT/'kaggle/biohub-detector-spatial-tta-selection-v1/biohub-detector-spatial-tta-selection-v1.ipynb'
    before=selection.read_bytes()
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-detector-spatial-tta-audit.py'))['build']()
    assert selection.read_bytes()==before
    assert meta['enable_internet'] is False and nb['metadata']['codex']['target_audit_opened'] is True
    assert nb['metadata']['codex']['authorized_for_submission'] is False
    assert '--embryo-audit' in ''.join(nb['cells'][-1]['source'])
    assignment=next(n for n in ast.parse(''.join(nb['cells'][1]['source'])).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(assignment.value)
    assert G['verify_gate'](runtime['frozen_selection_report.json'].encode(),runtime['split.json'].encode())==nb['metadata']['codex']['embryo_audit']
    for cell in nb['cells']: ast.parse(''.join(cell['source']))


def test_audit_scorer_binds_actual_kaggle_slug_and_immutable_notebook():
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-detector-spatial-tta-audit-scoring.py'))['build']()
    assert meta['kernel_sources']==['indarkarhana/biohub-detector-spatial-tta-embryo-audit-v1/1']
    assert not meta['enable_gpu'] and not meta['enable_internet']
    source=''.join(nb['cells'][-1]['source'])
    assignment=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='scoring_sources')
    bundle=ast.literal_eval(assignment.value)
    assert bundle['selection_launch.ipynb']==(ROOT/'kaggle/biohub-detector-spatial-tta-audit-v1/biohub-detector-spatial-tta-audit-v1.ipynb').read_text()
    assert 'research/embryo_audit_contract.py' in bundle
    assert "'biohub-detector-spatial-tta-embryo-audit-v1'" in source


def test_audit_summary_cannot_claim_full_target_or_submission():
    contract=G['contract'](split())
    rows=[dict(stem=s,adj_edge_jaccard=.6,edge_tp=10,edge_fp=2,edge_fn=3,division_tp=0,
        division_fp=1,division_fn=1,num_pred_nodes=200) for s in contract['stems']]
    shared=dict(embryo_audit=contract,checkpoint_sha256=G['CHECKPOINT_SHA'],target_audit_opened=True,authorized_for_submission=False)
    result=dict(**shared,status='scored_complete_embryo_audit',per_movie=rows,summary=dict(n=4),by_embryo={'44b6':{}})
    manifest=dict(**shared,status='completed',ground_truth_opened=False,
        records=[dict(stem=s,image_shape=[100,64,256,256],processed_frames=100) for s in contract['stems']])
    m=runpy.run_path(str(ROOT/'scripts/summarize-detector-spatial-tta-audit.py'))
    report=m['summarize'](result,manifest,split())
    assert report['unaudited_target_movies']==65 and not report['authorized_for_submission']
    result['target_audit_opened']=False
    with pytest.raises(ValueError): m['summarize'](result,manifest,split())
