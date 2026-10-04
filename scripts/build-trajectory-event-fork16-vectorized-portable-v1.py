"""Offline fork16 inference code, tested on frozen untrained controls only."""
import argparse
import ast
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha,validate_graph
from research.trajectory_event_release_v1 import check_event_stage


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path,allow_pickle=False) as data:return dict(data)


def portable_source():
    original=ROOT/'.biohub/cache/trajectory-event-anchor-portable-v1/event-trajectory.py'
    accelerated=ROOT/'research/trajectory_event_vectorized_dominance_v1.py'
    assert sha(original)=='c7f98a711adb4d3247b12297cde0b31a246043e906d8ea462a3e5f763f03d954'
    assert sha(accelerated)=='23db55460d7e7cd079d8b03e36c9219c64ab4e4c348c01ee30f1194b3d9da0f8'
    tree=ast.parse(original.read_text(encoding='utf-8'))
    function=next(n for n in ast.parse(accelerated.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='allowed_options')
    found_pruning=0;found_frames=0
    for index,node in enumerate(tree.body):
        if isinstance(node,ast.FunctionDef) and node.name=='allowed_options':
            node.name='reference';tree.body.insert(index+1,function);found_pruning+=1;break
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='frames':
            assert not node.keywords;node.keywords=[ast.keyword(arg='max_fork_children',value=ast.Constant(value=16))];found_frames+=1
    assert found_pruning==found_frames==1
    assert not any(isinstance(n,ast.ImportFrom) and (n.module or '').startswith('research') for n in ast.walk(tree))
    assert not any(isinstance(n,ast.FunctionDef) and n.name in ('fit','prepare','train') for n in ast.walk(tree))
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    started=time.monotonic();name='trajectory-event-fork16-vectorized-portable-v1'+('-smoke' if args.smoke else '')
    target=ROOT/'.biohub/cache'/name;receipt=ROOT/'reports/experiments'/(name+'.json')
    assert not target.exists() and not receipt.exists(),'Preserve existing build/replay'
    evaluator=runpy.run_path(str(ROOT/'scripts/evaluate-trajectory-event-fork16-held-v1.py'))
    contracts=evaluator['fixed_fit_contracts'](ROOT)
    code=portable_source();proofs={};tests=[]
    for held in ('44b6','6bba'):
        reference_name='trajectory-event-fork16-held-'+held+'-v1-control-preflight'
        reference=read(ROOT/'reports/experiments'/(reference_name+'.json'))
        replay_path=ROOT/'reports/experiments'/('trajectory-event-vectorized-replay-'+held+'-v1.json')
        replay=read(replay_path)
        assert reference['status']=='untrained_control_preflight_passed' and replay['status']=='complete_control_graph_replay_exact'
        assert replay['source_preflight_sha256']==sha(ROOT/'reports/experiments'/(reference_name+'.json'))
        assert replay['fit_contracts']==contracts and not reference['ground_truth_opened'] and not replay['ground_truth_opened']
        proofs[held]=dict(control_sha256=replay['source_preflight_sha256'],vectorized_replay_sha256=sha(replay_path))
        for stem in (reference['movies'][:1] if args.smoke else reference['movies']):
            tests.append((reference_name,reference,stem))
    if not args.smoke:
        smoke=read(ROOT/'reports/experiments/trajectory-event-fork16-vectorized-portable-v1-smoke.json')
        assert smoke['status']=='portable_source_smoke_passed' and smoke['proofs']==proofs and smoke['source_sha256']==sha(Path(__file__))
    target.mkdir();module=target/'event-trajectory.py';module.write_bytes(code.encode('utf-8'))
    if not args.smoke:assert sha(module)==smoke['code_sha256']
    runtime=runpy.run_path(str(module))
    report=dict(status='replaying_frozen_controls',code_sha256=sha(module),source_sha256=sha(Path(__file__)),
        proofs=proofs,fit_contracts=contracts,records=[],smoke=args.smoke,max_fork_children=16,
        learned_weights_included=False,training_data_in_bundle=False,annotations_in_bundle=False,
        ground_truth_opened=False,gpu_used=False,authorized_for_submission=False,kaggle_runtime_acceptance=False)
    def persist():
        report['seconds']=time.monotonic()-started
        text=json.dumps(report,indent=2,allow_nan=False)+'\n'
        receipt.write_text(text,encoding='utf-8');(target/'RESULT.json').write_text(text,encoding='utf-8')
    persist()
    try:
        for reference_name,reference,stem in tests:
            contract=read(ROOT/'.biohub/cache'/('trajectory-event-fork16-fit-'+reference['fit_embryo']+'-v1')/'CONTRACT.json')
            weights=np.asarray(contract['anchor'])
            located=[]
            for batch in range(8):
                prefix='trajectory-event-source-v1-b'+str(batch);folder=ROOT/'.biohub/cache'/(prefix+'-features')
                frozen=read(folder/'RESULT.json')
                if stem not in frozen['per_movie']:continue
                assert sha(folder/'RESULT.json')==reference['source_scope_receipts'][str(batch)]['feature_receipt_sha256']
                backup_path=ROOT/'reports/experiments'/(prefix+'-full-harvest.json')
                assert sha(backup_path)==reference['source_scope_receipts'][str(batch)]['backup_sha256']
                backup={r['path']:r['sha256'] for r in read(backup_path)['records']}
                base_path=folder/(stem+'-prediction.json');assert sha(base_path)==frozen['per_movie'][stem]['prediction_sha256']
                origin=ROOT/'.biohub/cache'/(prefix+'-full-output')/(stem+'-original')
                for filename in ('pre-postprocess.json','raw-candidates.npz'):
                    assert sha(origin/filename)==backup[stem+'-original/'+filename]
                feature_path=folder/(stem+'-features.npz');group_path=folder/(stem+'-groups.npz')
                assert sha(feature_path)==frozen['per_movie'][stem]['features_sha256']
                assert sha(group_path)==frozen['per_movie'][stem]['groups_sha256']
                located.append((read(origin/'pre-postprocess.json'),read(base_path),arrays(origin/'raw-candidates.npz'),arrays(feature_path),arrays(group_path)))
            assert len(located)==1
            initial,base,raw,edge,groups=located[0]
            regenerated=runtime['candidates'](initial,base)
            assert set(regenerated)==set(groups) and all(np.array_equal(regenerated[k],groups[k]) for k in groups)
            assert np.array_equal(runtime['edge_features'](initial,base,regenerated,raw['edges']),edge['features'])
            old=reference['inference'][stem]['untrained16'];old_path=ROOT/'.biohub/cache'/reference_name/(stem+'-untrained16.json')
            assert sha(old_path)==old['prediction_sha256']
            graph,details=runtime['refine'](initial,base,raw['coords'],raw['edges'],weights,per_frame_seconds=10,max_seconds=600)
            validate_graph(graph,100);check_event_stage(initial,base,graph,details,frames=100)
            assert not details['solver_fallbacks'] and not details['budget_exhausted']
            assert graph==read(old_path)
            assert {k:v for k,v in details.items() if k!='seconds'}=={k:v for k,v in old['details'].items() if k!='seconds'}
            report['records'].append(dict(stem=stem,exact_graph=True,exact_features=True,exact_solver_records=True,
                processed_frames=details['processed_frames'],seconds=details['seconds'],reference_prediction_sha256=old['prediction_sha256']))
            persist();print(json.dumps(report['records'][-1]),flush=True)
        assert evaluator['fixed_fit_contracts'](ROOT)==contracts
        report['status']='portable_source_smoke_passed' if args.smoke else 'portable_control_replay_passed'
        persist();print(json.dumps(report),flush=True)
    except BaseException as error:
        report.update(status='failed_requires_inspection',error=repr(error));persist();raise


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
