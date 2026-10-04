"""Final-weight offline replay; quality evaluation and submission remain separate."""
import argparse
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


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fit-embryo',choices=('44b6','6bba'),required=True)
    args=parser.parse_args();started=time.monotonic();reports=ROOT/'reports/experiments'
    evaluator=runpy.run_path(str(ROOT/'scripts/evaluate-trajectory-event-fork16-held-v1.py'))
    contracts=evaluator['fixed_fit_contracts'](ROOT)
    fit_name='trajectory-event-fork16-fit-'+args.fit_embryo+'-v1';fit_path=reports/(fit_name+'.json')
    fit=read(fit_path);fit_root=ROOT/'.biohub/cache'/fit_name;contract=read(fit_root/'CONTRACT.json')
    assert fit['status']=='source_event_training_complete' and fit['model_fitted'] and not fit['control_smoke']
    assert len(fit['epochs_completed'])==contract['epochs']==3 and fit['steps']==3*contract['cases']
    weights_path=fit_root/'final-weights.npz';assert sha(weights_path)==fit['weights_sha256']
    model=arrays(weights_path);assert int(model['max_fork_children'])==16
    held='6bba' if args.fit_embryo=='44b6' else '44b6'
    original_name='trajectory-event-fork16-held-'+held+'-v1-smoke'
    original_path=reports/(original_name+'.json');original=read(original_path)
    assert original['model_sha256']==fit['weights_sha256'] and not original['ground_truth_opened']
    assert 'learned16' in original['inference'][original['movies'][0]],'Original small-movie smoke must finish first'
    portable_root=ROOT/'.biohub/cache/trajectory-event-fork16-vectorized-portable-v1'
    proof_path=reports/'trajectory-event-fork16-vectorized-portable-v1.json';proof=read(proof_path)
    assert sha(proof_path)=='8bcd743a08941e7e77ab174053cdf912a15fc4fde31a8cc88cd092656b94463f'
    assert proof['status']=='portable_control_replay_passed'
    code_path=portable_root/'event-trajectory.py';assert sha(code_path)==proof['code_sha256']
    runtime=runpy.run_path(str(code_path));assert list(model['feature_names'])==list(runtime['FEATURES'])
    name='trajectory-event-fork16-learned-'+args.fit_embryo+'-portable-v1'
    target=ROOT/'.biohub/cache'/name;receipt=reports/(name+'.json')
    assert not target.exists() and not receipt.exists(),'Preserve prior learned runtime test'
    target.mkdir()
    model_path=target/'event-trajectory-model.json'
    model_path.write_text(json.dumps(dict(features=list(runtime['FEATURES']),weights=model['weights'].tolist(),max_fork_children=16,
        fit_sha256=sha(fit_path),weights_sha256=sha(weights_path),authorized_for_submission=False),indent=2),encoding='utf-8')
    result=dict(status='running_label_free_portable_smoke',fit_embryo=args.fit_embryo,held_embryo=held,
        code_sha256=sha(code_path),model_sha256=sha(model_path),weights_sha256=sha(weights_path),fit_contracts=contracts,
        original_smoke_path=str(original_path.relative_to(ROOT)),portable_control_proof_sha256=sha(proof_path),
        source_sha256=sha(Path(__file__)),ground_truth_opened=False,model_changed=False,quality_gain_established=False,
        gpu_used=False,kaggle_acceptance=False,authorized_for_submission=False,movies=original['movies'],inference={})
    def persist():
        result['seconds']=time.monotonic()-started
        text=json.dumps(result,indent=2,allow_nan=False)+'\n'
        receipt.write_text(text,encoding='utf-8');(target/'RESULT.json').write_text(text,encoding='utf-8')
    persist()
    try:
        for stem in original['movies']:
            assert stem not in contract['source_stems']
            located=[]
            for batch in range(8):
                prefix='trajectory-event-source-v1-b'+str(batch);folder=ROOT/'.biohub/cache'/(prefix+'-features')
                frozen=read(folder/'RESULT.json')
                if stem not in frozen['per_movie']:continue
                assert sha(folder/'RESULT.json')==original['source_scope_receipts'][str(batch)]['feature_receipt_sha256']
                base_path=folder/(stem+'-prediction.json');assert sha(base_path)==frozen['per_movie'][stem]['prediction_sha256']
                backup_path=reports/(prefix+'-full-harvest.json')
                assert sha(backup_path)==original['source_scope_receipts'][str(batch)]['backup_sha256']
                backup={r['path']:r['sha256'] for r in read(backup_path)['records']}
                origin=ROOT/'.biohub/cache'/(prefix+'-full-output')/(stem+'-original')
                for filename in ('pre-postprocess.json','raw-candidates.npz'):
                    assert sha(origin/filename)==backup[stem+'-original/'+filename]
                located.append((read(origin/'pre-postprocess.json'),read(base_path),arrays(origin/'raw-candidates.npz')))
            assert len(located)==1
            initial,base,raw=located[0]
            candidate,details=runtime['refine'](initial,base,raw['coords'],raw['edges'],model['weights'],per_frame_seconds=10,max_seconds=600)
            validate_graph(candidate,100);check_event_stage(initial,base,candidate,details,frames=100)
            path=target/(stem+'-prediction.json');path.write_text(json.dumps(candidate,sort_keys=True,allow_nan=False),encoding='utf-8')
            original_now=read(original_path);old=original_now['inference'].get(stem,{}).get('learned16')
            row=dict(prediction_sha256=sha(path),details=details,original_status=original_now['status'],reference_available=old is not None)
            result['inference'][stem]=row;persist() # Preserve timing/fallback evidence even if a check fails.
            assert not details['budget_exhausted'] and not details['solver_fallbacks'],'Portable inference incomplete'
            if old is not None:
                old_path=ROOT/'.biohub/cache'/original_name/(stem+'-learned16.json')
                assert sha(old_path)==old['prediction_sha256']
                assert candidate==read(old_path),'Learned portable graph differs from original complete graph'
                assert {k:v for k,v in details.items() if k!='seconds'}=={k:v for k,v in old['details'].items() if k!='seconds'}
                row['exact_original_graph_and_solver_records']=True
            elif stem==original['movies'][0]:raise AssertionError('Small learned reference disappeared')
            persist();print(json.dumps(dict(stem=stem,seconds=details['seconds'],reference_available=old is not None)),flush=True)
        assert evaluator['fixed_fit_contracts'](ROOT)==contracts
        result['status']='learned_portable_smoke_exact' if all(r.get('exact_original_graph_and_solver_records') for r in result['inference'].values()) else 'learned_portable_smoke_needs_original_comparison'
        persist();print(json.dumps(dict(status=result['status'],seconds=result['seconds'])),flush=True)
    except BaseException as error:
        result.update(status='failed_requires_inspection',error=repr(error));persist();raise


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
