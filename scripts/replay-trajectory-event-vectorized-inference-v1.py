"""Replay frozen complete control movies; compare graphs, not metric scores."""
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
from research.trajectory_event_vectorized_inference_v1 import refine


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path,allow_pickle=False) as data:return dict(data)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--held-embryo',choices=('44b6','6bba'),required=True)
    args=parser.parse_args();started=time.monotonic();reports=ROOT/'reports/experiments'
    evaluator=runpy.run_path(str(ROOT/'scripts/evaluate-trajectory-event-fork16-held-v1.py'))
    contracts=evaluator['fixed_fit_contracts'](ROOT)
    name='trajectory-event-vectorized-replay-'+args.held_embryo+'-v1'
    target=ROOT/'.biohub/cache'/name;receipt=reports/(name+'.json')
    assert not target.exists() and not receipt.exists()
    source_name='trajectory-event-fork16-held-'+args.held_embryo+'-v1-control-preflight'
    source_path=reports/(source_name+'.json');source=read(source_path)
    assert source['status']=='untrained_control_preflight_passed' and not source['ground_truth_opened']
    assert source['model_sha256'] is None and source['both_frozen_fit_contracts']==contracts
    for name,digest in source['helper_sha256'].items():assert sha(ROOT/'research'/name)==digest
    benchmark=read(reports/'trajectory-event-vectorized-dominance-v1.json')
    assert benchmark['status']=='source_pruning_masks_exact'
    for name,digest in benchmark['helper_sha256'].items():assert sha(ROOT/'research'/name)==digest
    contract=read(ROOT/'.biohub/cache'/('trajectory-event-fork16-fit-'+source['fit_embryo']+'-v1')/'CONTRACT.json')
    weights=np.asarray(contract['anchor'],dtype=np.float64)
    locations={}
    for batch in range(8):
        prefix='trajectory-event-source-v1-b'+str(batch);folder=ROOT/'.biohub/cache'/(prefix+'-features')
        frozen=read(folder/'RESULT.json');assert sha(folder/'RESULT.json')==source['source_scope_receipts'][str(batch)]['feature_receipt_sha256']
        backup_path=reports/(prefix+'-full-harvest.json');assert sha(backup_path)==source['source_scope_receipts'][str(batch)]['backup_sha256']
        backup={r['path']:r['sha256'] for r in read(backup_path)['records']}
        for stem in source['movies']:
            if stem not in frozen['per_movie']:continue
            initial=ROOT/'.biohub/cache'/(prefix+'-full-output')/(stem+'-original/pre-postprocess.json')
            assert sha(initial)==backup[stem+'-original/pre-postprocess.json']
            for suffix,key in (('-prediction.json','prediction_sha256'),('-groups.npz','groups_sha256'),('-features.npz','features_sha256')):
                assert sha(folder/(stem+suffix))==frozen['per_movie'][stem][key]
            locations[stem]=(folder,initial)
    assert set(locations)==set(source['movies'])
    target.mkdir();rows=[]
    for stem in source['movies']:
        folder,initial_path=locations[stem];initial=read(initial_path);base=read(folder/(stem+'-prediction.json'))
        old=source['inference'][stem]['untrained16'];old_path=ROOT/'.biohub/cache'/source_name/(stem+'-untrained16.json')
        assert sha(old_path)==old['prediction_sha256']
        candidate,details=refine(initial,base,arrays(folder/(stem+'-groups.npz')),arrays(folder/(stem+'-features.npz'))['features'],weights,
                                 per_frame_seconds=10,max_seconds=600)
        assert not details['solver_fallbacks'] and not details['budget_exhausted']
        validate_graph(candidate,100);check_event_stage(initial,base,candidate,details,frames=100)
        assert candidate==read(old_path),'Complete graph differs from frozen reference'
        assert {k:v for k,v in details.items() if k!='seconds'}=={k:v for k,v in old['details'].items() if k!='seconds'}
        path=target/(stem+'.json');path.write_text(json.dumps(candidate,sort_keys=True,allow_nan=False),encoding='utf-8')
        assert sha(path)==old['prediction_sha256']
        rows.append(dict(stem=stem,reference_seconds=old['details']['seconds'],vectorized_seconds=details['seconds'],
                         prediction_sha256=sha(path),processed_frames=details['processed_frames']))
        print(json.dumps(rows[-1]),flush=True)
    assert evaluator['fixed_fit_contracts'](ROOT)==contracts
    result=dict(status='complete_control_graph_replay_exact',rows=rows,source_preflight_sha256=sha(source_path),
        benchmark_sha256=sha(reports/'trajectory-event-vectorized-dominance-v1.json'),fit_contracts=contracts,
        source_sha256=sha(Path(__file__)),adapter_sha256=sha(ROOT/'research/trajectory_event_vectorized_inference_v1.py'),
        seconds=time.monotonic()-started,ground_truth_opened=False,model_changed=False,live_training_changed=False,
        kaggle_runtime_acceptance=False,authorized_for_submission=False)
    receipt.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (target/'RESULT.json').write_text(receipt.read_text(encoding='utf-8'),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
