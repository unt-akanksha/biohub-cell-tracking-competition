"""Diagnose local replay deadlines on a frozen public graph; never change GPU code."""
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    started=time.monotonic();stem='6bba_05db0fb1';name='trajectory-event-public-replay-budget-v1'
    target=ROOT/'.biohub/cache'/name;receipt=ROOT/'reports/experiments'/(name+'.json')
    assert not target.exists() and not receipt.exists()
    outputs=ROOT/'.biohub/cache/trajectory-event-anchor-production-v1-output'
    terminal=read(outputs/'trajectory-complete/result.json');harvest=read(ROOT/'reports/experiments/trajectory-event-anchor-production-v1-harvest.json')
    shards=[s for s,w in terminal['workers'].items() if stem in w['movies']];assert len(shards)==1
    folder=outputs/'trajectory-complete'/('shard-'+shards[0])/(stem+'-original')
    for name0 in ('pre-postprocess.json','structured-repaired-prediction.json','repaired-prediction.json','raw-candidates.npz','repair-details.json'):
        path=folder/name0;assert sha(path)==harvest['files'][path.relative_to(outputs).as_posix()]['sha256']
    bundle=ROOT/'.biohub/cache/trajectory-event-anchor-portable-v1';proof=read(bundle/'RESULT.json')
    code=bundle/'event-trajectory.py';assert sha(code)==proof['code_sha256']
    model=bundle/'event-trajectory-model.json';assert sha(model)==proof['model_sha256']
    spec=importlib.util.spec_from_file_location('budget_replay_event',code);module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module;spec.loader.exec_module(module)
    initial=read(folder/'pre-postprocess.json');baseline=read(folder/'structured-repaired-prediction.json');expected=read(folder/'repaired-prediction.json')
    with np.load(folder/'raw-candidates.npz',allow_pickle=False) as raw:
        groups=module.candidates(initial,baseline);matrix=module.edge_features(initial,baseline,groups,raw['edges'])
    target.mkdir();records={}
    report=dict(status='running',stem=stem,production_candidate_changed=False,annotations_opened=False,
        actual_kaggle_details=read(folder/'repair-details.json')['event_assignment'],
        model_sha256=sha(model),portable_code_sha256=sha(code),source_sha256=sha(Path(__file__)))
    def persist():
        report.update(records=records,seconds=time.monotonic()-started)
        text=json.dumps(report,indent=2)+'\n';(target/'RESULT.json').write_text(text,encoding='utf-8');receipt.write_text(text,encoding='utf-8')
    persist()
    for arm,per_frame,total in (('default_cpu',2.,120.),('extended_cpu_diagnostic',10.,600.)):
        graph,details=module.refine_prepared(initial,baseline,groups,matrix,read(model)['weights'],per_frame_seconds=per_frame,max_seconds=total)
        path=target/(arm+'-prediction.json');path.write_text(json.dumps(graph,sort_keys=True)+'\n',encoding='utf-8')
        pairs=lambda g:{(e['source_id'],e['target_id']) for e in g['edges']}
        records[arm]=dict(details,prediction_sha256=sha(path),exact_kaggle_graph=graph==expected,
            symmetric_edge_difference=len(pairs(graph)^pairs(expected)),cpu_per_frame_limit=per_frame,cpu_stage_limit=total)
        persist();print(json.dumps(dict(arm=arm,**{k:v for k,v in records[arm].items() if k!='frames'})),flush=True)
    report['status']='local_replay_budget_diagnosed';persist()


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
