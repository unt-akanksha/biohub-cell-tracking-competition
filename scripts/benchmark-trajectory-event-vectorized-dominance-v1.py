"""Exact pruning parity on frozen source arrays; no fits/GT/candidate changes."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.trajectory_event_dominance_v1 import allowed_options as reference
from research.trajectory_event_vectorized_dominance_v1 import allowed_options


def main():
    report=ROOT/'reports/experiments/trajectory-event-vectorized-dominance-v1.json'
    assert not report.exists(),'Preserve prior benchmark'
    evaluator=runpy.run_path(str(ROOT/'scripts/evaluate-trajectory-event-fork16-held-v1.py'))
    contracts=evaluator['fixed_fit_contracts'](ROOT)
    rng=np.random.default_rng(20260914);rows=[];started=time.monotonic()
    for embryo in ('44b6','6bba'):
        contract=json.loads((ROOT/'.biohub/cache'/('trajectory-event-fork16-fit-'+embryo+'-v1')/'CONTRACT.json').read_text())
        records=contract['case_records']
        # Uniform order coverage plus the largest vocabulary, no score selection.
        indices=sorted(set(np.linspace(0,len(records)-1,4,dtype=int).tolist()+[max(range(len(records)),key=lambda i:records[i]['options'])]))
        for index in indices:
            record=records[index];path=ROOT/'.biohub/cache'/record['path']
            assert sha(path)==record['sha256']
            with np.load(path,allow_pickle=False) as data:
                # Do not load targets, safe masks, margins, or ground-truth labels.
                case=dict(options=data['options'],nparents=int(data['nparents']),nchildren=int(data['nchildren']))
                features=data['features']
            assert features.shape==(len(case['options']),30)
            tests=(('anchor',np.asarray(contract['anchor'])),('zero',np.zeros(30)),('random',rng.normal(size=30)))
            for model,weights in tests:
                scores=features.astype(np.float64)@weights
                for mask_name,mask in (('all',None),('random_partial',rng.random(len(scores))>.2)):
                    begin=time.perf_counter();old,old_report=reference(case,scores,mask);old_seconds=time.perf_counter()-begin
                    begin=time.perf_counter();new,new_report=allowed_options(case,scores,mask);new_seconds=time.perf_counter()-begin
                    assert np.array_equal(old,new) and old_report==new_report,(embryo,index,model,mask_name)
                    rows.append(dict(embryo=embryo,case_index=index,stem=record['stem'],t=record['t'],
                        case_sha256=record['sha256'],options=len(scores),model=model,mask=mask_name,
                        reference_seconds=old_seconds,vectorized_seconds=new_seconds,
                        exact_mask_sha256=hashlib.sha256(old.tobytes()).hexdigest(),**old_report))
    assert evaluator['fixed_fit_contracts'](ROOT)==contracts
    old=sum(r['reference_seconds'] for r in rows);new=sum(r['vectorized_seconds'] for r in rows)
    result=dict(status='source_pruning_masks_exact',cases=len(rows)//6,comparisons=len(rows),rows=rows,
        reference_seconds=old,vectorized_seconds=new,component_speedup=old/new,
        seconds=time.monotonic()-started,fit_contracts=contracts,
        helper_sha256={name:sha(ROOT/'research'/name) for name in ('trajectory_event_dominance_v1.py','trajectory_event_vectorized_dominance_v1.py')},
        source_sha256=sha(Path(__file__)),source_labels_loaded=False,learned_checkpoints_loaded=False,
        live_training_changed=False,submission_changed=False,end_to_end_speedup_established=False)
    report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
