"""Real-source gradient/optimizer equivalence; ephemeral smoke weights only."""
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.trajectory_event_fast_case_store_v1 import CaseStore
from research.trajectory_event_fast_training_v1 import hinge as reference
from research.trajectory_event_vectorized_training_v1 import hinge
from research.trajectory_event_adam_state_v1 import initialize,update


def main():
    reports=ROOT/'reports/experiments';receipt=reports/'trajectory-event-vectorized-training-v1.json'
    assert not receipt.exists(),'Preserve previous benchmark'
    evaluator=runpy.run_path(str(ROOT/'scripts/evaluate-trajectory-event-fork16-held-v1.py'))
    contracts=evaluator['fixed_fit_contracts'](ROOT);started=time.monotonic();rows=[]
    for embryo,hard_stem,hard_t in (('44b6','44b6_eb2880fc',73),('6bba','6bba_48816121',17)):
        contract=json.loads((ROOT/'.biohub/cache'/('trajectory-event-fork16-fit-'+embryo+'-v1')/'CONTRACT.json').read_text())
        records=contract['case_records'];hard=next(r for r in records if r['stem']==hard_stem and r['t']==hard_t)
        selected=[records[0],hard];store=CaseStore(ROOT/'.biohub/cache',selected,cache_size=2,max_cache_bytes=128*1024**2)
        cases=[store[0],store[1]]
        anchor=np.asarray(contract['anchor']);penalty=np.asarray(contract['regularization'])
        old,new=initialize(anchor),initialize(anchor)
        for step,index in enumerate((0,1,0,1),1):
            case=cases[index]
            begin=time.perf_counter();loss_a,ga=reference(case,old['weights'],time_limit=10);old_seconds=time.perf_counter()-begin
            begin=time.perf_counter();loss_b,gb=hinge(case,new['weights'],time_limit=10);new_seconds=time.perf_counter()-begin
            assert loss_a==loss_b and np.array_equal(ga,gb),'Source hinge/gradient changed'
            update(old,ga+penalty*(old['weights']-anchor),.03)
            update(new,gb+penalty*(new['weights']-anchor),.03)
            assert all(np.array_equal(old[key],new[key]) for key in old),'Optimizer state changed'
            rows.append(dict(embryo=embryo,step=step,stem=selected[index]['stem'],t=selected[index]['t'],
                case_sha256=selected[index]['sha256'],options=len(case['problem']['options']),
                loss=loss_a,reference_seconds=old_seconds,vectorized_seconds=new_seconds,
                gradient_exact=True,weights_moments_and_step_exact=True))
    assert evaluator['fixed_fit_contracts'](ROOT)==contracts
    before=sum(r['reference_seconds'] for r in rows);after=sum(r['vectorized_seconds'] for r in rows)
    result=dict(status='real_source_gradient_optimizer_replay_exact',rows=rows,comparisons=len(rows),
        reference_seconds=before,vectorized_seconds=after,measured_hinge_speedup=before/after,
        seconds=time.monotonic()-started,fit_contracts=contracts,
        helper_sha256={n:sha(ROOT/'research'/n) for n in ('trajectory_event_vectorized_training_v1.py','trajectory_event_vectorized_inference_v1.py',
            'trajectory_event_vectorized_dominance_v1.py','trajectory_event_fast_training_v1.py','trajectory_event_adam_state_v1.py')},
        source_sha256=sha(Path(__file__)),source_training_labels_loaded=True,
        selection_or_validation_opened=False,intermediate_live_checkpoints_loaded=False,
        live_fits_changed=False,weights_exported=False,quality_gain_established=False)
    receipt.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
