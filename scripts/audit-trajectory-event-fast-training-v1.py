"""Real source replay, objective equivalence and bounded-memory training profile."""
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.trajectory_event_fast_case_store_v1 import CaseStore
from research.trajectory_event_assignment_v1 import hinge as reference_hinge
from research.trajectory_event_fast_training_v1 import hinge,fit


def main():
    started=time.monotonic();receipt=ROOT/'reports/experiments/trajectory-event-fast-training-v1-audit.json'
    assert not receipt.exists()
    root=ROOT/'.biohub/cache/trajectory-event-source-v1-b0-cases';manifest=json.loads((root/'RESULT.json').read_text())
    assert manifest['status']=='source_event_cases_prepared' and manifest['training_allowed'] and manifest['source_only']
    records=[dict(r,stem=stem) for stem,movie in manifest['per_movie'].items() if stem.startswith('6bba_')
             for r in movie['cases'] if r['reason']=='prepared']
    assert len(records)==673
    store=CaseStore(root,records,cache_size=len(records),max_cache_bytes=1536*1024**2)
    model=ROOT/'.biohub/cache/trajectory-event-learning-smoke-v1/epoch-3.npz'
    assert sha(model)=='508165d131dc0c2516cf91e0eafb5a88edf24358f6ced5b0436f29ccc6b733be'
    with np.load(model,allow_pickle=False) as data:weights=data['weights']
    pools_before=threadpool_info();tick=time.monotonic()
    for index in range(len(store)):
        case=store[index]
        assert case['x'].shape[1]==30 and case['n_constraints']==records[index]['n_constraints']
        if (index+1)%100==0:print(json.dumps(dict(event='source_cases_reconstructed',cases=index+1,cache_bytes=store.cache_bytes)),flush=True)
    preparation_seconds=time.monotonic()-tick
    selected=sorted(range(len(records)),key=lambda i:(-records[i]['options'],records[i]['path']))[:4]
    selected+= [next(i for i,r in enumerate(records) if r['stem']==stem and r['t']==t)
                for stem,t in (('6bba_6ca87370',67),('6bba_2819ca14',62))]
    rows=[]
    with threadpool_limits(limits=1,user_api='blas'):
        pools_limited=threadpool_info()
        for index in selected:
            case=store[index];tick=time.monotonic();old,old_gradient=reference_hinge(case,weights,time_limit=5.)
            old_seconds=time.monotonic()-tick;tick=time.monotonic();new,new_gradient=hinge(case,weights,time_limit=5.)
            new_seconds=time.monotonic()-tick
            assert abs(old-new)<1e-8,'Objective changed'
            row=dict(stem=records[index]['stem'],t=records[index]['t'],reference_hinge=old,reduced_hinge=new,
                     reference_seconds=old_seconds,reduced_seconds=new_seconds,
                     maximum_gradient_difference=float(np.max(np.abs(old_gradient-new_gradient))))
            rows.append(row);print(json.dumps(row),flush=True)
        # Fixed six-case functionality fit, not a promoted model or epoch search.
        small=[store[i] for i in selected];tick=time.monotonic()
        learned,history=fit(small,weights,np.r_[np.full(18,.1),np.full(12,.02)],epochs=1)
        fit_seconds=time.monotonic()-tick
        assert np.isfinite(learned).all() and len(history)==1
    result=dict(status='real_fast_training_functionality_passed',source_only=True,
        exact_saved_case_reconstructions=len(records),cache_bytes=store.cache_bytes,
        cache_items=len(store.cache),cache_limit_bytes=store.max_cache_bytes,preparation_seconds=preparation_seconds,
        paired_objectives=rows,small_fit_seconds=fit_seconds,small_fit_cases=len(selected),
        coefficients_not_exported=True,model_selected=False,authorized_for_submission=False,
        selection_or_validation_opened=False,blas_before=pools_before,blas_limited=pools_limited,
        source_sha256=sha(Path(__file__)),case_manifest_sha256=sha(root/'RESULT.json'),
        modules_sha256={n:sha(ROOT/'research'/n) for n in ('trajectory_event_fast_training_v1.py',
            'trajectory_event_fast_case_store_v1.py','trajectory_event_dominance_v1.py')},seconds=time.monotonic()-started)
    receipt.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],preparation_seconds=preparation_seconds,small_fit_seconds=fit_seconds,cache_bytes=store.cache_bytes)),flush=True)


if __name__=='__main__':main()
