"""Benchmark provable fork dominance on the largest frozen source problems."""
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.trajectory_event_assignment_v1 import problem,solve,EventSolveError
from research.trajectory_event_dominance_v1 import allowed_options


def main():
    started=time.monotonic();receipt=ROOT/'reports/experiments/trajectory-event-dominance-v1-audit.json'
    assert not receipt.exists()
    root=ROOT/'.biohub/cache/trajectory-event-source-v1-b0-cases'
    manifest=json.loads((root/'RESULT.json').read_text())
    assert manifest['status']=='source_event_cases_prepared' and manifest['source_only']
    model=ROOT/'.biohub/cache/trajectory-event-learning-smoke-v1/epoch-3.npz'
    assert sha(model)=='508165d131dc0c2516cf91e0eafb5a88edf24358f6ced5b0436f29ccc6b733be'
    with np.load(model,allow_pickle=False) as data:weights=data['weights']
    # Size-only selection, never source errors or agreement with labels.
    records=sorted([dict(r,stem=stem) for stem,movie in manifest['per_movie'].items()
                    for r in movie['cases'] if r['reason']=='prepared'],key=lambda r:(-r['options'],r['path']))[:8]
    rows=[]
    for row in records:
        path=root/row['path'];assert sha(path)==row['sha256']
        with np.load(path,allow_pickle=False) as data:
            case=problem(data['options'],int(data['nparents']),int(data['nchildren']))
            scores=data['features'].astype(np.float64)@weights
        tick=time.monotonic();keep,reduction=allowed_options(case,scores);reduction_seconds=time.monotonic()-tick
        result=dict(stem=row['stem'],t=row['t'],options=len(scores),reduction_seconds=reduction_seconds,**reduction)
        for arm,mask in (('reference',None),('reduced',keep)):
            tick=time.monotonic()
            try:
                choice=solve(case,scores,allowed=mask,time_limit=2.)
                result[arm]=dict(optimal=True,objective=float(scores@choice),seconds=time.monotonic()-tick)
            except EventSolveError as error:
                result[arm]=dict(optimal=False,error=str(error),seconds=time.monotonic()-tick)
        if result['reference']['optimal'] and result['reduced']['optimal']:
            difference=result['reduced']['objective']-result['reference']['objective']
            result['objective_difference']=difference
            assert abs(difference)<1e-7,'Dominance changed the optimum'
        rows.append(result);print(json.dumps(result),flush=True)
    report=dict(status='source_dominance_benchmark_complete',cases=rows,
        model_sha256=sha(model),case_manifest_sha256=sha(root/'RESULT.json'),
        source_sha256=sha(Path(__file__)),reduction_sha256=sha(ROOT/'research/trajectory_event_dominance_v1.py'),
        all_reduced_optimal=all(r['reduced']['optimal'] for r in rows),
        source_only=True,ground_truth_used=False,source_labels_used=False,model_changed=False,
        graph_predictions_exported=False,authorized_for_submission=False,seconds=time.monotonic()-started)
    receipt.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
