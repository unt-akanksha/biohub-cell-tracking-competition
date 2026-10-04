"""Check uniform 16-child fork coverage on every known incompatible source case.

Feasibility only: zero dummy features do not score, fit, or change predictions.
No source label may insert an inference-unavailable option.
"""
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.trajectory_event_candidates_v1 import frames
from research.trajectory_event_supervision_v1 import for_problem
from research.trajectory_event_fast_training_v1 import prepare


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def arrays(p):
    with np.load(p,allow_pickle=False) as data:return dict(data)


def main():
    started=time.monotonic()
    output=ROOT/'reports/experiments/trajectory-event-fork-coverage-v2.json'
    assert not output.exists()
    records=[];receipts={}
    for batch in (4,5):
        prefix='trajectory-event-source-v1-b'+str(batch)
        case_path=ROOT/'reports/experiments'/(prefix+'-cases.json');cases=read(case_path)
        assert cases['status']=='source_event_cases_prepared' and not cases['training_allowed']
        receipts[str(batch)]=sha(case_path)
        folder=ROOT/'.biohub/cache'/(prefix+'-features');labels_root=ROOT/'.biohub/cache'/(prefix+'-supervision')
        frozen=read(folder/'RESULT.json');labels_report=read(labels_root/'RESULT.json')
        assert sha(folder/'RESULT.json')==cases['feature_receipt_sha256']
        assert sha(labels_root/'RESULT.json')==cases['supervision_receipt_sha256']
        base=ROOT/'.biohub/cache'/(prefix+'-full-output')
        backup=read(ROOT/'reports/experiments'/(prefix+'-full-harvest.json'))
        assert backup['status']=='verified_backup'
        hashes={r['path']:r['sha256'] for r in backup['records']}
        found=0
        for stem,item in cases['per_movie'].items():
            times={r['t'] for r in item['cases'] if r['reason']=='incompatible_partial_constraints'}
            if not times:continue
            pp=folder/(stem+'-prediction.json');gp=folder/(stem+'-groups.npz')
            lp=labels_root/(stem+'-labels.npz');ip=base/(stem+'-original/pre-postprocess.json')
            assert sha(pp)==frozen['per_movie'][stem]['prediction_sha256']
            assert sha(gp)==frozen['per_movie'][stem]['groups_sha256']
            assert sha(lp)==labels_report['per_movie'][stem]['labels_sha256']
            assert sha(ip)==hashes[ip.relative_to(base).as_posix()]
            graph,initial,groups,labels=read(pp),read(ip),arrays(gp),arrays(lp)
            for budget in (8,16):
                observed=set()
                for case in frames(initial,graph,groups,max_fork_children=budget,selected_times=times):
                    targets,safe,mapping=for_problem(case,labels['target'],labels['safe'])
                    prepared,reason=prepare(case,np.zeros((len(case['options']),1)),targets,safe)
                    if budget==8:assert reason=='incompatible_partial_constraints'
                    records.append(dict(batch=batch,stem=stem,t=case['t'],max_fork_children=budget,
                        options=len(case['options']),feasible=prepared is not None,reason=reason,**mapping))
                    observed.add(case['t'])
                assert observed==times
            found+=len(times)
        assert found==cases['totals']['incompatible_partial_constraints']
    result=dict(status='known_source_incompatibilities_audited',records=records,
        all_known_cases_feasible_at_16=all(r['feasible'] for r in records if r['max_fork_children']==16),
        complete_source_vocabulary_audit=False,expanded_vocabulary_adopted=False,
        training_guards_unchanged=True,labels_changed=False,predictions_changed=False,
        candidate_changed=False,source_case_receipts=receipts,source_sha256=sha(Path(__file__)),
        seconds=time.monotonic()-started,authorized_for_submission=False)
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
