"""Diagnose source frame feasibility without changing labels or the candidate."""
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.trajectory_event_candidates_v1 import frames
from research.trajectory_event_supervision_v1 import for_problem
from research.trajectory_event_fast_features_v1 import FeatureContext
from research.trajectory_event_fast_training_v1 import prepare


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path,allow_pickle=False) as data:return dict(data)


def main():
    started=time.monotonic();stem='44b6_eb2880fc';t=73
    output=ROOT/'reports/experiments/trajectory-event-incompatible-source-v1.json'
    assert not output.exists()
    receipt=ROOT/'reports/experiments/trajectory-event-source-v1-b4-cases.json'
    assert sha(receipt)=='0bacb620fb9aa70645e9ecc751704bfaa1a6f92c9c31cd0bc289652a3f3b22e7'
    old=read(receipt);assert not old['training_allowed'] and old['totals']['incompatible_partial_constraints']==1
    folder=ROOT/'.biohub/cache/trajectory-event-source-v1-b4-features';frozen=read(folder/'RESULT.json')
    label_root=ROOT/'.biohub/cache/trajectory-event-source-v1-b4-supervision';label_report=read(label_root/'RESULT.json')
    assert sha(folder/'RESULT.json')==old['feature_receipt_sha256']
    assert sha(label_root/'RESULT.json')==old['supervision_receipt_sha256']
    pp=folder/(stem+'-prediction.json');gp=folder/(stem+'-groups.npz');fp=folder/(stem+'-features.npz')
    lp=label_root/(stem+'-labels.npz')
    for p,key in ((pp,'prediction'),(gp,'groups'),(fp,'features')):assert sha(p)==frozen['per_movie'][stem][key+'_sha256']
    assert sha(lp)==label_report['per_movie'][stem]['labels_sha256']
    base=ROOT/'.biohub/cache/trajectory-event-source-v1-b4-full-output'
    ip=base/(stem+'-original/pre-postprocess.json')
    harvest=read(ROOT/'reports/experiments/trajectory-event-source-v1-b4-full-harvest.json')
    assert sha(ip)=={r['path']:r['sha256'] for r in harvest['records']}[ip.relative_to(base).as_posix()]
    graph=read(pp);initial=read(ip);groups=arrays(gp);labels=arrays(lp)
    context=FeatureContext(graph,arrays(fp)['features']);records=[]
    for budget in (8,16):
        cases=list(frames(initial,graph,groups,max_fork_children=budget,selected_times={t}));assert len(cases)==1
        case=cases[0];targets,safe,mapping=for_problem(case,labels['target'],labels['safe'])
        prepared,reason=prepare(case,context.features(case),targets,safe)
        options={tuple(row) for row in case['options']};known=[]
        for parent in sorted(set(targets[targets>=0])):
            children=np.flatnonzero(targets==parent)
            option=(int(parent),int(children[0]),int(children[1]) if len(children)==2 else -1)
            known.append(dict(parent_id=int(case['parents'][parent]),
                child_ids=[int(case['children'][c]) for c in children],
                known_daughters=len(children),required_event_present=len(children)<=2 and option in options))
        records.append(dict(max_fork_children=budget,options=len(case['options']),reason=reason,
                            feasible=prepared is not None,known_parents=known,**mapping))
    assert records[0]['reason']=='incompatible_partial_constraints'
    report=dict(status='source_incompatibility_diagnosed',stem=stem,t=t,records=records,
        original_receipt_sha256=sha(receipt),source_sha256=sha(Path(__file__)),seconds=time.monotonic()-started,
        source_only=True,labels_changed=False,predictions_changed=False,training_guard_unchanged=True,
        production_candidate_unchanged=True,expanded_vocabulary_not_adopted=True,authorized_for_submission=False)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
