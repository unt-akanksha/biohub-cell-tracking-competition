"""Freeze every source-supervised frame as bounded on-disk event learning cases."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.trajectory_event_candidates_v1 import frames
from research.trajectory_event_features_v1 import FEATURES, features
from research.trajectory_event_supervision_v1 import for_problem
from research.trajectory_event_assignment_v1 import prepare


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path,allow_pickle=False) as data:return dict(data)


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,required=True);args=p.parse_args()
    name='trajectory-event-source-v1-b'+str(args.batch)
    feature_root=ROOT/'.biohub/cache'/(name+'-features')
    label_root=ROOT/'.biohub/cache'/(name+'-supervision')
    base=ROOT/'.biohub/cache'/(name+'-full-output')
    target=ROOT/'.biohub/cache'/(name+'-cases')
    receipt=ROOT/'reports/experiments'/(name+'-cases.json')
    assert not target.exists() and not receipt.exists(), 'Inspect prior attempt; do not restart'
    feature_report=read(feature_root/'RESULT.json');label_report=read(label_root/'RESULT.json')
    scope=ROOT/'.biohub/cache/trajectory-event-source-v1-plan'/('batch-'+str(args.batch))/'MOVIES.json'
    plan=read(scope)
    assert all(m['role']=='optimization' for m in plan['movies'])
    assert feature_report['status']=='current_candidate_source_features_frozen'
    assert label_report['status']=='source_event_supervision_prepared' and label_report['source_only']
    assert label_report['source_scope_sha256']==feature_report['source_scope_sha256']==sha(scope)
    assert label_report['feature_receipt_sha256']==sha(feature_root/'RESULT.json')
    backup=read(ROOT/'reports/experiments'/(name+'-full-harvest.json'))
    assert backup['status']=='verified_backup'
    hashes={r['path']:r['sha256'] for r in backup['records']}
    target.mkdir();started=time.monotonic();totals=Counter()
    report=dict(status='running',per_movie={},source_scope_sha256=sha(scope),source_only=True,
                selection_or_validation_opened=False,model_fitted=False,authorized_for_submission=False,
                feature_receipt_sha256=sha(feature_root/'RESULT.json'),supervision_receipt_sha256=sha(label_root/'RESULT.json'),
                source_sha256=sha(Path(__file__)),feature_names=list(FEATURES),
                helper_sha256={n:sha(ROOT/'research'/n) for n in ('trajectory_event_candidates_v1.py',
                    'trajectory_event_features_v1.py','trajectory_event_supervision_v1.py','trajectory_event_assignment_v1.py')})
    def persist():
        report.update(totals=dict(totals),seconds=time.monotonic()-started)
        text=json.dumps(report,indent=2)+'\n'
        (target/'RESULT.json').write_text(text,encoding='utf-8');receipt.write_text(text,encoding='utf-8')
    persist()
    try:
        for movie in plan['movies']:
            stem=movie['stem'];frozen=feature_report['per_movie'][stem]
            pp=feature_root/(stem+'-prediction.json');gp=feature_root/(stem+'-groups.npz');fp=feature_root/(stem+'-features.npz')
            lp=label_root/(stem+'-labels.npz');ip=base/(stem+'-original/pre-postprocess.json')
            assert sha(pp)==frozen['prediction_sha256'] and sha(gp)==frozen['groups_sha256'] and sha(fp)==frozen['features_sha256']
            assert sha(lp)==label_report['per_movie'][stem]['labels_sha256']
            assert sha(ip)==hashes[ip.relative_to(base).as_posix()]
            graph=read(pp);initial=read(ip);groups=arrays(gp);labels=arrays(lp);edge=arrays(fp)
            assert list(edge['feature_names'])==list(FEATURES[:18])
            selected={int(graph['nodes'][str(c)]['t']) for c,p in zip(groups['children'],labels['target']) if p>=0}
            rows=[];counts=Counter()
            for problem in frames(initial,graph,groups,selected_times=selected):
                expected,safe,mapping=for_problem(problem,labels['target'],labels['safe'])
                row=dict(t=problem['t'],**mapping)
                if not np.any(expected>=0):
                    row.update(reason='no_known_parent_constraints');rows.append(row)
                    counts['no_known_parent_constraints']+=1;continue
                x=features(problem,graph,edge['features'])
                prepared,reason=prepare(problem,x,expected,safe)
                row.update(reason=reason,options=len(problem['options']))
                counts.update(mapping);counts[reason]+=1
                if prepared is not None:
                    path=target/(stem+'-t'+str(problem['t'])+'.npz')
                    np.savez_compressed(path,options=problem['options'],features=x,targets=expected,safe=safe,
                        nparents=problem['nparents'],nchildren=problem['nchildren'],incumbent=problem['incumbent'],
                        allowed=prepared['allowed'],margin=prepared['margin'])
                    row.update(path=path.name,sha256=sha(path),bytes=path.stat().st_size,n_constraints=prepared['n_constraints'])
                    counts['case_bytes']+=row['bytes']
                rows.append(row)
            missing=sorted(selected-{r['t'] for r in rows})
            counts['known_frames_outside_editable_scope']=len(missing)
            report['per_movie'][stem]=dict(cases=rows,counts=dict(counts),missing_editable_times=missing,
                source_prior_counts=label_report['per_movie'][stem]['counts'])
            totals.update(counts);persist()
            print(json.dumps(dict(stem=stem,**counts)),flush=True)
        report['status']='source_event_cases_prepared'
        report['all_partial_constraints_compatible']=totals['incompatible_partial_constraints']==0
        report['training_allowed']=report['all_partial_constraints_compatible'] and totals['prepared']>0
        persist();print(json.dumps(dict(status=report['status'],**dict(totals),seconds=report['seconds'])),flush=True)
    except BaseException as error:
        report.update(status='failed',error=repr(error),training_allowed=False);persist();raise


if __name__=='__main__':main()
