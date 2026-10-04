"""Real, source-only fixed-budget event fit. No quality or submission claim."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.trajectory_event_candidates_v1 import frames
from research.trajectory_event_features_v1 import FEATURES, features
from research.trajectory_event_supervision_v1 import for_problem
from research.trajectory_event_assignment_v1 import prepare, hinge, solve
from research.trajectory_event_training_v1 import prior, fit


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path,allow_pickle=False) as values:
        return dict(values)


def diagnostic(cases, targets, weights):
    rows=[]
    for case, truth in zip(cases,targets):
        chosen=solve(case['problem'],case['x']@weights,time_limit=5.)
        parents=np.full(len(truth),-1,np.int64)
        for p,a,b in case['problem']['options'][chosen.astype(bool)]:
            for c in (a,b):
                if c>=0:parents[c]=p
        known=truth>=0
        rows.append(dict(hinge=hinge(case,weights,time_limit=5.)[0],
                         known=int(known.sum()),wrong_parent_or_birth=int((parents[known]!=truth[known]).sum()),
                         selected_forks=int(((case['problem']['options'][:,2]>=0)&chosen.astype(bool)).sum())))
    return dict(cases=rows,mean_hinge=float(np.mean([r['hinge'] for r in rows])),
                known=sum(r['known'] for r in rows),wrong_parent_or_birth=sum(r['wrong_parent_or_birth'] for r in rows))


def main():
    started=time.monotonic()
    target=ROOT/'.biohub/cache/trajectory-event-learning-smoke-v1'
    assert not target.exists(), 'Inspect existing attempt; do not silently restart'
    target.mkdir()
    report=dict(status='running',source_only=True,selection_or_validation_opened=False,
                authorized_for_submission=False,quality_gain_established=False,graph_predictions_exported=False)
    destination=ROOT/'reports/experiments/trajectory-event-learning-smoke-v1.json'
    assert not destination.exists()
    def persist():
        report['seconds']=time.monotonic()-started
        text=json.dumps(report,indent=2,allow_nan=False)+'\n'
        (target/'RESULT.json').write_text(text,encoding='utf-8')
        destination.write_text(text,encoding='utf-8')
    persist()
    try:
        name='trajectory-event-source-v1-b0'
        scope=ROOT/'.biohub/cache/trajectory-event-source-v1-plan/batch-0/MOVIES.json'
        source_plan=read(scope)
        assert all(m['role']=='optimization' for m in source_plan['movies'])
        stems=[m['stem'] for m in source_plan['movies'] if m['stem'].startswith('6bba_')][:2]
        assert len(stems)==2
        feature_root=ROOT/'.biohub/cache'/(name+'-features')
        label_root=ROOT/'.biohub/cache'/(name+'-supervision')
        frozen=read(feature_root/'RESULT.json');labels=read(label_root/'RESULT.json')
        assert frozen['status']=='current_candidate_source_features_frozen'
        assert labels['status']=='source_event_supervision_prepared' and labels['source_only']
        assert labels['source_scope_sha256']==frozen['source_scope_sha256']==sha(scope)
        assert labels['feature_receipt_sha256']==sha(feature_root/'RESULT.json')
        base=ROOT/'.biohub/cache'/(name+'-full-output')
        backup=read(ROOT/'reports/experiments'/(name+'-full-harvest.json'))
        assert backup['status']=='verified_backup'
        backup_hashes={r['path']:r['sha256'] for r in backup['records']}
        model=ROOT/'.biohub/cache/trajectory-structured-loss-v1-full/weights.npz'
        assert sha(model)=='e33fe1b79291ed89697db7a5ee6a839bc2ef34e23f504b27f1f29e44290107ba'
        counts=Counter()
        for stem in stems:counts.update(labels['per_movie'][stem]['counts'])
        anchor=prior(arrays(model)['6bba'],counts['annotated_division_parents'],counts['annotated_consecutive_parent_opportunities'])
        regularization=np.r_[np.full(18,.1),np.full(12,.02)]
        cases,truths,records=[],[],[]
        for stem in stems:
            files={suffix:feature_root/(stem+suffix) for suffix in ('-prediction.json','-groups.npz','-features.npz')}
            for suffix,key in (('-prediction.json','prediction_sha256'),('-groups.npz','groups_sha256'),('-features.npz','features_sha256')):
                assert sha(files[suffix])==frozen['per_movie'][stem][key]
            lp=label_root/(stem+'-labels.npz');assert sha(lp)==labels['per_movie'][stem]['labels_sha256']
            initial_path=base/(stem+'-original/pre-postprocess.json')
            assert sha(initial_path)==backup_hashes[initial_path.relative_to(base).as_posix()]
            initial=read(initial_path);graph=read(files['-prediction.json'])
            groups=arrays(files['-groups.npz']);edge=arrays(files['-features.npz']);label=arrays(lp)
            assert list(edge['feature_names'])==list(FEATURES[:18])
            by_event=defaultdict(list)
            for child,parent in zip(groups['children'],label['target']):
                if parent>=0:by_event[(int(graph['nodes'][str(child)]['t']),int(parent))].append(int(child))
            event_times={t for (t,p),children in by_event.items() if len(children)==2}
            known_times={t for t,p in by_event}
            ordinary=sorted(known_times-event_times,key=lambda t:hashlib.sha256((stem+':'+str(t)).encode()).hexdigest())[:1]
            selected=event_times|set(ordinary)
            assert selected
            for problem in frames(initial,graph,groups,selected_times=selected):
                x=features(problem,graph,edge['features'])
                expected,safe,mapping=for_problem(problem,label['target'],label['safe'])
                case,reason=prepare(problem,x,expected,safe)
                row=dict(stem=stem,t=problem['t'],event_frame=problem['t'] in event_times,
                         options=len(problem['options']),reason=reason,**mapping)
                if case is None:
                    raise ValueError('Smoke source frame rejected: '+json.dumps(row))
                path=target/(stem+'-t'+str(problem['t'])+'.npz')
                np.savez_compressed(path,options=problem['options'],features=x,targets=expected,safe=safe,
                                    nparents=problem['nparents'],nchildren=problem['nchildren'],
                                    incumbent=problem['incumbent'],allowed=case['allowed'],margin=case['margin'])
                row.update(case_sha256=sha(path),path=path.name,n_constraints=case['n_constraints'])
                cases.append(case);truths.append(expected);records.append(row)
                print(json.dumps(dict(event='case_prepared',**row)),flush=True)
            assert {r['t'] for r in records if r['stem']==stem}==selected, 'Selected frame vanished from editable graph'
        assert cases
        report.update(stems=stems,cases=records,source_scope_sha256=sha(scope),
                      feature_receipt_sha256=sha(feature_root/'RESULT.json'),supervision_receipt_sha256=sha(label_root/'RESULT.json'),
                      source_prior_counts=dict(divisions=counts['annotated_division_parents'],opportunities=counts['annotated_consecutive_parent_opportunities']),
                      anchor=anchor.tolist(),regularization=regularization.tolist(),epochs=3,learning_rate=.03,seed=20260914,
                      feature_names=list(FEATURES),inherited_model_sha256=sha(model),
                      source_sha256=sha(Path(__file__)),
                      modules_sha256={p:sha(ROOT/'research'/p) for p in (
                          'trajectory_event_training_v1.py','trajectory_event_assignment_v1.py',
                          'trajectory_event_candidates_v1.py','trajectory_event_features_v1.py','trajectory_event_supervision_v1.py')})
        report['before']=diagnostic(cases,truths,anchor);persist()
        def checkpoint(weights,row):
            path=target/('epoch-'+str(row['epoch'])+'.npz')
            np.savez_compressed(path,weights=weights,anchor=anchor,regularization=regularization,feature_names=np.asarray(FEATURES))
            row['weights_sha256']=sha(path)
            report.setdefault('epochs_completed',[]).append(row)
            persist();print(json.dumps(dict(event='epoch_complete',**row)),flush=True)
        weights,history=fit(cases,anchor,regularization,epochs=3,learning_rate=.03,seed=20260914,callback=checkpoint)
        assert len(history)==3 and np.isfinite(weights).all() and not np.array_equal(weights,anchor)
        report.update(status='real_source_event_training_smoke_complete',after=diagnostic(cases,truths,weights),
                      model_fitted=True,weights_changed=True,final_weights_sha256=sha(target/'epoch-3.npz'))
        persist()
        print(json.dumps(dict(status=report['status'],before=report['before'],after=report['after'],seconds=report['seconds'])),flush=True)
    except BaseException as error:
        report.update(status='failed',error=repr(error));persist();raise


if __name__=='__main__':main()
