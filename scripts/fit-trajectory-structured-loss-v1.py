"""Fixed source structured fits; graph predictions cannot consume label masks."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.public_d4_full_movie import sha
from research.trajectory_assignment_feasibility_v1 import problems
from research.trajectory_structured_loss_v1 import prepare,fit,infer
from research.trajectory_candidate_ranker_audit_v2 import evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('smoke','full'),required=True);args=p.parse_args()
    started=time.monotonic();target=ROOT/'.biohub/cache'/('trajectory-structured-loss-v1-'+args.mode)
    assert not target.exists()
    old=ROOT/'.biohub/cache/trajectory-candidate-ranker-v2-audit'
    old_result=json.loads((old/'RESULT.json').read_text())
    assert sha(old/'weights.npz')==old_result['weights_sha256']
    training=ROOT/'.biohub/cache/trajectory-candidate-ranker-v1'
    train_result=json.loads((training/'RESULT.json').read_text())
    inv_path=ROOT/'reports/experiments/trajectory-candidate-supervision-v1.json';inv=json.loads(inv_path.read_text())
    plan=ROOT/'.biohub/cache/trajectory-disagreement-source-v1-plan/MOVIES.json'
    assert sha(plan)==inv['source_scope_sha256']=='5c4b64793068565598537db6bb3af439513bde3d46e357ffeab59164bda020bf'
    stems=[m['stem'] for m in json.loads(plan.read_text())['movies']]
    source=ROOT/'.biohub/cache/trajectory-disagreement-source-v1-full-output'
    backup=json.loads((ROOT/'reports/experiments/trajectory-disagreement-source-v1-full-harvest.json').read_text())
    assert backup['status']=='verified_backup'
    for r in backup['records']:assert sha(source/r['path'])==r['sha256']
    contract=dict(source_sha256=sha(Path(__file__)),helper_sha256=sha(ROOT/'research/trajectory_structured_loss_v1.py'),
                  feasibility_helper_sha256=sha(ROOT/'research/trajectory_assignment_feasibility_v1.py'),
                  initial_weight_sha256=sha(old/'weights.npz'),source_scope_sha256=sha(plan),
                  steps=300,batch=4,learning_rate=.05,regularization=.01,seed=1729,
                  training_objective='partial-label latent structured hinge over feasible assignments',
                  source_only=True,selection_or_validation_read=False)
    if args.mode=='full':
        smoke=ROOT/'.biohub/cache/trajectory-structured-loss-v1-smoke/RESULT.json'
        proof=json.loads(smoke.read_text());assert proof['status']=='functionality_passed' and proof['contract']==contract
    target.mkdir();(target/'CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
    with np.load(old/'weights.npz',allow_pickle=False) as f:initial=dict(f)
    data=ROOT/'.biohub/cache/trajectory-candidate-supervision-v1'
    cases={};feasible={};matrices={};groups={};labels={};coverage={}
    for stem in stems:
        fp=training/(stem+'-features.npz');gp=data/(stem+'-candidates.npz');lp=data/(stem+'-labels.npz')
        assert sha(fp)==train_result['features_sha256'][stem]
        assert sha(gp)==inv['candidate_sha256'][stem] and sha(lp)==inv['label_sha256'][stem]
        with np.load(fp,allow_pickle=False) as f:matrices[stem]=f['features']
        with np.load(gp,allow_pickle=False) as f:groups[stem]=dict(f)
        with np.load(lp,allow_pickle=False) as f:labels[stem]=f['target'];safe=f['safe']
        final=json.loads((source/(stem+'-original')/'repaired-prediction.json').read_text())
        feasible[stem]=list(problems(final,groups[stem]));cases[stem]=[];count=Counter()
        for problem in feasible[stem]:
            case,reason=prepare(problem,matrices[stem],labels[stem],safe);count[reason]+=1
            if case is not None:cases[stem].append(case)
        coverage[stem]=dict(count)
        print(json.dumps(dict(stage='prepared',stem=stem,**count)),flush=True)
    def predictions(stem,weights):
        result=groups[stem]['current'].copy()
        for problem in feasible[stem]:result[problem['group_indices']]=infer(problem,matrices[stem],weights)
        return result
    def screen(stem,weights,baseline,tag):
        pred=predictions(stem,weights)
        path=target/(tag+'-'+stem+'.npy');np.save(path,pred,allow_pickle=False)
        # Choices persisted before labels enter this evaluation.
        report=evaluate(pred,groups[stem],labels[stem])
        prior=evaluate(predictions(stem,baseline),groups[stem],labels[stem])
        report['initial_joint_correct']=prior['learned_correct'];report['prediction_sha256']=sha(path)
        return report
    if args.mode=='smoke':
        held=stems[0];members={s:cases[s] for s in stems if s.startswith('44b6_') and s!=held}
        weights,fit_receipt=fit(members,initial['lomo_'+held],steps=2)
        measured=screen(held,weights,initial['lomo_'+held],'smoke')
        assert fit_receipt['updates']==2 and measured['queries']>0 and np.isfinite(weights).all()
        result=dict(status='functionality_passed',contract=contract,coverage=coverage,fit=fit_receipt,
                    held=measured,quality_gain_established=False,seconds=time.monotonic()-started)
    else:
        lomo={};source_summary={};cross={};weights={};fits={}
        for embryo in ('44b6','6bba'):
            members=[s for s in stems if s.startswith(embryo+'_')]
            for held in members:
                key='lomo_'+held
                w,fits[key]=fit({s:cases[s] for s in members if s!=held},initial[key])
                weights[key]=w;lomo[held]=screen(held,w,initial[key],'lomo')
                print(json.dumps(dict(stage='held_movie',stem=held,**lomo[held])),flush=True)
            def summary(rows):
                total={k:sum(r[k] for r in rows) for k in ('queries','learned_correct','current_correct','initial_joint_correct','learned_repairs','learned_breaks')}
                total['passed']=bool(total['learned_correct']>max(total['current_correct'],total['initial_joint_correct']) and all(r['learned_correct']>=r['current_correct'] for r in rows))
                return total
            source_summary[embryo]=summary([lomo[s] for s in members])
            weights[embryo],fits[embryo]=fit({s:cases[s] for s in members},initial[embryo])
            cross_rows={s:screen(s,weights[embryo],initial[embryo],'cross') for s in stems if s not in members}
            cross[embryo]=dict(per_movie=cross_rows,summary=summary(list(cross_rows.values())))
        np.savez_compressed(target/'weights.npz',**weights)
        qualified=[e for e in source_summary if source_summary[e]['passed'] and cross[e]['summary']['passed']]
        result=dict(status='source_structured_fits_complete',contract=contract,coverage=coverage,lomo=lomo,
                    source=source_summary,cross_embryo=cross,fits=fits,weights_sha256=sha(target/'weights.npz'),
                    qualified_for_full_movie_scoring=qualified,authorized_for_submission=False,
                    selection_or_validation_read=False,seconds=time.monotonic()-started)
    (target/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'reports/experiments'/('trajectory-structured-loss-v1-'+args.mode+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('fits','coverage','lomo','contract','cross_embryo')}))


if __name__=='__main__':main()
