"""Fixed low-dimensional morphology screen on unchanged cached triplets."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.native_division_morphology_v7 import features,fit,predict,calibrate,metrics


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def gate(candidate,controls):
    return bool(candidate['fp']==0 and candidate['tp']*2>=candidate['positive'] and
        all(candidate['tp']>=c['tp'] and candidate['balanced_nll']<c['balanced_nll'] for c in controls))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    started=time.monotonic();name='native-division-morphology-v7'+('-smoke' if args.smoke else '-full')
    target=ROOT/'.biohub/cache'/name;receipt=ROOT/'reports/experiments'/(name+'.json')
    assert not target.exists() and not receipt.exists(),'Inspect prior run, never overwrite/restart blindly'
    design=ROOT/'reports/experiments/native-division-morphology-v7-design.md'
    contract=dict(design_sha256=sha(design),source_sha256=sha(Path(__file__)),
        helper_sha256=sha(ROOT/'research/native_division_morphology_v7.py'),
        data_sha256=['7e0e7b1ca3c11b27319824fad3746b49f8a1cf519a73d3dad38e12d2ef8a0e06',
                    'e358a984eade85b96fbf2d77937ba0d57307f4813f44fb6c339346024d17df87'])
    if not args.smoke:
        smoke=read(ROOT/'reports/experiments/native-division-morphology-v7-smoke.json')
        assert smoke['status']=='real_source_feature_smoke_passed' and smoke['contract']==contract
    roots=[ROOT/'.biohub/cache/native-division-v3-data-bundle',ROOT/'.biohub/cache/native-division-additive-v5-full-output']
    records=[]
    for index,root in enumerate(roots):
        assert sha(root/'DATA.json')==contract['data_sha256'][index]
        for row in read(root/'DATA.json')['records']:
            if index:assert row['role']=='optimization' and row['positive']==row['triples']
            records.append((root,row))
    if args.smoke:
        records=[next((root,r) for root,r in records if r['embryo']==e and r['role']=='optimization' and r['positive']>0)
                 for e in ('44b6','6bba')]
    target.mkdir();blocks={key:[] for key in ('geometry','legacy_statistics','morphology')};labels=[];rows=[]
    result=dict(status='extracting_features',contract=contract,smoke=args.smoke,source_only_fitting=True,
        gpu_used=False,competition_test_opened=False,complete_movie_validation_opened=False,
        previously_exposed_selection_cohort=True,authorized_for_submission=False,records=[])
    def persist():
        result['seconds']=time.monotonic()-started
        text=json.dumps(result,indent=2,allow_nan=False)+'\n'
        receipt.write_text(text,encoding='utf-8');(target/'RESULT.json').write_text(text,encoding='utf-8')
    persist()
    try:
        for root,row in records:
            path=root/row['path'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
            with np.load(path,allow_pickle=False) as data:
                stored,triples,coords=data['patches'],data['triples'],data['coords']
                x=features(stored,triples,coords)
                swapped=features(stored,triples[:,[0,2,1]],coords[:,[0,2,1]])
                for key in x:assert np.array_equal(x[key],swapped[key]) and np.isfinite(x[key]).all()
                y=data['labels'];assert len(y)==row['triples'] and int(y.sum())==row['positive']
            for key in x:blocks[key].append(x[key])
            labels.append(y);rows.extend([dict(stem=row['stem'],embryo=row['embryo'],role=row['role'],transition=row['transition'])]*len(y))
            result['records'].append(dict(path=str(path.relative_to(ROOT)),sha256=row['sha256'],rows=len(y)))
        x={key:np.concatenate(value) for key,value in blocks.items()};y=np.concatenate(labels)
        roles=np.array([r['role'] for r in rows]);embryos=np.array([r['embryo'] for r in rows]);stems=np.array([r['stem'] for r in rows])
        feature_path=target/'features.npz';np.savez_compressed(feature_path,**x,labels=y,role=roles,embryo=embryos,stem=stems)
        result.update(rows=len(y),feature_sha256=sha(feature_path),feature_shapes={k:list(v.shape) for k,v in x.items()})
        if args.smoke:
            assert np.all(roles=='optimization')
            with np.load(feature_path,allow_pickle=False) as data:
                for key,value in x.items():assert np.array_equal(data[key],value)
            result['status']='real_source_feature_smoke_passed';persist();print(json.dumps(result),flush=True);return
        assert len(y)==3108
        assert int(y[roles=='optimization'].sum())==65 and int((y[roles=='optimization']==0).sum())==2305
        assert int(y[roles=='selection'].sum())==7 and int((y[roles=='selection']==0).sum())==731
        result['status']='fitting_source_heads';result['folds']=[];persist()
        for source in ('44b6','6bba'):
            train=(roles=='optimization')&(embryos==source);selection=(roles=='selection')&(embryos==source)
            opposite=(roles=='selection')&(embryos!=source)
            states={};source_metrics={};source_prob={};paths={}
            for arm in x:
                states[arm]=fit(x[arm][train],y[train]);p=predict(states[arm],x[arm][selection]);threshold=calibrate(y[selection],p)
                states[arm]['threshold']=threshold;source_prob[arm]=p;source_metrics[arm]=metrics(y[selection],p,threshold)
                path=target/(source+'-'+arm+'.npz');np.savez_compressed(path,**states[arm])
                with np.load(path,allow_pickle=False) as data:assert np.array_equal(p,predict(dict(data),x[arm][selection]))
                paths[arm]=dict(path=path.name,sha256=sha(path),iterations=states[arm]['iterations'])
            controls=('geometry','legacy_statistics')
            source_pass=gate(source_metrics['morphology'],[source_metrics[a] for a in controls])
            fold=dict(source=source,training_rows=int(train.sum()),training_positives=int(y[train].sum()),
                source_metrics=source_metrics,model_files=paths,source_pass=source_pass,opposite_opened=False,
                source_per_movie={a:{s:dict(rows=int((stems[selection]==s).sum()),positive=int(y[selection][stems[selection]==s].sum()),
                    predicted_positive=int((source_prob[a][stems[selection]==s]>=states[a]['threshold']).sum()),
                    tp=int(((source_prob[a][stems[selection]==s]>=states[a]['threshold'])&(y[selection][stems[selection]==s]==1)).sum()))
                    for s in sorted(set(stems[selection]))} for a in x})
            if source_pass:
                opposite_metrics={a:metrics(y[opposite],predict(states[a],x[a][opposite]),states[a]['threshold']) for a in x}
                fold.update(opposite_opened=True,opposite_metrics=opposite_metrics,
                    opposite_pass=gate(opposite_metrics['morphology'],[opposite_metrics[a] for a in controls]))
            result['folds'].append(fold);persist();print(json.dumps(fold),flush=True)
        result.update(status='fixed_morphology_screen_finished',any_candidate_passed=any(f.get('opposite_pass',False) for f in result['folds']),
            complete_movie_gain_established=False,automatic_followup_authorized=False)
        persist();print(json.dumps({k:v for k,v in result.items() if k not in ('records','folds')}),flush=True)
    except BaseException as error:
        result.update(status='failed_requires_inspection',error=repr(error));persist();raise


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
