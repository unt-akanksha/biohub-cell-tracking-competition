"""Frozen opposite-embryo full-movie evaluation, with an untrained16 ablation.

Run --smoke first: two label-free complete movies selected by feature-file size.
The full run reuses those exact predictions and freezes all others before GT.
Optional --control-preflight tests the immutable untrained control during fitting;
it never opens intermediate weights or GT and does not replace the learned smoke.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha,validate_graph
from research.trajectory_event_fork16_inference_v1 import refine,FEATURES
from research.trajectory_event_release_v1 import check_event_stage


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path,allow_pickle=False) as data:return dict(data)


def fixed_fit_contracts(root):
    """Other fit may still run, but neither schedule may change after scoring."""
    launch_path=root/'reports/experiments/trajectory-event-fork16-full-fit-launch-v1.json'
    assert sha(launch_path)=='9d58a5b50c276d505282477dec08a35b762bc91ed18a00b33c3eaee815d5436a'
    launch=read(launch_path)
    assert sha(root/'scripts/train-trajectory-event-fork16-source-v1.py')==launch['training_script_sha256']
    result={}
    for embryo,item in launch['workers'].items():
        path=root/'.biohub/cache'/('trajectory-event-fork16-fit-'+embryo+'-v1')/'CONTRACT.json'
        assert sha(path)==item['contract_sha256'],'A frozen fit contract changed'
        contract=read(path)
        assert contract['epochs']==launch['fixed_epochs']==3
        assert contract['source_only'] and not contract['source_score_based_epoch_selection']
        for name,digest in contract['helper_sha256'].items():
            assert sha(root/'research'/name)==digest,'A training helper changed'
        result[embryo]=sha(path)
    assert set(result)=={'44b6','6bba'}
    return result


def quality_checks(summaries,per_movie,movies):
    """No missing movies or undefined scores may masquerade as nonregression."""
    arms=('submitted8','untrained16','learned16')
    assert set(summaries)==set(per_movie)==set(arms)
    assert len(movies)==len(set(movies)) and movies
    assert all(set(per_movie[a])==set(movies) for a in arms)
    base,control,learned=[summaries[a] for a in arms]
    finite=all(s['n']==s['n_adj']==len(movies) and
        all(np.isfinite(s[k]) for k in ('score','edge_jaccard','adj_edge_jaccard'))
        for s in summaries.values())
    finite=finite and all(r['n']==r['n_adj']==1 and
        all(np.isfinite(r[k]) for k in ('score','edge_jaccard','adj_edge_jaccard'))
        for arm in per_movie.values() for r in arm.values())
    regressions={s:float(per_movie['learned16'][s]['score']-per_movie['submitted8'][s]['score']) for s in movies
                 if per_movie['learned16'][s]['score']<per_movie['submitted8'][s]['score']-1e-12}
    division_counts=lambda s:sum(s[k] for k in ('division_tp','division_fp','division_fn'))
    # The official scorer omits this term when there are no divisions at all.
    no_divisions=division_counts(base)==division_counts(learned)==0
    division_ok=no_divisions or (np.isfinite(base['division_jaccard']) and
        np.isfinite(learned['division_jaccard']) and learned['division_jaccard']>=base['division_jaccard'])
    checks=dict(finite_complete_scoring=finite,
        pooled_score_improves=learned['score']>base['score'],
        learning_beats_untrained_control=learned['score']>control['score'],
        raw_edge_nonregressing=learned['edge_jaccard']>=base['edge_jaccard'],
        division_jaccard_nonregressing=division_ok,every_movie_nonregressing=not regressions)
    return {k:bool(v) for k,v in checks.items()},regressions


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fit-embryo',choices=('44b6','6bba'),required=True)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke',action='store_true')
    mode.add_argument('--control-preflight',action='store_true');args=parser.parse_args()
    started=time.monotonic();reports=ROOT/'reports/experiments'
    frozen_fit_contracts=fixed_fit_contracts(ROOT)
    fit_name='trajectory-event-fork16-fit-'+args.fit_embryo+'-v1'
    fit_root=ROOT/'.biohub/cache'/fit_name;fit_path=reports/(fit_name+'.json')
    fitted,contract=read(fit_path),read(fit_root/'CONTRACT.json')
    assert sha(fit_root/'CONTRACT.json')==fitted['contract_sha256']
    assert contract['source_embryo']==args.fit_embryo and contract['source_only']
    assert contract['opposite_embryo_excluded_from_optimizer'] and not contract['selection_or_validation_used_for_training']
    model=None
    if args.control_preflight:
        weights={'untrained16':np.asarray(contract['anchor'],dtype=np.float64)}
    else:
        assert fitted['status']=='source_event_training_complete' and fitted['model_fitted']
        assert not fitted['control_smoke'] and fitted['steps']==contract['cases']*contract['epochs']
        assert len(fitted['epochs_completed'])==contract['epochs']==3
        model=fit_root/'final-weights.npz';assert sha(model)==fitted['weights_sha256']
        values=arrays(model);assert int(values['max_fork_children'])==16 and list(values['feature_names'])==list(FEATURES)
        weights={arm:values[key] for arm,key in (('untrained16','anchor'),('learned16','weights'))}
        assert np.array_equal(weights['untrained16'],np.asarray(contract['anchor'],dtype=np.float64))
    assert all(w.shape==(30,) and np.isfinite(w).all() for w in weights.values())
    held='6bba' if args.fit_embryo=='44b6' else '44b6'
    name='trajectory-event-fork16-held-'+held+'-v1'
    smoke_name=name+'-smoke'
    preflight_name=name+'-control-preflight'
    if args.smoke:name=smoke_name
    if args.control_preflight:name=preflight_name
    target=ROOT/'.biohub/cache'/name;receipt=reports/(name+'.json')
    assert not target.exists() and not receipt.exists(),'Inspect the existing evaluation; never restart blindly'
    references={}
    for key,expected,arm in (
        ('trajectory-event-anchor-source-v1','e1667c04fd1dd83d18932f63981e98e54ff0a2f72873932f55216987131b44fd','event_smoke'),
        ('trajectory-event-anchor-expanded-source-v1','d1e6adde78dde9a9eb324454cee90ec4973075df0c2458be253643d84e6d1894','anchor')):
        path=reports/(key+'.json');assert sha(path)==expected
        data=read(path);assert data['parameter_key']=='anchor' and not data['event_head_coefficients_fitted']
        for stem,row in data['inference'].items():
            references[stem]=dict(path=ROOT/'.biohub/cache'/key/(stem+'-prediction.json'),
                sha256=row['prediction_sha256'],score=data['per_movie_summaries'][arm][stem]['score'])
    inputs={};scopes={}
    for batch in range(8):
        prefix='trajectory-event-source-v1-b'+str(batch)
        folder=ROOT/'.biohub/cache'/(prefix+'-features');frozen=read(folder/'RESULT.json')
        scope=ROOT/'.biohub/cache/trajectory-event-source-v1-plan'/('batch-'+str(batch))/'MOVIES.json'
        plan=read(scope);assert sha(scope)==frozen['source_scope_sha256']
        assert all(m['role']=='optimization' for m in plan['movies'])
        backup_path=reports/(prefix+'-full-harvest.json');backup=read(backup_path)
        assert backup['status']=='verified_backup';hashes={r['path']:r['sha256'] for r in backup['records']}
        base=ROOT/'.biohub/cache'/(prefix+'-full-output')
        scopes[str(batch)]=dict(scope_sha256=sha(scope),feature_receipt_sha256=sha(folder/'RESULT.json'),backup_sha256=sha(backup_path))
        for movie in plan['movies']:
            stem=movie['stem']
            if not stem.startswith(held+'_'):continue
            assert stem not in contract['source_stems'] and stem not in inputs
            files={suffix:folder/(stem+suffix) for suffix in ('-prediction.json','-groups.npz','-features.npz')}
            for suffix,key in (('-prediction.json','prediction_sha256'),('-groups.npz','groups_sha256'),('-features.npz','features_sha256')):
                assert sha(files[suffix])==frozen['per_movie'][stem][key]
            initial=base/(stem+'-original/pre-postprocess.json')
            assert sha(initial)==hashes[initial.relative_to(base).as_posix()]
            assert sha(references[stem]['path'])==references[stem]['sha256']
            inputs[stem]=dict(files=files,initial=initial,feature_bytes=files['-features.npz'].stat().st_size)
    assert len(inputs)==(47 if held=='6bba' else 13)
    ordered=sorted(inputs,key=lambda s:(inputs[s]['feature_bytes'],s))
    smoke_stems=[ordered[0],ordered[-1]]
    movies=smoke_stems if args.smoke or args.control_preflight else sorted(inputs)
    helper_names=('trajectory_event_fork16_inference_v1.py','trajectory_event_pruned_inference_v1.py',
        'trajectory_event_candidates_v1.py','trajectory_event_fast_features_v1.py','trajectory_event_dominance_v1.py',
        'trajectory_event_assignment_v1.py','trajectory_event_release_v1.py',
        'trajectory_event_features_v1.py','trajectory_disagreement_data_v1.py','trajectory_runtime_v1.py')
    helpers={n:sha(ROOT/'research'/n) for n in helper_names}
    smoke=None
    if not args.smoke and not args.control_preflight:
        smoke=read(reports/(smoke_name+'.json'))
        assert smoke['status']=='complete_movie_inference_smoke_passed'
        assert smoke['model_sha256']==sha(model) and smoke['helper_sha256']==helpers and smoke['source_sha256']==sha(Path(__file__))
        assert smoke['movies']==smoke_stems and not smoke['ground_truth_opened']
    preflight=None
    preflight_path=reports/(preflight_name+'.json')
    if args.smoke and preflight_path.exists():
        preflight=read(preflight_path)
        assert preflight['status']=='untrained_control_preflight_passed' and preflight['model_sha256'] is None
        assert preflight['helper_sha256']==helpers and preflight['source_sha256']==sha(Path(__file__))
        assert preflight['both_frozen_fit_contracts']==frozen_fit_contracts
        assert preflight['source_scope_receipts']==scopes
        assert preflight['movies']==smoke_stems and not preflight['ground_truth_opened']
    target.mkdir()
    result=dict(status='freezing_predictions',fit_embryo=args.fit_embryo,held_embryo=held,source_only=True,
        movies=movies,all_held_movies=len(inputs),smoke_only=args.smoke,control_preflight=args.control_preflight,
        model_sha256=sha(model) if model is not None else None,
        fit_result_sha256=sha(fit_path),fit_contract_sha256=sha(fit_root/'CONTRACT.json'),
        both_frozen_fit_contracts=frozen_fit_contracts,
        helper_sha256=helpers,source_sha256=sha(Path(__file__)),source_scope_receipts=scopes,
        selection_or_validation_opened=False,ground_truth_opened=False,model_refitted=False,
        prior_backbone_and_initializer_exposure_disclosed=True,authorized_for_submission=False,
        local_diagnostic_per_frame_seconds=10,local_diagnostic_stage_seconds=600,
        kaggle_default_deadline_acceptance_not_established=True,inference={})
    def persist():
        result['seconds']=time.monotonic()-started
        text=json.dumps(result,indent=2,allow_nan=False)+'\n'
        for path in (receipt,target/'RESULT.json'):
            temporary=path.with_name(path.name+'.tmp')
            temporary.write_text(text,encoding='utf-8');os.replace(temporary,path)
    persist()
    try:
        for stem in movies:
            item=inputs[stem];initial=read(item['initial']);base=read(item['files']['-prediction.json'])
            groups=arrays(item['files']['-groups.npz']);edge=arrays(item['files']['-features.npz'])
            assert list(edge['feature_names'])==list(FEATURES[:18])
            result['inference'][stem]={}
            for arm,w in weights.items():
                path=target/(stem+'-'+arm+'.json')
                reused_preflight=False
                if smoke is not None and stem in smoke['inference']:
                    old=smoke['inference'][stem][arm]
                    old_path=ROOT/'.biohub/cache'/smoke_name/(stem+'-'+arm+'.json')
                    assert sha(old_path)==old['prediction_sha256']
                    candidate,details=read(old_path),old['details']
                    reused=True
                elif preflight is not None and arm=='untrained16':
                    old=preflight['inference'][stem][arm]
                    old_path=ROOT/'.biohub/cache'/preflight_name/(stem+'-'+arm+'.json')
                    assert sha(old_path)==old['prediction_sha256']
                    candidate,details=read(old_path),old['details']
                    reused=False;reused_preflight=True
                else:
                    candidate,details=refine(initial,base,groups,edge['features'],w,per_frame_seconds=10,max_seconds=600)
                    reused=False
                validate_graph(candidate,100);check_event_stage(initial,base,candidate,details,frames=100)
                assert not details['budget_exhausted'] and not details['solver_fallbacks'],'Incomplete exact event solve: '+stem+'/'+arm
                path.write_text(json.dumps(candidate,sort_keys=True,allow_nan=False),encoding='utf-8')
                result['inference'][stem][arm]=dict(prediction_sha256=sha(path),details=details,
                    reused_smoke=reused,reused_control_preflight=reused_preflight)
                persist()
            print(json.dumps(dict(event='held_movie_frozen',stem=stem)),flush=True)
        assert all(sha(ROOT/'research'/n)==digest for n,digest in helpers.items())
        assert fixed_fit_contracts(ROOT)==frozen_fit_contracts
        result['all_requested_predictions_frozen_before_scoring']=True
        if args.control_preflight:
            result['status']='untrained_control_preflight_passed';persist();return
        if args.smoke:
            result['status']='complete_movie_inference_smoke_passed';persist();return
        helper=runpy.run_path(str(ROOT/'scripts/score-public-d4-full-movie-v1.py'));scorer=helper['load_scorer']()
        import tracksdata as td
        from geff import GeffMetadata
        truth_root=ROOT/'.biohub/cache/competition-train-geffs-packed-v1'
        manifest=truth_root/'train_geff_cache_manifest.json'
        assert sha(manifest)=='744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9'
        files=read(manifest)['files'];rows={arm:[] for arm in ('submitted8','untrained16','learned16')}
        result['ground_truth_opened']=True;persist()
        for stem in movies:
            gt_files=[r for r in files if r['relative_path'].startswith(stem+'.geff/')];assert len(gt_files)==21
            for row in gt_files:assert sha(truth_root/'train'/row['relative_path'])==row['sha256']
            gt=truth_root/'train'/(stem+'.geff')
            for arm in rows:
                path=references[stem]['path'] if arm=='submitted8' else target/(stem+'-'+arm+'.json')
                expected=references[stem]['sha256'] if arm=='submitted8' else result['inference'][stem][arm]['prediction_sha256']
                assert sha(path)==expected
                truth=td.graph.IndexedRXGraph.from_geff(str(gt))[0];pred=helper['prediction_graph'](read(path))
                scored=scorer.evaluate(pred,truth,scale=(1.625,.40625,.40625),max_distance=7.)
                count=float(GeffMetadata.read(str(gt)).extra['estimated_number_of_nodes'])
                row=dict(scorer.per_sample_metrics(scored,count,scorer.node_recall(pred,truth)),stem=stem,embryo=held)
                if arm=='submitted8':assert abs(scorer.summarise([row])['score']-references[stem]['score'])<1e-12
                rows[arm].append(row)
            print(json.dumps(dict(event='held_movie_scored',stem=stem)),flush=True)
        summaries={a:scorer.summarise(r) for a,r in rows.items()}
        movies_summary={a:{r['stem']:scorer.summarise([r]) for r in rs} for a,rs in rows.items()}
        checks,regressions=quality_checks(summaries,movies_summary,movies)
        assert fixed_fit_contracts(ROOT)==frozen_fit_contracts
        result.update(status='held_embryo_complete_movie_scoring_finished',summaries=helper['finite'](summaries),
            per_movie_summaries=helper['finite'](movies_summary),rows=helper['finite'](rows),regressing_movies=regressions,
            quality_checks=checks,strict_quality_checks_pass=all(checks.values()),
            aggregate_diagnostic_pass=all(v for k,v in checks.items() if k!='every_movie_nonregressing'),
            independent_selection_test_authorized=False)
        persist();print(json.dumps(dict(status=result['status'],summaries=result['summaries'],checks=checks)),flush=True)
    except BaseException as error:
        result.update(status='evaluation_failed_requires_inspection',error=repr(error));persist();raise


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
