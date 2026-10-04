"""Complete-movie source diagnostic: seven fit movies and nine head-held-out movies."""
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha,validate_graph
from research.trajectory_event_pruned_inference_v1 import refine


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path,allow_pickle=False) as data:return dict(data)


def main():
    started=time.monotonic();name='trajectory-event-pilot-source-v2'
    target=ROOT/'.biohub/cache'/name;receipt=ROOT/'reports/experiments'/(name+'.json')
    assert not target.exists() and not receipt.exists()
    fit_root=ROOT/'.biohub/cache/trajectory-event-source-pilot-v2'
    fitted=read(fit_root/'RESULT.json');contract=read(fit_root/'CONTRACT.json')
    assert fitted['status']=='source_event_pilot_training_complete' and fitted['model_fitted']
    assert sha(fit_root/'CONTRACT.json')==fitted['contract_sha256'] and len(fitted['epochs_completed'])==3
    assert contract['epochs']==3 and contract['source_embryo']=='6bba' and contract['source_only']
    model=fit_root/'epoch-3.npz';assert sha(model)==fitted['weights_sha256']
    weights=arrays(model)['weights'];assert np.isfinite(weights).all() and weights.shape==(30,)
    runtime=ROOT/'reports/experiments/trajectory-event-pruned-runtime-v2.json';proof=read(runtime)
    assert proof['all_transitions_completed_without_fallback'] and proof['exact_on_shared_nonfallback_frames']
    assert sha(ROOT/'research/trajectory_event_pruned_inference_v1.py')==proof['fast_inference_sha256']
    fit_stems=set(contract['source_stems']);movies=[];inputs={}
    for batch in (0,1):
        prefix='trajectory-event-source-v1-b'+str(batch)
        folder=ROOT/'.biohub/cache'/(prefix+'-features');frozen=read(folder/'RESULT.json')
        scope=ROOT/'.biohub/cache/trajectory-event-source-v1-plan'/('batch-'+str(batch))/'MOVIES.json'
        plan=read(scope)
        assert sha(scope)==frozen['source_scope_sha256'] and frozen['status']=='current_candidate_source_features_frozen'
        assert all(m['role']=='optimization' for m in plan['movies'])
        base=ROOT/'.biohub/cache'/(prefix+'-full-output')
        backup=read(ROOT/'reports/experiments'/(prefix+'-full-harvest.json'))
        assert backup['status']=='verified_backup';hashes={r['path']:r['sha256'] for r in backup['records']}
        capacity=read(ROOT/'reports/experiments'/(prefix+'-capacity.json'))
        for item in plan['movies']:
            stem=item['stem'];assert stem not in inputs;movies.append(stem)
            inputs[stem]=dict(folder=folder,base=base,hashes=hashes,frozen=frozen['per_movie'][stem],
                              capacity=capacity['per_movie'][stem],scope_sha256=sha(scope),feature_receipt_sha256=sha(folder/'RESULT.json'))
    assert len(movies)==16 and len(fit_stems)==7 and fit_stems<=set(movies)
    target.mkdir();graphs={};inference={}
    report=dict(status='running',source_only=True,fit_movie_stems=sorted(fit_stems),
        event_head_unseen_movie_stems=sorted(set(movies)-fit_stems),model_sha256=sha(model),
        fit_contract_sha256=sha(fit_root/'CONTRACT.json'),fit_result_sha256=sha(fit_root/'RESULT.json'),
        runtime_proof_sha256=sha(runtime),inference_sha256=sha(ROOT/'research/trajectory_event_pruned_inference_v1.py'),
        source_sha256=sha(Path(__file__)),blas_thread_limit=1,model_refitted=False,
        selection_or_validation_opened=False,authorized_for_submission=False,quality_gain_established=False,
        public_backbone_independence_not_established=True)
    def persist():
        report.update(seconds=time.monotonic()-started,inference=inference)
        text=json.dumps(report,indent=2,allow_nan=False)+'\n'
        (target/'RESULT.json').write_text(text,encoding='utf-8');receipt.write_text(text,encoding='utf-8')
    persist()
    try:
        for stem in movies:
            item=inputs[stem];folder=item['folder'];frozen=item['frozen']
            pp=folder/(stem+'-prediction.json');gp=folder/(stem+'-groups.npz');fp=folder/(stem+'-features.npz')
            ip=item['base']/(stem+'-original/pre-postprocess.json')
            assert sha(pp)==frozen['prediction_sha256'] and sha(gp)==frozen['groups_sha256'] and sha(fp)==frozen['features_sha256']
            assert sha(ip)==item['hashes'][ip.relative_to(item['base']).as_posix()]
            baseline=read(pp);validate_graph(baseline,100)
            candidate,details=refine(read(ip),baseline,arrays(gp),arrays(fp)['features'],weights)
            validate_graph(candidate,100);assert candidate['nodes']==baseline['nodes']
            path=target/(stem+'-prediction.json');path.write_text(json.dumps(candidate,sort_keys=True,allow_nan=False),encoding='utf-8')
            inference[stem]=dict(details,prediction_sha256=sha(path),baseline_sha256=sha(pp),feature_receipt_sha256=item['feature_receipt_sha256'])
            graphs[stem]=dict(baseline=baseline,event_pilot=candidate);persist()
            print(json.dumps(dict(event='movie_frozen',stem=stem,head_held_out=stem not in fit_stems,
                **{k:v for k,v in details.items() if k!='frames'})),flush=True)
        helper=runpy.run_path(str(ROOT/'scripts/score-public-d4-full-movie-v1.py'));scorer=helper['load_scorer']()
        import tracksdata as td
        from geff import GeffMetadata
        truth_root=ROOT/'.biohub/cache/competition-train-geffs-packed-v1'
        manifest=truth_root/'train_geff_cache_manifest.json'
        assert sha(manifest)=='744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9'
        files=read(manifest)['files'];rows=dict(baseline=[],event_pilot=[])
        for stem in movies:
            selected=[r for r in files if r['relative_path'].startswith(stem+'.geff/')];assert len(selected)==21
            for r in selected:assert sha(truth_root/'train'/r['relative_path'])==r['sha256']
            path=truth_root/'train'/(stem+'.geff')
            for arm in rows:
                truth=td.graph.IndexedRXGraph.from_geff(str(path))[0];pred=helper['prediction_graph'](graphs[stem][arm])
                result=scorer.evaluate(pred,truth,scale=(1.625,.40625,.40625),max_distance=7.)
                if arm=='baseline':
                    for key in ('division_tp','division_fp','division_fn'):
                        assert getattr(result,key)==inputs[stem]['capacity'][key.replace('division_','official_division_')]
                count=float(GeffMetadata.read(str(path)).extra['estimated_number_of_nodes'])
                rows[arm].append(dict(scorer.per_sample_metrics(result,count,scorer.node_recall(pred,truth)),
                    stem=stem,embryo=stem.split('_')[0],head_held_out=stem not in fit_stems))
            print(json.dumps(dict(event='source_scored',stem=stem)),flush=True)
        def summaries(predicate):return {arm:scorer.summarise([r for r in values if predicate(r)]) for arm,values in rows.items()}
        total=summaries(lambda r:True);held=summaries(lambda r:r['head_held_out'])
        per_movie={arm:{r['stem']:scorer.summarise([r]) for r in values} for arm,values in rows.items()}
        regressions=[s for s in movies if per_movie['event_pilot'][s]['score']<per_movie['baseline'][s]['score']-1e-12]
        report.update(status='full_source_pilot_diagnostic_complete',per_movie_rows=helper['finite'](rows),
            summaries=helper['finite'](total),head_held_out_summaries=helper['finite'](held),
            fit_summaries=helper['finite'](summaries(lambda r:not r['head_held_out'])),
            by_embryo=helper['finite']({e:summaries(lambda r:r['embryo']==e) for e in ('44b6','6bba')}),
            per_movie_summaries=helper['finite'](per_movie),regressing_movies=regressions,
            all_refinement_transitions_completed_without_fallback=all(not r['budget_exhausted'] and not r['solver_fallbacks'] and r['processed_frames']==99 for r in inference.values()))
        persist();print(json.dumps(dict(status=report['status'],held_out=report['head_held_out_summaries'],regressions=regressions,seconds=report['seconds'])),flush=True)
    except BaseException as error:
        report.update(status='failed',error=repr(error));persist();raise


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
