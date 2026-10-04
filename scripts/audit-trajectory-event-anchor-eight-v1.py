"""Fixed eight-movie diagnostic with the failed selection gates preserved."""
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
    start=time.monotonic();name='trajectory-event-anchor-eight-v1'
    target=ROOT/'.biohub/cache'/name;receipt=ROOT/'reports/experiments'/(name+'.json')
    assert not target.exists() and not receipt.exists()
    selection_path=ROOT/'reports/experiments/trajectory-event-anchor-selection-v1.json'
    assert sha(selection_path)=='43e94177fe98c2ade5925aa78fd5c892a420a7d8323d1dfa986c75885d240fcb'
    selection=read(selection_path)
    assert selection['status']=='fixed_anchor_selection_diagnostic_complete'
    assert not selection['quality_checks_pass'] and selection['quality_checks']['pooled_score_improves']
    assert selection['all_predictions_frozen_before_scoring'] and not selection['selection_used_for_training']
    check_source=ROOT/'scripts/audit-trajectory-event-anchor-selection-v1.py'
    assert sha(check_source)=='fabe250eba93033d94847dc53a7dbede8770ed2e9dbb6ec2e21c077fb5f0648b'
    quality_checks=runpy.run_path(str(check_source))['quality_checks']
    model=ROOT/'.biohub/cache/trajectory-event-learning-smoke-v1/epoch-3.npz'
    assert sha(model)==selection['model_sha256'] and selection['parameter_key']=='anchor'
    weights=arrays(model)['anchor'];assert weights.shape==(30,) and np.isfinite(weights).all()
    runtime=ROOT/'research/trajectory_event_pruned_inference_v1.py'
    assert sha(runtime)==selection['inference_sha256']
    folder=ROOT/'.biohub/cache/trajectory-event-eight-v1-features';inputs=read(folder/'RESULT.json')
    assert inputs['status']=='eight_event_inputs_frozen_no_new_labels'
    assert inputs['source_and_selection_role_disjoint'] and not inputs['validation_labels_used']
    previous=ROOT/'.biohub/cache/trajectory-structured-eight-v1/RESULT.json'
    assert sha(previous)==inputs['baseline_score_receipt_sha256']
    expected=read(previous)['summaries']['structured'];assert len(inputs['per_movie'])==8
    target.mkdir();graphs={};evidence={}
    report=dict(status='running',diagnostic_only=True,selection_failure_preserved=True,
        selection_quality_checks=selection['quality_checks'],selection_result_sha256=sha(selection_path),
        model_sha256=sha(model),parameter_key='anchor',model_or_threshold_refitted=False,
        input_receipt_sha256=sha(folder/'RESULT.json'),inference_sha256=sha(runtime),
        source_sha256=sha(Path(__file__)),blas_thread_limit=1,
        source_and_selection_role_disjoint=True,prior_pipeline_validation_exposure_disclosed=True,
        public_backbone_independence_not_established=True,authorized_for_submission=False)

    def persist():
        report.update(seconds=time.monotonic()-start,inference=evidence)
        text=json.dumps(report,indent=2,allow_nan=False)+'\n'
        (target/'RESULT.json').write_text(text,encoding='utf-8');receipt.write_text(text,encoding='utf-8')

    persist()
    try:
        for stem,frozen in inputs['per_movie'].items():
            pp=folder/(stem+'-prediction.json');gp=folder/(stem+'-groups.npz');fp=folder/(stem+'-features.npz')
            ip=(ROOT/frozen['initial_path']).resolve()
            assert ip.is_relative_to(ROOT/'.biohub/cache/trajectory-overlap-cache-kaggle-v1-output')
            for path,key in ((pp,'prediction'),(gp,'groups'),(fp,'features'),(ip,'initial')):
                assert sha(path)==frozen[key+'_sha256']
            base=read(pp);validate_graph(base,100)
            pred,details=refine(read(ip),base,arrays(gp),arrays(fp)['features'],weights)
            validate_graph(pred,100);assert pred['nodes']==base['nodes']
            path=target/(stem+'-prediction.json')
            path.write_text(json.dumps(pred,sort_keys=True,allow_nan=False),encoding='utf-8')
            graphs[stem]=dict(baseline=base,anchor=pred)
            evidence[stem]=dict(details,prediction_sha256=sha(path),baseline_sha256=sha(pp));persist()
            print(json.dumps(dict(event='validation_movie_frozen',stem=stem,
                **{k:v for k,v in details.items() if k!='frames'})),flush=True)
        report['all_predictions_frozen_before_scoring']=True;persist()
        helper=runpy.run_path(str(ROOT/'scripts/score-public-d4-full-movie-v1.py'));scorer=helper['load_scorer']()
        import tracksdata as td
        from geff import GeffMetadata
        truth_root=ROOT/'.biohub/cache/competition-train-geffs-packed-v1'
        manifest=truth_root/'train_geff_cache_manifest.json'
        assert sha(manifest)=='744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9'
        files=read(manifest)['files'];rows=dict(baseline=[],anchor=[])
        for stem in graphs:
            selected=[r for r in files if r['relative_path'].startswith(stem+'.geff/')];assert len(selected)==21
            for r in selected:assert sha(truth_root/'train'/r['relative_path'])==r['sha256']
            path=truth_root/'train'/(stem+'.geff')
            for arm in rows:
                truth=td.graph.IndexedRXGraph.from_geff(str(path))[0]
                pred=helper['prediction_graph'](graphs[stem][arm])
                result=scorer.evaluate(pred,truth,scale=(1.625,.40625,.40625),max_distance=7.)
                count=float(GeffMetadata.read(str(path)).extra['estimated_number_of_nodes'])
                rows[arm].append(dict(scorer.per_sample_metrics(result,count,scorer.node_recall(pred,truth)),
                    stem=stem,embryo=stem.split('_')[0]))
            print(json.dumps(dict(event='validation_movie_scored',stem=stem)),flush=True)
        totals={a:scorer.summarise(v) for a,v in rows.items()}
        for key in expected:assert np.isclose(totals['baseline'][key],expected[key],rtol=0,atol=1e-12),key
        movies={a:{r['stem']:scorer.summarise([r]) for r in v} for a,v in rows.items()}
        embryos={e:{a:scorer.summarise([r for r in v if r['embryo']==e]) for a,v in rows.items()}
            for e in ('44b6','6bba')}
        checks=quality_checks(totals,movies,embryos)
        report.update(status='fixed_anchor_eight_diagnostic_complete',summaries=helper['finite'](totals),
            rows=helper['finite'](rows),per_movie_summaries=helper['finite'](movies),by_embryo=helper['finite'](embryos),
            quality_checks=checks,eight_movie_quality_checks_pass=all(checks.values()),
            overall_release_gates_pass=False,
            all_refinement_transitions_completed_without_fallback=all(r['processed_frames']==99
                and not r['solver_fallbacks'] and not r['budget_exhausted'] for r in evidence.values()))
        persist();print(json.dumps(dict(status=report['status'],summaries=report['summaries'],checks=checks)),flush=True)
    except BaseException as error:
        report.update(status='failed',error=repr(error));persist();raise


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
