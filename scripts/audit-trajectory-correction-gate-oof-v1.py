"""Fixed logistic gate with embryo-held-out source predictions and official scoring."""
import json
from pathlib import Path
import runpy
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha,validate_graph
from research.trajectory_correction_gate_v1 import apply_mask,FEATURES
from research.trajectory_correction_logistic_v1 import fit,predict


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def arrays(p):
    with np.load(p,allow_pickle=False) as data:return dict(data)


def main():
    started=time.monotonic();name='trajectory-correction-gate-oof-v1'
    target=ROOT/'.biohub/cache'/name;receipt=ROOT/'reports/experiments'/(name+'.json')
    assert not target.exists() and not receipt.exists()
    folder=ROOT/'.biohub/cache/trajectory-correction-source-v1';dataset=read(folder/'RESULT.json')
    assert dataset['status']=='source_correction_dataset_frozen' and len(dataset['records'])==60
    assert dataset['feature_names']==list(FEATURES) and not dataset['features_include_ids']
    assert sha(ROOT/'research/trajectory_correction_gate_v1.py')==dataset['gate_helper_sha256']
    data={}
    for stem,item in dataset['records'].items():
        fp=folder/(stem+'-features.npz');lp=folder/(stem+'-labels.json')
        assert sha(fp)==item['features_sha256'] and sha(lp)==item['labels_sha256']
        x=arrays(fp);assert list(x['feature_names'])==list(FEATURES)
        y=np.asarray([r['label'] for r in read(lp)])
        assert x['features'].shape==(len(y),len(FEATURES))
        data[stem]=dict(x=x['features'],y=y,embryo=item['embryo'])
    target.mkdir();models={};evidence={}
    report=dict(status='running',source_only=True,embryo_held_out_gate=True,
        prior_backbone_and_anchor_independence_not_established=True,
        model_family='standardized_L2_logistic',regularization=.1,threshold=.5,threshold_search=False,
        dataset_sha256=sha(folder/'RESULT.json'),source_sha256=sha(Path(__file__)),
        model_helper_sha256=sha(ROOT/'research/trajectory_correction_logistic_v1.py'),
        selection_or_validation_opened=False,authorized_for_submission=False)
    def persist():
        report.update(models=models,inference=evidence,seconds=time.monotonic()-started)
        text=json.dumps(report,indent=2,allow_nan=False)+'\n';(target/'RESULT.json').write_text(text,encoding='utf-8');receipt.write_text(text,encoding='utf-8')
    persist()
    try:
        for held_out in ('44b6','6bba'):
            fit_stems=sorted(s for s,r in data.items() if r['embryo']!=held_out)
            x=np.concatenate([data[s]['x'][data[s]['y']>=0] for s in fit_stems])
            y=np.concatenate([data[s]['y'][data[s]['y']>=0] for s in fit_stems])
            model=fit(x,y,regularization=.1)
            model.update(held_out_embryo=held_out,training_stems=fit_stems,feature_names=list(FEATURES))
            path=target/('held-out-'+held_out+'-model.json');path.write_text(json.dumps(model,indent=2)+'\n',encoding='utf-8')
            models[held_out]=dict(sha256=sha(path),training_movies=len(fit_stems),training_examples=len(y),class_counts=model['class_counts'])
            for stem in sorted(s for s,r in data.items() if r['embryo']==held_out):
                assert stem not in fit_stems
                item=dataset['records'][stem];baseline=ROOT/item['baseline_path'];candidate=ROOT/item['candidate_path']
                assert sha(baseline)==item['baseline_sha256'] and sha(candidate)==item['candidate_sha256']
                probability=predict(data[stem]['x'],model);mask=probability>=.5
                graph=apply_mask(read(ROOT/item['initial_path']),read(baseline),read(candidate),mask)
                validate_graph(graph,100)
                path=target/(stem+'-prediction.json');path.write_text(json.dumps(graph,sort_keys=True,allow_nan=False),encoding='utf-8')
                evidence[stem]=dict(prediction_sha256=sha(path),accepted=int(mask.sum()),rejected=int((~mask).sum()),held_out_embryo=held_out,
                    no_movie_labels_in_fit=True,gate_model_sha256=models[held_out]['sha256'])
                persist()
            print(json.dumps(dict(event='held_out_fold_frozen',embryo=held_out,**models[held_out])),flush=True)
        assert len(evidence)==60
        report['all_predictions_frozen_before_scoring']=True;persist()
        helper=runpy.run_path(str(ROOT/'scripts/score-public-d4-full-movie-v1.py'));scorer=helper['load_scorer']()
        import tracksdata as td
        from geff import GeffMetadata
        truth_root=ROOT/'.biohub/cache/competition-train-geffs-packed-v1';manifest=truth_root/'train_geff_cache_manifest.json'
        assert sha(manifest)=='744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9'
        files=read(manifest)['files'];rows=dict(baseline=[],anchor=[],gated=[])
        for index,name0 in ((0,'trajectory-event-anchor-source-v1'),(1,'trajectory-event-anchor-expanded-source-v1')):
            rp=ROOT/'reports/experiments'/(name0+'.json');assert sha(rp)==dataset['source_reports_sha256'][str(index)]
            previous=read(rp);prior=previous['per_movie'] if index==0 else previous['per_movie_rows']
            rows['baseline'].extend(prior['baseline']);rows['anchor'].extend(prior['event_smoke' if index==0 else 'anchor'])
        for stem,item in dataset['records'].items():
            matched=[r for r in files if r['relative_path'].startswith(stem+'.geff/')];assert len(matched)==21
            for r in matched:assert sha(truth_root/'train'/r['relative_path'])==r['sha256']
            truth_path=truth_root/'train'/(stem+'.geff');path=target/(stem+'-prediction.json')
            assert sha(path)==evidence[stem]['prediction_sha256']
            truth=td.graph.IndexedRXGraph.from_geff(str(truth_path))[0];pred=helper['prediction_graph'](read(path))
            evaluated=scorer.evaluate(pred,truth,scale=(1.625,.40625,.40625),max_distance=7.)
            count=float(GeffMetadata.read(str(truth_path)).extra['estimated_number_of_nodes'])
            rows['gated'].append(dict(scorer.per_sample_metrics(evaluated,count,scorer.node_recall(pred,truth)),stem=stem,embryo=item['embryo']))
            print(json.dumps(dict(event='oof_source_scored',stem=stem)),flush=True)
        assert all(len(v)==60 and len({r['stem'] for r in v})==60 for v in rows.values())
        summaries={a:scorer.summarise(v) for a,v in rows.items()}
        embryos={e:{a:scorer.summarise([r for r in v if r['embryo']==e]) for a,v in rows.items()} for e in ('44b6','6bba')}
        per_movie={a:{r['stem']:scorer.summarise([r]) for r in v} for a,v in rows.items()}
        checks=dict(finite_complete_scoring=all(s['n']==s['n_adj'] and np.isfinite(s['score']) for s in summaries.values()),
            improves_pooled_anchor=summaries['gated']['score']>summaries['anchor']['score'],
            both_embryos_nonregressing_vs_anchor=all(v['gated']['score']>=v['anchor']['score'] for v in embryos.values()),
            raw_edges_nonregressing=summaries['gated']['edge_jaccard']>=summaries['anchor']['edge_jaccard'],
            divisions_nonregressing=(summaries['gated']['division_tp']>=summaries['anchor']['division_tp'] and summaries['gated']['division_fp']<=summaries['anchor']['division_fp'] and summaries['gated']['division_fn']<=summaries['anchor']['division_fn']))
        report.update(status='source_oof_gate_scored',rows=helper['finite'](rows),summaries=helper['finite'](summaries),
            by_embryo=helper['finite'](embryos),per_movie_summaries=helper['finite'](per_movie),quality_checks=checks,
            eligible_for_separate_selection_test=all(checks.values()))
        persist();print(json.dumps(dict(status=report['status'],summaries=report['summaries'],checks=checks)),flush=True)
    except BaseException as error:
        report.update(status='failed',error=repr(error));persist();raise


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
