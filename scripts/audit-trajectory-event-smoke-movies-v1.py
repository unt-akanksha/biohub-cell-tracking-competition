"""Full source-movie diagnostic of smoke weights, never a promotion evaluation."""
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha, validate_graph
from research.trajectory_event_inference_v1 import refine


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path,allow_pickle=False) as data:return dict(data)


def main():
    start=time.monotonic();name='trajectory-event-smoke-movies-v1'
    target=ROOT/'.biohub/cache'/name;receipt=ROOT/'reports/experiments'/(name+'.json')
    assert not target.exists() and not receipt.exists()
    fit_root=ROOT/'.biohub/cache/trajectory-event-learning-smoke-v1';fit=read(fit_root/'RESULT.json')
    assert fit['status']=='real_source_event_training_smoke_complete' and fit['source_only']
    weights_path=fit_root/'epoch-3.npz';assert sha(weights_path)==fit['final_weights_sha256']
    weights=arrays(weights_path)['weights'];stems=fit['stems']
    feature_root=ROOT/'.biohub/cache/trajectory-event-source-v1-b0-features'
    frozen=read(feature_root/'RESULT.json');assert sha(feature_root/'RESULT.json')==fit['feature_receipt_sha256']
    source=ROOT/'.biohub/cache/trajectory-event-source-v1-b0-full-output'
    backup=read(ROOT/'reports/experiments/trajectory-event-source-v1-b0-full-harvest.json')
    assert backup['status']=='verified_backup';hashes={r['path']:r['sha256'] for r in backup['records']}
    target.mkdir();graphs={};evidence={}
    report=dict(status='running',source_only=True,training_movies_reused=True,independently_held_out=False,
                authorized_for_submission=False,quality_gain_established=False,
                model_sha256=sha(weights_path),fit_receipt_sha256=sha(fit_root/'RESULT.json'),
                source_sha256=sha(Path(__file__)),inference_sha256=sha(ROOT/'research/trajectory_event_inference_v1.py'))
    def persist():
        report.update(seconds=time.monotonic()-start,inference=evidence)
        text=json.dumps(report,indent=2,allow_nan=False)+'\n'
        (target/'RESULT.json').write_text(text,encoding='utf-8');receipt.write_text(text,encoding='utf-8')
    persist()
    try:
        for stem in stems:
            row=frozen['per_movie'][stem]
            pp=feature_root/(stem+'-prediction.json');gp=feature_root/(stem+'-groups.npz');fp=feature_root/(stem+'-features.npz')
            ip=source/(stem+'-original/pre-postprocess.json')
            assert sha(pp)==row['prediction_sha256'] and sha(gp)==row['groups_sha256'] and sha(fp)==row['features_sha256']
            assert sha(ip)==hashes[ip.relative_to(source).as_posix()]
            baseline=read(pp);validate_graph(baseline,100)
            prediction,details=refine(read(ip),baseline,arrays(gp),arrays(fp)['features'],weights)
            validate_graph(prediction,100);assert prediction['nodes']==baseline['nodes']
            path=target/(stem+'-prediction.json')
            path.write_text(json.dumps(prediction,sort_keys=True,allow_nan=False),encoding='utf-8')
            evidence[stem]=dict(details,prediction_sha256=sha(path),baseline_sha256=sha(pp))
            graphs[stem]=dict(baseline=baseline,event_smoke=prediction);persist()
            print(json.dumps(dict(event='full_movie_frozen',stem=stem,**{k:v for k,v in details.items() if k!='frames'})),flush=True)
        # Both complete graphs are persisted before fresh official source scoring.
        helper=runpy.run_path(str(ROOT/'scripts/score-public-d4-full-movie-v1.py'));scorer=helper['load_scorer']()
        import tracksdata as td
        from geff import GeffMetadata
        truth_root=ROOT/'.biohub/cache/competition-train-geffs-packed-v1'
        manifest=truth_root/'train_geff_cache_manifest.json'
        assert sha(manifest)=='744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9'
        truth_files=read(manifest)['files'];rows=dict(baseline=[],event_smoke=[])
        capacity=read(ROOT/'reports/experiments/trajectory-event-source-v1-b0-capacity.json')
        for stem in stems:
            selected=[r for r in truth_files if r['relative_path'].startswith(stem+'.geff/')];assert len(selected)==21
            for row in selected:assert sha(truth_root/'train'/row['relative_path'])==row['sha256']
            path=truth_root/'train'/(stem+'.geff')
            for arm in rows:
                truth=td.graph.IndexedRXGraph.from_geff(str(path))[0]
                prediction=helper['prediction_graph'](graphs[stem][arm])
                result=scorer.evaluate(prediction,truth,scale=(1.625,.40625,.40625),max_distance=7.)
                if arm=='baseline':
                    for key in ('division_tp','division_fp','division_fn'):
                        assert getattr(result,key)==capacity['per_movie'][stem][key.replace('division_','official_division_')]
                count=float(GeffMetadata.read(str(path)).extra['estimated_number_of_nodes'])
                row=dict(scorer.per_sample_metrics(result,count,scorer.node_recall(prediction,truth)),stem=stem,embryo=stem.split('_')[0])
                rows[arm].append(row)
                print(json.dumps(helper['finite'](dict(event='source_scored',arm=arm,**row))),flush=True)
        report.update(status='full_source_smoke_diagnostic_complete',per_movie=helper['finite'](rows),
            summaries=helper['finite']({arm:scorer.summarise(values) for arm,values in rows.items()}),
            per_movie_summaries=helper['finite']({arm:{r['stem']:scorer.summarise([r]) for r in values} for arm,values in rows.items()}),
            full_100_frame_graphs=True,ground_truth_opened=True,selection_or_validation_opened=False)
        persist();print(json.dumps(dict(status=report['status'],summaries=report['summaries'],seconds=report['seconds'])),flush=True)
    except BaseException as error:
        report.update(status='failed',error=repr(error));persist();raise


if __name__=='__main__':main()
