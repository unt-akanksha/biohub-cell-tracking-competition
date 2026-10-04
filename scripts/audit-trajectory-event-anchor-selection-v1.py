"""One fixed, source-selected anchor head on all ten role-disjoint selection movies."""
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.trajectory_runtime_v1 import sha, validate_graph
from research.trajectory_event_pruned_inference_v1 import refine


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def quality_checks(summaries, per_movie, by_embryo):
    """Never silently exclude an unscorable movie or erase a subgroup failure."""
    base, new = summaries['baseline'], summaries['anchor']
    all_summaries = list(summaries.values())
    all_summaries += [v for arm in per_movie.values() for v in arm.values()]
    all_summaries += [v for arms in by_embryo.values() for v in arms.values()]
    return dict(
        finite_complete_scoring=all(s['n'] == s['n_adj'] and all(
            isinstance(s[k], (int, float)) and np.isfinite(s[k])
            for k in ('score', 'edge_jaccard', 'adj_edge_jaccard', 'node_recall'))
            for s in all_summaries),
        pooled_score_improves=new['score'] > base['score'] + 1e-12,
        raw_edge_jaccard_nonregressing=new['edge_jaccard'] >= base['edge_jaccard'] - 1e-12,
        every_movie_nonregressing=all(per_movie['anchor'][s]['score'] >= b['score'] - 1e-12
                                      for s, b in per_movie['baseline'].items()),
        every_embryo_nonregressing=all(v['anchor']['score'] >= v['baseline']['score'] - 1e-12
                                      for v in by_embryo.values()),
        division_counts_nonregressing=(new['division_tp'] >= base['division_tp']
            and new['division_fp'] <= base['division_fp'] and new['division_fn'] <= base['division_fn']))


def main():
    start = time.monotonic()
    name = 'trajectory-event-anchor-selection-v1'
    target = ROOT / '.biohub/cache' / name
    receipt = ROOT / 'reports/experiments' / (name + '.json')
    assert not target.exists() and not receipt.exists()
    source_report = ROOT / 'reports/experiments/trajectory-event-anchor-source-v1.json'
    assert sha(source_report) == 'e1667c04fd1dd83d18932f63981e98e54ff0a2f72873932f55216987131b44fd'
    source = read(source_report)
    assert source['parameter_key'] == 'anchor' and not source['event_head_coefficients_fitted']
    for stem, base in source['per_movie_summaries']['baseline'].items():
        assert source['per_movie_summaries']['event_smoke'][stem]['score'] >= base['score'] - 1e-12
    model = ROOT / '.biohub/cache/trajectory-event-learning-smoke-v1/epoch-3.npz'
    assert sha(model) == '508165d131dc0c2516cf91e0eafb5a88edf24358f6ced5b0436f29ccc6b733be'
    weights = arrays(model)['anchor']
    assert weights.shape == (30,) and np.isfinite(weights).all()
    runtime = ROOT / 'research/trajectory_event_pruned_inference_v1.py'
    assert sha(runtime) == '281d42bcc9bb01dcc7695b765924d5c956e9094d05d6cdfcec105ef9e08d6b9d'
    folder = ROOT / '.biohub/cache/trajectory-event-selection-v1-features'
    assert sha(folder / 'RESULT.json') == '75258e792ca5eeb2d9763c20a43d2478ee55aa31ddbef0ef2ea596aa4f660d11'
    inputs = read(folder / 'RESULT.json')
    assert inputs['source_role_disjoint'] and not inputs['selection_labels_used']
    scope = ROOT / '.biohub/cache/trajectory-ranker-selection-v1-plan/MOVIES.json'
    assert sha(scope) == inputs['scope_sha256']
    movies = read(scope)['movies']
    assert len(movies) == 10 and all(m['role'] == 'selection' for m in movies)
    baseline_root = ROOT / '.biohub/cache/trajectory-ranker-selection-v1-full-output'
    previous = ROOT / '.biohub/cache/trajectory-structured-selection-v1/RESULT.json'
    assert sha(previous) == inputs['baseline_score_receipt_sha256']
    expected = read(previous)['summaries']['structured']
    target.mkdir()
    graphs, evidence = {}, {}
    report = dict(status='running',model_sha256=sha(model),parameter_key='anchor',
        source_diagnostic_sha256=sha(source_report),input_receipt_sha256=sha(folder/'RESULT.json'),
        inference_sha256=sha(runtime),source_sha256=sha(Path(__file__)),blas_thread_limit=1,
        selection_used_for_training=False,source_role_disjoint=True,
        prior_pipeline_selection_exposure_disclosed=True,public_backbone_independence_not_established=True,
        source_selected_fixed_variant=True,movie_id_routing=False,authorized_for_submission=False)

    def persist():
        report.update(seconds=time.monotonic()-start,inference=evidence)
        text=json.dumps(report,indent=2,allow_nan=False)+'\n'
        (target/'RESULT.json').write_text(text,encoding='utf-8')
        receipt.write_text(text,encoding='utf-8')

    persist()
    try:
        for movie in movies:
            stem=movie['stem']; frozen=inputs['per_movie'][stem]
            paths={key:folder/(stem+suffix) for key,suffix in (
                ('prediction','-prediction.json'),('groups','-groups.npz'),('features','-features.npz'))}
            for key,path in paths.items():assert sha(path)==frozen[key+'_sha256']
            initial=baseline_root/(stem+'-original/pre-postprocess.json')
            assert sha(initial)==frozen['initial_sha256']
            base=read(paths['prediction']);validate_graph(base,100)
            pred,details=refine(read(initial),base,arrays(paths['groups']),arrays(paths['features'])['features'],weights)
            validate_graph(pred,100);assert pred['nodes']==base['nodes']
            path=target/(stem+'-prediction.json')
            path.write_text(json.dumps(pred,sort_keys=True,allow_nan=False),encoding='utf-8')
            evidence[stem]=dict(details,prediction_sha256=sha(path),baseline_sha256=sha(paths['prediction']))
            graphs[stem]=dict(baseline=base,anchor=pred);persist()
            print(json.dumps(dict(event='selection_movie_frozen',stem=stem,
                **{k:v for k,v in details.items() if k!='frames'})),flush=True)
        # All predictions are frozen before loading any selection annotation.
        report['all_predictions_frozen_before_scoring']=True;persist()
        helper=runpy.run_path(str(ROOT/'scripts/score-public-d4-full-movie-v1.py'))
        scorer=helper['load_scorer']()
        import tracksdata as td
        from geff import GeffMetadata
        truth_root=ROOT/'.biohub/cache/competition-train-geffs-packed-v1'
        manifest=truth_root/'train_geff_cache_manifest.json'
        assert sha(manifest)=='744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9'
        files=read(manifest)['files'];rows=dict(baseline=[],anchor=[])
        for stem in graphs:
            selected=[r for r in files if r['relative_path'].startswith(stem+'.geff/')]
            assert len(selected)==21
            for r in selected:assert sha(truth_root/'train'/r['relative_path'])==r['sha256']
            path=truth_root/'train'/(stem+'.geff')
            for arm in rows:
                truth=td.graph.IndexedRXGraph.from_geff(str(path))[0]
                pred=helper['prediction_graph'](graphs[stem][arm])
                result=scorer.evaluate(pred,truth,scale=(1.625,.40625,.40625),max_distance=7.)
                count=float(GeffMetadata.read(str(path)).extra['estimated_number_of_nodes'])
                rows[arm].append(dict(scorer.per_sample_metrics(result,count,scorer.node_recall(pred,truth)),
                                      stem=stem,embryo=stem.split('_')[0]))
            print(json.dumps(dict(event='selection_movie_scored',stem=stem)),flush=True)
        totals={a:scorer.summarise(v) for a,v in rows.items()}
        for key in expected:assert np.isclose(totals['baseline'][key],expected[key],rtol=0,atol=1e-12),key
        per_movie={a:{r['stem']:scorer.summarise([r]) for r in v} for a,v in rows.items()}
        by_embryo={e:{a:scorer.summarise([r for r in v if r['embryo']==e]) for a,v in rows.items()}
                   for e in ('44b6','6bba')}
        checks=quality_checks(totals,per_movie,by_embryo)
        report.update(status='fixed_anchor_selection_diagnostic_complete',rows=helper['finite'](rows),
            summaries=helper['finite'](totals),per_movie_summaries=helper['finite'](per_movie),
            by_embryo=helper['finite'](by_embryo),quality_checks=checks,quality_checks_pass=all(checks.values()),
            all_refinement_transitions_completed_without_fallback=all(r['processed_frames']==99
                and not r['solver_fallbacks'] and not r['budget_exhausted'] for r in evidence.values()))
        persist();print(json.dumps(dict(status=report['status'],summaries=report['summaries'],checks=checks)),flush=True)
    except BaseException as error:
        report.update(status='failed',error=repr(error));persist();raise


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
