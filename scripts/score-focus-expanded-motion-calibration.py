"""Fit one fourteen-movie motion Gaussian then score complete source movies."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.focus_cached_motion import assemble_flow
from research.focus_residual_calibration import matched_residuals,fit,link
from research.focus_adaptation_labels import node_matches
BASE=runpy.run_path(str(ROOT/'scripts/score-focus-division-preserving-assignment.py'))
SOURCE=BASE['SOURCE'];FULL=SOURCE['FULL']
RUN='focus-expanded-motion-calibration-v1'
FILES=['scripts/score-focus-expanded-motion-calibration.py','research/focus_cached_motion.py',
    'research/focus_residual_calibration.py','research/focus_adaptation_labels.py',
    'research/focus_source_flow_comparison.py','scripts/score-focus-division-preserving-assignment.py',
    'scripts/score-focus-source-flow.py',f'reports/experiments/{RUN}-design.md']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import tracksdata as td
    from geff import GeffMetadata
    started=time.monotonic();cache=ROOT/'.biohub/cache'/RUN
    target=ROOT/f'reports/experiments/{RUN}-result.json';fit_path=ROOT/f'reports/experiments/{RUN}-fit.json'
    if cache.exists() or target.exists() or fit_path.exists():raise ValueError('Never overwrite or rerun a fitted experiment')
    frozen={p:sha(ROOT/p) for p in FILES}
    oldfitpath=ROOT/'reports/experiments/focus-motion-residual-fit-v1.json'
    priorpath=ROOT/'reports/experiments/focus-residual-calibration-v1-result.json'
    refpath=ROOT/'reports/experiments/focus-source-flow-v1-result.json'
    if (sha(oldfitpath)!='fcef805eabaa616102a4588437d0ea7cdcd5ef1cb7a7eeaa6d77138a7834c066'
        or sha(priorpath)!='b765d6105b48f31969bf149342ba7f87ae846f46bc1c5c9831fcd04fede0526e'
        or sha(refpath)!=BASE['REFERENCE_SHA']):raise ValueError('Frozen training/source references changed')
    oldfit=json.loads(oldfitpath.read_text());prior=json.loads(priorpath.read_text());reference=json.loads(refpath.read_text())
    spec=runpy.run_path(str(ROOT/'scripts/build-focus-expanded-summary.py'))['make_spec']()
    fitting=oldfit['training_stems']+spec['contract']['fitting_stems']
    split=json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text())['folds'][0]
    excluded=set(spec['contract']['unchanged_diagnostic_stems']+reference['contract']['source_stems']+split['audit_order'])
    if len(fitting)!=14 or len(set(fitting))!=14 or set(fitting)&excluded or not set(fitting)<=set(split['train']):raise ValueError('Fourteen fixed training-only movies required')
    metric=SOURCE['SCORER']['load_scorer'](ROOT/'.biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot')
    residuals=[];training=[]
    def collect(stem,coords,flow,evidence):
        truth=td.graph.IndexedRXGraph.from_geff(str(ROOT/'.biohub/cache/competition-train-geffs-packed-v1/train'/(stem+'.geff')))[0]
        mapping=node_matches(coords,truth);keys=td.DEFAULT_ATTR_KEYS
        edges=list(truth.edge_attrs().select(keys.EDGE_SOURCE,keys.EDGE_TARGET).iter_rows())
        values=matched_residuals(coords,flow,mapping,edges)
        if len(values)<50:raise ValueError('At least50 ordinary unique training residuals per movie required')
        row=dict(stem=stem,residuals=len(values),mean_um=values.mean(0).tolist(),variance_um2=values.var(0).tolist(),evidence=evidence)
        residuals.append(values);training.append(row)
        print(json.dumps(dict(stage='training_residuals',**row)),flush=True)
    for row in oldfit['records']:
        sample=ROOT/'.biohub/cache/kernel-outputs/focus-owned-flow-full-v1/focus_owned_flow_full/outputs'/row['stem']/'sampled_flow.npz'
        if sha(sample)!=row['sample_sha256']:raise ValueError('Original training motion changed')
        with np.load(sample,allow_pickle=False) as saved:coords,flow=saved['coords'].copy(),saved['backward_um'].copy()
        collect(row['stem'],coords,flow,dict(sample_sha256=sha(sample)))
        actual=training[-1]
        if actual['residuals']!=row['residuals'] or any(not np.allclose(actual[k],row[k],rtol=0,atol=1e-12) for k in ('mean_um','variance_um2')):raise ValueError('Original two training movies did not replay')
    replay=fit(np.concatenate(residuals))
    if replay!=oldfit['parameters']:raise ValueError('Original pooled fit exact replay failed')
    roots=[ROOT/'.biohub/cache/kernel-outputs/focus-adaptation-features-v1/focus_adaptation_features/outputs',
        ROOT/'.biohub/cache/kernel-outputs/focus-extra-fit-features-v1/focus_extra_fit_features/outputs']
    for group,output in zip(spec['feature_groups'],roots):
        for row in group['feature_records']:
            if row['role']!='fitting':raise ValueError('Fitting-only feature records required')
            stem=row['stem'];raw=output/'raw_detections'/(stem+'.npz');manifest_path=output/stem/'manifest.json'
            if sha(raw)!=row['raw_sha256'] or sha(manifest_path)!=row['manifest_sha256']:raise ValueError('Verified raw/packet manifest changed')
            with np.load(raw,allow_pickle=False) as saved:coords=saved['coords'].copy()
            manifest=json.loads(manifest_path.read_text())
            def packets():
                for t,p in enumerate(manifest['pairs']):
                    if p['file']!=f'{t:03d}.npz':raise ValueError('Exact packet ordering required')
                    path=output/stem/p['file']
                    if sha(path)!=p['sha256']:raise ValueError('Feature packet changed')
                    with np.load(path,allow_pickle=False) as saved:yield {k:saved[k].copy() for k in saved.files}
            flow=assemble_flow(coords,packets())
            collect(stem,coords,flow,dict(raw_sha256=sha(raw),manifest_sha256=sha(manifest_path),recovered_flow_sha256=hashlib.sha256(flow.tobytes()).hexdigest()))
    if [r['stem'] for r in training]!=fitting:raise ValueError('Whole declared fitting inventory required')
    parameters=fit(np.concatenate(residuals))
    fitted=dict(run_id=RUN,parameters=parameters,training_stems=fitting,records=training,
        original_two_fit_exactly_replayed=True,feature_spec=spec,source_hashes=frozen,
        source_labels_used=False,diagnostic_labels_used=False,new_target_movies_opened=0)
    fit_path.write_text(json.dumps(fitted,indent=2,allow_nan=False))
    if json.loads(fit_path.read_text())!=fitted:raise ValueError('Fit round trip changed')
    print(json.dumps(dict(stage='fit_frozen',parameters=parameters,fit_sha256=sha(fit_path))),flush=True)
    folder=ROOT/'.biohub/cache/kernel-outputs/focus-source-flow-v1/focus_source_flow'
    prepared,parents,policy=SOURCE['prepare'](folder,ROOT/'kaggle/biohub-focus-source-flow-v1/biohub-focus-source-flow-v1.ipynb',BASE['NOTEBOOK_SHA'])
    cache.mkdir();records=[]
    for stem in policy['source_stems']:
        sample=folder/'outputs'/stem/'sampled_flow.npz'
        with np.load(sample,allow_pickle=False) as saved:coords,flow=saved['coords'].copy(),saved['backward_um'].copy()
        edges=link(coords,flow,parameters);path=cache/(stem+'.npz')
        np.savez_compressed(path,coords=coords,edges=np.asarray([(s,d) for s,d,p in edges],dtype=np.int64).reshape(-1,2))
        records.append(dict(stem=stem,sample_sha256=sha(sample),candidate_sha256=sha(path),nodes=len(coords),edges=len(edges)))
    manifest=dict(run_id=RUN,fit_sha256=sha(fit_path),records=records,source_hashes=frozen,all_predictions_saved_before_source_gt=True)
    manifest_path=cache/'prelabel_manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2))
    rows={a:[] for a in ('parent','control','candidate')}
    for record in records:
        stem=record['stem'];path=cache/(stem+'.npz');sample=folder/'outputs'/stem/'sampled_flow.npz'
        if sha(path)!=record['candidate_sha256'] or sha(sample)!=record['sample_sha256']:raise ValueError('Persisted predictions changed')
        with np.load(path,allow_pickle=False) as saved:coords,edges=saved['coords'].copy(),saved['edges'].copy()
        with np.load(sample,allow_pickle=False) as saved:
            if not np.array_equal(coords,saved['coords']):raise ValueError('Raw nodes changed')
            expected=link(coords,saved['backward_um'],parameters)
        candidate=BASE['graph_from_arrays'](coords,edges);SOURCE['verify_graph'](candidate,coords,expected)
        truth_path=ROOT/'.biohub/cache/competition-train-geffs-packed-v1/train'/(stem+'.geff')
        for arm in rows:
            graph=parents[stem] if arm=='parent' else td.graph.IndexedRXGraph.from_geff(str(prepared[stem]['candidate']))[0] if arm=='control' else candidate
            truth=td.graph.IndexedRXGraph.from_geff(str(truth_path))[0]
            er=metric.evaluate(graph,truth,scale=(1.625,.40625,.40625),max_distance=7.)
            total=float(GeffMetadata.read(str(truth_path)).extra['estimated_number_of_nodes'])
            row=dict(metric.per_sample_metrics(er,total,SOURCE['SCORER']['diagnostic_node_recall'](metric,graph,truth)),stem=stem,embryo='6bba')
            rows[arm].append(row);print(json.dumps(dict(stage='source_scored',arm=arm,**row)),flush=True)
    summaries={a:metric.summarise(r) for a,r in rows.items()};comparison=BASE['comparison'](rows,summaries,reference)
    previous_deltas={k:summaries['candidate'][k]-prior['summaries']['candidate'][k] for k in ('score','edge_jaccard','division_tp')}
    conditions=dict(original_source_and_flow_gates=comparison['source_gate_passed'],
        improves_previous_calibrated_score=previous_deltas['score']>0,improves_previous_calibrated_raw_jaccard=previous_deltas['edge_jaccard']>0,
        previous_true_divisions_preserved=previous_deltas['division_tp']>=0)
    if frozen!={p:sha(ROOT/p) for p in FILES}:raise ValueError('Sources changed while running')
    result=dict(status='completed_expanded_motion_calibration',run_id=RUN,parameters=parameters,training_records=training,
        per_movie=rows,summaries=summaries,by_embryo={a:{'6bba':metric.summarise(r)} for a,r in rows.items()},
        comparison=comparison,previous_calibrated_deltas=previous_deltas,conditions=conditions,source_gate_passed=all(conditions.values()),
        fit_sha256=sha(fit_path),prelabel_manifest_sha256=sha(manifest_path),source_hashes=frozen,
        elapsed_seconds=time.monotonic()-started,gpu_seconds=0,new_target_movies_opened=0,authorized_for_submission=False,
        caveat='Source movies are exposed development; no independent or leaderboard claim.')
    target.write_text(json.dumps(FULL['finite_json'](result),indent=2,allow_nan=False));print(json.dumps(dict(conditions=conditions,summaries=summaries,previous_calibrated_deltas=previous_deltas)),flush=True)


if __name__=='__main__':main()
