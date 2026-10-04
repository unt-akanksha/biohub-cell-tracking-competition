"""Frozen conditional correction, complete-source current-official comparison."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.focus_conditional_variance_linking import link
from research.focus_conditional_motion_linking import link as constant_link
BASE=runpy.run_path(str(ROOT/'scripts/score-focus-division-preserving-assignment.py'))
SOURCE=BASE['SOURCE'];FULL=SOURCE['FULL'];RUN='focus-conditional-variance-source-v1'
FILES=['scripts/score-focus-conditional-variance.py','research/focus_conditional_variance_linking.py','research/focus_conditional_motion_linking.py',
    'research/focus_conditional_variance.py','research/focus_conditional_motion.py','research/focus_residual_calibration.py','scripts/verify-focus-conditional-variance.py',
    'scripts/score-focus-division-preserving-assignment.py','research/focus_source_flow_comparison.py',
    'scripts/score-focus-source-flow.py',f'reports/experiments/{RUN}-design.md']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import tracksdata as td
    from geff import GeffMetadata
    started=time.monotonic();cache=ROOT/'.biohub/cache'/RUN;target=ROOT/f'reports/experiments/{RUN}-result.json'
    if cache.exists() or target.exists():raise ValueError('Never overwrite fixed source experiment')
    frozen={p:sha(ROOT/p) for p in FILES}
    verified,model=runpy.run_path(str(ROOT/'scripts/verify-focus-conditional-variance.py'))['verify']()
    receipt=ROOT/'reports/experiments/focus-conditional-variance-v1-verification.json'
    if json.loads(receipt.read_text())!=verified:raise ValueError('Actual training-held-out verification required')
    refpath=ROOT/'reports/experiments/focus-source-flow-v1-result.json';priorpath=ROOT/'reports/experiments/focus-conditional-motion-source-v1-result.json'
    if sha(refpath)!=BASE['REFERENCE_SHA'] or sha(priorpath)!='998e43b168cd4402a081349d48b0cea9a0843783e72f22000c5e7c7c4ce8b400':raise ValueError('Frozen controls changed')
    reference=json.loads(refpath.read_text());prior=json.loads(priorpath.read_text())
    folder=ROOT/'.biohub/cache/kernel-outputs/focus-source-flow-v1/focus_source_flow'
    prepared,parents,policy=SOURCE['prepare'](folder,ROOT/'kaggle/biohub-focus-source-flow-v1/biohub-focus-source-flow-v1.ipynb',BASE['NOTEBOOK_SHA'])
    metric=SOURCE['SCORER']['load_scorer'](ROOT/'.biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot')
    cache.mkdir();records=[]
    for stem in policy['source_stems']:
        sample=folder/'outputs'/stem/'sampled_flow.npz'
        with np.load(sample,allow_pickle=False) as saved:coords,flow=saved['coords'].copy(),saved['backward_um'].copy()
        edges=link(coords,flow,model);path=cache/(stem+'.npz')
        np.savez_compressed(path,coords=coords,edges=np.asarray([(s,d) for s,d,p in edges],dtype=np.int64).reshape(-1,2))
        records.append(dict(stem=stem,sample_sha256=sha(sample),candidate_sha256=sha(path),nodes=len(coords),edges=len(edges)))
    manifest=dict(run_id=RUN,model_sha256=verified['final_model_sha256'],training_verification_sha256=sha(receipt),records=records,source_hashes=frozen,all_predictions_saved_before_source_gt=True)
    manifest_path=cache/'prelabel_manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2))
    rows={a:[] for a in ('parent','control','constant','candidate')}
    for record in records:
        stem=record['stem'];path=cache/(stem+'.npz');sample=folder/'outputs'/stem/'sampled_flow.npz'
        if sha(path)!=record['candidate_sha256'] or sha(sample)!=record['sample_sha256']:raise ValueError('Persisted prediction or original motion changed')
        with np.load(path,allow_pickle=False) as saved:coords,edges=saved['coords'].copy(),saved['edges'].copy()
        with np.load(sample,allow_pickle=False) as saved:
            if not np.array_equal(coords,saved['coords']):raise ValueError('Raw node geometry changed')
            expected=link(coords,saved['backward_um'],model)
            constant_expected=constant_link(coords,saved['backward_um'],model['mean_model'])
        candidate=BASE['graph_from_arrays'](coords,edges);SOURCE['verify_graph'](candidate,coords,expected)
        truth_path=ROOT/'.biohub/cache/competition-train-geffs-packed-v1/train'/(stem+'.geff')
        for arm in rows:
            graph=parents[stem] if arm=='parent' else td.graph.IndexedRXGraph.from_geff(str(prepared[stem]['candidate']))[0] if arm=='control' else BASE['graph_from_arrays'](coords,np.asarray([(s,d) for s,d,p in constant_expected],dtype=np.int64).reshape(-1,2)) if arm=='constant' else candidate
            truth=td.graph.IndexedRXGraph.from_geff(str(truth_path))[0]
            er=metric.evaluate(graph,truth,scale=(1.625,.40625,.40625),max_distance=7.)
            total=float(GeffMetadata.read(str(truth_path)).extra['estimated_number_of_nodes'])
            row=dict(metric.per_sample_metrics(er,total,SOURCE['SCORER']['diagnostic_node_recall'](metric,graph,truth)),stem=stem,embryo='6bba')
            rows[arm].append(row);print(json.dumps(dict(stage='source_scored',arm=arm,**row)),flush=True)
    summaries={a:metric.summarise(r) for a,r in rows.items()};comparison=BASE['comparison'](rows,summaries,reference)
    if rows['constant']!=prior['per_movie']['candidate'] or summaries['constant']!=prior['summaries']['candidate']:raise ValueError('Exact full-source constant uncertainty replay required')
    previous_deltas={k:summaries['candidate'][k]-prior['summaries']['candidate'][k] for k in ('score','edge_jaccard','division_tp')}
    conditions=dict(original_source_and_flow_gates=comparison['source_gate_passed'],
        improves_previous_calibrated_score=previous_deltas['score']>0,improves_previous_calibrated_raw_jaccard=previous_deltas['edge_jaccard']>0,
        previous_true_divisions_preserved=previous_deltas['division_tp']>=0)
    if frozen!={p:sha(ROOT/p) for p in FILES}:raise ValueError('Sources changed during evaluation')
    result=dict(status='completed_conditional_variance_source_comparison',run_id=RUN,model=model,model_sha256=verified['final_model_sha256'],
        training_verification_sha256=sha(receipt),per_movie=rows,summaries=summaries,by_embryo={a:{'6bba':metric.summarise(r)} for a,r in rows.items()},
        comparison=comparison,previous_calibrated_deltas=previous_deltas,conditions=conditions,source_gate_passed=all(conditions.values()),
        prelabel_manifest_sha256=sha(manifest_path),source_hashes=frozen,elapsed_seconds=time.monotonic()-started,
        gpu_seconds=0,new_target_movies_opened=0,authorized_for_submission=False,
        caveat='Exposed source-development movies only; held-out correction evidence is not independent whole-system validation.')
    target.write_text(json.dumps(FULL['finite_json'](result),indent=2,allow_nan=False));print(json.dumps(dict(conditions=conditions,summaries=summaries,previous_calibrated_deltas=previous_deltas)),flush=True)


if __name__=='__main__':main()
