"""Fit only two training caches, freeze predictions, then source comparison."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.focus_residual_calibration import TRAIN_STEMS,matched_residuals,fit,link
SOURCE=runpy.run_path(str(ROOT/'scripts/score-focus-source-flow.py'))
FULL=runpy.run_path(str(ROOT/'scripts/score-focus-owned-flow-full.py'))
BASE=runpy.run_path(str(ROOT/'scripts/score-focus-division-preserving-assignment.py'))
RUN='focus-residual-calibration-v1'
REF_SHA='896d3de86fa30fcd860f2c16073d47718118d94e1a43fe70710626c7b32ea169'
NB_SHA='d36fd6ef5fb460ae4964429aded2832b0d44db91a65836476ecea64639f30375'
TRAIN_REF_SHA='079abeeccec5f7a76e27b40c88063fae30bca4a155f3dae3bf3bc0311e4f6685'
TRAIN_NB_SHA='5af97d02de71448af3d4f2367db307a77212fb59ae0d1389f0a6beffb7e3db9f'
FILES=['research/focus_residual_calibration.py','scripts/score-focus-residual-calibration.py',
       'scripts/score-focus-source-flow.py','scripts/score-focus-owned-flow-full.py',
       'scripts/score-focus-division-preserving-assignment.py',
       'research/focus_source_flow_comparison.py','reports/experiments/focus-residual-calibration-v1-design.md']


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np
    import tracksdata as td
    from geff import GeffMetadata
    started=time.monotonic()
    cache=ROOT/'.biohub/cache'/RUN
    fit_path=ROOT/'reports/experiments/focus-motion-residual-fit-v1.json'
    result_path=ROOT/'reports/experiments'/f'{RUN}-result.json'
    if cache.exists() or fit_path.exists() or result_path.exists():
        raise ValueError('Existing experiment artifacts: no overwrite or refit')
    frozen={p:sha(ROOT/p) for p in FILES}
    ref_path=ROOT/'reports/experiments/focus-source-flow-v1-result.json'
    train_ref_path=ROOT/'reports/experiments/focus-owned-flow-full-v1-result.json'
    train_nb=ROOT/'kaggle/biohub-focus-owned-flow-full-v1/biohub-focus-owned-flow-full-v1.ipynb'
    if (sha(ref_path)!=REF_SHA or sha(train_ref_path)!=TRAIN_REF_SHA or sha(train_nb)!=TRAIN_NB_SHA):
        raise ValueError('Frozen source/training identities changed')
    reference=json.loads(ref_path.read_text()); train_reference=json.loads(train_ref_path.read_text())
    train_folder=ROOT/'.biohub/cache/kernel-outputs/focus-owned-flow-full-v1/focus_owned_flow_full'
    training_graphs,_,_=FULL['prepare'](train_folder,train_nb)
    folder=ROOT/'.biohub/cache/kernel-outputs/focus-source-flow-v1/focus_source_flow'
    prepared,parents,policy=SOURCE['prepare'](folder,ROOT/'kaggle/biohub-focus-source-flow-v1/biohub-focus-source-flow-v1.ipynb',NB_SHA)
    fold=json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text())['folds'][0]
    if any(s not in fold['train'] for s in TRAIN_STEMS) or set(TRAIN_STEMS)&set(policy['source_stems']+fold['audit_order']):
        raise ValueError('Calibration must use original training movies only')
    metric=SOURCE['SCORER']['load_scorer'](ROOT/'.biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot')
    residuals=[]; training_records=[]
    for stem in TRAIN_STEMS:
        graph=td.graph.IndexedRXGraph.from_geff(str(training_graphs[stem]['candidate']))[0]
        truth_path=ROOT/'.biohub/cache/competition-train-geffs-packed-v1/train'/(stem+'.geff')
        truth=td.graph.IndexedRXGraph.from_geff(str(truth_path))[0]
        er=metric.evaluate(graph,truth,scale=(1.625,.40625,.40625),max_distance=7.)
        expected=next(r for r in train_reference['per_movie']['candidate'] if r['stem']==stem)
        if any(getattr(er,k)!=expected[k] for k in er._fields): raise ValueError('Training graph score replay failed')
        sample=train_folder/'outputs'/stem/'sampled_flow.npz'
        with np.load(sample,allow_pickle=False) as data: coords,flow=data['coords'].copy(),data['backward_um'].copy()
        nodes=graph.node_attrs().sort('node_id')
        if not np.array_equal(nodes.select('t','z','y','x').to_numpy(),coords): raise ValueError('Training node ordering changed')
        mapping={i:g for i,g in enumerate(nodes[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])}
        gt_edges=list(truth.edge_attrs().select(td.DEFAULT_ATTR_KEYS.EDGE_SOURCE,td.DEFAULT_ATTR_KEYS.EDGE_TARGET).iter_rows())
        values=matched_residuals(coords,flow,mapping,gt_edges)
        if len(values)<50: raise ValueError('Insufficient matched ordinary training edges')
        residuals.append(values)
        training_records.append(dict(stem=stem,residuals=len(values),sample_sha256=sha(sample),
            mean_um=values.mean(axis=0).tolist(),variance_um2=values.var(axis=0).tolist(),
            matched_training_counts_replayed=True))
    parameters=fit(np.concatenate(residuals))
    fitted=dict(status='fitted_training_only_residual_gaussian',training_stems=list(TRAIN_STEMS),
        records=training_records,parameters=parameters,source_hashes=frozen,
        source_selection_labels_used=False,target_embryo_labels_used=False,
        model_checkpoint_sha256=policy['flow_checkpoint_sha256'],raw_detector_checkpoint_sha256='b14a7bd272f824adb1a1073bc3f2af17a95919d5a0c3f1d9011a8d82378d8f3a',
        authorized_for_submission=False)
    fit_path.write_text(json.dumps(fitted,indent=2,allow_nan=False))
    print(json.dumps(dict(fitting_complete=True,parameters=parameters,training_records=training_records)),flush=True)
    cache.mkdir()
    records=[]
    for stem in policy['source_stems']:
        sample=folder/'outputs'/stem/'sampled_flow.npz'
        with np.load(sample,allow_pickle=False) as data: coords,flow=data['coords'].copy(),data['backward_um'].copy()
        edges=link(coords,flow,parameters)
        target=cache/(stem+'.npz')
        np.savez_compressed(target,coords=coords,edges=np.asarray([(s,d) for s,d,p in edges],dtype=np.int64).reshape(-1,2))
        records.append(dict(stem=stem,nodes=len(coords),edges=len(edges),sample_sha256=sha(sample),candidate_sha256=sha(target)))
    manifest=dict(run_id=RUN,fit_sha256=sha(fit_path),reference_sha256=REF_SHA,records=records,
                  source_hashes=frozen,all_source_predictions_saved_before_source_gt=True)
    manifest_path=cache/'prelabel_manifest.json'; manifest_path.write_text(json.dumps(manifest,indent=2))
    arrays={}
    for record in records:
        stem=record['stem']; path=cache/(stem+'.npz')
        if sha(path)!=record['candidate_sha256']: raise ValueError('Persisted prediction changed')
        with np.load(path,allow_pickle=False) as data: coords,edges=data['coords'].copy(),data['edges'].copy()
        with np.load(folder/'outputs'/stem/'sampled_flow.npz',allow_pickle=False) as data:
            if not np.array_equal(coords,data['coords']): raise ValueError('Candidate moved/deleted raw nodes')
            expected_edges=link(coords,data['backward_um'],parameters)
        graph=BASE['graph_from_arrays'](coords,edges)
        SOURCE['verify_graph'](graph,coords,expected_edges)
        arrays[stem]=(coords,edges)
    rows={a:[] for a in ('parent','control','candidate')}
    for stem in policy['source_stems']:
        truth_path=ROOT/'.biohub/cache/competition-train-geffs-packed-v1/train'/(stem+'.geff')
        for arm in rows:
            if arm=='parent': graph=parents[stem]
            elif arm=='control': graph=td.graph.IndexedRXGraph.from_geff(str(prepared[stem]['candidate']))[0]
            else: graph=BASE['graph_from_arrays'](*arrays[stem])
            truth=td.graph.IndexedRXGraph.from_geff(str(truth_path))[0]
            er=metric.evaluate(graph,truth,scale=(1.625,.40625,.40625),max_distance=7.)
            total=float(GeffMetadata.read(str(truth_path)).extra['estimated_number_of_nodes'])
            row=dict(metric.per_sample_metrics(er,total,SOURCE['SCORER']['diagnostic_node_recall'](metric,graph,truth)),stem=stem,embryo='6bba')
            rows[arm].append(row); print(json.dumps(dict(arm=arm,**row)),flush=True)
    summaries={a:metric.summarise(r) for a,r in rows.items()}
    if frozen!={p:sha(ROOT/p) for p in FILES}: raise ValueError('Experiment source changed during execution')
    result=dict(status='completed_training_calibrated_focus_motion_comparison',run_id=RUN,
        parameters=parameters,per_movie=rows,summaries=summaries,by_embryo={a:{'6bba':metric.summarise(r)} for a,r in rows.items()},
        comparison=BASE['comparison'](rows,summaries,reference),fit_sha256=sha(fit_path),
        prelabel_manifest_sha256=sha(manifest_path),source_hashes=frozen,
        authoritative_scorer_commit='075fc5f5a52d11077f9dc2b074644618f26939e2',
        elapsed_seconds=time.monotonic()-started,gpu_seconds=0,new_target_movies_opened=0,
        authorized_for_submission=False,caveat='Parameters fit only two original training movies. Eight-source comparison is exposed development, not independent confirmation or leaderboard evidence.')
    result_path.write_text(json.dumps(FULL['finite_json'](result),indent=2,allow_nan=False))
    print(json.dumps(result['comparison']),flush=True)


if __name__=='__main__': main()
