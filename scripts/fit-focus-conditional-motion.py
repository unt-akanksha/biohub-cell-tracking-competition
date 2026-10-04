"""Persist verified training examples, then fixed leave-one-movie-out ridge."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.focus_conditional_motion import examples,fit,predict,metrics,heldout_gate
from research.focus_residual_calibration import fit as fit_global,matched_residuals
from research.focus_cached_motion import assemble_flow
from research.focus_adaptation_labels import node_matches
RUN='focus-conditional-motion-v1'
FILES=['scripts/fit-focus-conditional-motion.py','research/focus_conditional_motion.py','research/focus_residual_calibration.py',
    'research/focus_cached_motion.py','research/focus_adaptation_labels.py',f'reports/experiments/{RUN}-design.md']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import tracksdata as td
    started=time.monotonic();cache=ROOT/'.biohub/cache'/RUN;target=ROOT/f'reports/experiments/{RUN}-result.json'
    if cache.exists() or target.exists():raise ValueError('Never overwrite or refit completed work')
    frozen={p:sha(ROOT/p) for p in FILES};previous_path=ROOT/'reports/experiments/focus-expanded-motion-calibration-v1-fit.json'
    if sha(previous_path)!='379044ecc55dbba9bf59a216f46c7999858b805eca352b18873df2944c6e9ade':raise ValueError('Frozen fourteen-movie training evidence required')
    previous=json.loads(previous_path.read_text());spec=runpy.run_path(str(ROOT/'scripts/build-focus-expanded-summary.py'))['make_spec']()
    if spec!=previous['feature_spec']:raise ValueError('Actual verified feature evidence changed')
    roots=[ROOT/'.biohub/cache/kernel-outputs/focus-adaptation-features-v1/focus_adaptation_features/outputs',
        ROOT/'.biohub/cache/kernel-outputs/focus-extra-fit-features-v1/focus_extra_fit_features/outputs']
    features={r['stem']:(r,output) for g,output in zip(spec['feature_groups'],roots) for r in g['feature_records'] if r['role']=='fitting'}
    stems=previous['training_stems']
    if len(stems)!=14 or len(set(stems))!=14 or stems[2:]!=spec['contract']['fitting_stems']:raise ValueError('Fixed ordered fourteen training movies required')
    split=json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text())['folds'][0]
    if set(stems)&set(split['selection']+split['audit_order']+spec['contract']['unchanged_diagnostic_stems']) or not set(stems)<=set(split['train']):raise ValueError('Training-only partition required')
    runpy.run_path(str(ROOT/'scripts/score-independent-selection.py'))['load_scorer'](ROOT/'.biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot')
    cache.mkdir();records=[]
    for row in previous['records']:
        stem=row['stem']
        if stem not in features:
            sample=ROOT/'.biohub/cache/kernel-outputs/focus-owned-flow-full-v1/focus_owned_flow_full/outputs'/stem/'sampled_flow.npz'
            if sha(sample)!=row['evidence']['sample_sha256']:raise ValueError('Original complete flow changed')
            with np.load(sample,allow_pickle=False) as saved:coords,flow=saved['coords'].copy(),saved['backward_um'].copy()
        else:
            record,output=features[stem];raw=output/'raw_detections'/(stem+'.npz');manifest_path=output/stem/'manifest.json'
            if sha(raw)!=record['raw_sha256'] or sha(manifest_path)!=record['manifest_sha256']:raise ValueError('Verified raw/manifest changed')
            with np.load(raw,allow_pickle=False) as saved:coords=saved['coords'].copy()
            manifest=json.loads(manifest_path.read_text())
            def packets():
                for t,p in enumerate(manifest['pairs']):
                    if p['file']!=f'{t:03d}.npz':raise ValueError('Exact packet order required')
                    path=output/stem/p['file']
                    if sha(path)!=p['sha256']:raise ValueError('Packet bytes changed')
                    with np.load(path,allow_pickle=False) as saved:yield {k:saved[k].copy() for k in saved.files}
            flow=assemble_flow(coords,packets())
            if hashlib.sha256(flow.tobytes()).hexdigest()!=row['evidence']['recovered_flow_sha256']:raise ValueError('Reconstructed flow changed')
        truth=td.graph.IndexedRXGraph.from_geff(str(ROOT/'.biohub/cache/competition-train-geffs-packed-v1/train'/(stem+'.geff')))[0]
        mapping=node_matches(coords,truth);keys=td.DEFAULT_ATTR_KEYS;edges=list(truth.edge_attrs().select(keys.EDGE_SOURCE,keys.EDGE_TARGET).iter_rows())
        data=examples(coords,flow,mapping,edges);y=data['y']
        if not np.array_equal(y,matched_residuals(coords,flow,mapping,edges)):raise ValueError('Original residual extraction exact replay failed')
        if len(y)!=row['residuals'] or any(not np.allclose(values,row[k],rtol=0,atol=1e-12) for values,k in [(y.mean(0),'mean_um'),(y.var(0),'variance_um2')]):raise ValueError('Previous training statistics changed')
        path=cache/(stem+'.npz');np.savez_compressed(path,**data)
        with np.load(path,allow_pickle=False) as saved:
            if set(saved.files)!=set(data) or any(not np.array_equal(saved[k],data[k]) for k in data):raise ValueError('Saved examples changed')
        records.append(dict(stem=stem,sha256=sha(path),observations=len(y),exact_previous_residual_replay=True))
        print(json.dumps(dict(stage='training_examples_saved',**records[-1])),flush=True)
    manifest=dict(run_id=RUN,records=records,source_hashes=frozen,previous_training_fit_sha256=sha(previous_path),no_source_or_target_opened=True)
    manifest_path=cache/'examples_manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2))
    movies={}
    for r in records:
        path=cache/(r['stem']+'.npz')
        if sha(path)!=r['sha256']:raise ValueError('Frozen fitting examples changed')
        with np.load(path,allow_pickle=False) as saved:movies[r['stem']]={k:saved[k].copy() for k in saved.files}
    rows=[];folds=[]
    for held in stems:
        train=[s for s in stems if s!=held];x=np.concatenate([movies[s]['x'] for s in train]);y=np.concatenate([movies[s]['y'] for s in train])
        model=fit(x,y);baseline=fit_global(y);sample=movies[held]
        row=dict(stem=held,training_stems=train,baseline=metrics(sample['y'],baseline['mean_um'],baseline['variance_um2']),
            candidate=metrics(sample['y'],predict(model,sample['x']),model['variance_um2']))
        rows.append(row);folds.append(dict(held_out_stem=held,model=model,baseline=baseline))
        print(json.dumps(dict(stage='movie_held_out',**row)),flush=True)
    gate=heldout_gate(rows);final=None
    if gate['passed']:
        final_model=fit(np.concatenate([movies[s]['x'] for s in stems]),np.concatenate([movies[s]['y'] for s in stems]))
        path=cache/'model.json';path.write_text(json.dumps(final_model,indent=2,allow_nan=False));restored=json.loads(path.read_text())
        for sample in movies.values():
            if not np.array_equal(predict(final_model,sample['x']),predict(restored,sample['x'])):raise ValueError('Portable model replay changed')
        final=dict(file='model.json',sha256=sha(path),exact_prediction_replay=True)
    if frozen!={p:sha(ROOT/p) for p in FILES}:raise ValueError('Experiment sources changed')
    result=dict(status='completed_conditional_motion_heldout_screen',run_id=RUN,training_stems=stems,records=records,folds=folds,per_movie=rows,
        gate=gate,final_model=final,source_hashes=frozen,examples_manifest_sha256=sha(manifest_path),
        elapsed_seconds=time.monotonic()-started,gpu_seconds=0,source_movies_evaluated=0,new_target_movies_opened=0,authorized_for_submission=False,
        caveat='Leave-one-movie-out for motion correction only; underlying flow training/pretraining does not constitute independent whole-system validation.')
    target.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(dict(gate=gate,final_model=final)),flush=True)


if __name__=='__main__':main()
