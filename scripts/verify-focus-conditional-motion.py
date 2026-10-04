"""Recompute all fourteen frozen folds and final fit from saved training arrays."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.focus_conditional_motion import fit,predict,metrics,heldout_gate
from research.focus_residual_calibration import fit as global_fit
RUN='focus-conditional-motion-v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    path=ROOT/f'reports/experiments/{RUN}-result.json';cache=ROOT/'.biohub/cache'/RUN
    if sha(path)!='ad168df2c52620470e659436815b4fd8dd004095b385eee9e21a7b857c6cf226':raise ValueError('Frozen completed conditional result required')
    result=json.loads(path.read_text());manifest_path=cache/'examples_manifest.json'
    if any(sha(ROOT/p)!=v for p,v in result['source_hashes'].items()) or sha(manifest_path)!=result['examples_manifest_sha256']:raise ValueError('Actual runtime or examples manifest changed')
    manifest=json.loads(manifest_path.read_text())
    if manifest['records']!=result['records'] or manifest['source_hashes']!=result['source_hashes']:raise ValueError('Example evidence differs')
    samples={}
    for r in result['records']:
        path=cache/(r['stem']+'.npz')
        if sha(path)!=r['sha256']:raise ValueError('Persisted fitting arrays changed')
        with np.load(path,allow_pickle=False) as saved:samples[r['stem']]={k:saved[k].copy() for k in saved.files}
        if len(samples[r['stem']]['y'])!=r['observations']:raise ValueError('Observation inventory changed')
    stems=result['training_stems'];rows=[]
    if len(stems)!=14 or len(set(stems))!=14 or list(samples)!=stems:raise ValueError('Complete ordered fitting scope required')
    for held,fold,oldrow in zip(stems,result['folds'],result['per_movie']):
        train=[s for s in stems if s!=held];x=np.concatenate([samples[s]['x'] for s in train]);y=np.concatenate([samples[s]['y'] for s in train])
        model=fit(x,y);base=global_fit(y)
        if fold!=dict(held_out_stem=held,model=model,baseline=base):raise ValueError('Held-out fitting replay failed')
        sample=samples[held];row=dict(stem=held,training_stems=train,baseline=metrics(sample['y'],base['mean_um'],base['variance_um2']),
            candidate=metrics(sample['y'],predict(model,sample['x']),model['variance_um2']))
        if row!=oldrow:raise ValueError('Held-out evaluation replay failed')
        rows.append(row)
    if len(result['folds'])!=14 or len(result['per_movie'])!=14 or heldout_gate(rows)!=result['gate'] or not result['gate']['passed']:raise ValueError('Complete passing fixed held-out gate required')
    final=result['final_model'];path=cache/final['file']
    if final['file']!='model.json' or sha(path)!=final['sha256'] or final['exact_prediction_replay'] is not True:raise ValueError('Actual portable final model required')
    model=json.loads(path.read_text());expected=fit(np.concatenate([samples[s]['x'] for s in stems]),np.concatenate([samples[s]['y'] for s in stems]))
    if model!=expected:raise ValueError('Final all-training fit replay failed')
    if result['source_movies_evaluated']!=0 or result['new_target_movies_opened']!=0 or result['gpu_seconds']!=0 or result['authorized_for_submission'] is not False:raise ValueError('Fitting scope changed')
    return dict(status='verified_conditional_motion_training_folds',result_sha256=sha(ROOT/f'reports/experiments/{RUN}-result.json'),
        final_model_sha256=sha(path),exact_fold_replays=14,gate=result['gate'],authorized_for_submission=False),model


if __name__=='__main__':
    target=ROOT/f'reports/experiments/{RUN}-verification.json'
    if target.exists():raise ValueError('Never overwrite verified record')
    receipt,_=verify();target.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
