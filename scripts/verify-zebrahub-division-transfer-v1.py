"""Independent CPU replay of external labels, domain weights and source gates."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from research.frozen_image_head import predict,transform,stratum_weights,objective
from research.image_context_quality import metrics


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def packet(path):
    with np.load(path,allow_pickle=False) as a:return dict(a)


def main():
    output=ROOT/'reports/experiments/zebrahub-division-transfer-v1-verification.json'
    if output.exists():raise ValueError('Preserve verification')
    base=ROOT/'.biohub/cache/zebrahub-division-transfer-v1-output';full=base/'zebrahub-transfer-v1-full'
    code=ROOT/'.biohub/cache/zebrahub-division-transfer-v1-bundle'
    if sha(code/'CONTRACT.json')!='3ef608e8153d6f141643a427d647c6792d3ff127fe9623512661cf6cdf5140ae':raise ValueError('Contract mismatch')
    for name,digest in json.loads((code/'CONTRACT.json').read_text())['files'].items():
        if sha(code/name)!=digest:raise ValueError('Frozen source/data changed')
    inventories=list(base.glob('inventory-*.json'))
    if not inventories:raise ValueError('Missing remote inventory')
    for inventory in inventories:
        for row in json.loads(inventory.read_text())['files']:
            path=base/row['path']
            if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('Artifact changed')
    terminal=json.loads((full/'result.json').read_text())
    if terminal['status']!='rejected_source_selection' or terminal['held_out_embryo_scores_opened']:
        raise ValueError('Unexpected experiment outcome')
    if list(full.glob('held-out-*.npz')) or (full/'frozen-source-policy.json').exists():raise ValueError('Unexpected target access')
    external_manifest=json.loads((code/'EXTERNAL.json').read_text());external_parts=[]
    for record in external_manifest['files']:external_parts.append(packet(code/record['descriptor_path']))
    external_expected={k:np.concatenate([p[k] for p in external_parts]) for k in ('targets','eligible','weights')}
    evidence={}
    for source in ('44b6','6bba'):
        root=full/f'source-{source}';head=packet(root/'head.npz')
        ext=packet(root/'external-features.npz');bio=packet(root/'optimization-features.npz');selection=packet(root/'selection-features.npz')
        result=json.loads((root/'terminal.json').read_text())
        if sha(root/'head.npz')!=result['head_sha256']:raise ValueError('Selected head changed')
        for key,expected in external_expected.items():np.testing.assert_array_equal(ext[key],expected)
        for role,actual in (('optimization',bio),('selection',selection)):
            data=packet(ROOT/f'.biohub/cache/image-division-context-v2/{source}-{role}.npz')
            for key in ('targets','eligible','weights'):np.testing.assert_array_equal(actual[key],data[key])
        for data in (ext,bio,selection):
            if not np.isfinite(data['features']).all():raise ValueError('Nonfinite features')
            np.testing.assert_array_equal(data['features'][:,[1283+7,1283+8]],0.)
            np.testing.assert_array_equal(data['features'][:,[1283+16,1283+17]],1.)
        if int(head['external_rows'])!=7152 or float(head['external_domain_mass'])!=.5 or float(head['biohub_domain_mass'])!=.5 or float(head['penalty'])!=.01:
            raise ValueError('Training mass/penalty contract failed')
        mean=(ext['features'].mean(axis=0)+bio['features'].mean(axis=0))/2.
        variance=(np.square(ext['features']-mean).mean(axis=0)+np.square(bio['features']-mean).mean(axis=0))/2.
        np.testing.assert_allclose(head['mean'],mean,atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(head['scale'],np.maximum(np.sqrt(variance),.001),atol=1e-12,rtol=1e-12)
        data=np.concatenate((ext['features'],bio['features']));labels=np.r_[ext['targets'],bio['targets']]
        weights=np.concatenate([.5*stratum_weights(p['targets'],p['eligible'],p['weights']) for p in (ext,bio)])
        value,gradient=objective(np.r_[head['coefficients'],head['intercept']],transform(data,head),labels,weights,.01)
        np.testing.assert_allclose(value,head['objective'],atol=1e-10,rtol=1e-10)
        if np.max(np.abs(gradient))>1e-5:raise ValueError('Head stationarity verification failed')
        scores=predict(selection['features'],head);saved=packet(root/'source-selection.npz')
        np.testing.assert_allclose(scores,saved['scores'],atol=1e-12,rtol=1e-12)
        gate=selection['eligible'].astype(bool);row=metrics(selection['targets'][gate],scores[gate])
        for key in ('average_precision','zero_fp_tp','source_gate_passed'):
            if row[key]!=result['metrics'][key]:raise ValueError('Replayed source gate differs')
        np.testing.assert_allclose(row['binary_cross_entropy'],result['metrics']['binary_cross_entropy'],rtol=1e-12)
        evidence[source]=dict(source_metrics=row,maximum_absolute_objective_gradient=float(np.max(np.abs(gradient))),
                              exact_domain_masses=[.5,.5],penalty=.01,head_parameters=len(head['coefficients'])+1)
    result=dict(status='verified_rejection',run_id='zebrahub-division-transfer-v1',
        terminal_sha256=sha(full/'result.json'),remote_artifact_inventories=len(inventories),
        actual_leaderboard_score=None,authorized_for_submission=False,held_out_embryo_scores_opened=False,
        source_labels_replayed=True,external_labels_replayed=True,domain_normalization_replayed=True,
        velocity_domain_indicator_removed=True,head_stationarity_verified=True,evidence=evidence)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],result_sha256=sha(output),evidence=evidence)))


if __name__=='__main__':main()
