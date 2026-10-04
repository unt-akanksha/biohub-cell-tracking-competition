"""Verify exact candidate identity, append-only features and source-only fitted gates."""
import hashlib
import json
from pathlib import Path
import sys
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from research.comoving_division_head import predict,matrix,transform
from research.frozen_image_head import objective,stratum_weights
from research.image_context_quality import metrics


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with np.load(path,allow_pickle=False) as a:return dict(a)


def main():
    output=ROOT/'reports/experiments/comoving-division-head-v1-verification.json'
    if output.exists():raise ValueError('Preserve verifier receipt')
    data_root=ROOT/'.biohub/cache/comoving-division-features-v1'
    result_root=ROOT/'.biohub/cache/comoving-division-head-v1-output'
    manifest=json.loads((data_root/'MANIFEST.json').read_text());terminal=json.loads((result_root/'result.json').read_text())
    for source in (manifest,terminal):
        for name,digest in source['source_pins'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Source changed')
    if sha(data_root/'MANIFEST.json')!=terminal['source_manifest_sha256']:raise ValueError('Data contract mismatch')
    inventory=json.loads((ROOT/'.biohub/staging/biohub-relational-division-inventory-v3/relational_division_inventory_v3.json').read_text())
    frames=defaultdict(list)
    for row in inventory['examples']:
        if row['role'] in ('optimization','selection'):frames[(row['stem'],row['timepoint'])].append(row)
    original=ROOT/'.biohub/cache/graph-context-relational-v1/biohub_graph_context_relational_patches_v1'
    shard_manifest=json.loads((original/'graph_context_relational_patch_manifest.json').read_text())
    shard_pins={row['path']:row['sha256'] for row in shard_manifest['records']};loaded={}
    ids_verified=0;evidence={}
    for embryo in ('44b6','6bba'):
        packets={}
        for role in ('optimization','selection'):
            name=f'{embryo}-{role}.npz';path=data_root/name
            if sha(path)!=manifest['files'][name]['sha256']:raise ValueError('Motion packet changed')
            packet=read(path);packets[role]=packet
            old=read(ROOT/f'.biohub/cache/frozen-image-head-v1-output/frozen-head-v1-full/source-{embryo}/{role}-features.npz')
            for key in ('features','targets','eligible','weights'):np.testing.assert_array_equal(packet[key],old[key])
            index=json.loads((ROOT/f'.biohub/cache/image-division-context-v2/{embryo}-{role}-inventory.json').read_text())
            for row in index:
                name=row['source_path'];timepoint=int(Path(name).stem.rsplit('-t',1)[1]);number=row['source_row']
                candidate=frames[(row['stem'],timepoint)][number]
                if name not in loaded:
                    path=original/name
                    if sha(path)!=shard_pins[name]:raise ValueError('Original shard changed')
                    with np.load(path,allow_pickle=False) as a:loaded[name]=[a[k].copy() for k in ('parent_ids','existing_child_ids','proposed_child_ids')]
                actual=tuple(int(a[number]) for a in loaded[name])
                expected=tuple(int(candidate[k]) for k in ('parent_id','existing_child_id','proposed_child_id'))
                if actual!=expected:raise ValueError('Motion descriptors are attached to different candidates')
                ids_verified+=1
        evidence[embryo]={}
        for variant in ('control','motion'):
            folder=result_root/f'source-{embryo}';head=read(folder/f'{variant}-head.npz')
            result=terminal['source_models' if variant=='motion' else 'controls'][embryo]
            if sha(folder/f'{variant}-head.npz')!=result['head_sha256']:raise ValueError('Fitted head changed')
            x=matrix(packets['optimization'],variant=='motion')
            np.testing.assert_allclose(head['mean'],x.mean(axis=0),atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(head['scale'],np.maximum(x.std(axis=0),.001),atol=1e-12,rtol=1e-12)
            weights=stratum_weights(packets['optimization']['targets'],packets['optimization']['eligible'],packets['optimization']['weights'])
            obj,grad=objective(np.r_[head['coefficients'],head['intercept']],transform(x,head),packets['optimization']['targets'],weights,.01)
            np.testing.assert_allclose(obj,head['objective'],atol=1e-10)
            if abs(grad).max()>1e-5:raise ValueError('Stationarity verification failed')
            scores=predict(packets['selection'],head);saved=read(folder/f'{variant}-source-selection.npz')
            np.testing.assert_array_equal(scores,saved['scores'])
            mask=packets['selection']['eligible'].astype(bool);row=metrics(packets['selection']['targets'][mask],scores[mask])
            if row!=result['metrics']:raise ValueError('Source gate replay mismatch')
            evidence[embryo][variant]=dict(metrics=row,maximum_absolute_objective_gradient=float(abs(grad).max()))
    if terminal['held_out_embryo_scores_opened'] or terminal['status']!='rejected_source_selection':raise ValueError('Unexpected rejection transition')
    result=dict(status='verified_rejection',candidate_id_triples_verified=ids_verified,base_features_byte_equal=True,
        old_inference_masks_unchanged=True,source_only_normalization_replayed=True,source_scores_replayed=True,
        head_stationarity_verified=True,held_out_embryo_scores_opened=False,authorized_for_submission=False,
        terminal_sha256=sha(result_root/'result.json'),evidence=evidence)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='verified_rejection',candidate_id_triples_verified=ids_verified,result_sha256=sha(output))))


if __name__=='__main__':main()
