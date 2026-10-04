"""Optimization-only fixed-motion diagnostic; never alter candidate or score gates."""
import hashlib
import json
from pathlib import Path
import sys
import time
from collections import defaultdict,Counter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import zarr
from research.comoving_division_features import features


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def graph(path):
    g=zarr.open_group(str(path),mode='r')
    ids=np.asarray(g['nodes/ids'][:],dtype=np.int64)
    ts=np.asarray(g['nodes/props/t/values'][:],dtype=np.int64)
    xyz=np.stack([np.asarray(g[f'nodes/props/{a}/values'][:]) for a in ('z','y','x')],axis=1)
    nodes={int(i):(int(t),*map(float,p)) for i,t,p in zip(ids,ts,xyz)};incoming=defaultdict(list)
    for parent,child in np.asarray(g['edges/ids'][:],dtype=np.int64):incoming[int(child)].append(int(parent))
    return nodes,incoming


def main():
    start=time.monotonic();output=ROOT/'reports/experiments/comoving-division-geometry-v1-result.json'
    if output.exists():raise ValueError('Preserve completed diagnostic')
    inventory_path=ROOT/'.biohub/staging/biohub-relational-division-inventory-v3/relational_division_inventory_v3.json'
    if sha(inventory_path)!='94150632f5a80b2ef48a39743a425cbe1b8e57b1c131c19ef0bde3d97d1c783e':raise ValueError('Inventory changed')
    inventory=json.loads(inventory_path.read_text());rows=[r for r in inventory['examples'] if r['role']=='optimization']
    by_movie=defaultdict(list)
    for row in rows:by_movie[row['stem']].append(row)
    cache=ROOT/'.biohub/cache/competition-train-geffs-packed-v1'
    mp=cache/'train_geff_cache_manifest.json'
    if sha(mp)!='744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9':raise ValueError('GEFF manifest changed')
    for record in json.loads(mp.read_text())['files']:
        stem=record['relative_path'].split('.geff/',1)[0]
        if stem in by_movie and sha(cache/'train'/record['relative_path'])!=record['sha256']:raise ValueError('Annotation changed')
    counts={e:Counter() for e in ('44b6','6bba')};records=[]
    for index,(stem,examples) in enumerate(sorted(by_movie.items())):
        nodes,incoming=graph(cache/'train'/f'{stem}.geff');embryo=stem.split('_',1)[0];count=counts[embryo]
        for row in examples:
            vector,meta=features(nodes,incoming,row['parent_id'],row['existing_child_id'],row['proposed_child_id'])
            positive=bool(row['division_recovery_target']);label='positive' if positive else 'negative'
            count[label+'_rows']+=1;count[label+'_history_available']+=int(meta['available'])
            original=bool(row['inference_geometry_eligible']);count[label+'_original_geometry_ge3']+=int(original)
            if meta['available']:
                motion=meta['comoving_geometry'][6]>=3.
                count[label+'_comoving_geometry_ge3']+=int(motion)
                count[label+'_newly_ge3']+=int(motion and not original)
                count[label+'_no_longer_ge3']+=int(original and not motion)
            # Values are diagnostic only; no deployed eligibility mask is replaced.
            records.append(dict(stem=stem,parent_id=row['parent_id'],label=positive,original_eligible=original,
                source_geometry_score=row['biological_geometry_score'],motion_features=vector.tolist(),**meta))
        if (index+1)%25==0:print(json.dumps(dict(event='movies',complete=index+1,total=len(by_movie))),flush=True)
    result=dict(status='complete',run_id='comoving-division-geometry-v1',optimization_only=True,
        selection_audit_or_target_graphs_opened=False,model_fitting_performed=False,threshold_search=False,
        inference_masks_changed=False,authorized_for_submission=False,gpu_hours=0,source_inventory_sha256=sha(inventory_path),
        feature_source_sha256=sha(ROOT/'research/comoving_division_features.py'),generator_sha256=sha(Path(__file__)),
        by_embryo={e:dict(c) for e,c in counts.items()},records=records,elapsed_seconds=time.monotonic()-start)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='complete',counts=result['by_embryo'],elapsed_seconds=result['elapsed_seconds'],result_sha256=sha(output))))


if __name__=='__main__':main()
