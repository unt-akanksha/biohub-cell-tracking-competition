"""Append causal past-track descriptors to frozen original source feature banks."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from research.comoving_division_features import features


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start=time.monotonic();output=ROOT/'.biohub/cache/comoving-division-features-v1'
    if output.exists():raise ValueError('Preserve prepared data')
    source=ROOT/'.biohub/cache/frozen-image-head-v1-output/frozen-head-v1-full'
    if sha(source/'result.json')!='6206f476c9eec7fb335098e97b9e0df1db88874ee94ca691bd9d7937d91954bd':raise ValueError('Original source bank changed')
    bio=ROOT/'.biohub/cache/image-division-context-v2'
    manifest_path=bio/'MANIFEST.json'
    if sha(manifest_path)!='39032899ecea16e87bfb53d45909f06993d39ff0cfbb0bbb16a7b875f9dcecce':raise ValueError('Biohub source data changed')
    manifest=json.loads(manifest_path.read_text())
    inventory_path=ROOT/'.biohub/staging/biohub-relational-division-inventory-v3/relational_division_inventory_v3.json'
    if sha(inventory_path)!='94150632f5a80b2ef48a39743a425cbe1b8e57b1c131c19ef0bde3d97d1c783e':raise ValueError('Inventory changed')
    rows=[r for r in json.loads(inventory_path.read_text())['examples'] if r['role'] in ('optimization','selection')]
    by_frame={}
    for row in rows:by_frame.setdefault((row['stem'],row['timepoint']),[]).append(row)
    cache=ROOT/'.biohub/cache/competition-train-geffs-packed-v1';mp=cache/'train_geff_cache_manifest.json'
    if sha(mp)!='744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9':raise ValueError('GT contract changed')
    allowed={r['stem'] for r in rows};geff_pins={}
    for row in json.loads(mp.read_text())['files']:
        if row['relative_path'].split('.geff/',1)[0] in allowed:
            if sha(cache/'train'/row['relative_path'])!=row['sha256']:raise ValueError('Source track changed')
            geff_pins[row['relative_path']]=row['sha256']
    graph=runpy.run_path(str(ROOT/'scripts/audit-comoving-division-geometry-v1.py'))['graph']
    graphs={};output.mkdir();files={}
    source_pins={n:sha(ROOT/n) for n in ('research/comoving_division_features.py','scripts/prepare-comoving-division-features-v1.py',
                 'reports/experiments/comoving-division-head-v1-design.md')}
    for embryo in ('44b6','6bba'):
        for role in ('optimization','selection'):
            record=manifest['files'][f'{embryo}-{role}.npz'];ip=bio/record['inventory_path']
            if sha(ip)!=record['inventory_sha256']:raise ValueError('Row alignment inventory changed')
            index=json.loads(ip.read_text());bank=source/f'source-{embryo}'/f'{role}-features.npz'
            with np.load(bank,allow_pickle=False) as a:old=dict(a)
            with np.load(bio/f'{embryo}-{role}.npz',allow_pickle=False) as a:
                for k in ('targets','eligible','weights'):np.testing.assert_array_equal(old[k],a[k])
            values=[];availability=0;labels=[]
            for row in index:
                stem=row['stem'];timepoint=int(Path(row['source_path']).stem.rsplit('-t',1)[1])
                candidate=by_frame[(stem,timepoint)][row['source_row']]
                if candidate['role']!=role or candidate['embryo']!=embryo:raise ValueError('Source role mismatch')
                if stem not in graphs:graphs[stem]=graph(cache/'train'/f'{stem}.geff')
                vector,meta=features(*graphs[stem],candidate['parent_id'],candidate['existing_child_id'],candidate['proposed_child_id'])
                values.append(vector);labels.append(candidate['division_recovery_target']);availability+=int(meta['available'])
            np.testing.assert_array_equal(old['targets'],np.asarray(labels,dtype=np.float32))
            motion=np.asarray(values,dtype=np.float64)
            path=output/f'{embryo}-{role}.npz';np.savez_compressed(path,**old,motion_features=motion)
            files[path.name]=dict(rows=len(index),movies=len({r['stem'] for r in index}),available_history=availability,
                sha256=sha(path),bytes=path.stat().st_size,source_feature_bank_sha256=sha(bank),inventory_sha256=sha(ip))
            print(json.dumps(dict(event='motion_packet',embryo=embryo,role=role,**files[path.name])),flush=True)
    if any(sha(ROOT/n)!=digest for n,digest in source_pins.items()):raise ValueError('Frozen feature source changed')
    result=dict(status='complete',run_id='comoving-division-features-v1',files=files,source_pins=source_pins,
        geff_pins=geff_pins,original_audit_geffs_opened=False,final_probe_geffs_opened=False,
        source_selection_past_tracks_read=True,source_selection_scores_opened=False,
        old_inference_masks_unchanged=True,authorized_for_submission=False,elapsed_seconds=time.monotonic()-start)
    path=output/'MANIFEST.json';path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='complete',manifest_sha256=sha(path),elapsed_seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
