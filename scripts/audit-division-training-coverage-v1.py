"""Count annotation forks lost before image extraction, optimization movies only."""
import hashlib
import json
from pathlib import Path
import sys
import time
from collections import Counter,defaultdict

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import zarr
from research.learned_division_recovery import biological_geometry_score,VOXEL_SIZE_ZYX_UM


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start=time.monotonic()
    output=ROOT/'reports/experiments/division-training-coverage-v1-result.json'
    if output.exists():raise ValueError('Preserve completed audit')
    inventory_path=ROOT/'.biohub/staging/biohub-relational-division-inventory-v3/relational_division_inventory_v3.json'
    if sha(inventory_path)!='94150632f5a80b2ef48a39743a425cbe1b8e57b1c131c19ef0bde3d97d1c783e':
        raise ValueError('Historical inventory changed')
    inventory=json.loads(inventory_path.read_text())
    cache=ROOT/'.biohub/cache/competition-train-geffs-packed-v1'
    manifest_path=cache/'train_geff_cache_manifest.json'
    if sha(manifest_path)!='744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9':
        raise ValueError('Ground-truth cache contract changed')
    manifest=json.loads(manifest_path.read_text())
    # allocate_roles selects audit/selection only from movies with nonzero examples.
    # Thus zero-example non-final movies were optimization; do not invent new splits.
    roles={stem:'optimization' for stem in manifest['stems'] if stem not in inventory['final_probe_stems']}
    explicit={}
    for row in inventory['examples']:
        if row['stem'] in explicit and explicit[row['stem']]!=row['role']:raise ValueError('Inconsistent source role')
        explicit[row['stem']]=row['role']
    roles.update(explicit)
    training={stem for stem,role in roles.items() if role=='optimization'}
    # Verify only optimization GEFF files, not audit, selection, or final-probe graphs.
    pins={}
    for row in manifest['files']:
        stem=row['relative_path'].split('.geff/',1)[0]
        if stem not in training:continue
        path=cache/'train'/row['relative_path']
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('Optimization annotations changed')
        pins[row['relative_path']]=row['sha256']
    existing=defaultdict(dict)
    for row in inventory['examples']:
        if row['role']=='optimization' and row['division_recovery_target']:
            key=(row['parent_id'],*sorted((row['existing_child_id'],row['proposed_child_id'])))
            existing[row['stem']][key]=row
    by_embryo={e:Counter() for e in ('44b6','6bba')}
    events=[]
    for index,stem in enumerate(sorted(training)):
        embryo=stem.split('_',1)[0];count=by_embryo[embryo];count['movies']+=1
        group=zarr.open_group(str(cache/'train'/f'{stem}.geff'),mode='r')
        ids=np.asarray(group['nodes/ids'][:],dtype=np.int64)
        times=np.asarray(group['nodes/props/t/values'][:],dtype=np.int64)
        xyz=np.stack([np.asarray(group[f'nodes/props/{axis}/values'][:]) for axis in ('z','y','x')],axis=1)
        positions=xyz*VOXEL_SIZE_ZYX_UM
        rows={int(n):i for i,n in enumerate(ids)}
        edges=np.asarray(group['edges/ids'][:],dtype=np.int64)
        if len(rows)!=len(ids):raise ValueError('Duplicate GT node ids')
        outgoing=defaultdict(set)
        for parent,child in edges:
            if parent not in rows or child not in rows:raise ValueError('Dangling annotation edge')
            if times[rows[child]]!=times[rows[parent]]+1:raise ValueError('Nonadjacent annotated edge')
            outgoing[int(parent)].add(int(child))
        found=set()
        for parent,children in sorted(outgoing.items()):
            if len(children)!=2:continue
            first,second=sorted(children);found.add((parent,first,second));count['true_divisions']+=1
            delta1=positions[rows[first]]-positions[rows[parent]]
            delta2=positions[rows[second]]-positions[rows[parent]]
            score=biological_geometry_score(delta1,delta2)[0]
            first_distance=float(np.linalg.norm(delta1));second_distance=float(np.linalg.norm(delta2))
            sister_distance=float(np.linalg.norm(delta1-delta2))
            accepted=second_distance<=12. and sister_distance<=15. and score>=0.
            present=(parent,first,second) in existing[stem]
            if accepted!=present:raise ValueError('Historical positive extraction mismatch')
            eligible=accepted and score>=3.
            count['retained_training_positives']+=int(present)
            count['retained_inference_eligible_positives']+=int(eligible)
            count['removed_before_image_extraction']+=int(not present)
            count['parent_distance_rejections']+=int(second_distance>12.)
            count['sister_distance_rejections']+=int(sister_distance>15.)
            count['zero_example_movie_divisions']+=int(stem not in explicit)
            count['retained_low_geometry_positives']+=int(present and not eligible)
            events.append(dict(stem=stem,embryo=embryo,timepoint=int(times[rows[parent]]),
                parent_id=parent,children=[first,second],parent_center_zyx=xyz[rows[parent]].tolist(),
                daughter_centers_zyx=xyz[[rows[first],rows[second]]].tolist(),
                parent_distances_um=[first_distance,second_distance],sister_distance_um=sister_distance,
                geometry_score=score,retained=present,inference_geometry_eligible=eligible))
        if set(existing[stem])-found:raise ValueError('Historical positive not a true annotated division')
        if (index+1)%25==0:print(json.dumps(dict(event='movies',complete=index+1,total=len(training))),flush=True)
    result=dict(status='complete',run_id='division-training-coverage-v1',
        source_inventory_sha256=sha(inventory_path),source_manifest_sha256=sha(manifest_path),
        generator_sha256=sha(Path(__file__)),optimization_only=True,audit_geffs_opened=False,
        selection_geffs_opened=False,final_probe_geffs_opened=False,model_fitting_performed=False,
        metric_thresholds_changed=False,authorized_for_submission=False,gpu_hours=0,
        by_embryo={e:dict(c) for e,c in by_embryo.items()},events=events,
        optimization_input_hashes=pins,elapsed_seconds=time.monotonic()-start)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='complete',by_embryo=result['by_embryo'],elapsed_seconds=result['elapsed_seconds'],
                         result_sha256=sha(output))))


if __name__=='__main__':main()
