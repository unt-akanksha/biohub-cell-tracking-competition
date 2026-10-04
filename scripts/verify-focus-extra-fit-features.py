"""Fail-closed verification of additional fitting features before training."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.focus_cached_pair import validate_pair
RUN='focus-extra-fit-features-v1'
WORK='focus_extra_fit_features'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def staged_runtime(stage):
    identity=json.loads((stage/'staged_identity.json').read_text())
    nbpath=stage/('biohub-'+RUN+'.ipynb')
    if (identity['run_id']!=RUN or identity['status']!='staged_not_launched'
        or identity['declared_budget_seconds']!=3600 or sha(nbpath)!=identity['notebook_sha256']
        or sha(stage/'kernel-metadata.json')!=identity['metadata_sha256']
        or sha(ROOT/'scripts/build-focus-extra-fit-features.py')!=identity['builder_sha256']):
        raise ValueError('Exact prelaunch staged notebook/metadata/builder required')
    nb=json.loads(nbpath.read_text())
    node=next(n for n in ast.parse(''.join(nb['cells'][1]['source'])).body
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    return identity,ast.literal_eval(node.value)


def verify_packet(packet,coords,movie,t,pair):
    counts=validate_pair(packet,coords)
    if counts['source_frame']!=t or any(pair[k]!=v for k,v in counts.items()):
        raise ValueError('Packet frame/count/identity differs')
    labels=np.full(counts['target_nodes'],-1,np.int64)
    if movie['role']!='replay':
        row=movie['windows'][t]
        if row['source_frame']!=t:raise ValueError('Sparse label frame reordered')
        labels[np.asarray(row['columns'],np.int64)]=np.asarray(row['parent_rows'],np.int64)
    if not np.array_equal(packet['labels'],labels):raise ValueError('Cached labels differ from frozen audit')
    sampling=pair['sampling']
    if (sampling['sampled_nodes']!=counts['target_nodes'] or sampling['coordinates_modified'] is not False
        or sampling['nodes_deleted'] is not False):raise ValueError('Sampler changed raw proposals')
    return counts


def verify(folder):
    stage=ROOT/'kaggle'/('biohub-'+RUN);identity,runtime=staged_runtime(stage)
    spec=json.loads(runtime['features_spec.json'])
    if spec!=runpy.run_path(str(ROOT/'scripts/build-focus-extra-fit-features.py'))['make_spec']():
        raise ValueError('Scope differs from verified complete raw/label evidence')
    if spec['contract']['diagnostic_stems'] or any(r['role'] not in ('replay','fitting') for r in spec['movies']):
        raise ValueError('Additional fitting features only; original diagnostics remain untouched')
    work=folder/WORK;output=work/'outputs'
    for name,value in runtime.items():
        if (work/'runtime'/name).read_bytes()!=value.encode():raise ValueError('Actual runtime differs from immutable staged notebook')
    launcher=json.loads((work/'launcher_terminal.json').read_text())
    result=json.loads((output/'result.json').read_text())
    if (launcher['status']!='completed' or launcher['run_id']!=RUN or launcher['declared_budget_seconds']!=3600
        or not 0<launcher['elapsed_seconds']<=3600 or launcher['submission_performed'] is not False):
        raise ValueError('Complete bounded launcher required')
    expected=dict(neural=spec['probe_model_tensor_sha256'],flow='e82b7255fb2cded608a800fb6627ec4972043491e636e22a0dcbd66fc5c93779')
    if (result['status']!='completed_focus_adaptation_feature_cache' or result['before']!=expected or result['after']!=expected
        or result['replayed_stems']!=spec['contract']['replay_stems']
        or result['checkpoint_sha256']!='76f7da6e32c901e3e3f9d2ab870b8a22235b0b91086cfcf909aa41674896a144'
        or result['features_spec_sha256']!=hashlib.sha256(runtime['features_spec.json'].encode()).hexdigest()
        or result['runtime_source_hashes']!=json.loads(runtime['source_hashes.json'])
        or result['labels_from_verified_training_cache'] is not True or result['new_target_movies_opened']!=0
        or any(result[k] is not False for k in ('optimizer_run','authorized_for_submission','raw_gt_graphs_opened','source_selection_opened'))):
        raise ValueError('Feature terminal/model/replay identity differs')
    if json.loads((output/'features_spec.json').read_text())!=spec:raise ValueError('Actual specification differs')
    if [(r['stem'],r['role']) for r in result['records']]!=[(r['stem'],r['role']) for r in spec['movies']]:
        raise ValueError('Missing, extra or reordered movies')
    records=[]
    for movie,record in zip(spec['movies'],result['records']):
        stem=movie['stem'];raw=output/'raw_detections'/(stem+'.npz')
        if sha(raw)!=movie['raw_sha256'] or record['raw_sha256']!=movie['raw_sha256']:
            raise ValueError('Original raw nodes changed')
        with np.load(raw,allow_pickle=False) as data:coords=data['coords'].copy()
        frames=3 if movie['role']=='replay' else 100
        if (record['frames']!=frames or record['nodes']!=len(coords)
            or [p['file'] for p in record['pairs']]!=[f'{t:03d}.npz' for t in range(frames-1)]):
            raise ValueError('Incomplete feature movie')
        if json.loads((output/stem/'manifest.json').read_text())!=record:raise ValueError('Movie manifest differs')
        totals=dict(known_parent=0,known_absent=0,unknown=0)
        for t,pair in enumerate(record['pairs']):
            path=output/stem/pair['file']
            if sha(path)!=pair['sha256']:raise ValueError('Packet checksum mismatch')
            with np.load(path,allow_pickle=False) as data:packet={k:data[k].copy() for k in data.files}
            counts=verify_packet(packet,coords,movie,t,pair)
            for key in totals:totals[key]+=counts[key]
        records.append(dict(stem=stem,role=movie['role'],frames=frames,nodes=len(coords),pairs=frames-1,
            raw_sha256=movie['raw_sha256'],manifest_sha256=sha(output/stem/'manifest.json'),counts=totals))
    return dict(status='verified_focus_extra_fitting_feature_cache',run_id=RUN,records=records,contract=spec['contract'],
        notebook_sha256=identity['notebook_sha256'],staged_identity_sha256=sha(stage/'staged_identity.json'),
        worker_result_sha256=sha(output/'result.json'),launcher=launcher,all_raw_nodes_and_sparse_labels_exact=True,
        optimizer_run=False,authorized_for_submission=False,diagnostic_caveat=spec['contract']['diagnostic_caveat'])


if __name__=='__main__':
    path=ROOT/f'reports/experiments/{RUN}-result.json'
    if path.exists():raise ValueError('Never overwrite a verified result')
    result=verify(ROOT/f'.biohub/cache/kernel-outputs/{RUN}')
    path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
