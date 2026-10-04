"""Make source-only external pair descriptors, retaining verified images unchanged."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from research.zebrahub_division_transfer import external_examples,aligned_external
from research.image_division_context import image_context


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start=time.monotonic();output=ROOT/'.biohub/cache/zebrahub-division-transfer-v1'
    if output.exists():raise ValueError('Preserve earlier preparation')
    source=ROOT/'.biohub/staging/biohub-zebrahub-contextual-shards-v1-balanced'
    manifest_path=source/'DATASET_MANIFEST.json'
    if sha(manifest_path)!='b35738f215413f1ece403ba5c0601adea82e2540c65f37e6465de0d0755cb7bf':
        raise ValueError('External source contract changed')
    source_manifest=json.loads(manifest_path.read_text());block=source_manifest['training']
    if block['source']!='ZSNS004' or block['source_role']!='external_pretraining' or block['split']!='train':
        raise ValueError('Wrong external source role')
    pins={n:sha(ROOT/n) for n in ('research/zebrahub_division_transfer.py','research/image_division_context.py',
        'scripts/prepare-zebrahub-division-transfer-v1.py','reports/experiments/zebrahub-division-transfer-v1-design.md')}
    output.mkdir();(output/'external').mkdir();(output/'descriptors').mkdir()
    rows=[];totals=dict(examples=0,positives=0,eligible_positives=0,eligible_negatives=0)
    for index,record in enumerate(block['records']):
        path=source/record['path'];metadata_path=source/record['manifest_path']
        if sha(path)!=record['sha256'] or sha(metadata_path)!=record['manifest_sha256']:
            raise ValueError('External image or metadata changed')
        metadata=json.loads(metadata_path.read_text())
        if metadata['source']!='ZSNS004' or metadata['source_role']!='external_pretraining':raise ValueError('Invalid shard role')
        if metadata['patch_shape']!=[17,17,17] or metadata['patch_half_extent_um']!=[8.,8.,8.]:raise ValueError('Physical grid mismatch')
        with np.load(path,allow_pickle=False) as a:data=dict(a)
        examples=external_examples(data)
        if sum(r['target'] for r in examples)!=int(data['division_target'].sum()):raise ValueError('Lost external division')
        contexts=[];masks=[];cache={}
        for row in examples:
            parent=row['indices'][0]
            if parent not in cache:
                patch=data['source_patches'][parent,[1,1,2]]
                cache[parent]=image_context(patch,row['anchors'])
            context,mask=cache[parent];context=context.copy();context[:3]=row['anchors']
            contexts.append(context);masks.append(mask)
        arrays=dict(indices=np.asarray([r['indices'] for r in examples],dtype=np.int64),
            geometry=np.asarray([r['geometry'] for r in examples],dtype=np.float32),
            context=np.asarray(contexts,dtype=np.float16),mask=np.asarray(masks,dtype=bool),
            targets=np.asarray([r['target'] for r in examples],dtype=np.float32),
            eligible=np.asarray([r['eligible'] for r in examples],dtype=bool),weights=np.ones(len(examples),dtype=np.float32))
        # Exercise actual image indexing during preparation; no encoder/model is opened.
        probe=aligned_external(data['source_patches'],data['target_patches'],arrays['indices'][:2])
        if not np.isfinite(probe).all() or probe.shape[1:]!=(3,3,17,17,17):raise ValueError('Aligned image contract failed')
        name=Path(record['path']).name
        target=output/'external'/name;shutil.copyfile(path,target)
        descriptor=output/'descriptors'/name;np.savez_compressed(descriptor,**arrays)
        positives=arrays['targets']>.5;eligible=arrays['eligible']
        summary=dict(examples=len(examples),positives=int(positives.sum()),
            eligible_positives=int((positives&eligible).sum()),eligible_negatives=int((~positives&eligible).sum()))
        for key,value in summary.items():totals[key]+=value
        rows.append(dict(external_path='external/'+name,external_sha256=sha(target),external_bytes=target.stat().st_size,
            descriptor_path='descriptors/'+name,descriptor_sha256=sha(descriptor),descriptor_bytes=descriptor.stat().st_size,
            csv_timepoint=record['csv_timepoint'],source_manifest_sha256=record['manifest_sha256'],**summary))
        print(json.dumps(dict(event='external_pairs',complete=index+1,total=len(block['records']),**summary)),flush=True)
    if totals['positives']!=425:raise ValueError('External positive count changed')
    if any(sha(ROOT/name)!=digest for name,digest in pins.items()):raise ValueError('Frozen recipe changed')
    result=dict(status='complete',run_id='zebrahub-division-transfer-v1',source='ZSNS004',source_role='external_pretraining',
        source_manifest_sha256=sha(manifest_path),files=rows,totals=totals,source_pins=pins,
        external_validation_opened=False,external_audit_opened=False,competition_test_data_read=False,
        labels_are_public_tracking_derived=True,authorized_for_submission=False,elapsed_seconds=time.monotonic()-start)
    path=output/'EXTERNAL.json';path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='complete',**totals,elapsed_seconds=result['elapsed_seconds'],manifest_sha256=sha(path))),flush=True)


if __name__=='__main__':main()
