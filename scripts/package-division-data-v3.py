"""Merge disjoint legacy/event-enriched triplets and freeze encoder provenance."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output=ROOT/'.biohub/cache/native-division-v3-data-bundle';output.mkdir(exist_ok=False)
    records=[];seen=set();totals={};provenance={}
    for name,folder,status in [('legacy',ROOT/'.biohub/cache/native-division-v3-legacy','legacy_triplets_complete'),
                               ('extra',ROOT/'.biohub/cache/native-division-v3-extra-output','enrichment_complete')]:
        source=folder/'RESULT.json';manifest=json.loads(source.read_text())
        if manifest['status']!=status or manifest['sealed_audit_opened'] or manifest['target_pilot_labels_used']:raise ValueError('Unverified/forbidden triplet inputs')
        provenance[name]=sha(source)
        for r in manifest['records']:
            key=(r['stem'],r['transition'])
            if key in seen:raise ValueError('Duplicate transition across enrichment and legacy')
            seen.add(key);p=folder/r['path']
            if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('Triplet artifact mismatch')
            destination=output/name/r['path'];destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,destination)
            record={**r,'path':destination.relative_to(output).as_posix(),'origin':name};records.append(record)
            total=totals.setdefault(r['embryo']+'-'+r['role'],dict(positive=0,negative=0,triples=0))
            for k in total:total[k]+=r[k]
    training_path=ROOT/'reports/experiments/native-correspondence-v2-training-full-result.json'
    if sha(training_path)!='42224d3762ee3c7ab9bc5c6997e5efd5cf72c5eb5e848b1925b69c4500ef7a46':raise ValueError('Source encoder training changed')
    trained=json.loads(training_path.read_text());encoders=[]
    for member in trained['members']:
        encoders.append(dict(embryo=member['embryo'],family=member['family'],sha256=member['weights_sha256'],
            path='/tmp/biohub-image-context-v2.ScdSdY/native-correspondence-v2-training-full/'+member['embryo']+'-'+member['family']+'/best.pt'))
    result=dict(run_id='native-division-v3',status='triplets_merged',records=records,totals=totals,
        input_manifest_sha256=provenance,encoders=encoders,encoder_training_sha256=sha(training_path),
        encoder_policy='For each reciprocal fold use only encoders trained on that source embryo; old parent heads are discarded',
        old_source44_parent_models_promoted=False,target_pilot_labels_used=False,authorized_for_submission=False)
    (output/'DATA.json').write_text(json.dumps(result,indent=2)+'\n')
    archive=output.with_suffix('.tar')
    with tarfile.open(archive,'w') as tar:
        for p in sorted(output.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(output).as_posix())
    receipt=dict(status=result['status'],totals=totals,packets=len(records),data_sha256=sha(output/'DATA.json'),
                 archive_sha256=sha(archive),archive_bytes=archive.stat().st_size)
    (ROOT/'reports/experiments/native-division-v3-data-build.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
