"""Freeze original-pipeline source collection after exact image inventory recovery."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source,old,new):
    assert source.count(old)==1
    return source.replace(old,new)


def main():
    p=argparse.ArgumentParser();p.add_argument('--image-manifest',type=Path,required=True);args=p.parse_args()
    plan=ROOT/'.biohub/cache/trajectory-disagreement-source-v1-plan/MOVIES.json'
    assert sha(plan)=='5c4b64793068565598537db6bb3af439513bde3d46e357ffeab59164bda020bf'
    movies=json.loads(plan.read_text());stems=[m['stem'] for m in movies['movies']]
    images=json.loads(args.image_manifest.read_text())
    assert images['status']=='complete' and images['stems']==stems and len(images['records'])==816
    assert images['private_plan_sha256']=='9e405d2c7274f6a1a8640716d45a2285d6f388ebcb7b241eda4890a54d2bd64b'
    assert images['bytes']==3738158140 and not images['ground_truth_opened']
    source=ROOT/'.biohub/cache/trajectory-division-full-movie-v1-bundle'
    assert sha(source/'CONTRACT.json')=='7d4d1bc9f81f37fc2a8d23bd2449fc53f99bfb23eff39d02460b884661f8bdd1'
    contract=json.loads((source/'CONTRACT.json').read_text())
    for name,digest in contract['bundle_sha256'].items():assert sha(source/name)==digest
    target=ROOT/'.biohub/cache/trajectory-disagreement-source-v1-bundle';target.mkdir(exist_ok=False)
    for name in contract['bundle_sha256']:shutil.copy2(source/name,target/name)
    helper=target/'trajectory_division_full_movie_v1.py'
    text=replace_once(helper.read_text(encoding='utf-8'),
        "STEMS = ('44b6_12dfb391', '44b6_267148e4', '6bba_062c8d37', '6bba_07e24132')",'STEMS = '+repr(tuple(stems)))
    compile(text,str(helper),'exec');helper.write_bytes(text.encode())
    runner=target/'run-trajectory-division-full-movie-v1.py'
    text=replace_once(runner.read_text(encoding='utf-8'),"len(image_manifest['records']) != 408","len(image_manifest['records']) != 816")
    text=replace_once(text,"run_id='trajectory-division-full-movie-v1'","run_id='trajectory-disagreement-source-v1'")
    compile(text,str(runner),'exec');runner.write_bytes(text.encode())
    contract.update(run_id='trajectory-disagreement-source-v1',stems=stems,
        image_manifest_sha256=sha(args.image_manifest),source_scope_sha256=sha(plan),
        source_only_collection=True,motion_model_validation_movies_excluded=False,
        purpose='Source fitting corpus: original neural/ILP/postprocess disagreements; no labels or model fitting in this run',
        model_training_performed=False,validation_result=False)
    contract['full']['movies']=8
    contract['bundle_sha256']={name:sha(target/name) for name in contract['bundle_sha256']}
    (target/'CONTRACT.json').write_bytes((json.dumps(contract,indent=2)+'\n').encode())
    archive=target.with_suffix('.tar')
    with tarfile.open(archive,'w') as tar:tar.add(target,arcname=target.name)
    receipt=dict(status='source_collection_staged',contract_sha256=sha(target/'CONTRACT.json'),
        archive_sha256=sha(archive),bytes=archive.stat().st_size,images_sha256=sha(args.image_manifest),
        movies=stems,smoke_required_before_full=True,full_wall_cap_seconds=3600,
        model_training_performed=False,authorized_for_submission=False)
    (ROOT/'reports/experiments/trajectory-disagreement-source-v1-build.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
