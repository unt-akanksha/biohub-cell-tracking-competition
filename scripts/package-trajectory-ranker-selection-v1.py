"""Package unchanged inference for ten frozen selection movies, no labels."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def replace(text,old,new):
    assert text.count(old)==1
    return text.replace(old,new)


def main():
    scope=ROOT/'.biohub/cache/trajectory-ranker-selection-v1-plan'
    plan=scope/'MOVIES.json'
    assert sha(plan)=='1d9cbb244bbdc23d631c45811ccd222bb2e29d5b70a2cd6f05fa87aeb5ff297e'
    stems=[m['stem'] for m in json.loads(plan.read_text())['movies']]
    image_path=scope/'IMAGE_MANIFEST.json';images=json.loads(image_path.read_text())
    assert images['status']=='complete' and images['stems']==stems and len(images['records'])==1020
    assert images['private_plan_sha256']=='db11cb7929b3369c0363a34112f8c4bd514dbd5e1ffd2f9a53499cee991c5791'
    assert images['bytes']==4333540727 and not images['ground_truth_opened']
    base=ROOT/'.biohub/cache/trajectory-division-full-movie-v1-bundle'
    assert sha(base/'CONTRACT.json')=='7d4d1bc9f81f37fc2a8d23bd2449fc53f99bfb23eff39d02460b884661f8bdd1'
    contract=json.loads((base/'CONTRACT.json').read_text())
    for name,digest in contract['bundle_sha256'].items():assert sha(base/name)==digest
    target=ROOT/'.biohub/cache/trajectory-ranker-selection-v1-bundle';target.mkdir(exist_ok=False)
    for name in contract['bundle_sha256']:shutil.copy2(base/name,target/name)
    helper=target/'trajectory_division_full_movie_v1.py'
    text=replace(helper.read_text(encoding='utf-8'),
        "STEMS = ('44b6_12dfb391', '44b6_267148e4', '6bba_062c8d37', '6bba_07e24132')",'STEMS = '+repr(tuple(stems)))
    compile(text,str(helper),'exec');helper.write_bytes(text.encode())
    runner=target/'run-trajectory-division-full-movie-v1.py'
    text=replace(runner.read_text(encoding='utf-8'),"len(image_manifest['records']) != 408","len(image_manifest['records']) != 1020")
    text=replace(text,"run_id='trajectory-division-full-movie-v1'","run_id='trajectory-ranker-selection-v1'")
    compile(text,str(runner),'exec');runner.write_bytes(text.encode())
    contract.update(run_id='trajectory-ranker-selection-v1',stems=stems,image_manifest_sha256=sha(image_path),
                    source_scope_sha256=sha(plan),selection_predictions_only=True,model_training_performed=False,
                    purpose='Original baseline selection graphs for fixed trajectory ranker; no GT or trained ranker applied in collection')
    contract['full']['movies']=10
    contract['bundle_sha256']={name:sha(target/name) for name in contract['bundle_sha256']}
    (target/'CONTRACT.json').write_bytes((json.dumps(contract,indent=2)+'\n').encode())
    archive=target.with_suffix('.tar')
    with tarfile.open(archive,'w') as tar:tar.add(target,arcname=target.name)
    receipt=dict(status='selection_collection_staged',contract_sha256=sha(target/'CONTRACT.json'),
                 archive_sha256=sha(archive),images_sha256=sha(image_path),bytes=archive.stat().st_size,
                 movies=stems,full_wall_cap_seconds=3600,smoke_required=True,authorized_for_submission=False)
    (ROOT/'reports/experiments/trajectory-ranker-selection-v1-build.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
