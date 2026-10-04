"""Stage the frozen prospective candidate only after cross-embryo admission."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.native_graph_repair_v2 import select_model_groups


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    reports=ROOT/'reports/experiments'
    training_path=reports/'native-correspondence-v2-training-full-result.json'
    cross_path=reports/'native-correspondence-v2-cross-embryo-result.json'
    training=json.loads(training_path.read_text());cross=json.loads(cross_path.read_text())
    if training['status']!='training_complete' or cross['status']!='diagnostic_complete' or cross['training_result_sha256']!=sha(training_path):
        raise ValueError('Require exact completed training and frozen cross diagnostic')
    chosen=select_model_groups(training,cross)
    if not chosen:
        print(json.dumps(dict(status='not_staged_no_admitted_native_group')));return
    target=ROOT/'.biohub/cache/native-repair-pilot-v2-bundle';target.mkdir(exist_ok=False)
    files=['research/native_correspondence_data_v2.py','research/native_correspondence_models_v2.py',
        'research/native_correspondence_inference_v2.py','research/visual_correspondence_models_v1.py',
        'research/native_graph_repair_v2.py','research/trajectory_runtime_v1.py','scripts/run-native-repair-pilot-v2.py']
    for relative in files:
        path=target/relative;path.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(ROOT/relative,path)
    image_manifest=ROOT/'.biohub/cache/trajectory-division-archive-plan-v1/IMAGE_MANIFEST.json'
    if sha(image_manifest)!='5bac85637ee2fa1ae6c7d960c11f88778781cf30b1aceafe9897273fa6d63543':raise ValueError('Image manifest changed')
    shutil.copy2(image_manifest,target/'IMAGE_MANIFEST.json')
    parent=ROOT/'.biohub/cache/trajectory-overlap-cache-full-v1-output'
    baseline=json.loads((parent/'result.json').read_text())
    if baseline['status']!='complete_prelabel_predictions' or baseline['contract_sha256']!='1b0124fd435802c47fffd072ed323d4bfde3aa95785db05d7a7eb9610fbdb7d1':
        raise ValueError('Accepted exact FP32 graph baseline changed')
    image_root='/tmp/biohub-image-context-v2.ScdSdY/trajectory-division-images-v1'
    movies=[];(target/'graphs').mkdir()
    for stem in ('44b6_12dfb391','6bba_062c8d37','44b6_267148e4','6bba_07e24132'):
        row=baseline['movies'][stem];source=parent/f'shard-{row["shard"]}'/(stem+'-original')/'repaired-prediction.json'
        if sha(source)!=row['original']['repaired_sha256']:raise ValueError('Parent graph changed')
        relative='graphs/'+stem+'.json';shutil.copy2(source,target/relative)
        movies.append(dict(stem=stem,graph=relative,images=image_root+'/train/'+stem+'.zarr'))
    for model in chosen:
        model['path']='/tmp/biohub-image-context-v2.ScdSdY/native-correspondence-v2-training-full/'+model['embryo']+'-'+model['family']+'/best.pt'
    contract=dict(run_id='native-repair-pilot-v2',models=chosen,movies=movies,image_root=image_root,
        files=[dict(path=p.relative_to(target).as_posix(),sha256=sha(p)) for p in sorted(target.rglob('*')) if p.is_file()],
        training_result_sha256=sha(training_path),cross_result_sha256=sha(cross_path),
        prediction_policy='native-graph-repair-v2-design.md: posterior>=.99, free parents, preserve divisions, no new nodes',
        design_sha256=sha(reports/'native-graph-repair-v2-design.md'),ground_truth_included=False,
        validation_parent_graphs_only=True,authorized_for_submission=False)
    (target/'PILOT.json').write_text(json.dumps(contract,indent=2)+'\n')
    archive=target.with_suffix('.tar')
    with tarfile.open(archive,'w') as tar:
        for path in sorted(target.rglob('*')):
            if path.is_file():tar.add(path,arcname=path.relative_to(target).as_posix())
    receipt=dict(status='staged_requires_smoke',pilot_sha256=sha(target/'PILOT.json'),archive_sha256=sha(archive),
                 archive_bytes=archive.stat().st_size,models=chosen,movies=[m['stem'] for m in movies])
    (reports/'native-repair-pilot-v2-build.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
