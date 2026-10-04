"""Hash-bound source-only smoke bundle; no Kaggle GPU or training GT."""
import hashlib
import argparse
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--full',action='store_true');args=parser.parse_args()
    target=ROOT/('.biohub/cache/dense-warp-v1-full-bundle' if args.full else '.biohub/cache/dense-warp-v1-bundle');target.mkdir(exist_ok=False)
    names=['scripts/smoke-dense-warp-v1.py','scripts/build-native-correspondence-v2.py',
           'research/dense_warp_correspondence_v1.py','research/kaggle_archive_ranges.py',
           'research/native_correspondence_data_v2.py','research/visual_correspondence_data_v1.py']
    files={name:ROOT/name for name in names}
    if args.full:
        files['scripts/train-dense-warp-v1.py']=ROOT/'scripts/train-dense-warp-v1.py'
        files['SMOKE.json']=ROOT/'reports/experiments/dense-warp-v1-smoke-result.json'
    public=ROOT/'.biohub/cache/public-d4-preflight-v1'
    old=json.loads((public/'MANIFEST.json').read_text())
    for name in ('primary.pth','temporal_unet.py','simple_node_transformer.py','train_unet_transformer.py','public_d4_preflight.py'):
        assert sha(public/name)==old['files'][name]['sha256']
        files['public/'+name]=public/name
    for name in ('MOVIES.json','PRIVATE_ARCHIVE_PLAN.json'):
        files['plan/'+name]=ROOT/('.biohub/cache/dense-warp-v1-full-plan' if args.full else '.biohub/cache/dense-warp-v1-plan')/name
    records=[]
    for name,source in files.items():
        dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        records.append(dict(path=name,sha256=sha(dest),bytes=dest.stat().st_size))
    contract=dict(run_id='dense-warp-v1-full' if args.full else 'dense-warp-v1',files=records,smoke_steps=25,hard_stop_seconds=1800 if args.full else 600,
                  authorized_for_submission=False,unknown_license_weights_used=False)
    (target/'CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
    archive=target.with_suffix('.tar')
    with tarfile.open(archive,'w') as tf:tf.add(target,arcname=target.name)
    receipt=dict(status='staged_smoke',contract_sha256=sha(target/'CONTRACT.json'),archive_sha256=sha(archive),bytes=archive.stat().st_size)
    (ROOT/('reports/experiments/dense-warp-v1-full-build.json' if args.full else 'reports/experiments/dense-warp-v1-build.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
