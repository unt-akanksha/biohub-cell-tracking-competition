"""Bind new positive extraction to existing data, source recipes and image archive."""
import hashlib
import argparse
import json
from pathlib import Path
import shutil
import tarfile
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--revision',default='r2');args=parser.parse_args()
    if args.revision!='r2':raise ValueError('Only the named functionality repair is authorized')
    output=ROOT/'.biohub/cache'/('native-division-additive-v5-bundle-'+args.revision);output.mkdir(exist_ok=False)
    files=['scripts/extract-division-additive-v5.py','scripts/build-native-correspondence-v2.py',
           'research/native_correspondence_data_v2.py','research/visual_correspondence_data_v1.py','research/kaggle_archive_ranges.py',
           'reports/experiments/native-division-additive-v5-design.md']
    for relative in files:
        path=output/relative;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,path)
    (output/'plan').mkdir()
    shutil.copy2(ROOT/'.biohub/cache/native-division-additive-v5-plan/RECIPES.json',output/'plan/RECIPES.json')
    shutil.copy2(ROOT/'.biohub/cache/native-division-proposal-audit-v4-plan/PRIVATE_ARCHIVE_PLAN.json',output/'plan/PRIVATE_ARCHIVE_PLAN.json')
    old=ROOT/'.biohub/cache/native-division-v3-data-bundle/DATA.json'
    if sha(old)!='7e0e7b1ca3c11b27319824fad3746b49f8a1cf519a73d3dad38e12d2ef8a0e06':raise ValueError('Old data changed')
    shutil.copy2(old,output/'plan/OLD_DATA.json')
    contract=dict(run_id='native-division-additive-v5',max_seconds=300,authorized_for_submission=False,
                  files=[dict(path=p.relative_to(output).as_posix(),sha256=sha(p)) for p in sorted(output.rglob('*')) if p.is_file()])
    (output/'CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
    archive=output.with_suffix('.tar')
    with tarfile.open(archive,'w') as tf:
        for p in sorted(output.rglob('*')):
            if p.is_file():tf.add(p,arcname=p.relative_to(output).as_posix())
    receipt=dict(status='additive_extraction_packaged',archive_sha256=sha(archive),contract_sha256=sha(output/'CONTRACT.json'),bytes=archive.stat().st_size)
    (ROOT/'reports/experiments'/('native-division-additive-v5-build-'+args.revision+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
