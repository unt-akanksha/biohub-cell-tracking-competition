"""Package the exact tested code and private authorized plan for owned EC2."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    target=ROOT/'.biohub/cache/native-correspondence-v2-bundle'
    target.mkdir(exist_ok=False)
    files=['research/kaggle_archive_ranges.py','research/native_correspondence_data_v2.py',
           'research/visual_correspondence_data_v1.py','scripts/build-native-correspondence-v2.py']
    records=[]
    for relative in files:
        path=target/relative; path.parent.mkdir(exist_ok=True,parents=True)
        shutil.copy2(ROOT/relative,path)
        records.append(dict(path=relative,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    plan=target/'plan'; plan.mkdir()
    for name in ('MOVIES.json','PRIVATE_ARCHIVE_PLAN.json'):
        shutil.copy2(ROOT/'.biohub/cache/native-correspondence-v2-plan'/name, plan/name)
    (plan/'CODE.json').write_text(json.dumps(dict(files=records),indent=2)+'\n')
    archive=target.with_suffix('.tar')
    with tarfile.open(archive,'w') as tar:
        for path in sorted(target.rglob('*')):
            if path.is_file(): tar.add(path,arcname=path.relative_to(target).as_posix())
    print(json.dumps(dict(status='packaged',bytes=archive.stat().st_size,
                         archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                         code_sha256=hashlib.sha256((plan/'CODE.json').read_bytes()).hexdigest())))


if __name__=='__main__': main()
