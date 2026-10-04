"""Hash-bound new enrichment code; reuse the tested archive/native image helper."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output=ROOT/'.biohub/cache/native-division-v3-extra-bundle';output.mkdir(exist_ok=False)
    files=['scripts/build-division-extra-v3.py','scripts/build-native-correspondence-v2.py',
        'research/native_correspondence_data_v2.py','research/native_division_data_v3.py',
        'research/visual_correspondence_data_v1.py','research/kaggle_archive_ranges.py']
    for relative in files:
        p=output/relative;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,p)
    (output/'plan').mkdir()
    for name in ('MOVIES.json','PRIVATE_ARCHIVE_PLAN.json'):shutil.copy2(ROOT/'.biohub/cache/native-division-v3-extra-plan'/name,output/'plan'/name)
    contract=dict(run_id='native-division-v3-extra',max_seconds=600,max_output_bytes=512*1024**2,
        files=[dict(path=p.relative_to(output).as_posix(),sha256=sha(p)) for p in sorted(output.rglob('*')) if p.is_file()],
        source_only=True,selection_unchanged=True,authorized_for_submission=False)
    (output/'CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
    archive=output.with_suffix('.tar')
    with tarfile.open(archive,'w') as tar:
        for p in sorted(output.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(output).as_posix())
    print(json.dumps(dict(status='enrichment_packaged',archive_sha256=sha(archive),contract_sha256=sha(output/'CONTRACT.json'),bytes=archive.stat().st_size)))


if __name__=='__main__':main()
