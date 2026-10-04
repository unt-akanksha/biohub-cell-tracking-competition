"""Freeze temporal extraction code and source frame access plan."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output=ROOT/'.biohub/cache/native-division-context-v6-bundle';output.mkdir(exist_ok=False)
    files=['scripts/extract-division-context-v6.py','scripts/build-native-correspondence-v2.py',
           'research/native_division_context_data_v6.py','research/native_correspondence_data_v2.py',
           'research/visual_correspondence_data_v1.py','research/kaggle_archive_ranges.py',
           'reports/experiments/native-division-context-v6-design.md']
    for relative in files:
        path=output/relative;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,path)
    (output/'plan').mkdir()
    for name in ('MOVIES.json','PRIVATE_ARCHIVE_PLAN.json'):
        shutil.copy2(ROOT/'.biohub/cache/native-division-context-v6-plan'/name,output/'plan'/name)
    contract=dict(run_id='native-division-context-v6',max_seconds=1200,max_output_bytes=512*1024**2,
                  files=[dict(path=p.relative_to(output).as_posix(),sha256=sha(p)) for p in sorted(output.rglob('*')) if p.is_file()],
                  authorized_for_submission=False)
    (output/'CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
    archive=output.with_suffix('.tar')
    with tarfile.open(archive,'w') as tf:
        for p in sorted(output.rglob('*')):
            if p.is_file():tf.add(p,arcname=p.relative_to(output).as_posix())
    receipt=dict(status='context_extraction_packaged',archive_sha256=sha(archive),contract_sha256=sha(output/'CONTRACT.json'),bytes=archive.stat().st_size)
    (ROOT/'reports/experiments/native-division-context-v6-build.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
