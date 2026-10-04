"""Package the source-only image proposal audit with exact archive bindings."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output=ROOT/'.biohub/cache/native-division-proposal-audit-v4-bundle';output.mkdir(exist_ok=False)
    files=['scripts/audit-division-proposals-v4.py','scripts/build-native-correspondence-v2.py',
           'research/native_division_proposal_audit_v4.py','research/native_correspondence_data_v2.py',
           'research/visual_correspondence_data_v1.py','research/kaggle_archive_ranges.py',
           'reports/experiments/native-division-proposal-audit-v4-design.md']
    for relative in files:
        path=output/relative;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,path)
    (output/'plan').mkdir()
    for name in ('MOVIES.json','PRIVATE_ARCHIVE_PLAN.json'):
        shutil.copy2(ROOT/'.biohub/cache/native-division-proposal-audit-v4-plan'/name,output/'plan'/name)
    contract=dict(run_id='native-division-proposal-audit-v4',max_seconds=600,source_only=True,authorized_for_submission=False,
                  files=[dict(path=p.relative_to(output).as_posix(),sha256=sha(p)) for p in sorted(output.rglob('*')) if p.is_file()])
    (output/'CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
    archive=output.with_suffix('.tar')
    with tarfile.open(archive,'w') as tf:
        for p in sorted(output.rglob('*')):
            if p.is_file():tf.add(p,arcname=p.relative_to(output).as_posix())
    receipt=dict(status='proposal_audit_packaged',archive_sha256=sha(archive),contract_sha256=sha(output/'CONTRACT.json'),bytes=archive.stat().st_size)
    (ROOT/'reports/experiments/native-division-proposal-audit-v4-build.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
