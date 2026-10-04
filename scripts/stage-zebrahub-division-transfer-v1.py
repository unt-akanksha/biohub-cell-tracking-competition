"""Freeze external-only pair inputs and code; reuse remote Biohub/warm starts."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    output=ROOT/'.biohub/cache/zebrahub-division-transfer-v1-bundle'
    archive=ROOT/'.biohub/cache/zebrahub-division-transfer-v1-bundle.tar.gz'
    if output.exists() or archive.exists():raise ValueError('Preserve frozen bundle')
    prepared=ROOT/'.biohub/cache/zebrahub-division-transfer-v1'
    if sha(prepared/'EXTERNAL.json')!='ce4d395302104b0c5dbf447a53e411adb48ac4cfea7fbe02eac5765bae78b971':raise ValueError('Prepared data changed')
    external=json.loads((prepared/'EXTERNAL.json').read_text())
    for name,digest in external['source_pins'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Prepared source recipe changed')
    base=ROOT/'.biohub/cache/image-context-pilot-v2-bundle'
    if sha(base/'BUNDLE.json')!='467076cacc2587444267f6e14f4edb93dd578c27fb1eff76ffbc06cbb290bdb6':raise ValueError('Base contract changed')
    base_manifest=json.loads((base/'BUNDLE.json').read_text())
    output.mkdir();shutil.copyfile(prepared/'EXTERNAL.json',output/'EXTERNAL.json')
    for name,digest in base_manifest['files'].items():
        if not name.startswith('research/') or not name.endswith('.py'):continue
        if sha(base/name)!=digest:raise ValueError('Base code changed')
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(base/name,target)
    for name in ('research/frozen_image_head.py','research/learned_division_recovery.py',
                 'research/zebrahub_division_transfer.py','research/zebrahub_transfer_head.py',
                 'scripts/run-zebrahub-division-transfer-v1.py','scripts/prepare-zebrahub-division-transfer-v1.py',
                 'reports/experiments/zebrahub-division-transfer-v1-design.md'):
        target=output/(name if name.startswith('research/') else Path(name).name)
        shutil.copyfile(ROOT/name,target)
    for record in external['files']:
        for key in ('external','descriptor'):
            name=record[key+'_path'];source=prepared/name
            if sha(source)!=record[key+'_sha256']:raise ValueError('Prepared artifact changed')
            target=output/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    contract=dict(run_id='zebrahub-division-transfer-v1',files={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()},
        external_manifest_sha256=sha(prepared/'EXTERNAL.json'),source_biohub_manifest_sha256='39032899ecea16e87bfb53d45909f06993d39ff0cfbb0bbb16a7b875f9dcecce',
        source_base_sha256=sha(base/'BUNDLE.json'),authorized_for_submission=False)
    path=output/'CONTRACT.json';path.write_text(json.dumps(contract,indent=2)+'\n')
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for p in sorted(output.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(output).as_posix())
    print(json.dumps(dict(status='staged',contract_sha256=sha(path),files=len(contract['files']),archive_sha256=sha(archive),bytes=archive.stat().st_size)))


if __name__=='__main__':main()
