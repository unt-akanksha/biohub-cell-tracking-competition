"""Freeze a functionality-only temporal model smoke, separate from data extraction."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output=ROOT/'.biohub/cache/native-division-context-v6-model-bundle';output.mkdir(exist_ok=False)
    files=['scripts/smoke-division-context-model-v6.py','research/native_division_context_model_v6.py',
           'research/native_correspondence_models_v2.py','research/visual_correspondence_models_v1.py']
    for name in files:
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    contract=dict(run_id='native-division-context-v6-model-smoke',max_seconds=120,
                  files=[dict(path=p.relative_to(output).as_posix(),sha256=sha(p)) for p in sorted(output.rglob('*')) if p.is_file()],
                  authorized_for_submission=False)
    (output/'CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
    archive=output.with_suffix('.tar')
    with tarfile.open(archive,'w') as tf:
        for p in sorted(output.rglob('*')):
            if p.is_file():tf.add(p,arcname=p.relative_to(output).as_posix())
    print(json.dumps(dict(status='model_smoke_packaged',archive_sha256=sha(archive),contract_sha256=sha(output/'CONTRACT.json'),bytes=archive.stat().st_size)))


if __name__=='__main__':main()
