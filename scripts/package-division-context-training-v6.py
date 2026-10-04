"""Freeze paired temporal training and original source-encoder provenance."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output=ROOT/'.biohub/cache/native-division-context-v6-training-bundle-r2';output.mkdir(exist_ok=False)
    files=['scripts/train-division-context-v6.py','scripts/train-division-heads-v3.py',
           'research/native_division_context_model_v6.py','research/native_correspondence_models_v2.py',
           'research/visual_correspondence_models_v1.py','reports/experiments/native-division-context-v6-training-design.md']
    for relative in files:
        path=output/relative;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,path)
    data=ROOT/'.biohub/cache/native-division-v3-data-bundle/DATA.json'
    if sha(data)!='7e0e7b1ca3c11b27319824fad3746b49f8a1cf519a73d3dad38e12d2ef8a0e06':raise ValueError('Original source encoders changed')
    (output/'ENCODERS.json').write_text(json.dumps(json.loads(data.read_text())['encoders'],indent=2)+'\n')
    contract=dict(run_id='native-division-context-v6-training',max_seconds=1800,successful_updates=12000,
                  files=[dict(path=p.relative_to(output).as_posix(),sha256=sha(p)) for p in sorted(output.rglob('*')) if p.is_file()],
                  authorized_for_submission=False)
    (output/'CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
    archive=output.with_suffix('.tar')
    with tarfile.open(archive,'w') as tf:
        for p in sorted(output.rglob('*')):
            if p.is_file():tf.add(p,arcname=p.relative_to(output).as_posix())
    receipt=dict(status='temporal_training_packaged',archive_sha256=sha(archive),contract_sha256=sha(output/'CONTRACT.json'),bytes=archive.stat().st_size)
    (ROOT/'reports/experiments/native-division-context-v6-training-build-r2.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
