"""Freeze source-only admission, fixed mixtures, code and budgets before fits."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    target=ROOT/'.biohub/cache/native-correspondence-v2-training-r3-bundle'; target.mkdir(exist_ok=False)
    files=['research/visual_correspondence_models_v1.py','research/native_correspondence_models_v2.py',
           'research/visual_correspondence_data_v1.py','scripts/train-native-correspondence-v2.py']
    records=[]
    for relative in files:
        path=target/relative; path.parent.mkdir(exist_ok=True,parents=True); shutil.copy2(ROOT/relative,path)
        records.append(dict(path=relative,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    contract=dict(run_id='native-correspondence-v2',files=records,max_steps_per_member=8000,batch_size=12,
        optimizer='AdamW lr2e-4 cosine2e-6 WD.01 AMPfp16 clip2',seed=20260914,
        source_gate='>=2% lower selection NLL; no fewer correct; missing-parent NLL nonregression',
        cross_embryo_gate='Only source-admitted models; require correct>=distance and NLL<distance; missing NLL<=distance',
        mixture='equal probabilities only if both members independently source-admitted; no target weight/threshold search',
        promotion='complete-movie patched official score, per-embryo and worst-movie nonregression, no metric hacks, runtime acceptance',
        total_watchdog_seconds=5400,projected_launch_cap_seconds=5100,source_selection_frequency=500,
        native_scales_um=[.8125,1.625,3.25],jitter_um=.8125,missing_parent_dropout=.2,
        random_initialization=True,previous_failed_weights_used=False,cudnn_benchmark=False,authorized_for_submission=False)
    (target/'TRAINING.json').write_text(json.dumps(contract,indent=2)+'\n')
    archive=target.with_suffix('.tar')
    with tarfile.open(archive,'w') as tar:
        for path in sorted(target.rglob('*')):
            if path.is_file(): tar.add(path,arcname=path.relative_to(target).as_posix())
    print(json.dumps(dict(status='packaged',archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                         training_contract_sha256=hashlib.sha256((target/'TRAINING.json').read_bytes()).hexdigest())))


if __name__=='__main__': main()
