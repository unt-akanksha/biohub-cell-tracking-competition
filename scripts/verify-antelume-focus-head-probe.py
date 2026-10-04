"""Recompute actual cloud/Kaggle logit differences from recovered matrices."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder=ROOT/'.biohub/cache/antelume-focus-head-replay-v1';target=ROOT/'reports/experiments/antelume-focus-head-replay-v1-result.json'
    if target.exists():raise ValueError('Never overwrite cloud replay evidence')
    identity=json.loads((folder/'staged_identity.json').read_text());result=json.loads((folder/'result.json').read_text())
    if sha(folder/'probe.zip')!=identity['archive_sha256']:raise ValueError('Exact staged probe required')
    with zipfile.ZipFile(folder/'probe.zip') as archive:
        manifest=json.loads(archive.read('manifest.json'));records=[]
        if (result['status']!='completed_antelume_frozen_head_replay' or result['model_before']!=manifest['model_tensor_sha256']
            or result['model_after']!=result['model_before'] or not 0<result['elapsed_seconds']<=180
            or result['new_target_movies_opened']!=0 or any(result[k] is not False for k in
                ('optimizer_run','new_training_data_opened','source_selection_opened','authorized_for_submission'))):raise ValueError('Exact frozen bounded probe terminal required')
        for item in manifest['pairs']:
            path=folder/('actual_'+item['stem']+'_'+str(item['frame'])+'.npz')
            with np.load(path,allow_pickle=False) as saved:
                if saved.files!=['logits']:raise ValueError('Exact real-logit artifact required')
                actual=saved['logits'].copy()
            with np.load(io.BytesIO(archive.read(item['reference'])),allow_pickle=False) as saved:reference=saved['neural_'+str(item['frame'])].copy()
            if actual.shape!=reference.shape or actual.dtype!=reference.dtype or not np.isfinite(actual).all():raise ValueError('Actual finite logit shape/dtype changed')
            row=dict(stem=item['stem'],frame=item['frame'],shape=list(actual.shape),exact_kaggle_replay=bool(np.array_equal(actual,reference)),
                max_abs_error=float(np.abs(actual-reference).max()),real_parent_argmax_changes=int((actual.argmax(0)!=reference.argmax(0)).sum()))
            records.append(row)
        if records!=result['records'] or result['exact_kaggle_replay']!=all(r['exact_kaggle_replay'] for r in records):
            raise ValueError('Recovered matrices disagree with reported replay')
    output=dict(status='verified_antelume_head_replay_comparison',worker=result,archive_sha256=identity['archive_sha256'],
        worker_result_sha256=sha(folder/'result.json'),matrix_sha256={p.name:sha(p) for p in sorted(folder.glob('actual_*.npz'))},
        qualifies_for_original_exact_replay_protocol=result['exact_kaggle_replay'],authorized_for_submission=False)
    target.write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))


if __name__=='__main__':main()
