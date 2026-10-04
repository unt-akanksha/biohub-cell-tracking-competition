"""Replay all14 uncertainty fits and actual final model from frozen arrays."""
import json
import os
from pathlib import Path
import runpy
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.focus_conditional_motion import predict as mean_predict, metrics as baseline_metrics
from research.focus_conditional_variance import fit, predict, metrics, gate


def verify():
    runner = runpy.run_path(str(ROOT / 'scripts/fit-focus-conditional-variance.py'))
    run, sha = runner['RUN'], runner['sha']
    path = ROOT / f'reports/experiments/{run}-result.json'
    result = json.loads(path.read_text())
    if any(sha(ROOT / p) != v for p, v in result['source_hashes'].items()):
        raise ValueError('Executed uncertainty implementation changed')
    verified, original_final, previous, samples = runner['load']()
    stems = previous['training_stems']
    if (result['prior_verification'] != verified or result['training_stems'] != stems
            or len(result['folds']) != 14 or len(result['per_movie']) != 14):
        raise ValueError('Exact14fold training provenance required')
    cache = ROOT / '.biohub/cache' / run
    rows = []
    for held, oldfold, record, actual in zip(stems, previous['folds'], result['folds'], result['per_movie']):
        train = [s for s in stems if s != held]
        x = np.concatenate([samples[s]['x'] for s in train])
        y = np.concatenate([samples[s]['y'] for s in train])
        model_path = cache / (held + '-model.json')
        model = json.loads(model_path.read_text())
        if (record != dict(held_out_stem=held, model_file=model_path.name, model_sha256=sha(model_path))
                or model != fit(oldfold['model'], x, y, 'fitting')):
            raise ValueError('Actual saved fitting-only variance model replay differs')
        sample = samples[held]
        mean = mean_predict(oldfold['model'], sample['x'])
        row = dict(stem=held, training_stems=train,
                   baseline=baseline_metrics(sample['y'], mean, oldfold['model']['variance_um2']),
                   candidate=metrics(sample['y'], mean, predict(model, sample['x'])))
        if row != actual:
            raise ValueError('Exact actual held-out uncertainty metrics differ')
        rows.append(row)
    decision = gate(rows)
    if decision != result['gate'] or not decision['passed']:
        raise ValueError('Complete passing unchanged screen required')
    final = result['final_model']
    model_path = cache / 'model.json'
    if final != dict(file='model.json', sha256=sha(model_path), exact_prediction_replay=True):
        raise ValueError('Actual final model export required')
    model = json.loads(model_path.read_text())
    x = np.concatenate([samples[s]['x'] for s in stems])
    y = np.concatenate([samples[s]['y'] for s in stems])
    if model != fit(original_final, x, y, 'fitting'):
        raise ValueError('Final all-fitting model replay differs')
    if any(result[k] != 0 for k in ('gpu_seconds', 'source_movies_evaluated', 'new_target_movies_opened')) or result['authorized_for_submission'] is not False:
        raise ValueError('Training-only resource/evaluation scope required')
    return dict(status='verified_conditional_variance_training_folds', result_sha256=sha(path),
                final_model_sha256=sha(model_path), exact_fold_replays=14, gate=decision,
                authorized_for_submission=False), model


if __name__ == '__main__':
    path = ROOT / 'reports/experiments/focus-conditional-variance-v1-verification.json'
    if path.exists():
        raise ValueError('Never overwrite actual verification')
    receipt, _ = verify()
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False))
    print(json.dumps(receipt, indent=2))
