"""Verify saved LOMO models, complete held-out predictions and training objectives."""
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
from research.focus_candidate_ranker import combine, metrics, validate, objective


def verify():
    runner = runpy.run_path(str(ROOT / 'scripts/fit-focus-candidate-ranker.py'))
    run, sha = runner['RUN'], runner['sha']
    path = ROOT / f'reports/experiments/{run}-lomo.json'
    result = json.loads(path.read_text())
    if any(sha(ROOT / p) != value for p, value in result['source_hashes'].items()):
        raise ValueError('Actual ranking method changed')
    evidence_path = ROOT / f'reports/experiments/{run}-data-smoke.json'
    if sha(evidence_path) != result['data_smoke_sha256']:
        raise ValueError('Actual full-candidate data provenance changed')
    evidence, samples = runner['load']()
    records = {r['stem']: r for r in evidence['records']}
    if [r['held_out'] for r in result['folds']] != list(samples):
        raise ValueError('Exact twelve held-out movies required')
    cache = ROOT / '.biohub/cache' / run / 'lomo'
    folds, objectives = [], []
    for row in result['folds']:
        held = row['held_out']
        train = [s for s in samples if s != held]
        model_path = cache / (held + '-model.json')
        saved = json.loads(model_path.read_text())
        if (sha(model_path) != row['model_sha256'] or saved['held_out'] != held
                or saved['training_stems'] != train or row['training_stems'] != train
                or json.loads((cache / (held + '-result.json')).read_text()) != row):
            raise ValueError('Actual saved fold identities differ')
        model = saved['model']
        data = combine([samples[s] for s in train])
        ids = validate(data)
        theta = np.asarray(model['theta'], float)
        value, gradient = objective(theta, data, ids)
        initial, _ = objective(np.zeros(8), data, ids)
        if (model['observations'] != len(data['starts']) or model['choices'] != len(data['offset'])
                or model['fitting_parent'] != int(data['present'].sum())
                or not np.isfinite(gradient).all() or abs(value - model['objective']) > 1e-7
                or abs(initial - model['initial_objective']) > 1e-7 or value > initial + 1e-8):
            raise ValueError('Actual fitting-only objective/count replay differs')
        objectives.append(dict(held_out=held, objective=value, gradient_inf_norm=float(np.abs(gradient).max())))
        del data, ids
        actual = metrics(samples[held], model)
        if (actual != row['candidate'] or records[held]['physical'] != row['physical']
                or records[held]['neural'] != row['neural']):
            raise ValueError('Exact held-out candidate and control replay differs')
        folds.append(row)
    gate = runner['gate'](folds)
    if gate != result['gate']:
        raise ValueError('Fixed screening decision differs')
    final = result['final_model']
    if gate['passed']:
        model_path = cache / 'model.json'
        if final != dict(file='model.json', sha256=sha(model_path), exact_prediction_replay=True):
            raise ValueError('Actual final model export required')
        model = json.loads(model_path.read_text())
        data = combine(list(samples.values()))
        ids = validate(data)
        value, _ = objective(np.asarray(model['theta']), data, ids)
        if abs(value - model['objective']) > 1e-7 or model['observations'] != 10915:
            raise ValueError('Final all-fitting objective differs')
    elif final is not None or (cache / 'model.json').exists():
        raise ValueError('Failed screen must not export a final promoted model')
    if any(result[k] != 0 for k in ('diagnostic_movies_opened', 'source_movies_opened', 'new_target_movies_opened', 'gpu_seconds')) or result['authorized_for_submission'] is not False:
        raise ValueError('Fitting-only resource/evaluation scope required')
    return dict(status='verified_full_candidate_ranker_lomo', result_sha256=sha(path),
                data_smoke_sha256=sha(evidence_path), complete_groups=evidence['total_groups'],
                complete_choices=evidence['total_choices'], all_saved_fold_predictions_replayed=True,
                training_objectives=objectives, gate=gate, final_model=final,
                authorized_for_submission=False,
                verification_scope='Exact stored model objectives, full candidate predictions, labels and gate arithmetic; not an independent optimizer refit')


if __name__ == '__main__':
    target = ROOT / 'reports/experiments/focus-candidate-ranker-v1-verification.json'
    if target.exists():
        raise ValueError('Never overwrite completed verification')
    result = verify()
    target.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k != 'training_objectives'}, indent=2))
