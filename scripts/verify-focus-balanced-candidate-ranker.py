"""Replay weighted fitting objectives, actual stored decisions and fixed gates."""
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
from research.focus_candidate_ranker import combine, metrics, validate
from research.focus_balanced_candidate_ranker import class_weights, objective, frame_packet, error_counts


def verify_fit(data, model):
    ids = validate(data)
    weights, absent_weight = class_weights(data, 'fitting')
    theta = np.asarray(model['theta'], float)
    if (model['role'] != 'fitting' or model['parent_weight'] != 1.
            or model['weight_rule'] != 'sqrt(fitting_parent/fitting_absent)'
            or model['absent_weight'] != absent_weight or model['ridge'] != 1.
            or model['observations'] != len(data['starts']) or model['choices'] != len(data['offset'])
            or model['fitting_parent'] != int(data['present'].sum())
            or model['fitting_absent'] != int((data['present'] == 0).sum())
            or theta.shape != (8,) or not np.isfinite(theta).all() or (theta[4:7] > .5).any()):
        raise ValueError('Actual fitting-only weights, complete counts and coefficients required')
    value, gradient = objective(theta, data, ids, weights)
    initial, _ = objective(np.zeros(8), data, ids, weights)
    scores = data['offset'] + data['features'] @ theta
    independent = .5 * np.dot(theta[1:], theta[1:])
    for start, size, chosen, weight in zip(data['starts'], data['sizes'], data['chosen'], weights):
        independent += weight * (np.logaddexp.reduce(scores[start:start + size]) - scores[chosen])
    if (abs(value - model['objective']) > 1e-7 or abs(value - independent) > 1e-7
            or abs(initial - model['initial_objective']) > 1e-7 or value > initial + 1e-8
            or not np.isfinite(gradient).all()):
        raise ValueError('Saved weighted objective or independent complete-group replay differs')
    return dict(objective=value, independent_group_objective=float(independent),
                gradient_inf_norm=float(np.abs(gradient).max()), absent_weight=absent_weight,
                fitting_parent=model['fitting_parent'], fitting_absent=model['fitting_absent'])


def verify():
    runner = runpy.run_path(str(ROOT / 'scripts/fit-focus-balanced-candidate-ranker.py'))
    run, sha, report, cache = (runner[k] for k in ('RUN', 'sha', 'REPORT', 'CACHE'))
    result_path = report / f'{run}-lomo.json'
    result = json.loads(result_path.read_text())
    smoke = runner['require_smoke']()
    if (result['status'] != 'completed_balanced_candidate_ranker_lomo' or result['run_id'] != run
            or result['source_hashes'] != runner['sources']()
            or result['smoke_sha256'] != sha(report / f'{run}-smoke.json')
            or result['data_smoke_sha256'] != runner['DATA_SHA']
            or result['prior_result_sha256'] != runner['PRIOR_SHA']
            or result['prior_verification_sha256'] != runner['VERIFY_SHA']):
        raise ValueError('Actual immutable balanced experiment and upstream identities required')
    evidence, samples, previous = runner['load']()
    records = {r['stem']: r for r in evidence['records']}
    prior = {r['held_out']: r for r in previous['folds']}
    smoke_choices = []
    for stem, sample in samples.items():
        for frame in np.unique(sample['source_frame']):
            keep = sample['source_frame'] == frame
            labels = sample['present'][keep]
            if labels.min() == 0 and labels.max() == 1:
                smoke_choices.append((-int((sample['sizes'][keep] - 1).sum()), stem, int(frame)))
    choices, stem, frame = min(smoke_choices)
    if (smoke['stem'], smoke['source_frame'], smoke['real_choices']) != (stem, frame, -choices):
        raise ValueError('Frozen largest-both-class smoke selection differs')
    packet = frame_packet(samples[stem], frame)
    smoke_model = json.loads((cache / 'smoke-model.json').read_text())
    if smoke_model != smoke['model'] or smoke['groups'] != len(packet['starts']):
        raise ValueError('Actual saved smoke model/packet differs')
    smoke_fit = verify_fit(packet, smoke_model)
    if [r['held_out'] for r in result['folds']] != list(samples):
        raise ValueError('All twelve original movie folds in original order required')
    fits = []
    for row in result['folds']:
        held = row['held_out']
        train_stems = [s for s in samples if s != held]
        path = cache / 'lomo' / (held + '-model.json')
        saved = json.loads(path.read_text())
        if (sha(path) != row['model_sha256'] or saved['held_out'] != held
                or saved['training_stems'] != train_stems or row['training_stems'] != train_stems
                or json.loads((cache / 'lomo' / (held + '-result.json')).read_text()) != row):
            raise ValueError('Actual fold file, complete fitting identities or results differ')
        training = combine([samples[s] for s in train_stems])
        fit_result = verify_fit(training, saved['model'])
        fits.append(dict(held_out=held, **fit_result))
        del training
        if (metrics(samples[held], saved['model']) != row['candidate']
                or error_counts(samples[held], saved['model']) != row['error_counts']
                or records[held]['physical'] != row['physical'] or records[held]['neural'] != row['neural']
                or prior[held]['candidate'] != row['unweighted_ranker']):
            raise ValueError('Exact complete held-out predictions and controls differ')
    gate = runner['PREVIOUS']['gate'](result['folds'])
    if result['gate'] != gate:
        raise ValueError('Unchanged fixed gate arithmetic differs')
    final = result['final_model']
    final_fit = None
    final_path = cache / 'lomo/model.json'
    if gate['passed']:
        if final != dict(file='model.json', sha256=sha(final_path), exact_prediction_replay=True):
            raise ValueError('Actual complete final export required after pass')
        model = json.loads(final_path.read_text())
        final_fit = verify_fit(combine(list(samples.values())), model)
        for sample in samples.values():
            metrics(sample, model)
    elif final is not None or final_path.exists():
        raise ValueError('Failed experiment must not export a final promoted model')
    if (any(result[k] != 0 for k in ('gpu_seconds', 'diagnostic_movies_opened', 'source_movies_opened',
                                    'new_target_movies_opened')) or result['authorized_for_submission'] is not False):
        raise ValueError('Fitting-only CPU scope required')
    errors = {key: sum(r['error_counts'][key] for r in result['folds'])
              for key in result['folds'][0]['error_counts']}
    return dict(status='verified_balanced_candidate_ranker_lomo', result_sha256=sha(result_path),
                verifier_sha256=sha(Path(__file__)), smoke_sha256=result['smoke_sha256'],
                complete_groups=evidence['total_groups'], complete_choices=evidence['total_choices'],
                smoke_fit=smoke_fit, fitting_objectives=fits, final_fit=final_fit,
                all_saved_held_out_predictions_replayed=True,
                independent_weighted_group_losses_replayed=True, error_counts=errors,
                gate=gate, final_model=final, authorized_for_submission=False,
                scope='Actual stored parameters and objectives; no independent optimizer refit or global optimum claim')


if __name__ == '__main__':
    target = ROOT / 'reports/experiments/focus-balanced-candidate-ranker-v1-verification.json'
    if target.exists():
        raise ValueError('Never overwrite completed verification')
    result = verify()
    target.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k != 'fitting_objectives'}, indent=2), flush=True)
