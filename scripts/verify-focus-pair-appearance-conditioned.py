"""Independent original-objective/held-out replay for all conditioned full folds."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.focus_pair_appearance import merge_stats, projector
from research.focus_pair_appearance_head import blocks, objective, metrics, validate_samples
from research.focus_pair_appearance_conditioning import hessian, coordinate_map

TRAIN = runpy.run_path(str(ROOT / 'scripts/fit-focus-pair-appearance-conditioned.py'))
CACHE, REPORT, RUN, sha = (TRAIN[k] for k in ('CACHE', 'REPORT', 'RUN', 'sha'))


def check_optimizer_state(state, transform, model):
    theta, beta = np.asarray(state['theta']), np.asarray(state['beta'])
    if (state['success'] is not True or theta.shape != (72,) or beta.shape != (72,)
            or not np.isfinite(theta).all() or not np.isfinite(beta).all()
            or (theta[4:7] > .5).any() or state['theta'] != model['theta']
            or state['iterations'] != model['iterations'] or state['evaluations'] != model['evaluations']
            or state['objective'] != model['objective']):
        raise ValueError('Actual successful original-coordinate model and optimizer state required')
    np.testing.assert_allclose(transform @ beta, theta, rtol=1e-12, atol=1e-12)


def main():
    started = time.monotonic()
    target = REPORT / f'{RUN}-verification.json'
    if target.exists():
        raise ValueError('Never overwrite completed recovery verification')
    launch = json.loads((CACHE / 'launch.json').read_text())
    if launch['source_hashes'] != TRAIN['sources']():
        raise ValueError('Frozen launched scientific and numerical method required')
    result_path = CACHE / 'full/result.json'
    result = json.loads(result_path.read_text())
    if result['status'] != 'completed' or result['failure'] is not None or len(result['folds']) != 12:
        raise ValueError('All twelve converged completed folds required')
    evidence, samples, moments, controls, weighted = TRAIN['load']()
    lda = {r['held_out']: r for r in json.loads((TRAIN['PRIOR']['CACHE'] / 'lda/result.json').read_text())['folds']}
    stems = list(samples)
    if [r['held_out'] for r in result['folds']] != stems or result['final_model'] is not None:
        raise ValueError('Exact complete correction screen without premature final model required')
    checks = []
    for row in result['folds']:
        held = row['held_out']
        training_stems = [s for s in stems if s != held]
        fitting = [samples[s] for s in training_stems]
        projection = projector(merge_stats([moments[s] for s in training_stems], 'fitting'), 'fitting')
        path = CACHE / 'full' / (held+'-model.json')
        saved = json.loads(path.read_text())
        model = saved['model']
        counts = validate_samples(fitting)
        weight = float(np.sqrt(counts['parents']/counts['absent']))
        if (sha(path) != row['model_sha256'] or saved['held_out'] != held
                or saved['training_stems'] != training_stems or row['training_stems'] != training_stems
                or model['projection'] != projection or model['counts'] != counts
                or model['absent_weight'] != weight or model['parent_weight'] != 1.
                or model['arm'] != 'full' or json.loads((path.with_name(held+'-result.json')).read_text()) != row):
            raise ValueError('Exact fold-only fitting identity/projection/weights required')
        provider = lambda: blocks(fitting, projection, 'full')
        theta = np.asarray(model['theta'])
        value, gradient = objective(theta, provider, weight)
        if abs(value-model['objective']) > 1e-7:
            raise ValueError('Actual original-theta training objective differs')
        optimizer = CACHE / 'full' / (held+'-optimizer')
        state = json.loads((optimizer / 'optimizer-terminal.json').read_text())
        with np.load(optimizer / 'coordinates.npz', allow_pickle=False) as data:
            stored_hessian, stored_transform = data['hessian'].copy(), data['transform'].copy()
        check_optimizer_state(state, stored_transform, model)
        np.testing.assert_allclose(gradient, state['original_gradient'], rtol=1e-7, atol=1e-7)
        # Recompute all fitting curvature from original streamed features, not cached transformed files.
        actual_hessian, coverage = hessian(np.zeros(72), provider, weight, 'fitting')
        np.testing.assert_allclose(actual_hessian, stored_hessian, rtol=1e-9, atol=1e-7)
        rebuilt_transform, _ = coordinate_map(actual_hessian, 'fitting')
        np.testing.assert_allclose(rebuilt_transform, stored_transform, rtol=1e-8, atol=1e-9)
        if coverage != row['execution']['coverage']:
            raise ValueError('Every original curvature group/candidate must be accounted for')
        iterations = list(range(25, model['iterations']+1, 25))
        checkpoint_paths = sorted(optimizer.glob('iterate-*.json'))
        if [p.name for p in checkpoint_paths] != [f'iterate-{n:06d}.json' for n in iterations]:
            raise ValueError('All declared optimizer checkpoints must be preserved')
        for checkpoint_path, n in zip(checkpoint_paths, iterations):
            checkpoint = json.loads(checkpoint_path.read_text())
            if checkpoint['iteration'] != n or checkpoint['authorized_for_submission'] is not False:
                raise ValueError('Partial optimizer state must never be a submission model')
            np.testing.assert_allclose(stored_transform @ checkpoint['beta'], checkpoint['theta'], rtol=1e-12, atol=1e-12)
        actual = metrics([samples[held]], model)
        independent = dict(loss_sum=0., known_parent=0, known_absent=0, correct_parent=0, correct_absent=0)
        for block in blocks([samples[held]], projection, 'full'):
            score = block['offset'] + block['x'] @ theta
            for start, size, chosen, parent in zip(block['starts'], block['sizes'], block['chosen'], block['present']):
                group = score[start:start+size]
                independent['loss_sum'] += float(np.logaddexp.reduce(group)-score[chosen])
                key = 'parent' if parent else 'absent'
                independent['known_'+key] += 1
                independent['correct_'+key] += int(int(np.argmax(group))+start == chosen)
        independent['nll'] = independent['loss_sum'] / (independent['known_parent']+independent['known_absent'])
        if actual != row['candidate'] or any(abs(independent[k]-actual[k]) > 1e-7 for k in independent):
            raise ValueError('Every held-out loss/decision must replay independently')
        if (row['physical'] != controls[held]['physical'] or row['neural'] != controls[held]['neural']
                or row['weighted_ranker'] != weighted[held]['candidate'] or row['lda'] != lda[held]['candidate']):
            raise ValueError('All frozen original and comparison controls required')
        checks.append(dict(held_out=held, model_sha256=sha(path), original_training_objective=value,
                           gradient_inf_norm=float(np.abs(gradient).max()), projection_and_hessian_refit=True,
                           optimizer_checkpoints=len(checkpoint_paths), candidate=actual))
        print(json.dumps(checks[-1]), flush=True)
    decision = TRAIN['BASE']['gate'](result['folds'])
    if decision != result['gate'] or TRAIN['sources']() != launch['source_hashes']:
        raise ValueError('Original acceptance and frozen source must replay')
    record = dict(status='host_verified_conditioned_full_pair_appearance_lomo',
                  source_hashes=launch['source_hashes'], verifier_sha256=sha(Path(__file__)),
                  result_sha256=sha(result_path), checks=checks, gate=decision, optimizer_refit=False,
                  diagnostic_movies_opened=0, source_movies_opened=0, new_target_movies_opened=0,
                  gpu_seconds=0, authorized_for_submission=False, elapsed_seconds=time.monotonic()-started)
    target.write_text(json.dumps(record, indent=2, allow_nan=False))
    print(json.dumps(dict(gate=decision, elapsed_seconds=record['elapsed_seconds']), indent=2), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        main()
    elif not sys.argv[1:]:
        subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--worker'], cwd=ROOT,
                       timeout=900, check=True)
    else:
        raise ValueError('No arguments supported')
