"""CPU cache-equivalence smoke and first-fold profile; no held-out scoring."""
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
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.focus_pair_appearance import merge_stats, projector
from research.focus_pair_appearance_head import blocks, objective
from research.focus_pair_appearance_prepared import Prepared
from research.focus_balanced_candidate_ranker import frame_packet

VERIFY = runpy.run_path(str(ROOT / 'scripts/verify-focus-pair-appearance.py'))
sha, read_stats = VERIFY['sha'], VERIFY['read_stats']
RUN = 'focus-pair-appearance-prepared-v1'
CACHE = ROOT / '.biohub/cache' / RUN
REPORT = ROOT / 'reports/experiments'


def load():
    evidence, data_path = VERIFY['load_receipt']()
    if sha(data_path) != '4e2eb70fa910025184a53cc2549293658496ad77803bc403a82137254af9c22f':
        raise ValueError('Frozen verified appearance data required')
    verified = REPORT / 'focus-pair-appearance-v1-verification.json'
    if sha(verified) != '823d8f64a4615ada45a220036a7125551ea3a997438d46b14d864bdaa988cab9':
        raise ValueError('Independent descriptor verification required')
    samples, moments = {}, {}
    for row in evidence['records']:
        stem = row['stem']
        base_path = ROOT / '.biohub/cache/focus-candidate-ranker-v1' / (stem + '.npz')
        app_path = ROOT / '.biohub/cache/focus-pair-appearance-v1' / (stem + '-appearance.npy')
        moment_path = app_path.with_name(stem + '-moments.json')
        if (sha(base_path) != row['base_sha256'] or sha(app_path) != row['descriptor_sha256']
                or sha(moment_path) != row['moments_sha256']):
            raise ValueError('Actual fitting arrays or moments changed')
        samples[stem] = (VERIFY['BUILD']['arrays'](base_path), np.load(app_path, mmap_mode='r', allow_pickle=False))
        moments[stem] = read_stats(moment_path)
    return evidence, samples, moments


def main():
    started = time.monotonic()
    destination = REPORT / f'{RUN}-profile.json'
    if destination.exists() or CACHE.exists():
        raise ValueError('Never overwrite completed or partial preparation profile')
    source_paths = ['research/focus_pair_appearance_prepared.py', 'scripts/profile-focus-pair-appearance-prepared.py',
                    f'reports/experiments/{RUN}-design.md']
    sources = {name: sha(ROOT / name) for name in source_paths}
    evidence, samples, moments = load()
    CACHE.mkdir()
    base, app = samples[evidence['stress']['stem']]
    indices = np.flatnonzero(base['source_frame'] == evidence['stress']['source_frame'])
    start = int(base['starts'][indices[0]])
    end = int(base['starts'][indices[-1]] + base['sizes'][indices[-1]])
    small = [(frame_packet(base, evidence['stress']['source_frame']), app[start:end])]
    smoke = []
    for arm in ('full', 'lda'):
        model_path = ROOT / '.biohub/cache/focus-pair-appearance-v1' / (arm + '-smoke-model.json')
        record = next(a for a in evidence['stress']['arms'] if a['arm'] == arm)
        if sha(model_path) != record['model_sha256']:
            raise ValueError('Frozen actual smoke head required')
        model = json.loads(model_path.read_text())
        prepared = Prepared(small, model['projection'], arm, CACHE / ('smoke-' + arm), 'fitting')
        for original, cached in zip(blocks(small, model['projection'], arm), prepared()):
            if any(not np.array_equal(original[k], cached[k]) for k in original):
                raise ValueError('Exact cached descriptors, labels and group identities must replay')
        for theta in (np.zeros(prepared.dimension), np.asarray(model['theta'])):
            old = objective(theta, lambda: blocks(small, model['projection'], arm), model['absent_weight'])
            new = objective(theta, prepared, model['absent_weight'])
            if old[0] != new[0] or not np.array_equal(old[1], new[1]):
                raise ValueError('Exact prepared objective and gradient required')
        bounds = [(None, None)] * prepared.dimension
        bounds[4:7] = [(None, .5)] * 3
        tick = time.monotonic()
        result = minimize(objective, np.zeros(prepared.dimension), args=(prepared, model['absent_weight']),
                          jac=True, method='L-BFGS-B', bounds=bounds,
                          options=dict(maxiter=500, gtol=1e-8, ftol=1e-12))
        if not result.success or result.fun != model['objective'] or not np.array_equal(result.x, model['theta']):
            raise ValueError('Exact optimized CPU smoke coefficients required')
        smoke.append(dict(arm=arm, exact_array_objective_gradient_and_fitted_theta=True,
                          elapsed_seconds=time.monotonic()-tick, iterations=int(result.nit)))
    held = evidence['records'][0]['stem']
    train = [s for s in samples if s != held]
    projection = projector(merge_stats([moments[s] for s in train], 'fitting'), 'fitting')
    training = [samples[s] for s in train]
    parents = sum(int(b['present'].sum()) for b, _ in training)
    absent = sum(len(b['present']) for b, _ in training) - parents
    weight = float(np.sqrt(parents / absent))
    old_profile_path = REPORT / 'focus-pair-appearance-v1-profile.json'
    if sha(old_profile_path) != '28cee34f9d75a11c112668dbadf8d7626db96e80b4ad2e54783e0f42e57dd46c':
        raise ValueError('Exact previous full-size CPU profile required')
    old_profile = json.loads(old_profile_path.read_text())
    if projection != old_profile['first_fold_projection']:
        raise ValueError('Same fitting-fold projection required')
    trials = []
    for arm in ('full', 'lda'):
        tick = time.monotonic()
        prepared = Prepared(training, projection, arm, CACHE / ('profile-' + arm), 'fitting')
        preparation_seconds = time.monotonic() - tick
        timings = []
        for _ in range(3):
            tick = time.monotonic()
            value, gradient = objective(np.zeros(prepared.dimension), prepared, weight)
            timings.append(time.monotonic() - tick)
        old = next(t for t in old_profile['trials'] if t['arm'] == arm)
        if value != old['zero_objective'] or abs(np.linalg.norm(gradient) - old['gradient_norm']) > 1e-8:
            raise ValueError('Actual complete first-fold objective/profile changed')
        trials.append(dict(arm=arm, preparation_seconds=preparation_seconds, bytes=prepared.bytes,
                           objective_seconds=timings, zero_objective=value, gradient_norm=float(np.linalg.norm(gradient))))
        print(json.dumps(trials[-1]), flush=True)
    if any(sha(ROOT / name) != value for name, value in sources.items()):
        raise ValueError('Prepared method changed during profile')
    result = dict(status='verified_prepared_cpu_equivalence_and_first_fold_profile', source_hashes=sources,
                  smoke=smoke, trials=trials, held_out_not_evaluated=held, training_stems=train,
                  head_quality_evaluated=False, authorized_for_submission=False, gpu_seconds=0,
                  data_sha256='4e2eb70fa910025184a53cc2549293658496ad77803bc403a82137254af9c22f',
                  elapsed_seconds=time.monotonic()-started)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        main()
    elif not sys.argv[1:]:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'], cwd=ROOT, timeout=300, check=True)
    else:
        raise ValueError('No arguments supported')
