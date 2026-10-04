"""Smoke an equivalent coordinate system, then audit first-fold curvature only."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
from types import SimpleNamespace

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.focus_candidate_ranker import pack
from research.focus_pair_appearance import descriptors, merge_stats, projector
from research.focus_pair_appearance_head import blocks, objective
from research.focus_pair_appearance_resident import Resident
from research.focus_pair_appearance_conditioning import hessian, coordinate_map, transformed_objective

PRIOR = runpy.run_path(str(ROOT / 'scripts/fit-focus-pair-appearance-resident.py'))
sha = PRIOR['sha']
RUN = 'focus-pair-appearance-conditioning-v1'
CACHE = ROOT / '.biohub/cache' / RUN
REPORT = ROOT / 'reports/experiments'


def spectrum(matrix):
    values = np.linalg.eigvalsh(matrix)
    if values[0] <= 0 or not np.isfinite(values).all():
        raise ValueError('Strictly positive finite fitting curvature required')
    return dict(minimum=float(values[0]), maximum=float(values[-1]), condition=float(values[-1]/values[0]))


def main():
    started = time.monotonic()
    target = REPORT / f'{RUN}-profile.json'
    if target.exists() or CACHE.exists():
        raise ValueError('Never overwrite partial or completed numerical profile')
    old_path = REPORT / 'focus-pair-appearance-lomo-v1-result.json'
    if sha(old_path) != 'dadaa3293fbb0b99252bae246186a7e336f2259a154b05010b3d60c995919bbf':
        raise ValueError('Actual terminal full-fit failure required')
    prior_result = json.loads(old_path.read_text())
    failed = next(a for a in prior_result['arms'] if a['arm'] == 'full')
    if (failed['status'] != 'fit_failed' or failed['folds'] or failed['gate'] is not None
            or failed['failure']['held_out'] != '6bba_6479435d'
            or 'ITERATIONS REACHED LIMIT' not in failed['failure']['error']):
        raise ValueError('Unscored first-fitting-fold optimizer failure required')
    frozen = {p: sha(ROOT / p) for p in ('research/focus_pair_appearance_conditioning.py',
        'scripts/profile-focus-pair-appearance-conditioning.py', 'tests/test_focus_pair_appearance_conditioning.py',
        f'reports/experiments/{RUN}-design.md')}
    runtime = ROOT / '.biohub/cache/kernel-outputs/focus-pair-appearance-gpu-smoke-v1/focus_pair_appearance_gpu_smoke/runtime'
    hashes = json.loads((runtime / 'source_hashes.json').read_text())
    if any(sha(runtime / p) != value for p, value in hashes.items()):
        raise ValueError('Actual original smoke packet and CPU controls required')
    with np.load(runtime / 'pair.npz', allow_pickle=False) as saved:
        packet = {k: saved[k].copy() for k in saved.files}
    spec = json.loads((runtime / 'spec.json').read_text())
    base = pack(packet, spec['motion_parameters'], 'fitting')
    app = descriptors(packet, np.flatnonzero(packet['labels'] >= 0))
    CACHE.mkdir()
    smokes = []
    for arm in ('full', 'lda'):
        cpu = json.loads((runtime / (arm + '-cpu-model.json')).read_text())
        prepared = list(blocks([(base, app)], cpu['projection'], arm))
        provider = lambda: iter(prepared)
        dimension = len(cpu['theta'])
        curvature, coverage = hessian(np.zeros(dimension), provider, cpu['absent_weight'], 'fitting')
        transform, bounds = coordinate_map(curvature, 'fitting')
        tick = time.monotonic()
        result = minimize(transformed_objective, np.zeros(dimension),
                          args=(transform, objective, provider, cpu['absent_weight']), jac=True,
                          method='L-BFGS-B', bounds=bounds,
                          options=dict(maxiter=500, gtol=1e-8, ftol=1e-12))
        seconds = time.monotonic()-tick
        theta = transform @ result.x
        state = dict(theta=theta.tolist(), beta=result.x.tolist(), success=bool(result.success),
                     message=str(result.message), objective=float(result.fun), iterations=int(result.nit),
                     evaluations=int(result.nfev), elapsed_seconds=seconds)
        (CACHE / (arm + '-smoke-fit.json')).write_text(json.dumps(state, indent=2, allow_nan=False))
        block = prepared[0]
        original_scores = (block['offset'] + block['x'] @ np.asarray(cpu['theta'])).reshape(len(base['starts']), -1)
        actual_scores = (block['offset'] + block['x'] @ theta).reshape(len(base['starts']), -1)
        if (not result.success or not np.isfinite(theta).all() or (theta[4:7] > .5).any()
                or abs(result.fun-cpu['objective']) > 1e-5
                or not np.array_equal(original_scores.argmax(axis=1), actual_scores.argmax(axis=1))):
            raise ValueError('Equivalent coordinate smoke optimizer/objective/decisions failed')
        value, gradient = objective(theta, provider, cpu['absent_weight'])
        if abs(value-result.fun) > 1e-8:
            raise ValueError('Original-coordinate objective must replay')
        smokes.append(dict(arm=arm, **state, original_objective=cpu['objective'],
                           exact_target_choices=True, before=spectrum(curvature),
                           after=spectrum(transform.T @ curvature @ transform), coverage=coverage))
        print(json.dumps(smokes[-1]), flush=True)
    evidence, samples, moments, _, _ = PRIOR['load']()
    held = evidence['records'][0]['stem']
    train = [s for s in samples if s != held]
    training = [samples[s] for s in train]
    projection = projector(merge_stats([moments[s] for s in train], 'fitting'), 'fitting')
    folder = PRIOR['CACHE'] / 'full' / (held + '-prepared')
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest['projection'] != projection or manifest['arm'] != 'full' or manifest['dimension'] != 72:
        raise ValueError('Exact failed-fold prepared transform required')
    prepared = SimpleNamespace(samples=training, paths=[folder/f'{i:02d}.npy' for i in range(len(train))],
                               bytes=manifest['bytes'])
    resident = Resident(prepared)
    replayed = 0
    for old, actual in zip(blocks(training, projection, 'full'), resident()):
        if any(not np.array_equal(old[k], actual[k]) for k in old):
            raise ValueError('Every original prepared value/group must replay')
        replayed += len(old['x'])
    if replayed != 4460090:
        raise ValueError('Exact full first-fold coverage required')
    counts = manifest['counts']
    weight = float(np.sqrt(counts['parents']/counts['absent']))
    value, gradient = objective(np.zeros(72), resident, weight)
    if value != 9174.180132778369 or abs(np.linalg.norm(gradient)-12511.007119549158) > 1e-8:
        raise ValueError('Original first-fold zero objective/gradient changed')
    tick = time.monotonic()
    curvature, coverage = hessian(np.zeros(72), resident, weight, 'fitting')
    seconds = time.monotonic()-tick
    transform, bounds = coordinate_map(curvature, 'fitting')
    before, after = spectrum(curvature), spectrum(transform.T @ curvature @ transform)
    admission = after['condition'] * 10 < before['condition']
    arrays_path = CACHE / 'first-fold-curvature.npz'
    np.savez_compressed(arrays_path, hessian=curvature, transform=transform)
    result = dict(status='completed_smoke_and_first_fold_conditioning_profile', source_hashes=frozen,
                  prior_failure_sha256=sha(old_path), smokes=smokes, held_out_not_evaluated=held,
                  training_stems=train, projection=projection, counts=counts, coverage=coverage,
                  zero_objective=value, curvature_seconds=seconds, before=before, after=after,
                  coordinate_bounds=bounds, numerical_recovery_admitted=admission,
                  first_fold_curvature_sha256=sha(arrays_path), full_fold_head_optimizer_run=False,
                  quality_evaluated=False, gpu_seconds=0, authorized_for_submission=False,
                  elapsed_seconds=time.monotonic()-started)
    if any(sha(ROOT / p) != value for p, value in frozen.items()):
        raise ValueError('Frozen numerical method changed')
    target.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k not in ('projection', 'smokes', 'coordinate_bounds')}, indent=2), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        main()
    elif not sys.argv[1:]:
        subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--worker'], cwd=ROOT,
                       timeout=300, check=True)
    else:
        raise ValueError('No arguments supported')
