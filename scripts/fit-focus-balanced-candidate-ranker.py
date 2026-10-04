"""Smoke first, then fixed rare-null-weighted full-candidate movie-held-out fits."""
import hashlib
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
from research.focus_candidate_ranker import combine, metrics, validate
from research.focus_balanced_candidate_ranker import class_weights, objective, fit, frame_packet, error_counts

RUN = 'focus-balanced-candidate-ranker-v1'
PREVIOUS = runpy.run_path(str(ROOT / 'scripts/fit-focus-candidate-ranker.py'))
CACHE = ROOT / '.biohub/cache' / RUN
REPORT = ROOT / 'reports/experiments'
DATA_SHA = 'c0376562166ca50ffdb12e9f79aff7bc867a217c5f69104454e723e45c3b8ccf'
PRIOR_SHA = '1682bf7167f5119bad272a26f035b15a1cfe25ae4c4ce6db7cc4a13a30446c92'
VERIFY_SHA = '6044fabdc1ada181d6a7fd6d583e7998fc4ad474bc21936734376b055de864cf'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    return {p: sha(ROOT / p) for p in (
        'scripts/fit-focus-balanced-candidate-ranker.py',
        'research/focus_balanced_candidate_ranker.py', 'research/focus_candidate_ranker.py',
        'scripts/fit-focus-candidate-ranker.py', 'tests/test_focus_balanced_candidate_ranker.py',
        f'reports/experiments/{RUN}-design.md')}


def load():
    for name, expected in [('data-smoke', DATA_SHA), ('lomo', PRIOR_SHA), ('verification', VERIFY_SHA)]:
        if sha(REPORT / f'focus-candidate-ranker-v1-{name}.json') != expected:
            raise ValueError('Pinned actual previous data/model verification changed')
    previous = json.loads((REPORT / 'focus-candidate-ranker-v1-lomo.json').read_text())
    if any(sha(ROOT / p) != value for p, value in previous['source_hashes'].items()):
        raise ValueError('Frozen original ranker experiment changed')
    evidence, samples = PREVIOUS['load']()
    return evidence, samples, previous


def smoke():
    started = time.monotonic()
    target = REPORT / f'{RUN}-smoke.json'
    if target.exists() or CACHE.exists():
        raise ValueError('Never overwrite started or completed smoke')
    frozen = sources()
    _, samples, _ = load()
    choices = []
    for stem, data in samples.items():
        for frame in np.unique(data['source_frame']):
            keep = data['source_frame'] == frame
            present = data['present'][keep]
            if present.min() == 0 and present.max() == 1:
                choices.append((-int((data['sizes'][keep] - 1).sum()), stem, int(frame)))
    _, stem, frame = min(choices)
    packet = frame_packet(samples[stem], frame)
    ids = validate(packet)
    weights, _ = class_weights(packet, 'fitting')
    direction = np.arange(1, 9, dtype=float)
    direction /= np.linalg.norm(direction)
    eps = 1e-6
    _, gradient = objective(np.zeros(8), packet, ids, weights)
    finite = (objective(eps * direction, packet, ids, weights)[0]
              - objective(-eps * direction, packet, ids, weights)[0]) / (2 * eps)
    analytic = float(gradient @ direction)
    relative_error = abs(finite - analytic) / max(1., abs(analytic))
    if relative_error > 2e-6:
        raise ValueError('Real weighted candidate gradient failed')
    fit_started = time.monotonic()
    model = fit(packet, 'fitting')
    fit_seconds = time.monotonic() - fit_started
    CACHE.mkdir()
    path = CACHE / 'smoke-model.json'
    path.write_text(json.dumps(model, indent=2, allow_nan=False))
    restored = json.loads(path.read_text())
    if model != restored or metrics(packet, model) != metrics(packet, restored):
        raise ValueError('Actual smoke model/predictions changed on reload')
    if sources() != frozen:
        raise ValueError('Method changed during smoke')
    result = dict(status='verified_balanced_candidate_real_smoke', run_id=RUN,
                  source_hashes=frozen, data_smoke_sha256=DATA_SHA, prior_result_sha256=PRIOR_SHA,
                  prior_verification_sha256=VERIFY_SHA, stem=stem, source_frame=frame,
                  groups=len(packet['starts']), real_choices=int((packet['sizes'] - 1).sum()),
                  directional_derivative=analytic, finite_difference=finite,
                  gradient_relative_error=relative_error, model_sha256=sha(path),
                  model=model, fit_seconds=fit_seconds, exact_parameter_prediction_reload=True,
                  quality_evaluated=False, gpu_seconds=0, elapsed_seconds=time.monotonic() - started)
    target.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2), flush=True)


def require_smoke():
    path = REPORT / f'{RUN}-smoke.json'
    result = json.loads(path.read_text())
    if (result['status'] != 'verified_balanced_candidate_real_smoke' or result['source_hashes'] != sources()
            or result['data_smoke_sha256'] != DATA_SHA or result['prior_result_sha256'] != PRIOR_SHA
            or result['prior_verification_sha256'] != VERIFY_SHA
            or result['exact_parameter_prediction_reload'] is not True or result['quality_evaluated'] is not False
            or result['gradient_relative_error'] > 2e-6
            or sha(CACHE / 'smoke-model.json') != result['model_sha256']):
        raise ValueError('Actual fixed-method real smoke required before full fitting')
    return result


def train():
    started = time.monotonic()
    target = REPORT / f'{RUN}-lomo.json'
    cache = CACHE / 'lomo'
    if target.exists() or cache.exists():
        raise ValueError('Never overwrite completed or partial weighted experiment')
    require_smoke()
    frozen = sources()
    evidence, samples, previous = load()
    prior = {r['held_out']: r for r in previous['folds']}
    records = {r['stem']: r for r in evidence['records']}
    cache.mkdir()
    folds = []
    for held_out in samples:
        tick = time.monotonic()
        train_stems = [stem for stem in samples if stem != held_out]
        data = combine([samples[stem] for stem in train_stems])
        model = fit(data, 'fitting')
        path = cache / (held_out + '-model.json')
        path.write_text(json.dumps(dict(training_stems=train_stems, held_out=held_out, model=model),
                                   indent=2, allow_nan=False))
        restored = json.loads(path.read_text())
        if restored['model'] != model:
            raise ValueError('Actual fold model changed before held-out evaluation')
        del data
        row = dict(held_out=held_out, training_stems=train_stems, model_sha256=sha(path),
                   physical=records[held_out]['physical'], neural=records[held_out]['neural'],
                   unweighted_ranker=prior[held_out]['candidate'],
                   candidate=metrics(samples[held_out], restored['model']),
                   error_counts=error_counts(samples[held_out], restored['model']),
                   elapsed_seconds=time.monotonic() - tick)
        folds.append(row)
        (cache / (held_out + '-result.json')).write_text(json.dumps(row, indent=2, allow_nan=False))
        print(json.dumps(dict(held_out=held_out, candidate=row['candidate'], absent_weight=model['absent_weight'],
                              elapsed_seconds=row['elapsed_seconds'])), flush=True)
    decision = PREVIOUS['gate'](folds)
    final = None
    if decision['passed']:
        model = fit(combine(list(samples.values())), 'fitting')
        path = cache / 'model.json'
        path.write_text(json.dumps(model, indent=2, allow_nan=False))
        restored = json.loads(path.read_text())
        if any(metrics(s, restored) != metrics(s, model) for s in samples.values()):
            raise ValueError('Final portable predictions changed on reload')
        final = dict(file='model.json', sha256=sha(path), exact_prediction_replay=True)
    if sources() != frozen:
        raise ValueError('Frozen weighted method changed during experiment')
    result = dict(status='completed_balanced_candidate_ranker_lomo', run_id=RUN,
                  source_hashes=frozen, smoke_sha256=sha(REPORT / f'{RUN}-smoke.json'),
                  data_smoke_sha256=DATA_SHA, prior_result_sha256=PRIOR_SHA,
                  prior_verification_sha256=VERIFY_SHA, folds=folds, gate=decision, final_model=final,
                  diagnostic_movies_opened=0, source_movies_opened=0, new_target_movies_opened=0,
                  gpu_seconds=0, authorized_for_submission=False, elapsed_seconds=time.monotonic() - started)
    target.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(dict(gate=decision, final_model=final, elapsed_seconds=result['elapsed_seconds']), indent=2), flush=True)


if __name__ == '__main__':
    args = sys.argv[1:]
    if args in (['smoke', '--worker'], ['train', '--worker']):
        {'smoke': smoke, 'train': train}[args[0]]()
    elif args in (['smoke'], ['train']):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), args[0], '--worker'], cwd=ROOT,
                       env=dict(os.environ), timeout=300 if args[0] == 'smoke' else 900, check=True)
    else:
        raise ValueError('Use smoke or train; no implicit repeated jobs')
