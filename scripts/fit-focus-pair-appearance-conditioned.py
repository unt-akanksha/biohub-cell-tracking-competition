"""Full72D correction-only LOMO using the admitted equivalent optimizer coordinates."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.focus_pair_appearance import merge_stats, projector
from research.focus_pair_appearance_head import blocks, metrics
from research.focus_pair_appearance_resident import Resident
from research.focus_pair_appearance_conditioned_fit import fit

PRIOR = runpy.run_path(str(ROOT / 'scripts/fit-focus-pair-appearance-resident.py'))
BASE, sha = PRIOR['BASE'], PRIOR['sha']
RUN = 'focus-pair-appearance-conditioned-lomo-v1'
CACHE = ROOT / '.biohub/cache' / RUN
REPORT = ROOT / 'reports/experiments'


def sources():
    names = list(PRIOR['sources']()) + ['research/focus_pair_appearance_conditioning.py',
        'research/focus_pair_appearance_conditioned_fit.py', 'scripts/fit-focus-pair-appearance-conditioned.py',
        f'reports/experiments/{RUN}-design.md']
    return {p: sha(ROOT / p) for p in names}


def load():
    profile_path = REPORT / 'focus-pair-appearance-conditioning-v1-profile.json'
    if sha(profile_path) != 'c1b4af86d8a21cd6a84512dc05e76d8860e883e3731db3976c5e819f83d795c9':
        raise ValueError('Actual admitted first-fold numerical profile required')
    profile = json.loads(profile_path.read_text())
    if (not profile['numerical_recovery_admitted'] or profile['quality_evaluated'] is not False
            or any(sha(ROOT / p) != value for p, value in profile['source_hashes'].items())):
        raise ValueError('Frozen tested coordinate method required')
    failure_path = REPORT / 'focus-pair-appearance-lomo-v1-result.json'
    if sha(failure_path) != profile['prior_failure_sha256']:
        raise ValueError('Terminal original failure required; never race a live old fit')
    if sha(REPORT / 'focus-pair-appearance-lomo-v1-lda-verification.json') != 'c3045123571fa1daccf23f3cd784d751ca240f9bf384b942a1a5562337f704f4':
        raise ValueError('Actual verified LDA comparison required')
    return PRIOR['load']()


def first_prepared(training, projection, held):
    folder = PRIOR['CACHE'] / 'full' / (held + '-prepared')
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest['projection'] != projection or manifest['arm'] != 'full':
        raise ValueError('Exact original training-fold prepared projection required')
    prepared = SimpleNamespace(samples=training, paths=[folder/f'{i:02d}.npy' for i in range(len(training))], bytes=manifest['bytes'])
    resident = Resident(prepared)
    count = 0
    for expected, actual in zip(blocks(training, projection, 'full'), resident()):
        if any(not np.array_equal(expected[k], actual[k]) for k in expected):
            raise ValueError('Exact first-fold original cached transform/group replay required')
        count += len(expected['x'])
    if count != manifest['counts']['choices']:
        raise ValueError('Complete cached first-fold choices required')
    return prepared


def main():
    started = time.monotonic()
    target = REPORT / f'{RUN}-result.json'
    if target.exists() or CACHE.exists():
        raise ValueError('Never overwrite completed or partial conditioned run')
    frozen = sources()
    evidence, samples, moments, controls, weighted = load()
    lda_path = PRIOR['CACHE'] / 'lda/result.json'
    if sha(lda_path) != '9d3713fae680f67ef149444f32c176c3705cfaed74b7a9122efba6a5ea59a779':
        raise ValueError('Exact frozen LDA folds required')
    lda = {r['held_out']: r for r in json.loads(lda_path.read_text())['folds']}
    CACHE.mkdir()
    (CACHE / 'launch.json').write_text(json.dumps(dict(run_id=RUN, source_hashes=frozen,
        declared_cpu_seconds=9000, gpu_seconds=0, arm_order=['full']), indent=2))
    folder = CACHE / 'full'
    folder.mkdir()
    folds, failure = [], None
    for index, held in enumerate(samples):
        tick = time.monotonic()
        train = [s for s in samples if s != held]
        training = [samples[s] for s in train]
        projection = projector(merge_stats([moments[s] for s in train], 'fitting'), 'fitting')
        print(json.dumps(dict(held_out=held, status='conditioned_fit_started')), flush=True)
        try:
            prepared = first_prepared(training, projection, held) if index == 0 else None
            model, execution = fit(training, projection, folder / (held+'-optimizer'), 'fitting', prepared=prepared,
                progress=lambda r: print(json.dumps(dict(held_out=held, **r)), flush=True))
        except Exception as error:
            failure = dict(held_out=held, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
            (folder / (held+'-failure.json')).write_text(json.dumps(failure, indent=2))
            print(json.dumps(failure), flush=True)
            break
        path = folder / (held+'-model.json')
        path.write_text(json.dumps(dict(held_out=held, training_stems=train, model=model), indent=2, allow_nan=False))
        restored = json.loads(path.read_text())
        if restored['model'] != model:
            raise ValueError('Persisted model changed before held-out scoring')
        row = dict(held_out=held, training_stems=train, model_sha256=sha(path),
                   physical=controls[held]['physical'], neural=controls[held]['neural'],
                   weighted_ranker=weighted[held]['candidate'], lda=lda[held]['candidate'],
                   candidate=metrics([samples[held]], restored['model']), execution=execution,
                   elapsed_seconds=time.monotonic()-tick)
        folds.append(row)
        (folder / (held+'-result.json')).write_text(json.dumps(row, indent=2, allow_nan=False))
        print(json.dumps(row), flush=True)
        if sources() != frozen:
            raise ValueError('Frozen scientific or numerical method changed')
    decision = BASE['gate'](folds) if len(folds) == 12 else None
    result = dict(arm='full', status='completed' if failure is None else 'fit_failed', folds=folds,
                  failure=failure, gate=decision, final_model=None, elapsed_seconds=time.monotonic()-started)
    (folder / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    if sources() != frozen:
        raise ValueError('Frozen method changed during run')
    target.write_text(json.dumps(dict(status='completed_conditioned_full_feature_screen', run_id=RUN,
        source_hashes=frozen, result=result, diagnostic_movies_opened=0, source_movies_opened=0,
        new_target_movies_opened=0, gpu_seconds=0, authorized_for_submission=False,
        elapsed_seconds=time.monotonic()-started), indent=2, allow_nan=False))
    print(json.dumps(dict(status=result['status'], gate=decision, elapsed_seconds=result['elapsed_seconds']), indent=2), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        main()
    elif not sys.argv[1:]:
        subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--worker'], cwd=ROOT,
                       timeout=9000, check=True)
    else:
        raise ValueError('No arguments supported')
