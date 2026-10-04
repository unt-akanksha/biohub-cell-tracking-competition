"""Sequential frozen LDA/full pair-appearance LOMO, with resumable fold artifacts."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.focus_candidate_ranker import metrics as physical_metrics
from research.focus_pair_appearance import merge_stats, projector
from research.focus_pair_appearance_head import metrics
from research.focus_pair_appearance_resident_fit import fit

PREPARED = runpy.run_path(str(ROOT / 'scripts/profile-focus-pair-appearance-prepared.py'))
BASE = runpy.run_path(str(ROOT / 'scripts/fit-focus-candidate-ranker.py'))
sha = PREPARED['sha']
RUN = 'focus-pair-appearance-lomo-v1'
CACHE = ROOT / '.biohub/cache' / RUN
REPORT = ROOT / 'reports/experiments'


def sources():
    return {p: sha(ROOT / p) for p in (
        'research/focus_pair_appearance_resident_fit.py', 'scripts/fit-focus-pair-appearance-resident.py',
        'research/focus_pair_appearance_resident.py', 'research/focus_pair_appearance_prepared.py',
        'research/focus_pair_appearance.py', 'research/focus_pair_appearance_head.py',
        'research/focus_candidate_ranker.py', 'scripts/fit-focus-candidate-ranker.py',
        'scripts/profile-focus-pair-appearance-prepared.py',
        f'reports/experiments/{RUN}-design.md')}


def load():
    profile_path = REPORT / 'focus-pair-appearance-resident-v1-profile.json'
    if sha(profile_path) != 'e9d65e8e07f867dfd4de4cc5698daf03a84f0317a874937ec6d86101ac2f3b6d':
        raise ValueError('Exact resident complete first-fold profile required')
    profile = json.loads(profile_path.read_text())
    if any(sha(ROOT / p) != v for p, v in profile['source_hashes'].items()):
        raise ValueError('Frozen resident transform implementation changed')
    prepared_path = REPORT / 'focus-pair-appearance-prepared-v1-profile.json'
    if sha(prepared_path) != profile['prepared_profile_sha256']:
        raise ValueError('Exact CPU smoke equivalence required')
    prepared = json.loads(prepared_path.read_text())
    if any(sha(ROOT / p) != v for p, v in prepared['source_hashes'].items()):
        raise ValueError('Frozen prepared implementation changed')
    evidence, samples, moments = PREPARED['load']()
    original_path = REPORT / 'focus-candidate-ranker-v1-data-smoke.json'
    weighted_path = REPORT / 'focus-balanced-candidate-ranker-v1-lomo.json'
    if (sha(original_path) != 'c0376562166ca50ffdb12e9f79aff7bc867a217c5f69104454e723e45c3b8ccf'
            or sha(weighted_path) != 'dba19c4210c29941ca8056f49089ebac9d25eae8151c47931282af8366f5fe7c'):
        raise ValueError('Exact original and weighted fitting controls required')
    records = {r['stem']: r for r in json.loads(original_path.read_text())['records']}
    weighted = {r['held_out']: r for r in json.loads(weighted_path.read_text())['folds']}
    for stem, (base, _) in samples.items():
        if physical_metrics(base) != records[stem]['physical']:
            raise ValueError('Actual original complete physical control changed')
    return evidence, samples, moments, records, weighted


def main():
    started = time.monotonic()
    target = REPORT / f'{RUN}-result.json'
    if CACHE.exists() or target.exists():
        raise ValueError('Never overwrite completed or partial LOMO run; inspect live state before recovery')
    frozen = sources()
    evidence, samples, moments, controls, weighted = load()
    CACHE.mkdir()
    (CACHE / 'launch.json').write_text(json.dumps(dict(run_id=RUN, source_hashes=frozen,
        declared_cpu_seconds=9000, gpu_seconds=0, arm_order=['lda', 'full']), indent=2))
    arms = []
    for arm in ('lda', 'full'):
        arm_started = time.monotonic()
        folder = CACHE / arm
        folder.mkdir()
        folds = []
        failure = None
        for held in samples:
            tick = time.monotonic()
            train = [s for s in samples if s != held]
            projection = projector(merge_stats([moments[s] for s in train], 'fitting'), 'fitting')
            progress = lambda p: print(json.dumps(dict(arm=arm, held_out=held, **p)), flush=True)
            print(json.dumps(dict(arm=arm, held_out=held, status='fitting_started')), flush=True)
            try:
                model, execution = fit([samples[s] for s in train], projection, arm,
                                       folder / (held + '-prepared'), 'fitting', progress=progress)
            except Exception as error:
                failure = dict(held_out=held, error_type=type(error).__name__, error=str(error),
                               traceback=traceback.format_exc(), elapsed_seconds=time.monotonic()-tick)
                (folder / (held + '-failure.json')).write_text(json.dumps(failure, indent=2))
                print(json.dumps(dict(arm=arm, status='arm_stopped_without_retuning', **failure)), flush=True)
                break
            # Persist fold-only parameters before opening the correction-held-out movie for evaluation.
            model_path = folder / (held + '-model.json')
            model_path.write_text(json.dumps(dict(held_out=held, training_stems=train, model=model), indent=2, allow_nan=False))
            restored = json.loads(model_path.read_text())
            if restored['model'] != model:
                raise ValueError('Saved fold model changed before evaluation')
            row = dict(held_out=held, training_stems=train, model_sha256=sha(model_path),
                       physical=controls[held]['physical'], neural=controls[held]['neural'],
                       weighted_ranker=weighted[held]['candidate'],
                       candidate=metrics([samples[held]], restored['model']), execution=execution,
                       elapsed_seconds=time.monotonic()-tick)
            folds.append(row)
            (folder / (held + '-result.json')).write_text(json.dumps(row, indent=2, allow_nan=False))
            print(json.dumps(dict(arm=arm, **row)), flush=True)
            if sources() != frozen:
                raise ValueError('Frozen method changed during fitting')
        decision = BASE['gate'](folds) if len(folds) == 12 else None
        result = dict(arm=arm, status='completed' if failure is None else 'fit_failed', folds=folds,
                      failure=failure, gate=decision, final_model=None,
                      elapsed_seconds=time.monotonic()-arm_started)
        (folder / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        arms.append(result)
        print(json.dumps(dict(arm=arm, status=result['status'], gate=decision)), flush=True)
    if sources() != frozen:
        raise ValueError('Frozen method changed')
    result = dict(status='completed_sequential_pair_appearance_screen', run_id=RUN, source_hashes=frozen,
                  arms=arms, gpu_seconds=0, diagnostic_movies_opened=0, source_movies_opened=0,
                  new_target_movies_opened=0, authorized_for_submission=False,
                  elapsed_seconds=time.monotonic()-started, declared_cpu_seconds=9000)
    target.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(dict(status=result['status'], elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        main()
    elif not sys.argv[1:]:
        subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--worker'], cwd=ROOT,
                       timeout=9000, check=True)
    else:
        raise ValueError('No arguments supported')
