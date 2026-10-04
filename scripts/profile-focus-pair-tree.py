"""One fitting-only full-size tree residual profile with time/RAM watchdogs."""
import ctypes
import hashlib
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
from research.focus_pair_appearance import projector, merge_stats
from research.focus_pair_appearance_head import blocks, validate_samples
from research.focus_pair_tree import fit, derivatives, validate_groups

RUN = 'focus-pair-tree-first-fold-v1'
CACHE = ROOT / '.biohub/cache' / RUN
REPORT = ROOT / 'reports/experiments'
HELD = '6bba_6479435d'
MAX_SECONDS = 900
MAX_WORKING_BYTES = 6 * 1024**3


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    smoke = json.loads((REPORT/'focus-pair-tree-v1-smoke.json').read_text())
    names = list(smoke['source_hashes']) + ['scripts/profile-focus-pair-tree.py',
             f'reports/experiments/{RUN}-design.md']
    return {p: sha(ROOT/p) for p in names}


def load_training():
    smoke_path = REPORT/'focus-pair-tree-v1-smoke.json'
    verified_path = REPORT/'focus-pair-appearance-conditioned-lomo-v1-verification.json'
    if (sha(smoke_path) != '41fd6c6e9d8ea8cfa67a0b6d6d4917958e430fd543eaaebf1fac3e3e70836e2c'
            or sha(verified_path) != '370fb67ec26f99842be51ebd89b26a55494b96959a88366805a328b5512e7450'):
        raise ValueError('Actual real smoke and independent full linear verification required')
    smoke, verified = [json.loads(p.read_text()) for p in (smoke_path, verified_path)]
    if any(sha(ROOT/p) != value for p, value in {**smoke['source_hashes'], **verified['source_hashes']}.items()):
        raise ValueError('Frozen verified baseline and tree implementation required')
    receipt = REPORT/'focus-pair-appearance-v1-data-smoke.json'
    if sha(receipt) != '4e2eb70fa910025184a53cc2549293658496ad77803bc403a82137254af9c22f':
        raise ValueError('Exact original fitting-only descriptor receipt required')
    evidence = json.loads(receipt.read_text())
    record = next(r for r in verified['checks'] if r['held_out'] == HELD)
    baseline_path = ROOT/'.biohub/cache/focus-pair-appearance-conditioned-lomo-v1/full'/f'{HELD}-model.json'
    if sha(baseline_path) != record['model_sha256']:
        raise ValueError('Exactly matching held-out linear model required')
    baseline = json.loads(baseline_path.read_text())
    stems = [r['stem'] for r in evidence['records'] if r['stem'] != HELD]
    if len(stems) != 11 or baseline['held_out'] != HELD or baseline['training_stems'] != stems:
        raise ValueError('Exactly the other eleven movies, not a global baseline required')
    helper = runpy.run_path(str(ROOT/'scripts/verify-focus-pair-appearance.py'))
    training, moments, inputs = [], [], []
    for row in evidence['records']:
        stem = row['stem']
        if stem == HELD:
            continue
        base_path = ROOT/'.biohub/cache/focus-candidate-ranker-v1'/f'{stem}.npz'
        app_path = ROOT/'.biohub/cache/focus-pair-appearance-v1'/f'{stem}-appearance.npy'
        moments_path = app_path.with_name(f'{stem}-moments.json')
        paths = [(base_path, row['base_sha256']), (app_path, row['descriptor_sha256']),
                 (moments_path, row['moments_sha256'])]
        if any(sha(p) != expected for p, expected in paths):
            raise ValueError('Complete original training array identity required')
        training.append((helper['BUILD']['arrays'](base_path), np.load(app_path, mmap_mode='r', allow_pickle=False)))
        moments.append(helper['read_stats'](moments_path))
        inputs.append(dict(stem=stem, hashes={str(p.relative_to(ROOT)): v for p, v in paths}))
    model = baseline['model']
    counts = validate_samples(training)
    if (model['counts'] != counts or projector(merge_stats(moments, 'fitting'), 'fitting') != model['projection']
            or model['absent_weight'] != np.sqrt(counts['parents']/counts['absent'])):
        raise ValueError('Fold-only baseline projection, counts and class weight required')
    return training, model, inputs, record['model_sha256']


def worker():
    started = time.monotonic()
    frozen = sources()
    training, baseline, inputs, baseline_sha = load_training()
    count = baseline['counts']['choices']
    x = np.lib.format.open_memmap(CACHE/'features.npy', mode='w+', dtype=np.float32, shape=(count, 72))
    margin = np.lib.format.open_memmap(CACHE/'margin.npy', mode='w+', dtype=np.float64, shape=(count,))
    parts = {k: [] for k in ('starts', 'sizes', 'chosen', 'present')}
    position = 0
    for block in blocks(training, baseline['projection'], 'full'):
        end = position+len(block['offset'])
        x[position:end] = block['x']
        margin[position:end] = block['offset']+block['x'] @ np.asarray(baseline['theta'])
        if np.any(block['x'][block['null_rows']] != 0) or np.any(margin[position:end][block['null_rows']] != -4.5):
            raise ValueError('Original null representation changed')
        for key in parts:
            parts[key].append(block[key]+position if key in ('starts', 'chosen') else block[key])
        position = end
    x.flush()
    margin.flush()
    groups = {k: np.concatenate(v) for k, v in parts.items()}
    if position != count:
        raise ValueError('Every original candidate must be prepared')
    validate_groups(**groups, rows=count)
    weight = baseline['absent_weight']
    value, _, _ = derivatives(margin, **groups, absent_weight=weight)
    penalty = .5*np.dot(baseline['theta'][1:], baseline['theta'][1:])
    if abs(value+penalty-baseline['objective']) > 1e-7:
        raise ValueError('Original full linear training objective must replay before residual fit')
    np.savez_compressed(CACHE/'groups.npz', **groups)
    preparation_seconds = time.monotonic()-started
    print(json.dumps(dict(status='complete_fitting_preparation', choices=count,
                          groups=len(groups['starts']), initial_loss=value,
                          preparation_seconds=preparation_seconds)), flush=True)
    tick = time.monotonic()
    model = fit(x, margin, **groups, absent_weight=weight, output=CACHE/'fit', role='fitting',
                progress=lambda r: print(json.dumps(dict(**r, fitting_seconds=time.monotonic()-tick)), flush=True))
    fit_seconds = time.monotonic()-tick
    if sources() != frozen:
        raise ValueError('Frozen smoke/profile method changed')
    result = dict(status='completed_first_fold_fitting_only_tree_profile', source_hashes=frozen,
                  held_out_not_loaded_or_scored=HELD, training_stems=[r['stem'] for r in inputs], inputs=inputs,
                  baseline_sha256=baseline_sha, counts=baseline['counts'], absent_weight=weight,
                  tree_sha256=sha(CACHE/'fit/portable.json'), native_sha256=sha(CACHE/'fit/native.ubj'),
                  prepared_hashes={p: sha(CACHE/p) for p in ('features.npy', 'margin.npy', 'groups.npz')},
                  initial_fitting_loss=value, final_fitting_loss=model['final_loss'],
                  native_portable_max_error=model['native_portable_max_error'],
                  preparation_seconds=preparation_seconds, fit_seconds=fit_seconds,
                  diagnostic_movies_opened=0, source_movies_opened=0, new_target_movies_opened=0,
                  held_out_evaluated=False, gpu_seconds=0, authorized_for_submission=False,
                  elapsed_seconds=time.monotonic()-started)
    (CACHE/'worker-result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k not in ('inputs', 'source_hashes')}, indent=2), flush=True)


class ProcessCounters(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
        (n, ctypes.c_size_t) for n in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
        'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]


def peak_memory(process):
    counters = ProcessCounters()
    counters.cb = ctypes.sizeof(counters)
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(int(process._handle)), ctypes.byref(counters), counters.cb):
        if process.poll() is not None:
            return 0
        raise ctypes.WinError()
    return int(counters.PeakWorkingSetSize)


def main():
    if os.name != 'nt':
        raise ValueError('This bounded local profile expects Windows memory accounting')
    destination = REPORT/f'{RUN}-profile.json'
    if destination.exists() or CACHE.exists():
        raise ValueError('Never overwrite partial or completed profile')
    CACHE.mkdir()
    (CACHE/'launch.json').write_text(json.dumps(dict(source_hashes=sources(),
        maximum_seconds=MAX_SECONDS, maximum_working_bytes=MAX_WORKING_BYTES), indent=2))
    started, peak, failure = time.monotonic(), 0, None
    process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--worker'], cwd=ROOT)
    try:
        while process.poll() is None:
            peak = max(peak, peak_memory(process))
            if peak > MAX_WORKING_BYTES or time.monotonic()-started > MAX_SECONDS:
                raise RuntimeError('Declared local profile time/RAM limit exceeded')
            time.sleep(.25)
        if process.returncode:
            raise RuntimeError(f'Profile worker exited {process.returncode}')
    except BaseException:
        failure = traceback.format_exc()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
    receipt = dict(status='completed' if failure is None else 'failed', failure=failure,
                   process_exit_code=process.returncode, peak_working_bytes=peak,
                   elapsed_seconds=time.monotonic()-started)
    if failure is None:
        receipt['worker'] = json.loads((CACHE/'worker-result.json').read_text())
        receipt['worker_result_sha256'] = sha(CACHE/'worker-result.json')
    destination.write_text(json.dumps(receipt, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in receipt.items() if k != 'worker'}, indent=2), flush=True)
    if failure:
        raise RuntimeError(failure)


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        worker()
    elif not sys.argv[1:]:
        main()
    else:
        raise ValueError('No arguments supported')
