"""Fixed twelve-fold nonlinear candidate ranking, sequential bounded CPU workers."""
import ctypes
import gc
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
from research.focus_pair_appearance_head import blocks, validate_samples, metrics as linear_metrics
from research.focus_pair_tree import fit, portable_predict, derivatives, validate_groups

RECOVERY = runpy.run_path(str(ROOT/'scripts/recover-focus-pair-tree-profile.py'))
LINEAR = runpy.run_path(str(ROOT/'scripts/fit-focus-pair-appearance-conditioned.py'))
sha = RECOVERY['sha']
RUN = 'focus-pair-tree-lomo-v1'
CACHE, REPORT = ROOT/'.biohub/cache'/RUN, ROOT/'reports/experiments'
MAX_SECONDS, FOLD_SECONDS, MAX_BYTES = 7200, 900, 6*1024**3


def sources():
    names = list(RECOVERY['sources']())+['scripts/fit-focus-pair-tree-lomo.py',
        'tests/test_focus_pair_tree_lomo.py', f'reports/experiments/{RUN}-design.md']
    return {p: sha(ROOT/p) for p in names}


def evidence():
    path = REPORT/'focus-pair-tree-first-fold-recovery-v1-profile.json'
    if sha(path) != 'fefb07cf2f38aafa90eaab53d6f672a02d81b34e46869f4e326faa5c867400cb':
        raise ValueError('Actual completed resource recovery required')
    receipt = json.loads(path.read_text())
    if (receipt['status'] != 'completed' or receipt['failure'] is not None
            or not receipt['actual_worker_pid_confirmed'] or receipt['peak_working_bytes'] > MAX_BYTES
            or any(sha(ROOT/p) != v for p, v in receipt['worker']['source_hashes'].items())):
        raise ValueError('Successful frozen actual-worker resource profile required')
    full_path = LINEAR['CACHE']/'full/result.json'
    if sha(full_path) != 'c32589575cd1d225b63c93bbcea8e9f332f629798e499b767c90a22549ee3919':
        raise ValueError('Exact twelve original full linear models/controls required')
    full = json.loads(full_path.read_text())
    return receipt, {r['held_out']: r for r in full['folds']}


def model_metrics(sample, baseline, tree):
    result = dict(loss_sum=0., known_parent=0, known_absent=0, correct_parent=0, correct_absent=0)
    for block in blocks([sample], baseline['projection'], 'full'):
        score = block['offset']+block['x'] @ np.asarray(baseline['theta'])+portable_predict(tree, block['x'])
        if not np.isfinite(score).all():
            raise ValueError('All original real/null logits must be finite')
        maximum = np.maximum.reduceat(score, block['starts'])
        normalizer = maximum+np.log(np.add.reduceat(np.exp(score-maximum[block['ids']]), block['starts']))
        rows = np.arange(len(score))
        choice = np.minimum.reduceat(np.where(score == maximum[block['ids']], rows, len(rows)), block['starts'])
        present = block['present'].astype(bool)
        correct = choice == block['chosen']
        result['loss_sum'] += float((normalizer-score[block['chosen']]).sum())
        for label, keep in (('parent', present), ('absent', ~present)):
            result['known_'+label] += int(keep.sum())
            result['correct_'+label] += int((correct & keep).sum())
    result['nll'] = result['loss_sum']/(result['known_parent']+result['known_absent'])
    return result


def prepare(training, baseline, folder):
    count = validate_samples(training)['choices']
    x = np.lib.format.open_memmap(folder/'features.npy', mode='w+', dtype=np.float32, shape=(count, 72))
    margin = np.lib.format.open_memmap(folder/'margin.npy', mode='w+', dtype=np.float64, shape=(count,))
    parts = {k: [] for k in ('starts', 'sizes', 'chosen', 'present')}
    position = 0
    for block in blocks(training, baseline['projection'], 'full'):
        end = position+len(block['offset'])
        x[position:end] = block['x']
        margin[position:end] = block['offset']+block['x'] @ np.asarray(baseline['theta'])
        for key in parts:
            parts[key].append(block[key]+position if key in ('starts', 'chosen') else block[key])
        position = end
    x.flush()
    margin.flush()
    groups = {k: np.concatenate(v) for k, v in parts.items()}
    validate_groups(**groups, rows=count)
    if position != count:
        raise ValueError('Complete candidate preparation required')
    value = derivatives(margin, **groups, absent_weight=baseline['absent_weight'])[0]
    penalty = .5*np.dot(baseline['theta'][1:], baseline['theta'][1:])
    if abs(value+penalty-baseline['objective']) > 1e-7:
        raise ValueError('Original baseline fitting objective changed')
    np.savez_compressed(folder/'groups.npz', **groups)
    return x, margin, groups


def worker(held):
    started, folder = time.monotonic(), CACHE/held
    (folder/'worker-pid.tmp').write_text(json.dumps(dict(pid=os.getpid())))
    (folder/'worker-pid.tmp').replace(folder/'worker-pid.json')
    frozen = sources()
    profile, records = evidence()
    if held not in records:
        raise ValueError('Only predeclared fitting-movie folds allowed')
    baseline_path = LINEAR['CACHE']/'full'/f'{held}-model.json'
    saved = json.loads(baseline_path.read_text())
    baseline, train = saved['model'], saved['training_stems']
    if (sha(baseline_path) != records[held]['model_sha256'] or saved['held_out'] != held
            or train != [s for s in records if s != held]):
        raise ValueError('Exactly matched fold-only linear baseline required')
    reused = held == RECOVERY['OLD']['HELD']
    if reused:
        tree_path = RECOVERY['CACHE']/'fit/portable.json'
        if sha(tree_path) != profile['worker']['tree_sha256'] or profile['worker']['baseline_sha256'] != sha(baseline_path):
            raise ValueError('Exact recovered first-fold tree/baseline required')
        tree = json.loads(tree_path.read_text())
        import xgboost as xgb
        prefix_path = RECOVERY['OLD']['CACHE']/'fit/partial-075.ubj'
        if sha(prefix_path) != profile['worker']['checkpoint_sha256']:
            raise ValueError('Preserved native prefix identity changed')
        prefix = xgb.Booster(params={'nthread': 2}, model_file=prefix_path)
        if tree['trees'][:75] != [json.loads(t) for t in prefix.get_dump(dump_format='json')]:
            raise ValueError('Recovered100-tree model must preserve its actual75-tree prefix')
        fit_seconds = 0.
    else:
        _, samples, moments, _, _ = LINEAR['load']()
        training = [samples[s] for s in train]
        if (validate_samples(training) != baseline['counts']
                or projector(merge_stats([moments[s] for s in train], 'fitting'), 'fitting') != baseline['projection']):
            raise ValueError('Exact training-fold representation and counts required')
        x, margin, groups = prepare(training, baseline, folder)
        del training, samples, moments
        gc.collect()
        tick = time.monotonic()
        tree = fit(x, margin, **groups, absent_weight=baseline['absent_weight'], output=folder/'fit', role='fitting',
                   progress=lambda r: print(json.dumps(dict(held_out=held, **r, fit_seconds=time.monotonic()-tick)), flush=True))
        fit_seconds = time.monotonic()-tick
        del x, margin, groups
        gc.collect()
    # Commit the entire composite before evaluating its correction-held-out movie.
    model_path = folder/'model.json'
    model_path.write_text(json.dumps(dict(held_out=held, training_stems=train, baseline=baseline, tree=tree,
        baseline_sha256=sha(baseline_path), reused_profile=reused, authorized_for_submission=False), indent=2, allow_nan=False))
    restored = json.loads(model_path.read_text())
    receipt_path = REPORT/'focus-pair-appearance-v1-data-smoke.json'
    if sha(receipt_path) != '4e2eb70fa910025184a53cc2549293658496ad77803bc403a82137254af9c22f':
        raise ValueError('Actual original held-out array receipt required')
    data_receipt = json.loads(receipt_path.read_text())
    input_row = next(r for r in data_receipt['records'] if r['stem'] == held)
    base_path = ROOT/'.biohub/cache/focus-candidate-ranker-v1'/f'{held}.npz'
    app_path = ROOT/'.biohub/cache/focus-pair-appearance-v1'/f'{held}-appearance.npy'
    if sha(base_path) != input_row['base_sha256'] or sha(app_path) != input_row['descriptor_sha256']:
        raise ValueError('Original complete held-out candidate arrays required')
    with np.load(base_path, allow_pickle=False) as data:
        base = {k: data[k].copy() for k in data.files}
    sample = (base, np.load(app_path, mmap_mode='r', allow_pickle=False))
    if linear_metrics([sample], baseline) != records[held]['candidate']:
        raise ValueError('Actual held-out full linear control changed')
    row = dict(held_out=held, training_stems=train, model_sha256=sha(model_path), reused_profile=reused,
               physical=records[held]['physical'], neural=records[held]['neural'],
               weighted_ranker=records[held]['weighted_ranker'], lda=records[held]['lda'],
               full_linear=records[held]['candidate'], candidate=model_metrics(sample, restored['baseline'], restored['tree']),
               fit_seconds=fit_seconds, elapsed_seconds=time.monotonic()-started)
    if sources() != frozen:
        raise ValueError('Frozen method changed while fitting or scoring')
    (folder/'result.json').write_text(json.dumps(row, indent=2, allow_nan=False))
    print(json.dumps(row), flush=True)


class MemoryStatus(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)]+[(n, ctypes.c_ulonglong) for n in
        ('total_physical', 'available_physical', 'total_page', 'available_page', 'total_virtual', 'available_virtual', 'available_extended')]


def available_memory():
    state = MemoryStatus()
    state.length = ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
        raise ctypes.WinError()
    return int(state.available_physical)


def main():
    target = REPORT/f'{RUN}-result.json'
    if target.exists() or CACHE.exists():
        raise ValueError('Never overwrite complete or partial LOMO experiment')
    profile, records = evidence()
    frozen, started = sources(), time.monotonic()
    CACHE.mkdir()
    (CACHE/'launch.json').write_text(json.dumps(dict(source_hashes=frozen, maximum_seconds=MAX_SECONDS,
        maximum_fold_seconds=FOLD_SECONDS, maximum_worker_bytes=MAX_BYTES, stems=list(records)), indent=2))
    folds, executions, failure = [], [], None
    for held in records:
        if available_memory() < 7*1024**3:
            failure = dict(held_out=held, error='Less than7GiB local memory available; no job started')
            break
        folder = CACHE/held
        folder.mkdir()
        code = f'import runpy; sys.argv=[{str(Path(__file__).resolve())!r}, "--worker", {held!r}]; runpy.run_path(sys.argv[0], run_name="__main__")'
        process = subprocess.Popen(RECOVERY['direct_command'](code), cwd=ROOT)
        tick, peak, confirmed = time.monotonic(), 0, False
        try:
            while process.poll() is None:
                pid_path = folder/'worker-pid.json'
                if not confirmed and pid_path.exists():
                    if json.loads(pid_path.read_text())['pid'] != process.pid:
                        raise ValueError('Monitor actual numerical worker PID, not a launcher')
                    confirmed = True
                peak = max(peak, RECOVERY['OLD']['peak_memory'](process))
                if (peak > MAX_BYTES or time.monotonic()-tick > FOLD_SECONDS
                        or time.monotonic()-started > MAX_SECONDS):
                    raise RuntimeError('Declared CPU time/RAM limit exceeded')
                time.sleep(.25)
            if process.returncode or not confirmed:
                raise RuntimeError(f'Fold worker exited {process.returncode}, PID confirmed={confirmed}')
            folds.append(json.loads((folder/'result.json').read_text()))
            executions.append(dict(held_out=held, peak_working_bytes=peak,
                actual_worker_pid_confirmed=confirmed, elapsed_seconds=time.monotonic()-tick))
            (folder/'execution.json').write_text(json.dumps(executions[-1], indent=2))
            if sources() != frozen:
                raise ValueError('Frozen complete-run source changed')
        except BaseException:
            failure = dict(held_out=held, error=traceback.format_exc(), peak_working_bytes=peak)
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            (folder/'failure.json').write_text(json.dumps(failure, indent=2))
            break
    decision = LINEAR['BASE']['gate'](folds) if len(folds) == 12 and failure is None else None
    result = dict(status='completed' if failure is None else 'stopped', run_id=RUN, source_hashes=frozen,
        folds=folds, executions=executions, failure=failure, gate=decision, final_model=None,
        diagnostic_movies_opened=0, source_movies_opened=0, new_target_movies_opened=0, gpu_seconds=0,
        authorized_for_submission=False, elapsed_seconds=time.monotonic()-started)
    target.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k not in ('source_hashes', 'folds', 'executions')}, indent=2), flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(sys.argv[2])
    elif not sys.argv[1:]:
        main()
    else:
        raise ValueError('Only controlled parent or declared fold worker supported')
