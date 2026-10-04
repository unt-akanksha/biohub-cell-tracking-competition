"""Recover the owned75-tree prefix with direct-interpreter memory monitoring."""
import ctypes
import gc
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
from research.focus_pair_tree_resume import resume
from research.focus_candidate_ranker import pack
from research.focus_pair_appearance import descriptors
from research.focus_pair_appearance_head import blocks
from research.focus_pair_tree import portable_predict

OLD = runpy.run_path(str(ROOT/'scripts/profile-focus-pair-tree.py'))
RUN = 'focus-pair-tree-first-fold-recovery-v1'
CACHE, REPORT = ROOT/'.biohub/cache'/RUN, ROOT/'reports/experiments'
MAX_BYTES, MAX_SECONDS = 6*1024**3, 900


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    names = list(OLD['sources']())+['research/focus_pair_tree_resume.py',
        'scripts/recover-focus-pair-tree-profile.py', f'reports/experiments/{RUN}-design.md']
    return {p: sha(ROOT/p) for p in names}


def direct_command(source):
    # Bypass Windows venv redirector while explicitly retaining only its package path.
    # -S excludes global site-packages; the real worker PID is Popen.pid.
    packages = str(Path(sys.prefix)/'Lib/site-packages')
    code = f'import sys; sys.path.insert(0, {packages!r}); '+source
    return [sys._base_executable, '-S', '-c', code]


def guard_smoke():
    process = subprocess.Popen(direct_command('import os,time; block=bytearray(64*1024**2); print(os.getpid(), flush=True); time.sleep(10)'),
                               cwd=ROOT, stdout=subprocess.PIPE, text=True)
    started, peak = time.monotonic(), 0
    try:
        while process.poll() is None and time.monotonic()-started < 5:
            peak = max(peak, OLD['peak_memory'](process))
            if peak > 32*1024**2:
                break
            time.sleep(.05)
        if peak <= 32*1024**2:
            raise ValueError('Direct worker allocation was not detected by memory guard')
    finally:
        if process.poll() is None:
            process.terminate()
        stdout, _ = process.communicate(timeout=10)
    # Allocation may be observed immediately before the print; PID identity is
    # also recorded explicitly by the actual numerical worker before data access.
    printed = stdout.strip()
    if printed and int(printed) != process.pid:
        raise ValueError('Unexpected interpreter redirector child')
    return dict(monitored_pid=process.pid, printed_pid=int(printed) if printed else None,
                peak_working_bytes=peak, detected_over_32_mib=True, owned_child_terminated=True)


def numerical_smoke():
    runtime = ROOT/'.biohub/cache/kernel-outputs/focus-pair-appearance-gpu-smoke-v1/focus_pair_appearance_gpu_smoke/runtime'
    with np.load(runtime/'pair.npz', allow_pickle=False) as data:
        packet = {k: data[k].copy() for k in data.files}
    baseline = json.loads((runtime/'full-cpu-model.json').read_text())
    parameters = json.loads((runtime/'spec.json').read_text())['motion_parameters']
    base = pack(packet, parameters, 'fitting')
    app = descriptors(packet, np.flatnonzero(packet['labels'] >= 0))
    block = next(blocks([(base, app)], baseline['projection'], 'full'))
    margin = block['offset']+block['x'] @ np.asarray(baseline['theta'])
    groups = {k: block[k] for k in ('starts', 'sizes', 'chosen', 'present')}
    original_dir = ROOT/'.biohub/cache/focus-pair-tree-v1/fit'
    expected_model = json.loads((original_dir/'portable.json').read_text())
    result = resume(block['x'], margin, **groups, absent_weight=baseline['absent_weight'],
                    checkpoint=original_dir/'partial-075.ubj', output=CACHE/'smoke-fit', role='fitting')
    expected, actual = [portable_predict(m, block['x']) for m in (expected_model, result)]
    if not np.array_equal(actual, expected) or result['trees'] != expected_model['trees']:
        raise ValueError('75+25 resume must exactly reproduce original100-tree real smoke')
    return dict(exact_original_100_trees=True, exact_original_smoke_scores=True,
                final_loss=result['final_loss'], original_loss=expected_model['final_loss'])


def worker():
    (CACHE/'worker-pid.json').write_text(json.dumps(dict(pid=os.getpid())))
    started, frozen = time.monotonic(), sources()
    if sha(REPORT/'focus-pair-tree-first-fold-v1-profile.json') != 'c68365d93644a8bff69ce5ea0acf7a10227877d9985e3c951bffb71fbb37092f':
        raise ValueError('Original stopped profile receipt required')
    old_cache = OLD['CACHE']
    checkpoint = old_cache/'fit/partial-075.ubj'
    if sha(checkpoint) != 'cf98863fc4ac97338a8a10349ad28f27482ce5caada877fbf5f1e51ecf4a7241':
        raise ValueError('Exact preserved75-tree model required')
    smoke = numerical_smoke()
    (CACHE/'resume-smoke.json').write_text(json.dumps(smoke, indent=2))
    print(json.dumps(smoke), flush=True)
    training, baseline, inputs, baseline_sha = OLD['load_training']()
    x = np.load(old_cache/'features.npy', mmap_mode='r', allow_pickle=False)
    margin = np.load(old_cache/'margin.npy', mmap_mode='r', allow_pickle=False)
    with np.load(old_cache/'groups.npz', allow_pickle=False) as saved:
        groups = {k: saved[k].copy() for k in saved.files}
    position, group_position = 0, 0
    for block in blocks(training, baseline['projection'], 'full'):
        end, group_end = position+len(block['offset']), group_position+len(block['starts'])
        if (not np.array_equal(x[position:end], block['x'].astype(np.float32))
                or not np.array_equal(margin[position:end], block['offset']+block['x'] @ np.asarray(baseline['theta']))):
            raise ValueError('Every preserved feature/margin must replay from original11 fitting movies')
        for key in groups:
            expected = block[key]+position if key in ('starts', 'chosen') else block[key]
            if not np.array_equal(groups[key][group_position:group_end], expected):
                raise ValueError('Every original known candidate group must replay')
        position, group_position = end, group_end
    if position != len(x) or group_position != len(groups['starts']):
        raise ValueError('Complete preserved fitting rows required')
    # Release the redundant ~1.5GB source maps/base arrays before DMatrix construction.
    # Features, labels, baseline margins, tree bins/settings and remaining rounds are unchanged.
    del training, block
    gc.collect()
    print(json.dumps(dict(status='all_prepared_arrays_replayed_sources_released', choices=position)), flush=True)
    tick = time.monotonic()
    model = resume(x, margin, **groups, absent_weight=baseline['absent_weight'], checkpoint=checkpoint,
                   output=CACHE/'fit', role='fitting',
                   progress=lambda r: print(json.dumps(dict(**r, recovery_seconds=time.monotonic()-tick)), flush=True))
    if sources() != frozen:
        raise ValueError('Frozen recovery source changed')
    record = dict(status='recovered_complete_100_tree_first_fitting_fold', source_hashes=frozen,
        baseline_sha256=baseline_sha, checkpoint_sha256=sha(checkpoint), inputs=inputs, counts=baseline['counts'],
        tree_sha256=sha(CACHE/'fit/portable.json'), native_sha256=sha(CACHE/'fit/native.ubj'),
        prepared_hashes={p: sha(old_cache/p) for p in ('features.npy', 'margin.npy', 'groups.npz')},
        numerical_smoke=smoke, source_arrays_released_before_tree_fit=True,
        first75_rounds_reused=True, remaining_rounds=25, initial_loss=model['initial_loss'],
        prefix_loss=model['prefix_loss'], final_loss=model['final_loss'],
        native_portable_max_error=model['native_portable_max_error'], recovery_seconds=time.monotonic()-tick,
        held_out_not_loaded_or_scored=OLD['HELD'], diagnostic_movies_opened=0, source_movies_opened=0,
        new_target_movies_opened=0, gpu_seconds=0, authorized_for_submission=False,
        elapsed_seconds=time.monotonic()-started)
    (CACHE/'worker-result.json').write_text(json.dumps(record, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in record.items() if k not in ('source_hashes', 'inputs')}, indent=2), flush=True)


def main():
    destination = REPORT/f'{RUN}-profile.json'
    if destination.exists() or CACHE.exists():
        raise ValueError('Never overwrite recovery artifacts')
    CACHE.mkdir()
    guard = guard_smoke()
    (CACHE/'guard-smoke.json').write_text(json.dumps(guard, indent=2))
    (CACHE/'launch.json').write_text(json.dumps(dict(source_hashes=sources(), maximum_seconds=MAX_SECONDS,
        maximum_working_bytes=MAX_BYTES, guard=guard), indent=2))
    command = direct_command(f'import runpy; sys.argv=[{str(Path(__file__).resolve())!r}, "--worker"]; runpy.run_path(sys.argv[0], run_name="__main__")')
    process = subprocess.Popen(command, cwd=ROOT)
    started, peak, failure, confirmed = time.monotonic(), 0, None, False
    try:
        while process.poll() is None:
            pid_path = CACHE/'worker-pid.json'
            if not confirmed and pid_path.exists():
                if json.loads(pid_path.read_text())['pid'] != process.pid:
                    raise ValueError('Memory guard must monitor actual numerical worker PID')
                confirmed = True
            peak = max(peak, OLD['peak_memory'](process))
            if peak > MAX_BYTES or time.monotonic()-started > MAX_SECONDS:
                raise RuntimeError('Declared real-worker time/RAM limit exceeded')
            time.sleep(.25)
        if process.returncode or not confirmed:
            raise RuntimeError(f'Recovery exited {process.returncode}; PID confirmed={confirmed}')
    except BaseException:
        failure = traceback.format_exc()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
    result = dict(status='completed' if failure is None else 'failed', failure=failure,
                  process_exit_code=process.returncode, actual_worker_pid_confirmed=confirmed,
                  peak_working_bytes=peak, elapsed_seconds=time.monotonic()-started, guard_smoke=guard)
    if failure is None:
        result['worker'] = json.loads((CACHE/'worker-result.json').read_text())
        result['worker_result_sha256'] = sha(CACHE/'worker-result.json')
    destination.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k != 'worker'}, indent=2), flush=True)
    if failure:
        raise RuntimeError(failure)


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        worker()
    elif not sys.argv[1:]:
        main()
    else:
        raise ValueError('No arguments supported')
