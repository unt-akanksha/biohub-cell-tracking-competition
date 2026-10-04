"""Run the completed-screen verifier with real-worker time/RAM accounting."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
TRAIN = runpy.run_path(str(ROOT/'scripts/fit-focus-pair-tree-lomo.py'))
RECOVERY = TRAIN['RECOVERY']
CACHE = ROOT/'.biohub/cache/focus-pair-tree-lomo-v1-verification-control'
MAX_SECONDS, MAX_BYTES = 1800, 4*1024**3


def main():
    if Path(sys.prefix).resolve() != (ROOT/'.biohub/cache/pair-tree-venv').resolve():
        raise ValueError('Use isolated pair-tree-venv for the native verifier')
    result = json.loads((ROOT/'reports/experiments/focus-pair-tree-lomo-v1-result.json').read_text())
    if result['status'] != 'completed' or result['failure'] is not None or len(result['folds']) != 12:
        raise ValueError('Wait for all12 terminal completed folds before verification')
    output = ROOT/'reports/experiments/focus-pair-tree-lomo-v1-verification.json'
    if output.exists() or CACHE.exists():
        raise ValueError('Never race or overwrite a verifier attempt')
    if TRAIN['available_memory']() < 5*1024**3:
        raise ValueError('At least5GiB free local memory required before verification')
    verifier = ROOT/'scripts/verify-focus-pair-tree-lomo.py'
    # This exact source already passed the actual first-fold native/portable smoke.
    smoke = json.loads((ROOT/'reports/experiments/focus-pair-tree-lomo-v1-verifier-smoke.json').read_text())
    if TRAIN['sha'](verifier) != smoke['source_sha256']:
        raise ValueError('Actual smoke-tested verifier source required')
    CACHE.mkdir()
    temporary, pid_path = CACHE/'worker-pid.tmp', CACHE/'worker-pid.json'
    code = (f'import os,json,pathlib,runpy; pathlib.Path({str(temporary)!r}).write_text(json.dumps(dict(pid=os.getpid()))); '
            f'pathlib.Path({str(temporary)!r}).replace({str(pid_path)!r}); '
            f'sys.argv=[{str(verifier)!r}, "--worker"]; runpy.run_path(sys.argv[0], run_name="__main__")')
    started, peak, confirmed, failure = time.monotonic(), 0, False, None
    process = subprocess.Popen(RECOVERY['direct_command'](code), cwd=ROOT)
    (CACHE/'launch.json').write_text(json.dumps(dict(pid=process.pid, maximum_seconds=MAX_SECONDS,
        maximum_bytes=MAX_BYTES, verifier_sha256=TRAIN['sha'](verifier),
        controller_sha256=TRAIN['sha'](Path(__file__))), indent=2))
    try:
        while process.poll() is None:
            if not confirmed and pid_path.exists():
                if json.loads(pid_path.read_text())['pid'] != process.pid:
                    raise ValueError('Memory accounting must monitor the real verifier, not a launcher')
                confirmed = True
            peak = max(peak, RECOVERY['OLD']['peak_memory'](process))
            if peak > MAX_BYTES or time.monotonic()-started > MAX_SECONDS:
                raise RuntimeError('Verifier real-worker time/RAM limit exceeded')
            time.sleep(.25)
        if process.returncode or not confirmed or not output.exists():
            raise RuntimeError(f'Incomplete verifier: exit={process.returncode}, PID confirmed={confirmed}')
    except BaseException:
        failure = traceback.format_exc()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
    receipt = dict(status='completed' if failure is None else 'failed', failure=failure,
        peak_working_bytes=peak, elapsed_seconds=time.monotonic()-started,
        actual_worker_pid_confirmed=confirmed, verifier_result_sha256=TRAIN['sha'](output) if output.exists() else None,
        gpu_seconds=0, authorized_for_submission=False)
    (CACHE/'result.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2), flush=True)
    if failure:
        raise RuntimeError(failure)


if __name__ == '__main__':
    main()
