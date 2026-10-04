"""One-shot cache harvest -> sequential flow GPU -> local CPU comparison."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / '.biohub/automation/focus-source-route-v1'
CACHE_REF = 'indarkarhana/biohub-focus-source-cache-v1'
FLOW_REF = 'indarkarhana/biohub-focus-source-flow-v1'
CACHE_SHA = 'd65eac4ce6354829b13fd92176c3163b45a50efabe2c326f5b5322d7d73f322b'
SOURCE_FILES = [
    'scripts/verify-focus-source-cache.py', 'scripts/build-focus-source-flow.py',
    'scripts/run-focus-source-flow.py', 'scripts/score-focus-source-flow.py',
    'research/focus_source_flow_contract.py', 'research/focus_source_flow_comparison.py',
    'research/independent_motion_prior.py', 'research/backward_flow_linking.py',
    'scripts/score-independent-selection.py', 'scripts/score-focus-owned-flow-full.py',
    'scripts/run-owned-detector-evaluation-queue.py']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate():
    path = ROOT / 'kaggle/biohub-focus-source-cache-v1/biohub-focus-source-cache-v1.ipynb'
    if sha(path) != CACHE_SHA:
        raise ValueError('Frozen running source-cache notebook changed')
    python = ROOT / '.biohub/cache/graph-analysis-venv/Scripts/python.exe'
    if not python.is_file():
        raise ValueError('Working CPU graph environment required')
    for name in SOURCE_FILES:
        if not (ROOT / name).is_file():
            raise ValueError('Source route implementation missing: ' + name)
    return python


def source_snapshot():
    return {p: sha(ROOT / p) for p in SOURCE_FILES}


def wait_complete(helper, ref, deadline):
    previous = None
    while time.monotonic() < deadline:
        try:
            output = helper.command(['kaggle', 'kernels', 'status', ref])
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            helper.log('status_observation_failed', ref=ref, error=str(exc)[-800:], remote_terminal_not_inferred=True)
            time.sleep(60)
            continue
        status = helper.status_value(output, ref)
        if status != previous:
            helper.log('kernel_status', ref=ref, status=status)
        previous = status
        if status == 'COMPLETE':
            state = helper.INSPECT(ref)
            if state['present'] is not True or state['current_version_number'] != 1:
                raise ValueError('Exact completed version1 required')
            return
        time.sleep(60)
    raise TimeoutError('Source route wait deadline reached; existing remote jobs left untouched')


def launch_flow(helper, python):
    if helper.INSPECT(FLOW_REF)['present']:
        raise ValueError('Source-flow kernel already exists; no duplicate launch')
    helper.command([python, ROOT / 'scripts/build-focus-source-flow.py'])
    folder = ROOT / 'kaggle/biohub-focus-source-flow-v1'
    meta = json.loads((folder / 'kernel-metadata.json').read_text())
    notebook = folder / meta['code_file']
    nb = json.loads(notebook.read_text())
    if (meta['id'] != FLOW_REF or meta['enable_gpu'] is not True
            or meta['enable_internet'] is not False or meta['enable_tpu'] is not False
            or nb['metadata']['codex']['declared_budget_seconds'] != 3600
            or nb['metadata']['codex']['contract']['new_target_movies_opened'] != 0
            or max(len(meta['title']),len(meta['id'].split('/')[1])) > 50):
        raise ValueError('Exact offline bounded source-flow metadata required')
    notebook_sha = sha(notebook)
    remaining = helper.gpu_remaining(helper.command(['kaggle', 'quota']))
    helper.log('fresh_quota_authorization', ref=FLOW_REF, remaining_hours=remaining,
               declared_hours=1, reserve_hours=8)
    helper.log('push_intent', ref=FLOW_REF, notebook_sha256=notebook_sha)
    output = helper.command(['kaggle', 'kernels', 'push', '-p', folder,
                             '--accelerator', 'NvidiaTeslaT4', '--timeout', '3600'])
    helper.verify_push(output, FLOW_REF)
    helper.log('push_accepted', ref=FLOW_REF, version=1, notebook_sha256=notebook_sha)
    return notebook_sha


def run():
    python = validate()
    frozen = source_snapshot()
    (STATE / 'frozen_sources.json').write_text(json.dumps(frozen, indent=2))
    for name in ('OMP_NUM_THREADS', 'POLARS_MAX_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '2'
    spec = importlib.util.spec_from_file_location('focus_source_queue_helpers', ROOT / 'scripts/run-owned-detector-evaluation-queue.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.STATE = STATE
    deadline = time.monotonic() + 3*3600
    original_command = helper.command
    def bounded_command(args, timeout=300):
        left = deadline-time.monotonic()
        if left <= 0:
            raise TimeoutError('Source route deadline reached')
        if source_snapshot() != frozen:
            raise ValueError('Source route code changed while waiting; no continuation')
        return original_command(args, timeout=min(timeout, left))
    helper.command = bounded_command
    wait_complete(helper, CACHE_REF, deadline)
    helper.harvest(CACHE_REF, 'focus-source-cache-v1', r'(focus_source_cache_terminal|verified_runtime_identity|raw_detections/).*')
    helper.command([python, ROOT / 'scripts/verify-focus-source-cache.py'])
    cache = json.loads((ROOT / 'reports/experiments/focus-source-cache-v1-result.json').read_text())
    helper.log('source_cache_verified', source_frames=cache['complete_source_frames'],
               smoke_frames=cache['smoke_frames_replayed'], nodes=sum(r['nodes'] for r in cache['records'] if r['scope']=='source_selection'))
    validate()
    notebook_sha = launch_flow(helper, python)
    wait_complete(helper, FLOW_REF, deadline)
    helper.command(['kaggle', 'kernels', 'output', FLOW_REF + '/1', '-p',
                    ROOT / '.biohub/cache/kernel-outputs/focus-source-flow-v1',
                    '--file-pattern', r'^focus_source_flow/.*', '-q'], timeout=900)
    helper.log('local_cpu_score_started', notebook_sha256=notebook_sha)
    helper.command([python, ROOT / 'scripts/score-focus-source-flow.py', '--notebook-sha256', notebook_sha], timeout=1800)
    result = json.loads((ROOT / 'reports/experiments/focus-source-flow-v1-result.json').read_text())
    helper.log('source_comparison_completed', comparison=result['comparison'], summaries=result['summaries'],
               submission_performed=False, new_target_movies_opened=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    validate()
    if not args.execute:
        print('Source route implementation present; no external action or GPU launch')
        raise SystemExit(0)
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'started.json').open('x', encoding='utf-8') as handle:
        json.dump(dict(pid=os.getpid(), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), handle)
    status, error = 'failed', None
    try:
        run()
        status = 'completed'
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        (STATE / 'terminal.json').write_text(json.dumps(dict(status=status, error=error,
            submission_performed=False, new_target_movies_opened=0,
            finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))))
