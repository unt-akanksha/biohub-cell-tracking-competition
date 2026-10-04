"""Build the eight immutable source batches with at most two local CPU workers."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.trajectory_runtime_v1 import sha


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    reports = ROOT / 'reports/experiments'
    name = 'trajectory-event-fork16-cases-controller-v1'
    receipt, logs = reports / (name + '.json'), ROOT / '.biohub/cache' / name
    assert not receipt.exists() and not logs.exists(), 'Observe existing controller; never duplicate'
    audit_path = reports / 'trajectory-event-source-vocabulary16-v1.json'
    audit = read(audit_path)
    assert audit['status'] == 'full_source_vocabulary_audited' and audit['all_partial_constraints_compatible']
    assert len(audit['per_movie']) == 60 and audit['totals']['known_frames_outside_editable_scope'] == 0
    smoke_path = reports / 'trajectory-event-fork16-learning-smoke-v1.json'
    smoke = read(smoke_path)
    assert smoke['status'] == 'expanded_vocabulary_learning_functionality_passed'
    script = ROOT / 'scripts/prepare-trajectory-event-fork16-cases-v1.py'
    pending = [4, 5, 0, 1, 2, 3, 6, 7]
    for batch in pending:
        stem = 'trajectory-event-source-v1-b' + str(batch) + '-cases-fork16'
        assert not (reports / (stem + '.json')).exists() and not (ROOT / '.biohub/cache' / stem).exists()
    logs.mkdir()
    started = time.monotonic()
    result = dict(status='running', controller_pid=os.getpid(), cpu_workers=2, gpu_used=False,
        source_only=True, selection_or_validation_opened=False, model_fitted=False,
        production_candidate_changed=False, full_source_audit_sha256=sha(audit_path),
        small_learning_smoke_sha256=sha(smoke_path), case_builder_sha256=sha(script),
        source_sha256=sha(Path(__file__)), active={}, completed={}, failed={}, pending=pending.copy())
    active = {}
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    def persist():
        result.update(updated_utc=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic()-started,
            pending=pending.copy(), active={str(b):dict(pid=p.pid, log=str(path.relative_to(ROOT)),
                seconds=time.monotonic()-tick) for b,(p,stream,path,tick) in active.items()})
        receipt.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    persist()
    while pending or active:
        while pending and len(active) < 2 and not result['failed']:
            batch = pending.pop(0)
            assert sha(script) == result['case_builder_sha256'], 'Case builder changed while scheduled'
            path = logs / ('batch-' + str(batch) + '.log')
            stream = path.open('x', encoding='utf-8')
            process = subprocess.Popen([sys.executable, str(script), '--batch', str(batch)],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            active[batch] = (process, stream, path, time.monotonic())
            persist()
            print(json.dumps(dict(event='case_batch_started', batch=batch, pid=process.pid)), flush=True)
        for batch, (process, stream, path, tick) in list(active.items()):
            code = process.poll()
            if code is None and time.monotonic()-tick > 3600:
                # This controller owns this exact local Python process only.
                process.terminate()
                code = process.wait(timeout=30)
                result['failed'][str(batch)] = dict(reason='owned_cpu_worker_one_hour_cap', exit_code=code)
            if code is None:
                continue
            stream.close()
            del active[batch]
            evidence_path = reports / ('trajectory-event-source-v1-b' + str(batch) + '-cases-fork16.json')
            if code:
                result['failed'][str(batch)] = dict(reason='case_builder_failed', exit_code=code, log_sha256=sha(path))
            else:
                evidence = read(evidence_path)
                assert evidence['status'] == 'source_event_cases_prepared' and evidence['training_allowed']
                assert evidence['max_fork_children'] == 16
                assert evidence['full_source_coverage_sha256'] == sha(audit_path)
                result['completed'][str(batch)] = dict(receipt_sha256=sha(evidence_path), totals=evidence['totals'],
                    movies=list(evidence['per_movie']), seconds=time.monotonic()-tick, log_sha256=sha(path))
            persist()
            print(json.dumps(dict(event='case_batch_finished', batch=batch, exit_code=code)), flush=True)
        if result['failed'] and not active:
            result['status'] = 'failed_no_more_batches_scheduled'
            persist()
            raise RuntimeError('Inspect failed batch; completed data remains preserved')
        if active:
            time.sleep(2)
    assert set(result['completed']) == {str(i) for i in range(8)}
    assert sum(len(v['movies']) for v in result['completed'].values()) == 60
    result.update(status='complete_source_corpus_prepared',
        total_case_bytes=sum(v['totals']['case_bytes'] for v in result['completed'].values()),
        total_cases=sum(v['totals']['prepared'] for v in result['completed'].values()))
    persist()
    print(json.dumps({k:v for k,v in result.items() if k not in ('completed', 'active')}), flush=True)


if __name__ == '__main__':
    main()
