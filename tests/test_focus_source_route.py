import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
M = runpy.run_path(str(ROOT / 'scripts/run-focus-source-route.py'))
H = runpy.run_path(str(ROOT / 'scripts/run-owned-detector-evaluation-queue.py'))


def helper(tmp_path, remaining='10.00', present=False, receipt=None):
    calls = []
    folder = tmp_path / 'kaggle/biohub-focus-source-flow-v1'
    folder.mkdir(parents=True)
    meta = dict(id=M['FLOW_REF'], title='biohub-focus-source-flow-v1', code_file='flow.ipynb',
                enable_gpu=True, enable_internet=False, enable_tpu=False)
    (folder / 'kernel-metadata.json').write_text(json.dumps(meta))
    (folder / 'flow.ipynb').write_text(json.dumps(dict(metadata=dict(codex=dict(
        declared_budget_seconds=3600, contract=dict(new_target_movies_opened=0))))))
    def command(args):
        words = list(map(str, args))
        calls.append(words)
        if words[:2] == ['kaggle', 'quota']:
            return f'GPU 20.00h {remaining}h 30.00h 2026-09-12'
        if words[:3] == ['kaggle', 'kernels', 'push']:
            return receipt if receipt is not None else 'Kernel version 1 successfully pushed. https://www.kaggle.com/code/' + M['FLOW_REF']
        return ''
    h = SimpleNamespace(command=command, INSPECT=lambda ref: dict(present=present),
                        gpu_remaining=H['gpu_remaining'], verify_push=H['verify_push'], log=lambda *a, **kw: None)
    return h, calls


def test_fresh_quota_immediately_precedes_single_push(tmp_path, monkeypatch):
    h, calls = helper(tmp_path)
    function = M['launch_flow']
    monkeypatch.setitem(function.__globals__, 'ROOT', tmp_path)
    result = function(h, 'python')
    assert len(result) == 64
    assert calls[-2] == ['kaggle', 'quota']
    assert calls[-1][:3] == ['kaggle', 'kernels', 'push']
    assert sum(c[:3] == ['kaggle', 'kernels', 'push'] for c in calls) == 1


def test_reserve_violation_never_pushes(tmp_path, monkeypatch):
    h, calls = helper(tmp_path, remaining='8.99')
    function = M['launch_flow']
    monkeypatch.setitem(function.__globals__, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='reserve'):
        function(h, 'python')
    assert not any(c[:3] == ['kaggle', 'kernels', 'push'] for c in calls)


def test_existing_remote_kernel_never_relaunches(tmp_path, monkeypatch):
    h, calls = helper(tmp_path, present=True)
    function = M['launch_flow']
    monkeypatch.setitem(function.__globals__, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='duplicate'):
        function(h, 'python')
    assert calls == []


def test_ambiguous_push_is_not_retried(tmp_path, monkeypatch):
    h, calls = helper(tmp_path, receipt='Connection dropped after upload')
    function = M['launch_flow']
    monkeypatch.setitem(function.__globals__, 'ROOT', tmp_path)
    with pytest.raises(RuntimeError, match='ambiguous'):
        function(h, 'python')
    assert sum(c[:3] == ['kaggle', 'kernels', 'push'] for c in calls) == 1


def test_wait_observation_failure_does_not_mean_remote_terminal(monkeypatch):
    events, calls = [], []
    def command(args):
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError('temporary status transport failure')
        return M['CACHE_REF'] + ' has status "KernelWorkerStatus.COMPLETE"'
    h = SimpleNamespace(command=command, status_value=H['status_value'],
                        log=lambda event, **kw: events.append(event),
                        INSPECT=lambda ref: dict(present=True, current_version_number=1))
    function = M['wait_complete']
    monkeypatch.setattr(function.__globals__['time'], 'sleep', lambda seconds: None)
    function(h, M['CACHE_REF'], function.__globals__['time'].monotonic()+60)
    assert len(calls) == 2 and 'status_observation_failed' in events


def test_wait_rejects_actual_failed_run_and_wrong_version(monkeypatch):
    function = M['wait_complete']
    h = SimpleNamespace(command=lambda args: M['CACHE_REF'] + ' has status "KernelWorkerStatus.ERROR"',
                        status_value=H['status_value'], log=lambda *a, **kw: None)
    with pytest.raises(RuntimeError, match='terminal'):
        function(h, M['CACHE_REF'], function.__globals__['time'].monotonic()+60)
    h.command = lambda args: M['CACHE_REF'] + ' has status "KernelWorkerStatus.COMPLETE"'
    h.INSPECT = lambda ref: dict(present=True, current_version_number=2)
    with pytest.raises(ValueError, match='version1'):
        function(h, M['CACHE_REF'], function.__globals__['time'].monotonic()+60)
