from pathlib import Path
import runpy
import pytest
ROOT = Path(__file__).resolve().parents[1]
M = runpy.run_path(str(ROOT/'scripts/run-owned-detector-evaluation-queue.py'))


def test_quota_parses_remaining_not_used():
    assert M['gpu_remaining']('GPU       15.63h  14.37h     30.00h  2026-09-12T00:00:00\n')==14.37
    assert M['gpu_remaining']('GPU 21.00h 9.00h 30.00h tomorrow\n')==9


@pytest.mark.parametrize('text',['GPU 21h 8.99h 30h tomorrow','GPU unavailable','TPU 0h 20h 20h tomorrow',
                                'GPU 0h 10h 30h x\nGPU 0h 10h 30h x'])
def test_quota_fails_closed(text):
    with pytest.raises(ValueError): M['gpu_remaining'](text)


def test_status_requires_exact_ref_and_supported_state():
    ref=M['FIT']
    assert M['status_value'](ref+' has status "KernelWorkerStatus.COMPLETE"',ref)=='COMPLETE'
    with pytest.raises(RuntimeError): M['status_value'](ref+' has status "KernelWorkerStatus.ERROR"',ref)
    with pytest.raises(ValueError): M['status_value']('another has status "KernelWorkerStatus.COMPLETE"',ref)


def test_successful_exit_is_not_sufficient_push_receipt():
    ref='indarkarhana/biohub-owned-detector-pu-selection-v1'
    good=f'Kernel version 1 successfully pushed. Please check progress at https://www.kaggle.com/code/{ref}'
    M['verify_push'](good,ref)
    for bad in ('Error: slots exhausted',good.replace('version 1','version 2'),good.replace(ref,'other/slug')):
        with pytest.raises(RuntimeError): M['verify_push'](bad,ref)


def test_queue_is_sequential_gpu_and_never_opens_audit_or_submits(monkeypatch):
    run = M['run']
    events = []
    monkeypatch.setitem(run.__globals__,'wait_complete',lambda ref,deadline:events.append(('wait',ref)))
    monkeypatch.setitem(run.__globals__,'harvest',lambda ref,*args:events.append(('harvest',ref)))
    monkeypatch.setitem(run.__globals__,'log',lambda *args,**kwargs:None)
    monkeypatch.setitem(run.__globals__,'command',lambda args: '{}')
    def launch(arm,kind,gpu):
        events.append(('launch',arm,kind,gpu))
        return arm+'-'+kind
    monkeypatch.setitem(run.__globals__,'launch',launch)
    run()
    gpu_launches=[r for r in events if r[0]=='launch' and r[-1]]
    assert gpu_launches==[('launch','sparse','selection',True),('launch','pu','selection',True)]
    assert events.index(('wait','sparse-selection'))<events.index(gpu_launches[1])
    assert events.index(('wait',M['FIT']))<events.index(gpu_launches[0])
    assert all('audit' not in str(r) and 'submit' not in str(r) for r in events)
