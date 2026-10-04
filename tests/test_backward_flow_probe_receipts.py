import copy
import hashlib
import json
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFY = runpy.run_path(str(ROOT/'scripts/summarize-backward-flow-probe.py'))['verify']


def fixture():
    split = (ROOT/'research/independent_real_baseline_v1_split.json').read_bytes()
    train = json.loads(split)['folds'][0]['train']
    diag = train[::5]
    def diagnostic(absolute,endpoint):
        rows = [dict(stem=s,n=2,absolute=absolute,endpoint=endpoint,zero_absolute=6.,zero_endpoint=4.,
            median_absolute=5.4,median_endpoint=3.6,boundary_excluded=0) for s in diag]
        counts = {k:sum(r[k] for r in rows) for k in rows[0] if k != 'stem'}
        return dict(per_movie=rows,counts=counts,mae_um=absolute/6,endpoint_um=endpoint/2,
                    zero_mae_um=1.,zero_endpoint_um=2.,median_mae_um=.9,median_endpoint_um=1.8)
    result = dict(status='completed_probe_not_tracking_candidate',steps=100,
        identity=dict(max_steps=100,seed=20260910,fitting_stems=[s for s in train if s not in diag],
            diagnostic_stems=diag,split_sha256=hashlib.sha256(split).hexdigest(),public_checkpoint_loaded=False,
            selection_opened=False,target_audit_opened=False,authorized_for_submission=False,parameters=100,fitting_windows=1000),
        checkpoint_sha256='a'*64,input_hashes=['b'*64]*200,strict_reload_identical=True,
        authorized_for_submission=False,small_fit_gate_passed=True,
        history=[dict(step=s,observed_links=2,boundary_excluded=0,loss_um=.5,grad_norm=1.,elapsed_seconds=s)
                 for s in range(1,101)],before=diagnostic(6.,4.),after=diagnostic(3.,2.))
    terminal = dict(status='completed',run_id='backward-flow-probe-v1',declared_budget_seconds=3600,
                    elapsed_seconds=200.,submission_performed=False)
    return result,terminal,split


def test_valid_probe_never_authorizes_submission():
    report = VERIFY(*fixture())
    assert report['small_fit_gate_passed'] and report['authorized_for_submission'] is False
    assert report['diagnostic_links'] == 48


@pytest.mark.parametrize('damage',['scope','controls','history','counts','gate','hashes','runtime'])
def test_tampered_receipts_rejected(damage):
    result,terminal,split = fixture()
    if damage == 'scope':
        result['identity']['fitting_stems'].append(result['identity']['diagnostic_stems'][0])
    elif damage == 'controls':
        result['after']['median_mae_um'] = .8
    elif damage == 'history':
        result['history'].pop()
    elif damage == 'counts':
        result['after']['counts']['n'] += 1
    elif damage == 'gate':
        result['small_fit_gate_passed'] = False
    elif damage == 'hashes':
        result['input_hashes'].pop()
    else:
        terminal['elapsed_seconds'] = 3601
    with pytest.raises(ValueError):
        VERIFY(result,terminal,split)


def test_full_fit_requires_actual_probe_replay():
    verify_fit = runpy.run_path(str(ROOT/'scripts/summarize-backward-flow-fit.py'))['verify']
    probe,terminal,split = fixture()
    result = copy.deepcopy(probe)
    result['steps'] = result['identity']['max_steps'] = 1000
    result['input_hashes'] = ['b'*64]*2000
    result['history'] = [dict(step=s,observed_links=2,boundary_excluded=0,loss_um=.5,grad_norm=1.,elapsed_seconds=s)
                         for s in range(1,1001)]
    result.update(probe_inputs_replayed=True,step100_diagnostic=copy.deepcopy(probe['after']))
    terminal['run_id'] = 'backward-flow-fit-v1'
    terminal['elapsed_seconds'] = 1100
    assert verify_fit(result,terminal,split,probe)['steps'] == 1000
    result['input_hashes'][5] = 'e'*64
    with pytest.raises(ValueError,match='replay'):
        verify_fit(result,terminal,split,probe)
