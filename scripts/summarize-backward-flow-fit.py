"""Verify the1000step fit and exact replay before considering graph evaluation."""
import ast
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(str(ROOT/'scripts/summarize-backward-flow-probe.py'))


def verify(result,terminal,split_bytes,probe):
    report = BASE['verify'](result,terminal,split_bytes,steps=1000,run_id='backward-flow-fit-v1')
    if (result['probe_inputs_replayed'] is not True or result['input_hashes'][:200] != probe['input_hashes']
        or result['before'] != probe['before']):
        raise ValueError('Independent full fit must exactly replay probe inputs and initial diagnostics')
    step100 = result['step100_diagnostic']
    for key in ('mae_um','endpoint_um','zero_mae_um','zero_endpoint_um','median_mae_um','median_endpoint_um'):
        if abs(step100[key]-probe['after'][key]) > 1e-6:
            raise ValueError('Step100 probe diagnostics did not reproduce')
    report.update(status='verified_full_motion_fit_not_tracking_validation',probe_inputs_replayed=True,
                  step100_errors={k:step100[k] for k in ('mae_um','endpoint_um')})
    return report


if __name__ == '__main__':
    cache = ROOT/'.biohub/cache/kernel-outputs/backward-flow-fit-v1/backward_flow_fit'
    probe = ROOT/'.biohub/cache/kernel-outputs/backward-flow-probe-v1/backward_flow_probe/outputs/result.json'
    paths = dict(result=cache/'outputs/result.json',terminal=cache/'launcher_terminal.json',probe=probe)
    report = verify(*(json.loads(paths[k].read_text()) for k in ('result','terminal')),
        (ROOT/'research/independent_real_baseline_v1_split.json').read_bytes(),json.loads(probe.read_text()))
    notebook = ROOT/'kaggle/biohub-backward-flow-fit-v1/biohub-backward-flow-fit-v1.ipynb'
    source = ''.join(json.loads(notebook.read_text())['cells'][1]['source'])
    for name,filename in [('sources','source_hashes.json'),('runtime_sources','runtime_hashes.json')]:
        assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
            and isinstance(n.targets[0],ast.Name) and n.targets[0].id == name)
        expected = {k:hashlib.sha256(v.encode()).hexdigest() for k,v in ast.literal_eval(assignment.value).items()}
        if json.loads((cache/filename).read_text()) != expected:
            raise ValueError('Immutable fit notebook source receipt mismatch')
    report['source_sha256'] = {key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in paths.items()}
    report['notebook_sha256'] = hashlib.sha256(notebook.read_bytes()).hexdigest()
    (ROOT/'reports/experiments/backward-flow-fit-v1-result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
